"""分支与 KV 缓存存储。

设计要点
--------
- 每个分支（:class:`Branch`）持有自己的 token 序列，并指向一个 :class:`KvState`。
- :class:`KvState` 是一份物理 K/V 数据（NumPy 数组），通过引用计数在分支间
  共享；分叉是零拷贝的——新分支直接指向同一份 KV，只做引用计数 +1。
- 追加 token 时若当前 KV 被多个分支引用，则先做写时复制（copy-on-write），
  保证“各分支只使用自己的上下文”，追加不会影响共享同一份 KV 的其他分支。
- 分支可以从任一已有 token 处分叉（offset <= 当前长度）：截断 fork 复用底层
  物理数组，只按自己的视图长度读取，不产生新的缓存占用。
- 容量按“实际保存的 token 位置数”计量，即所有仍存活的物理 KvState 的
  有效长度之和；共享不重复计数，截断不占额外空间。
- 容量超限的提交在改动任何状态之前完成检查，整批拒绝（原子失败），
  原分支内容与全局占用量保持不变。
- 每份 KV 在最后一个引用它的分支关闭后释放（引用计数归零即删除）。

线程模型：单进程服务，所有公开方法在同一把可重入锁内完成“检查 + 修改”，
保证每个操作原子可见，且不存在多锁获取顺序导致的死锁。每步仅为小规模
NumPy 运算，串行化的开销可以忽略。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

import numpy as np

from . import model as model_mod


class CapacityExceeded(Exception):
    """追加整批 token 后会超出缓存上限。"""


@dataclass
class KvState:
    """一份物理 K/V 缓存，可被多个分支共享。"""

    K: np.ndarray | None
    V: np.ndarray | None
    length: int  # 物理数组的有效长度（截断分叉的分支只读取其中的前若干行）
    refcount: int = 0


@dataclass
class Branch:
    id: str
    token_ids: list[int]
    kv: KvState | None = None


class BranchStore:
    def __init__(self, model: dict, max_cached_tokens: int):
        self._model = model
        self._max_cached_tokens = int(max_cached_tokens)
        self._branches: dict[str, Branch] = {}
        self._occupied = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 元数据

    @property
    def max_cached_tokens(self) -> int:
        return self._max_cached_tokens

    def usage(self) -> dict:
        with self._lock:
            return {
                "occupied_tokens": self._occupied,
                "max_cached_tokens": self._max_cached_tokens,
                "branch_count": len(self._branches),
                "kv_state_count": len(self._live_kv()),
            }

    def _live_kv(self) -> list[KvState]:
        seen: dict[int, KvState] = {}
        for b in self._branches.values():
            if b.kv is not None and id(b.kv) not in seen:
                seen[id(b.kv)] = b.kv
        return list(seen.values())

    def _get_branch(self, branch_id: str) -> Branch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise KeyError(branch_id)
        return branch

    def get_branch(self, branch_id: str) -> Branch:
        """按 ID 取分支，不存在则抛 KeyError。"""
        with self._lock:
            return self._get_branch(branch_id)

    # ------------------------------------------------------------- 引用计数管理

    def _retain(self, kv: KvState) -> KvState:
        kv.refcount += 1
        return kv

    def _release(self, kv: KvState) -> None:
        """释放一次引用；最后一个引用释放后删除该 KV 并回收占用量。"""
        kv.refcount -= 1
        if kv.refcount <= 0:
            # 占用量按有效长度计量；被截断视图“隐藏”的行本就不计入占用。
            self._occupied -= kv.length
            # 去掉对大数组的引用，使其可被 NumPy/GC 回收。
            kv.K = None
            kv.V = None
            kv.length = 0

    def _new_kv(self, K: np.ndarray, V: np.ndarray, length: int) -> KvState:
        kv = KvState(K=K, V=V, length=length, refcount=1)
        self._occupied += length
        return kv

    # --------------------------------------------------------------- 分支生命周期

    def create_branch(
        self, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray | None]:
        """提交一批 token 建立上下文，返回新分支与最后位置的 next-token logits。"""
        with self._lock:
            branch = Branch(id=uuid.uuid4().hex, token_ids=list(token_ids))
            logits = None
            if token_ids:
                # 建立上下文等价于从空 KV 开始整批追加；先做容量预检。
                if self._occupied + len(token_ids) > self._max_cached_tokens:
                    raise CapacityExceeded(
                        f"creating context with {len(token_ids)} tokens exceeds "
                        f"limit {self._max_cached_tokens} (occupied {self._occupied})"
                    )
                K, V = model_mod.init_kv(self._model["d_model"])
                for tid in token_ids:
                    logits, K, V = model_mod.step(self._model, tid, K, V)
                branch.kv = self._new_kv(K, V, length=len(token_ids))
            self._branches[branch.id] = branch
            return branch, logits

    def fork(
        self, parent_id: str, offset: int | None
    ) -> tuple[Branch, np.ndarray | None]:
        """从父分支的任一已有 token 处分叉，零拷贝继承 KV。

        offset 为保留的 token 数：新分支上下文 = 父分支前 offset 个 token。
        省略 offset 时继承完整上下文。
        """
        with self._lock:
            parent = self._get_branch(parent_id)
            if offset is None:
                offset = len(parent.token_ids)
            if offset < 0 or offset > len(parent.token_ids):
                raise ValueError(
                    f"offset {offset} out of range [0, {len(parent.token_ids)}]"
                )

            child = Branch(
                id=uuid.uuid4().hex, token_ids=list(parent.token_ids[:offset])
            )
            if parent.kv is not None and offset > 0:
                # 零拷贝：共享底层物理数组，分支只按自己的视图长度读取。
                child.kv = self._retain(parent.kv)
                assert offset <= parent.kv.length
            logits = self._logits_at(child)
            self._branches[child.id] = child
            return child, logits

    def append(
        self, branch_id: str, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray]:
        """向分支批量追加 token；超限时整批拒绝，状态保持不变。"""
        with self._lock:
            if not token_ids:
                raise ValueError("token batch must not be empty")

            branch = self._get_branch(branch_id)
            model = self._model
            batch = len(token_ids)
            # 分支自身上下文长度；共享 KV 的物理长度可能更大（截断分叉）。
            view = len(branch.token_ids)

            # ---- 容量预检（在任何状态修改之前）----
            # 共享 KV：写时复制出新状态，初始占 view 个位置（旧状态仍被
            #   其他分支引用，占用保持不变），随后再增长 batch。
            # 独占 KV：原地增长 batch 个位置（视图短于物理数组时先裁剪，
            #   只会减少占用，因此 base + batch 是安全上界）。
            # 空上下文：新建 KV，占 batch 个位置。
            current = branch.kv
            if current is None:
                projected = self._occupied + batch
            elif current.refcount > 1:
                projected = self._occupied + view + batch
            else:
                projected = self._occupied + batch
            if projected > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"appending {batch} tokens would raise occupancy "
                    f"{self._occupied} -> {projected}, limit {self._max_cached_tokens}"
                )

            # ---- 预检通过，执行追加 ----
            logits = None
            if current is None:
                K, V = model_mod.init_kv(model["d_model"])
                for tid in token_ids:
                    logits, K, V = model_mod.step(model, tid, K, V)
                branch.kv = self._new_kv(K, V, length=batch)
            else:
                if current.refcount > 1:
                    # 写时复制：只复制分支视图内的有效部分。
                    K = current.K[:view].copy()
                    V = current.V[:view].copy()
                    old = current
                    branch.kv = self._new_kv(K, V, length=view)
                    self._release(old)  # 共享中，释放后旧状态仍存活
                elif current.length > view:
                    # 独占但来自截断分叉：先裁掉视图之外的陈旧位置。
                    current.K = current.K[:view].copy()
                    current.V = current.V[:view].copy()
                    self._occupied -= current.length - view
                    current.length = view
                kv = branch.kv
                K, V = kv.K, kv.V
                for tid in token_ids:
                    logits, K, V = model_mod.step(model, tid, K, V)
                kv.K, kv.V = K, V
                kv.length = K.shape[0]
                self._occupied += batch

            branch.token_ids.extend(token_ids)
            assert logits is not None
            return branch, logits

    def close(self, branch_id: str) -> None:
        """关闭分支；若它是某份 KV 的最后一个引用，则该 KV 立即释放。"""
        with self._lock:
            branch = self._branches.pop(branch_id, None)
            if branch is None:
                raise KeyError(branch_id)
            if branch.kv is not None:
                self._release(branch.kv)
                branch.kv = None

    # -------------------------------------------------------------- logits 回放

    def _logits_at(self, branch: Branch) -> np.ndarray | None:
        """计算分支当前最后一个 token 的 next-token logits。

        基于继承的 KV 直接回放最后一步，结果与对整段 token 重算一致。
        """
        if not branch.token_ids or branch.kv is None:
            return None
        kv = branch.kv
        view = len(branch.token_ids)
        assert view <= kv.length
        last_id = branch.token_ids[-1]
        return model_mod.logits_from_kv(
            self._model, last_id, kv.K[:view], kv.V[:view]
        )

    # ------------------------------------------------------------------- 调试

    def debug_snapshot(self) -> dict:
        """返回内部状态快照，供测试与排障核对引用计数/占用量不变量。"""
        with self._lock:
            live = self._live_kv()
            counted = sum(kv.length for kv in live)
            return {
                "usage": self.usage(),
                "occupied_matches_live_kv": counted == self._occupied,
                "branches": [
                    {
                        "id": b.id,
                        "tokens": list(b.token_ids),
                        "view_length": len(b.token_ids),
                        "kv_length": b.kv.length if b.kv is not None else 0,
                        "kv_refcount": b.kv.refcount if b.kv is not None else 0,
                        "kv_id": id(b.kv) if b.kv is not None else None,
                    }
                    for b in self._branches.values()
                ],
            }
