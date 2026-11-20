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
- 评分（:meth:`BranchStore.score_candidates`）对分支只读：接纳请求时快照
  分支当时可见的 KV 视图，并为候选部分预留临时 KV（按最长候选的 token 数），
  预留计入同一缓存上限。评分计算在锁外进行；期间预留与借用的 KV 继续计入
  占用，因此关闭分支不会提前回收被借用的 KV，其他续写也按剩余容量判定。
  评分借用不计入写时复制判定（它只是保活，不构成上下文共享）。无论成功
  失败，结束后归还预留并释放评分持有的引用。

线程模型：单进程服务，所有公开方法在同一把可重入锁内完成“检查 + 修改”，
保证每个操作原子可见，且不存在多锁获取顺序导致的死锁。评分只在接纳与
收尾两个短临界区内持锁，计算本身在锁外，因此评分期间仍可关闭/续写分支。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

import numpy as np

from . import model as model_mod


class CapacityExceeded(Exception):
    """追加整批 token 或评分临时预留会超出缓存上限。"""


@dataclass(frozen=True)
class CandidateScore:
    """单条候选的评分结果：输入序号 + 逐 token 条件对数概率均值。"""

    index: int
    score: float


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
        # 评分临时 KV 的预留量：与 _occupied 共用同一上限，但尚未产生物理
        # KvState；接纳评分请求时按最长候选的 token 数一次性预留。
        self._reserved = 0
        # 评分期间额外持有的 KV 引用，按物理 KV id 记录持有次数；分支被
        # 关闭后仍保持存活，计入 live KV 统计，但不重复计入占用。
        self._scoring_holds: dict[int, list] = {}  # id -> [KvState, hold_count]
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 元数据

    @property
    def max_cached_tokens(self) -> int:
        return self._max_cached_tokens

    def _used(self) -> int:
        """已占用 + 已预留；所有容量判定都以此为准。"""
        return self._occupied + self._reserved

    def usage(self) -> dict:
        with self._lock:
            return {
                "occupied_tokens": self._occupied,
                "reserved_tokens": self._reserved,
                "total_tokens": self._used(),
                "max_cached_tokens": self._max_cached_tokens,
                "branch_count": len(self._branches),
                "kv_state_count": len(self._live_kv()),
            }

    def _live_kv(self) -> list[KvState]:
        seen: dict[int, KvState] = {}
        for b in self._branches.values():
            if b.kv is not None and id(b.kv) not in seen:
                seen[id(b.kv)] = b.kv
        for key, (kv, _count) in self._scoring_holds.items():
            seen.setdefault(key, kv)
        return list(seen.values())

    def _hold_kv(self, kv: KvState) -> KvState:
        """评分借用一份 KV：引用计数 +1，并登记在评分持有表中。"""
        assert kv.K is not None  # 调用方在持锁状态下确认过
        self._retain(kv)
        entry = self._scoring_holds.get(id(kv))
        if entry is None:
            self._scoring_holds[id(kv)] = [kv, 1]
        else:
            entry[1] += 1
        return kv

    def _unhold_kv(self, kv: KvState) -> None:
        """归还评分借用：注销持有登记并释放引用（最后一个引用则回收）。"""
        entry = self._scoring_holds[id(kv)]
        entry[1] -= 1
        if entry[1] <= 0:
            del self._scoring_holds[id(kv)]
        self._release(kv)

    def _branch_refcount(self, kv: KvState) -> int:
        """仅统计分支持有的引用数，忽略评分借用。

        评分借用只是“保持存活”，不构成上下文共享；否则评分期间对该分支
        的续写会被误判为需要写时复制，凭空多占缓存。
        """
        hold = self._scoring_holds.get(id(kv))
        holds = hold[1] if hold is not None else 0
        return kv.refcount - holds

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
                # 预检必须计入评分临时预留（预留与占用共用同一上限）。
                used = self._used()
                if used + len(token_ids) > self._max_cached_tokens:
                    raise CapacityExceeded(
                        f"creating context with {len(token_ids)} tokens exceeds "
                        f"limit {self._max_cached_tokens} (occupied {self._occupied}, "
                        f"reserved {self._reserved})"
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
                projected = self._used() + batch
            elif self._branch_refcount(current) > 1:
                projected = self._used() + view + batch
            else:
                projected = self._used() + batch
            if projected > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"appending {batch} tokens would raise usage "
                    f"{self._used()} -> {projected}, limit {self._max_cached_tokens} "
                    f"(occupied {self._occupied}, reserved {self._reserved})"
                )

            # ---- 预检通过，执行追加 ----
            logits = None
            # 评分借用的引用不计入“共享”判定：只有其他分支持有才算 COW。
            exclusive = self._branch_refcount(current) <= 1 if current is not None else True
            if current is None:
                K, V = model_mod.init_kv(model["d_model"])
                for tid in token_ids:
                    logits, K, V = model_mod.step(model, tid, K, V)
                branch.kv = self._new_kv(K, V, length=batch)
            else:
                if not exclusive:
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

    # ------------------------------------------------------------------- 评分

    def score_candidates(
        self, branch_id: str, candidates: list[list[int]]
    ) -> list[CandidateScore]:
        """对分支只读评分：返回每条候选的平均逐 token 条件对数概率，降序排列。

        语义
        ----
        - 接纳请求时快照分支当时可见的 KV 视图；评分只用这份快照，后续对
          分支的追加/关闭都不影响本次结果。
        - 候选部分使用临时 KV：按最长候选的 token 数预留，预留与已有 KV
          共用同一上限，余量不足时在修改任何状态之前抛 :class:`CapacityExceeded`。
        - 评分计算在锁外进行；期间预留继续计入占用，借用的上下文 KV 通过
          引用计数保持存活（分支被关闭也不会提前回收），其他续写按剩余容量判定。
        - 无论成功或异常，收尾时都归还预留并释放借用的 KV 引用。
        """
        if not candidates or any(not cand for cand in candidates):
            raise ValueError("candidates must be non-empty token sequences")
        vocab = int(self._model["vocab_size"])
        for i, cand in enumerate(candidates):
            for tid in cand:
                if tid < 0 or tid >= vocab:
                    raise ValueError(
                        f"candidate {i}: token id {tid} out of range [0, {vocab - 1}]"
                    )

        # ---- 接纳（短临界区）：校验、容量预检、快照并借用 KV ----
        with self._lock:
            branch = self._get_branch(branch_id)
            if not branch.token_ids or branch.kv is None:
                raise ValueError("branch context must contain at least one token")

            max_len = max(len(cand) for cand in candidates)
            if self._used() + max_len > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"scoring {max_len} candidate tokens would raise usage "
                    f"{self._used()} -> {self._used() + max_len}, "
                    f"limit {self._max_cached_tokens} (occupied {self._occupied}, "
                    f"reserved {self._reserved})"
                )

            context_tokens = list(branch.token_ids)
            borrowed = self._hold_kv(branch.kv)  # 保持存活，关闭分支也不回收
            # 在持锁状态下取定视图切片：之后即便分支被追加（原地增长会
            # 重新绑定 K/V 数组）或被关闭，本次评分读取的仍是接纳时的前缀。
            view = len(context_tokens)
            base_K = borrowed.K[:view]
            base_V = borrowed.V[:view]
            self._reserved += max_len

        # ---- 计算（锁外）：其余请求可继续关闭/续写分支 ----
        try:
            model = self._model
            # 上下文中最后一个 token 的 next-token 分布，是每条候选的首个
            # token 条件概率的来源。
            context_logits = model_mod.logits_from_kv(
                model, context_tokens[-1], base_K, base_V
            )
            results: list[CandidateScore] = []
            for index, cand in enumerate(candidates):
                logits = context_logits
                K, V = base_K, base_V
                total = 0.0
                for tid in cand:
                    total += float(model_mod.log_softmax(logits)[tid])
                    logits, K, V = model_mod.step(model, tid, K, V)
                results.append(CandidateScore(index=index, score=total / len(cand)))
            # 分数降序；同分保持输入顺序（稳定排序 + 序号键）。
            results.sort(key=lambda item: (-item.score, item.index))
            return results
        finally:
            # ---- 收尾（短临界区）：归还预留，释放评分持有的 KV 引用 ----
            with self._lock:
                self._reserved -= max_len
                self._unhold_kv(borrowed)

    # ------------------------------------------------------------------- 调试

    def debug_snapshot(self) -> dict:
        """返回内部状态快照，供测试与排障核对引用计数/占用量不变量。"""
        with self._lock:
            live = self._live_kv()
            counted = sum(kv.length for kv in live)
            return {
                "usage": self.usage(),
                "occupied_matches_live_kv": counted == self._occupied,
                "scoring_holds": sum(count for _kv, count in self._scoring_holds.values()),
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
