"""存储层面：分叉复用 KV、引用计数释放、写时复制、截断分叉、超限原子拒绝。"""

import threading

import numpy as np
import pytest

from app.model import build_model, forward_full
from app.store import BranchStore, CapacityExceeded


MODEL = build_model()


def _assert_logits_match(store, branch):
    logits = store._logits_at(branch)
    full = forward_full(MODEL, branch.token_ids)
    np.testing.assert_allclose(logits, full, rtol=1e-5, atol=1e-6)


def _assert_invariants(store):
    snap = store.debug_snapshot()
    assert snap["occupied_matches_live_kv"]
    for b in snap["branches"]:
        assert b["view_length"] <= b["kv_length"]
        assert b["kv_refcount"] >= 1
    assert snap["usage"]["occupied_tokens"] <= snap["usage"]["max_cached_tokens"]


def test_create_context_and_usage():
    store = BranchStore(MODEL, max_cached_tokens=128)
    branch, logits = store.create_branch([1, 2, 3, 4])
    assert branch.token_ids == [1, 2, 3, 4]
    assert store.usage()["occupied_tokens"] == 4
    _assert_logits_match(store, branch)
    _assert_invariants(store)


def test_fork_shares_kv_without_extra_occupancy():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4])
    occupied_before = store.usage()["occupied_tokens"]

    child, child_logits = store.fork(parent.id, None)
    # 零拷贝：占用量不变，两个分支指向同一个物理 KvState，引用计数为 2。
    assert store.usage()["occupied_tokens"] == occupied_before
    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[parent.id]["kv_id"] == snap[child.id]["kv_id"]
    assert snap[child.id]["kv_refcount"] == 2
    np.testing.assert_allclose(
        child_logits, forward_full(MODEL, [1, 2, 3, 4]), rtol=1e-5, atol=1e-6
    )
    _assert_invariants(store)


def test_fork_at_any_token_offset_and_recompute_consistency():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([5, 6, 7, 8, 9])

    for offset in range(1, 6):
        child, logits = store.fork(parent.id, offset)
        assert child.token_ids == [5, 6, 7, 8, 9][:offset]
        np.testing.assert_allclose(
            logits,
            forward_full(MODEL, [5, 6, 7, 8, 9][:offset]),
            rtol=1e-5,
            atol=1e-6,
        )
        _assert_logits_match(store, child)
    _assert_invariants(store)


def test_append_on_shared_kv_uses_copy_on_write():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3])
    child, _ = store.fork(parent.id, None)

    # 在子分支追加：父分支不受影响（COW），各自上下文独立。
    _, logits = store.append(child.id, [4, 5])
    assert child.token_ids == [1, 2, 3, 4, 5]
    assert parent.token_ids == [1, 2, 3]
    _assert_logits_match(store, child)
    _assert_logits_match(store, parent)

    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[parent.id]["kv_id"] != snap[child.id]["kv_id"]
    assert snap[parent.id]["kv_refcount"] == 1
    assert snap[child.id]["kv_refcount"] == 1
    # 两份物理 KV：长度 3 与 5。
    assert store.usage()["occupied_tokens"] == 8
    _assert_invariants(store)


def test_truncated_fork_then_append():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4, 5])
    child, _ = store.fork(parent.id, offset=2)  # 只继承 [1, 2]

    # 截断分叉不增加占用（仍只有一份物理 KV）。
    assert store.usage()["occupied_tokens"] == 5
    _assert_logits_match(store, child)

    _, logits = store.append(child.id, [9])
    assert child.token_ids == [1, 2, 9]
    assert parent.token_ids == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(
        logits, forward_full(MODEL, [1, 2, 9]), rtol=1e-5, atol=1e-6
    )
    _assert_logits_match(store, child)
    _assert_logits_match(store, parent)
    # COW 出新 KV（视图长度 2 + 追加 1 = 3），父 KV 仍为 5。
    assert store.usage()["occupied_tokens"] == 8
    _assert_invariants(store)


def test_close_releases_kv_after_last_reference():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4])
    child, _ = store.fork(parent.id, None)

    # 关闭父分支：KV 仍被子分支引用，占用不释放。
    store.close(parent.id)
    assert store.usage()["occupied_tokens"] == 4
    assert child.kv is not None and child.kv.refcount == 1

    # 关闭最后一个引用：KV 释放，占用归零。
    store.close(child.id)
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["branch_count"] == 0
    assert store.debug_snapshot()["occupied_matches_live_kv"]


def test_capacity_exceeded_rejects_whole_batch_atomically():
    store = BranchStore(MODEL, max_cached_tokens=10)
    branch, _ = store.create_branch([1, 2, 3])  # 占用 3
    tokens_before = list(branch.token_ids)
    occupied_before = store.usage()["occupied_tokens"]

    # 追加 8 个会到 11 > 10：整批拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(branch.id, [4, 5, 6, 7, 8, 9, 10, 11])

    # 原分支与占用量保持原值。
    assert branch.token_ids == tokens_before
    assert store.usage()["occupied_tokens"] == occupied_before == 3
    _assert_invariants(store)

    # 恰好到上限的批次成功。
    store.append(branch.id, [4, 5, 6, 7, 8, 9, 10])
    assert store.usage()["occupied_tokens"] == 10
    _assert_logits_match(store, branch)

    # 再追加一个也被拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(branch.id, [0])
    assert store.usage()["occupied_tokens"] == 10
    _assert_invariants(store)


def test_capacity_exceeded_with_cow_counts_new_state():
    """共享 KV 上追加：容量预检必须把 COW 产生的新 KV 长度计入。"""
    store = BranchStore(MODEL, max_cached_tokens=10)
    parent, _ = store.create_branch([1, 2, 3, 4, 5, 6])  # 占用 6
    child, _ = store.fork(parent.id, None)  # 共享，仍为 6

    # 子分支追加 5 个：COW 后新 KV 为 6+5=11，占用 6+11=17 > 10 -> 拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(child.id, [7, 8, 9, 10, 11])
    # 原子失败：仍是共享状态，占用不变。
    assert store.usage()["occupied_tokens"] == 6
    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[parent.id]["kv_id"] == snap[child.id]["kv_id"]

    # 追加 4 个：新 KV 长度 10，总占用 6+10=16 > 10 仍拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(child.id, [7, 8, 9, 10])
    assert store.usage()["occupied_tokens"] == 6

    # 关闭父分支后独占，追加 4 个：占用 6+4=10，成功。
    store.close(parent.id)
    store.append(child.id, [7, 8, 9, 10])
    assert store.usage()["occupied_tokens"] == 10
    _assert_logits_match(store, child)
    _assert_invariants(store)


def test_create_context_over_capacity_rejected():
    store = BranchStore(MODEL, max_cached_tokens=4)
    with pytest.raises(CapacityExceeded):
        store.create_branch([1, 2, 3, 4, 5])
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["branch_count"] == 0


def test_concurrent_appends_to_different_branches_stay_isolated():
    store = BranchStore(MODEL, max_cached_tokens=10_000)
    root, _ = store.create_branch([1, 2])
    children = [store.fork(root.id, None)[0] for _ in range(8)]
    store.close(root.id)  # 子分支各自持有引用

    errors: list[Exception] = []

    def worker(branch, token):
        try:
            for _ in range(20):
                store.append(branch.id, [token])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(b, i + 10))
        for i, b in enumerate(children)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i, b in enumerate(children):
        token = i + 10
        assert b.token_ids == [1, 2] + [token] * 20
        _assert_logits_match(store, b)
    _assert_invariants(store)


def test_concurrent_close_and_append_different_branches():
    """一个分支关闭、另一个分支追加，互不干扰。"""
    store = BranchStore(MODEL, max_cached_tokens=10_000)
    a, _ = store.create_branch([1, 2, 3])
    b, _ = store.create_branch([4, 5, 6])
    errors: list[Exception] = []

    def closer():
        try:
            store.close(a.id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def appender():
        try:
            for _ in range(50):
                store.append(b.id, [7])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=closer)
    t2 = threading.Thread(target=appender)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert errors == []
    assert b.token_ids == [4, 5, 6] + [7] * 50
    _assert_logits_match(store, b)
    assert store.usage()["occupied_tokens"] == 53
    _assert_invariants(store)


def test_concurrent_mixed_fork_append_close():
    """多线程随机混合 fork/append/close，全程不变量必须成立。"""
    import random

    store = BranchStore(MODEL, max_cached_tokens=100_000)
    root, _ = store.create_branch([1, 2])
    errors: list[Exception] = []

    def worker(seed):
        rng = random.Random(seed)
        parent = root.id
        for _ in range(30):
            try:
                op = rng.choice(["fork", "append", "close", "fork_append"])
                if op == "fork":
                    child, _ = store.fork(parent, None)
                    parent = child.id
                elif op == "fork_append":
                    child, _ = store.fork(parent, rng.choice([None, 1, 2]))
                    store.append(child.id, [rng.randrange(MODEL["vocab_size"])])
                    parent = child.id
                elif op == "append":
                    store.append(parent, [rng.randrange(MODEL["vocab_size"])])
                else:
                    store.close(parent)
                    child, _ = store.fork(root.id, None)
                    parent = child.id
            except KeyError:
                # 目标分支已被其他线程关闭：从根重新开始；根也没了就新建。
                try:
                    child, _ = store.fork(root.id, None)
                except KeyError:
                    child, _ = store.create_branch([1, 2])
                parent = child.id
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    _assert_invariants(store)
    # 存活分支的 logits 仍与整段重算一致（上下文未被其他线程污染）。
    for branch in list(store.debug_snapshot()["branches"]):
        live = store.get_branch(branch["id"])
        if live.token_ids:
            _assert_logits_match(store, live)


def test_unknown_branch_raises_keyerror():
    store = BranchStore(MODEL, max_cached_tokens=128)
    with pytest.raises(KeyError):
        store.append("nope", [1])
    with pytest.raises(KeyError):
        store.close("nope")
    with pytest.raises(KeyError):
        store.fork("nope", None)
