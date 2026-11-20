"""模型层面：增量解码与整段重算的 next-token logits 必须一致。"""

import numpy as np

from app.model import build_model, forward_full, init_kv, logits_from_kv, step


MODEL = build_model()


def test_single_token_matches_full():
    logits, _, _ = step(MODEL, 3, *init_kv(MODEL["d_model"]))
    full = forward_full(MODEL, [3])
    np.testing.assert_allclose(logits, full, rtol=1e-5, atol=1e-6)


def test_incremental_equals_full_recompute_long_sequence():
    rng = np.random.default_rng(7)
    ids = rng.integers(0, MODEL["vocab_size"], size=40).tolist()

    K, V = init_kv(MODEL["d_model"])
    inc_logits = None
    for tid in ids:
        inc_logits, K, V = step(MODEL, tid, K, V)

    full_logits = forward_full(MODEL, ids)
    np.testing.assert_allclose(inc_logits, full_logits, rtol=1e-5, atol=1e-6)

    # 回放最后一步（分叉路径所用函数）也必须一致。
    replay = logits_from_kv(MODEL, ids[-1], K, V)
    np.testing.assert_allclose(replay, full_logits, rtol=1e-5, atol=1e-6)


def test_different_prefixes_diverge():
    """不同上下文产生不同 logits（确认注意力确实看到了历史）。"""
    a = forward_full(MODEL, [1, 2, 3])
    b = forward_full(MODEL, [4, 5, 3])  # 末位相同、前缀不同
    assert not np.allclose(a, b, atol=1e-4)


def test_same_prefix_same_logits():
    a = forward_full(MODEL, [10, 20, 30])
    K, V = init_kv(MODEL["d_model"])
    for tid in [10, 20, 30]:
        logits, K, V = step(MODEL, tid, K, V)
    np.testing.assert_allclose(logits, a, rtol=1e-5, atol=1e-6)
