"""固定权重的单层因果注意力模型（纯 NumPy 实现）。

模型结构：token embedding -> 单头因果自注意力 -> 残差 + 输出投影 -> logits。
权重在进程启动时由固定随机种子生成，之后不再变化。

同时提供两种前向路径：
- :func:`forward_full`：对整段 token 重新计算，作为一致性基准；
- :func:`init_kv` / :func:`step`：增量解码，复用历史 K/V 缓存。
两条路径对“最后一个位置”的 next-token logits 数值一致。
"""

from __future__ import annotations

import numpy as np

DEFAULT_VOCAB_SIZE = 64
DEFAULT_D_MODEL = 16
DEFAULT_SEED = 20261120


def build_model(
    seed: int = DEFAULT_SEED,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    d_model: int = DEFAULT_D_MODEL,
) -> dict:
    """根据固定种子构造模型权重（float32）。"""
    rng = np.random.default_rng(seed)
    scale = np.float32(0.1)

    def randn(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "emb": randn((vocab_size, d_model)),
        "wq": randn((d_model, d_model)),
        "wk": randn((d_model, d_model)),
        "wv": randn((d_model, d_model)),
        "wo": randn((d_model, d_model)),
        "wout": randn((d_model, vocab_size)),
        "bout": np.zeros(vocab_size, dtype=np.float32),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def init_kv(d_model: int) -> tuple[np.ndarray, np.ndarray]:
    """返回空的 K/V 缓存，形状分别为 [0, d_model]。"""
    empty = np.zeros((0, d_model), dtype=np.float32)
    return empty, empty.copy()


def logits_from_kv(
    model: dict, token_id: int, K: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """给定已有 K/V（最后一行必须是 token_id 自身的 k/v），计算 next-token logits。

    增量解码与分叉点回放共用本函数，保证两条路径计算完全相同。
    """
    d = np.float32(model["d_model"])
    x = model["emb"][token_id]  # [D]

    # 当前 token 是序列最后一个位置，K/V 中的所有位置都在它之前（含自身），
    # 因此因果掩码天然满足，不需要再 mask。
    q = x @ model["wq"]  # [D]
    scores = (K @ q) / np.sqrt(d)  # [t]
    weights = _softmax(scores)  # [t]
    context = weights @ V  # [D]
    hidden = x + context @ model["wo"]  # 残差连接
    return (hidden @ model["wout"] + model["bout"]).astype(np.float32)


def step(
    model: dict, token_id: int, K: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """追加一个 token，返回 (next-token logits, 新 K, 新 V)。"""
    x = model["emb"][token_id][None, :]  # [1, D]
    k = x @ model["wk"]  # [1, D]
    v = x @ model["wv"]
    K_new = np.concatenate([K, k], axis=0)
    V_new = np.concatenate([V, v], axis=0)
    logits = logits_from_kv(model, token_id, K_new, V_new)
    return logits, K_new, V_new


def forward_full(model: dict, token_ids: list[int]) -> np.ndarray:
    """对整段 token 做一次性因果注意力前向，返回最后位置的 next-token logits。"""
    if not token_ids:
        raise ValueError("forward_full requires at least one token")

    ids = np.asarray(token_ids, dtype=np.int64)
    x = model["emb"][ids]  # [t, D]
    q_all = x @ model["wq"]
    k_all = x @ model["wk"]
    v_all = x @ model["wv"]

    t = x.shape[0]
    d = np.float32(model["d_model"])
    scores = (q_all @ k_all.T) / np.sqrt(d)  # [t, t]

    # 因果掩码：禁止关注未来位置
    causal_mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores = np.where(causal_mask, np.float32(-1e9), scores)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    context = weights @ v_all  # [t, D]
    hidden = x + context @ model["wo"]
    logits = hidden[-1] @ model["wout"] + model["bout"]
    return logits.astype(np.float32)
