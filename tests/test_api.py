"""HTTP API 端到端测试（Starlette TestClient，单进程内真实 ASGI 调用）。"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import BranchStore
from app import main as main_module


@pytest.fixture()
def client():
    # 每个测试使用全新的 store，容量固定为 12，便于构造超限场景。
    main_module.STORE = BranchStore(main_module.MODEL, max_cached_tokens=12)
    return TestClient(app)


def test_health_and_docs(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_create_fork_append_close_flow_and_logits_consistency(client):
    # 1) 建立上下文
    r = client.post("/branches", json={"token_ids": [1, 2, 3, 4]})
    assert r.status_code == 201
    body = r.json()
    root_id = body["branch_id"]
    assert body["length"] == 4
    assert body["usage"]["occupied_tokens"] == 4
    # 服务端返回的增量 logits 与整段重算 logits 一致
    np.testing.assert_allclose(
        body["logits"], body["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )

    # 2) 完整分叉：零拷贝，占用不变，logits 一致
    r = client.post(f"/branches/{root_id}/fork", json={})
    assert r.status_code == 200
    child = r.json()
    assert child["usage"]["occupied_tokens"] == 4
    np.testing.assert_allclose(
        child["logits"], child["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )

    # 3) 从第 2 个 token 处截断分叉
    r = client.post(f"/branches/{root_id}/fork", json={"offset": 2})
    assert r.status_code == 200
    truncated = r.json()
    assert truncated["token_ids"] == [1, 2]
    assert truncated["usage"]["occupied_tokens"] == 4  # 共享，不增加占用
    np.testing.assert_allclose(
        truncated["logits"], truncated["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )

    # 4) 子分支批量追加（COW），父分支不受影响
    r = client.post(f"/branches/{child['branch_id']}/tokens", json={"token_ids": [5, 6]})
    assert r.status_code == 200
    grown = r.json()
    assert grown["token_ids"] == [1, 2, 3, 4, 5, 6]
    assert grown["usage"]["occupied_tokens"] == 10  # 父 KV 4 + 子 KV 6
    np.testing.assert_allclose(
        grown["logits"], grown["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )

    root_now = client.get("/branches").json()
    root_entry = next(b for b in root_now["branches"] if b["id"] == root_id)
    assert root_entry["tokens"] == [1, 2, 3, 4]

    # 5) 关闭父分支：截断/完整子分支仍存活，KV 不被提前释放
    r = client.delete(f"/branches/{root_id}")
    assert r.status_code == 200
    assert r.json()["usage"]["occupied_tokens"] == 10

    # 6) 关闭所有分支后占用归零
    for bid in (child["branch_id"], truncated["branch_id"]):
        client.delete(f"/branches/{bid}")
    assert client.get("/stats").json()["occupied_tokens"] == 0


def test_capacity_exceeded_returns_413_and_preserves_state(client):
    r = client.post("/branches", json={"token_ids": [1, 2, 3]})
    branch_id = r.json()["branch_id"]
    assert r.json()["usage"]["occupied_tokens"] == 3

    # 12 - 3 = 9 的余量；提交 10 个 token 超限
    r = client.post(
        f"/branches/{branch_id}/tokens",
        json={"token_ids": list(range(4, 14))},
    )
    assert r.status_code == 413
    err = r.json()
    assert err["error"] == "capacity_exceeded"
    # 失败响应同样返回当前用量，且用量未变
    assert err["usage"]["occupied_tokens"] == 3

    state = client.get("/branches").json()
    entry = next(b for b in state["branches"] if b["id"] == branch_id)
    assert entry["tokens"] == [1, 2, 3]

    # 恰好 9 个的批次成功，达到上限
    r = client.post(
        f"/branches/{branch_id}/tokens",
        json={"token_ids": list(range(4, 13))},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["occupied_tokens"] == 12


def test_validation_errors_and_404(client):
    # 空追加批次 -> 422
    r = client.post("/branches", json={"token_ids": [1]})
    bid = r.json()["branch_id"]
    assert client.post(f"/branches/{bid}/tokens", json={"token_ids": []}).status_code == 422

    # 越界 token -> 422
    assert client.post("/branches", json={"token_ids": [main_module.VOCAB_SIZE]}).status_code == 422
    assert client.post("/branches", json={"token_ids": [-1]}).status_code == 422

    # 不存在的分支
    assert client.post("/branches/missing/fork", json={}).status_code == 404
    assert client.post("/branches/missing/tokens", json={"token_ids": [1]}).status_code == 404
    assert client.delete("/branches/missing").status_code == 404

    # 非法 offset
    r = client.post(f"/branches/{bid}/fork", json={"offset": 99})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_offset"


def test_empty_context_branch(client):
    r = client.post("/branches", json={"token_ids": []})
    assert r.status_code == 201
    body = r.json()
    assert body["length"] == 0
    assert body["logits"] is None
    assert body["full_recompute_logits"] is None
    assert body["usage"]["occupied_tokens"] == 0

    # 空上下文分支可以继续追加
    r = client.post(f"/branches/{body['branch_id']}/tokens", json={"token_ids": [7, 8]})
    assert r.status_code == 200
    out = r.json()
    assert out["token_ids"] == [7, 8]
    np.testing.assert_allclose(
        out["logits"], out["full_recompute_logits"], rtol=1e-5, atol=1e-6
    )


def test_fork_from_empty_context(client):
    r = client.post("/branches", json={"token_ids": []})
    bid = r.json()["branch_id"]
    r = client.post(f"/branches/{bid}/fork", json={})
    assert r.status_code == 200
    assert r.json()["length"] == 0


# --------------------------------------------------------------------- 评分


def test_score_sorts_candidates_descending(client):
    r = client.post("/branches", json={"token_ids": [1, 2, 3, 4]})
    bid = r.json()["branch_id"]

    # 同一候选重复两次：结果里应保持输入顺序（0 在 3 之前）。
    candidates = [[5], [7, 8], [9, 10, 11], [5]]
    r = client.post(
        f"/branches/{bid}/score",
        json={"branch_id": bid, "candidates": candidates},
    )
    assert r.status_code == 200
    body = r.json()
    indices = [entry["index"] for entry in body["scores"]]
    scores = [entry["score"] for entry in body["scores"]]
    assert sorted(indices) == [0, 1, 2, 3]
    assert scores == sorted(scores, reverse=True)
    assert indices.index(0) < indices.index(3)
    assert body["usage"]["reserved_tokens"] == 0
    assert body["usage"]["occupied_tokens"] == 4

    # 与整段重算的 log-softmax 均值一致。
    from app.model import forward_full

    for entry in body["scores"]:
        cand = candidates[entry["index"]]
        total = 0.0
        for i, tid in enumerate(cand):
            logits = forward_full(main_module.MODEL, [1, 2, 3, 4] + cand[:i])
            shifted = logits.astype(np.float64) - np.max(logits)
            total += float(
                (shifted - np.log(np.sum(np.exp(shifted))))[tid]
            )
        assert entry["score"] == pytest.approx(total / len(cand), rel=1e-6, abs=1e-6)

    # 评分对分支只读：状态与 token 未变。
    state = client.get("/branches").json()
    entry = next(b for b in state["branches"] if b["id"] == bid)
    assert entry["tokens"] == [1, 2, 3, 4]


def test_score_capacity_exceeded_returns_413(client):
    # 容量 12：上下文 10，最长候选 3 -> 13 > 12，413。
    r = client.post("/branches", json={"token_ids": list(range(10))})
    bid = r.json()["branch_id"]

    r = client.post(
        f"/branches/{bid}/score",
        json={"branch_id": bid, "candidates": [[1], [2, 3, 4]]},
    )
    assert r.status_code == 413
    err = r.json()
    assert err["error"] == "capacity_exceeded"
    assert err["usage"]["occupied_tokens"] == 10
    assert err["usage"]["reserved_tokens"] == 0

    # 最长候选恰好 2 -> 12，成功；评分后无残留。
    r = client.post(
        f"/branches/{bid}/score",
        json={"branch_id": bid, "candidates": [[1], [2, 3]]},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["occupied_tokens"] == 10
    assert r.json()["usage"]["reserved_tokens"] == 0


def test_score_validation_and_errors(client):
    r = client.post("/branches", json={"token_ids": [1, 2, 3]})
    bid = r.json()["branch_id"]

    # 空候选列表 / 空候选 / 越界 token / 上下文为空 -> 422
    assert (
        client.post(f"/branches/{bid}/score", json={"branch_id": bid, "candidates": []}).status_code
        == 422
    )
    assert (
        client.post(
            f"/branches/{bid}/score", json={"branch_id": bid, "candidates": [[]]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/branches/{bid}/score",
            json={"branch_id": bid, "candidates": [[main_module.VOCAB_SIZE]]},
        ).status_code
        == 422
    )
    empty = client.post("/branches", json={"token_ids": []}).json()["branch_id"]
    assert (
        client.post(
            f"/branches/{empty}/score", json={"branch_id": empty, "candidates": [[1]]}
        ).status_code
        == 422
    )

    # body 的 branch_id 与路径不一致 -> 422
    assert (
        client.post(
            f"/branches/{bid}/score",
            json={"branch_id": "other", "candidates": [[1]]},
        ).status_code
        == 422
    )

    # 不存在的分支 -> 404
    missing = client.post(
        "/branches/missing/score",
        json={"branch_id": "missing", "candidates": [[1]]},
    )
    assert missing.status_code == 404
    assert missing.json()["error"] == "branch_not_found"


def test_score_returns_usage_and_leaves_no_reservation_after_close(client):
    r = client.post("/branches", json={"token_ids": [1, 2, 3]})
    bid = r.json()["branch_id"]
    r = client.post(
        f"/branches/{bid}/score", json={"branch_id": bid, "candidates": [[4, 5]]}
    )
    assert r.status_code == 200
    usage = r.json()["usage"]
    assert usage["occupied_tokens"] == 3
    assert usage["reserved_tokens"] == 0
    assert usage["kv_state_count"] == 1

    # 关闭分支后占用归零，说明评分未遗留任何引用。
    assert client.delete(f"/branches/{bid}").status_code == 200
    assert client.get("/stats").json()["occupied_tokens"] == 0
    assert client.get("/branches").json()["occupied_matches_live_kv"]
