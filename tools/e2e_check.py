"""端到端试跑：对运行中的 uvicorn 服务发真实 HTTP 请求。

覆盖：建立上下文 / 零拷贝分叉 / 截断分叉 / 追加 COW 隔离 /
超限整批拒绝（413）/ 关闭释放 KV / 并发追加隔离 / logits 与整段重算一致。
"""
import json
import threading
import urllib.request

import numpy as np

from app.model import build_model, forward_full

BASE = "http://127.0.0.1:8000"
MODEL = build_model()
ok = True


def call(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, cond, extra=""):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name, extra)
    ok = ok and cond


def check_logits(name, body, tokens):
    full = forward_full(MODEL, tokens)
    match = np.allclose(body["logits"], full, rtol=1e-5, atol=1e-6)
    check(name + " logits == full recompute", match,
          f"max|diff|={np.max(np.abs(np.array(body['logits']) - full)):.2e}")


# 1) 建立上下文（6 个 token）
s, root = call("POST", "/branches", {"token_ids": [1, 2, 3, 4, 5, 6]})
check("create context 201", s == 201)
check("occupancy == 6", root["usage"]["occupied_tokens"] == 6, str(root["usage"]))
check_logits("create", root, [1, 2, 3, 4, 5, 6])
rid = root["branch_id"]

# 2) 完整分叉：零拷贝，占用不变，引用计数 2
s, child = call("POST", f"/branches/{rid}/fork", {})
snap = call("GET", "/branches")[1]
kv_ids = {b["id"]: b["kv_id"] for b in snap["branches"]}
refcounts = {b["id"]: b["kv_refcount"] for b in snap["branches"]}
check("full fork keeps occupancy 6", child["usage"]["occupied_tokens"] == 6)
check("full fork shares physical KV", kv_ids[rid] == kv_ids[child["branch_id"]])
check("shared KV refcount == 2", refcounts[child["branch_id"]] == 2)
check_logits("full fork", child, [1, 2, 3, 4, 5, 6])
cid = child["branch_id"]

# 3) 从第 3 个 token 处截断分叉
s, trunc = call("POST", f"/branches/{rid}/fork", {"offset": 3})
check("truncated fork tokens", trunc["token_ids"] == [1, 2, 3])
check("truncated fork no extra occupancy", trunc["usage"]["occupied_tokens"] == 6)
check_logits("truncated fork", trunc, [1, 2, 3])
tid = trunc["branch_id"]

# 4) 子分支批量追加 -> COW；父分支与截断分支不受影响
s, grown = call("POST", f"/branches/{cid}/tokens", {"token_ids": [7, 8]})
check("append 200", s == 200)
check("child tokens extended", grown["token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8])
# 物理 KV：父 6（共享给截断分支）+ 子 8 = 14
check("occupancy after COW == 14", grown["usage"]["occupied_tokens"] == 14,
      str(grown["usage"]))
check_logits("append", grown, [1, 2, 3, 4, 5, 6, 7, 8])
snap = call("GET", "/branches")[1]
entries = {b["id"]: b for b in snap["branches"]}
check("parent branch untouched", entries[rid]["tokens"] == [1, 2, 3, 4, 5, 6])
check("truncated branch untouched", entries[tid]["tokens"] == [1, 2, 3])

# 5) 截断分支追加（独占视图裁剪路径的前置：它仍与父共享 -> COW, view=3+1）
s, t2 = call("POST", f"/branches/{tid}/tokens", {"token_ids": [9]})
# 新 KV 长度 4；总占用 = 6(父) + 8(子) + 4(截断) = 18
check("truncated append occupancy == 18", t2["usage"]["occupied_tokens"] == 18,
      str(t2["usage"]))
check_logits("truncated append", t2, [1, 2, 3, 9])

# 6) 超限：上限 20，当前 18，再追加 3 个 -> 21 > 20，整批拒绝（413）
before = call("GET", "/branches")[1]
s, err = call("POST", f"/branches/{cid}/tokens", {"token_ids": [10, 11, 12]})
check("over-limit batch rejected 413", s == 413, str(err))
check("error code capacity_exceeded", err["error"] == "capacity_exceeded")
check("413 reports unchanged occupancy 18", err["usage"]["occupied_tokens"] == 18)
after = call("GET", "/branches")[1]
check("state identical after rejection",
      json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True))

# 恰好 2 个 -> 20，成功
s, ok2 = call("POST", f"/branches/{cid}/tokens", {"token_ids": [10, 11]})
check("batch reaching exactly limit succeeds", s == 200 and
      ok2["usage"]["occupied_tokens"] == 20, str(ok2["usage"]))
# 再追加 1 个也被拒绝，占用仍为 20
s, err2 = call("POST", f"/branches/{cid}/tokens", {"token_ids": [12]})
check("one more token rejected at limit", s == 413 and
      err2["usage"]["occupied_tokens"] == 20)

# 7) 关闭：先关父分支（其 KV 已无其他引用？截断分支已 COW -> 是，释放 6）
s, closed = call("DELETE", f"/branches/{rid}")
# 占用 20 - 6 = 14
check("close parent releases its KV (14)", closed["usage"]["occupied_tokens"] == 14,
      str(closed["usage"]))
# 关闭不存在的分支 -> 404
s, nf = call("DELETE", "/branches/does-not-exist")
check("close unknown -> 404", s == 404)

# 8) 并发：新建根分支并分叉 5 份，各自并发追加不同 token，检查隔离
#    每份独占 KV 最终长 4，5 份恰好 20，不超上限。
s, base = call("POST", "/branches", {"token_ids": [0]})
call("DELETE", f"/branches/{cid}")   # 释放 10
call("DELETE", f"/branches/{tid}")   # 释放 4
# 现在只剩 base 根 KV(1)，占用 1
forks = []
for _ in range(5):
    s, f = call("POST", f"/branches/{base['branch_id']}/fork", {})
    forks.append(f["branch_id"])
call("DELETE", f"/branches/{base['branch_id']}")  # 根关闭，5 个分叉各持引用

errors = []
def worker(bid, token):
    try:
        for _ in range(3):
            code, body = call("POST", f"/branches/{bid}/tokens", {"token_ids": [token]})
            if code != 200:
                errors.append((bid, code, body))
    except Exception as e:  # noqa: BLE001
        errors.append((bid, -1, str(e)))

threads = [threading.Thread(target=worker, args=(b, 20 + i)) for i, b in enumerate(forks)]
for t in threads: t.start()
for t in threads: t.join()
check("concurrent appends all 200", not errors, str(errors[:3]))
snap = call("GET", "/branches")[1]
entries = {b["id"]: b for b in snap["branches"]}
isolated = all(entries[b]["tokens"] == [0, 20 + i, 20 + i, 20 + i]
               for i, b in enumerate(forks))
check("concurrent branches keep isolated contexts", isolated)
check("concurrent run occupancy == 20 (5 KV x 4)",
      snap["usage"]["occupied_tokens"] == 20, str(snap["usage"]))
check("all invariants hold", snap["occupied_matches_live_kv"])

for b in forks:
    call("DELETE", f"/branches/{b}")
final = call("GET", "/stats")[1]
check("all KV released at end (occupied 0)", final["occupied_tokens"] == 0, str(final))

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
