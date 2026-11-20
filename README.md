# NumPy 单头因果注意力 · 分支化解码 API

单进程 FastAPI 解码服务，内置固定权重（纯 NumPy 实现）的单层因果注意力模型，
支持从同一上下文分叉出多条候选、批量追加 token、关闭分支、管理共享的
K/V 缓存，以及对候选序列打分并排序。

## 模型

- 结构：token embedding → 单头因果自注意力（softmax，缩放点积）→ 残差 + 输出投影 → logits
- 权重由固定种子 `MODEL_SEED` 生成（默认 `20261120`），进程内恒定
- 默认词表 64、隐藏维度 16（可用环境变量调整，见下）
- 提供两条前向路径，next-token logits 数值一致（测试容差 rtol=1e-5）：
  - 增量路径：`model.step`，复用历史 K/V，每步只计算新 token
  - 基准路径：`model.forward_full`，对整段 token 一次性重算并施加因果掩码

## 分支与缓存语义

- **建立上下文**：`POST /branches` 提交 token ID 序列（允许空数组）
- **分叉**：`POST /branches/{id}/fork`，`offset` 可指向任一已有 token 位置
  （省略则继承完整上下文）。分叉零拷贝复用继承的物理 KV，仅引用计数 +1，
  不增加缓存占用
- **批量追加**：`POST /branches/{id}/tokens`
  - 共享 KV 上追加先做写时复制（COW），只复制该分支视图内的有效部分，
    其他分支上下文不受影响
  - 截断分叉后的分支追加时，先裁剪视图之外的陈旧位置再增长
- **关闭**：`DELETE /branches/{id}`；某份 KV 的最后一个引用关闭后立即释放
- **容量**：按实际保存的 token 位置数计（所有存活物理 KV 的有效长度之和；
  共享不重复计数）。超限的提交在任何状态修改之前检查，**整批拒绝**（HTTP 413），
  原分支内容与全局占用量保持不变
- **并发**：单进程内所有操作为同一把可重入锁保护的原子操作，
  各分支只使用自己的上下文
- 每次成功响应都返回当前缓存用量；建立上下文 / 分叉 / 追加还返回
  `logits`（增量）与 `full_recompute_logits`（整段重算），两者一致

## 候选评分

`POST /branches/{id}/score`，body：

```json
{"branch_id": "<id>", "candidates": [[5], [7, 8], [9, 10, 11]]}
```

- **只读**：对分支不做任何修改。请求被接纳时快照分支当时可见的 KV 视图，
  之后对该分支的追加/关闭都不会改变本次结果
- **评分**：从上下文末尾的 next-token 分布开始，逐 token 取候选 token 的
  条件对数概率（log-softmax），返回每条候选的**均值**
- **排序**：按分数从高到低返回 `{"index": 输入序号, "score": 分数}`，
  同分按输入顺序
- **临时 KV**：候选部分使用独立临时 KV，按**最长候选**的 token 数预留，
  与已有 KV 共用同一上限；余量不足在改动任何状态之前返回 `413`
- **预留计入占用**：评分进行期间预留继续计入容量，关闭分支/续写照常按
  剩余容量判定；被借用的上下文 KV 在评分结束前保持存活（关闭分支也不会
  提前回收）
- **收尾**：无论成功或失败都归还预留，并释放评分持有的 KV 引用，
  `reserved_tokens` 回到 0
- 评分计算不持锁，端点以同步方式在线程池执行，评分期间其他请求仍可处理

缓存用量现在同时给出 `occupied_tokens`（分支实际占用）、`reserved_tokens`
（评分临时预留）与 `total_tokens`（两者之和，容量判定依据），上限约束
`total_tokens <= MAX_CACHED_TOKENS`。

## 用 Docker Compose 启动

```bash
docker compose up --build
```

- 服务：http://localhost:8000
- Swagger UI：http://localhost:8000/docs （`/` 自动重定向到 `/docs`）
- OpenAPI：http://localhost:8000/openapi.json
- 容器固定单 uvicorn worker，保证进程内缓存与锁状态一致

### 可配置环境变量（docker-compose.yml）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CACHED_TOKENS` | `128` | 缓存上限（实际保存的 token 位置数） |
| `MODEL_VOCAB_SIZE` | `64` | 词表大小 |
| `MODEL_D_MODEL` | `16` | 模型隐藏维度 |
| `MODEL_SEED` | `20261120` | 固定权重随机种子 |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/branches` | 建立上下文，body `{"token_ids": [...]}` |
| POST | `/branches/{id}/fork` | 分叉，body `{"offset": 3}` 或 `{}` |
| POST | `/branches/{id}/tokens` | 批量追加，body `{"token_ids": [...]}` |
| POST | `/branches/{id}/score` | 候选评分，body `{"branch_id": id, "candidates": [[...], ...]}` |
| DELETE | `/branches/{id}` | 关闭分支 |
| GET | `/stats` | 缓存用量 |
| GET | `/branches` | 调试快照（引用计数、物理 KV ID、占用不变量） |
| GET | `/health` | 健康检查 |

在 Swagger 中依次调用示例：

1. `POST /branches`，body `{"token_ids": [1, 2, 3, 4]}` → 记下返回的 `branch_id`
2. `POST /branches/{id}/fork`，body `{}` → 完整分叉，占用不变
3. `POST /branches/{id}/fork`，body `{"offset": 2}` → 从第 2 个 token 处分叉
4. `POST /branches/{id}/tokens`，body `{"token_ids": [5, 6]}` → 批量追加
5. `POST /branches/{id}/score`，
   body `{"branch_id": "<id>", "candidates": [[7], [8, 9]]}` → 返回按分数
   降序的 `index`/`score`
6. 构造超过 `MAX_CACHED_TOKENS` 的批次或最长候选 → 返回 413，`usage` 保持原值
7. `DELETE /branches/{id}` → 关闭，最后一个引用释放后占用下降

## 本地开发与测试（无 Docker 时）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest            # 37 个单元/API 测试
MAX_CACHED_TOKENS=20 .venv/bin/python -m uvicorn app.main:app --port 8000
.venv/bin/python tools/e2e_check.py   # 对运行中的服务做真实 HTTP 端到端核对
```

`requirements.txt` 为镜像运行依赖；`requirements-dev.txt` 额外包含 pytest 与
httpx（Starlette TestClient 所需）。

## 目录结构

```
app/
  model.py   # 固定权重模型、增量 step、整段重算 forward_full
  store.py   # 分支/KvState、引用计数、COW、容量控制、并发锁
  main.py    # FastAPI 端点、请求校验、错误响应
tests/       # pytest：模型一致性 / 存储语义 / 并发 / HTTP API
tools/       # 端到端核对脚本（非镜像内容）
Dockerfile, docker-compose.yml, requirements.txt
```

## 错误响应

- `413 capacity_exceeded`：批次或评分临时 KV 超限被拒绝，响应含当前 `usage`
- `404 branch_not_found`：分支不存在（可能已被关闭）
- `422`：请求体校验失败（空追加批次、token 越界、offset 越界、空候选、
  空上下文评分、body 与路径 `branch_id` 不一致）
