"""单进程解码 API（FastAPI）。

端点
----
- POST /branches              提交 token ID 建立上下文
- POST /branches/{id}/fork    从任一已有 token 处分叉（零拷贝继承 KV）
- POST /branches/{id}/tokens  批量追加 token（超限整批拒绝）
- DELETE /branches/{id}       关闭分支（最后一个引用关闭后释放 KV）
- GET  /branches              列出分支与内部引用计数（调试）
- GET  /stats                 缓存用量
- GET  /docs                  Swagger UI

所有成功响应都带缓存用量；建立上下文 / 分叉 / 追加还返回与整段重算
（model.forward_full）一致的下一 token logits。
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from .model import (
    DEFAULT_D_MODEL,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    build_model,
    forward_full,
)
from .store import BranchStore, CapacityExceeded


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


VOCAB_SIZE = _env_int("MODEL_VOCAB_SIZE", DEFAULT_VOCAB_SIZE)
D_MODEL = _env_int("MODEL_D_MODEL", DEFAULT_D_MODEL)
SEED = _env_int("MODEL_SEED", DEFAULT_SEED)
MAX_CACHED_TOKENS = _env_int("MAX_CACHED_TOKENS", 128)

MODEL = build_model(seed=SEED, vocab_size=VOCAB_SIZE, d_model=D_MODEL)
STORE = BranchStore(MODEL, max_cached_tokens=MAX_CACHED_TOKENS)


# --------------------------------------------------------------------- 请求模型


class CreateBranchRequest(BaseModel):
    token_ids: list[int] = Field(
        ..., description="用于建立上下文的 token ID 序列；可为空数组"
    )

    @field_validator("token_ids")
    @classmethod
    def _check_range(cls, value: list[int]) -> list[int]:
        for tid in value:
            if tid < 0 or tid >= VOCAB_SIZE:
                raise ValueError(f"token id {tid} out of range [0, {VOCAB_SIZE - 1}]")
        return value


class ForkRequest(BaseModel):
    offset: int | None = Field(
        default=None,
        description="保留父分支前 offset 个 token；省略或 null 表示继承完整上下文",
    )

    @field_validator("offset")
    @classmethod
    def _non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("offset must be >= 0")
        return value


class AppendTokensRequest(BaseModel):
    token_ids: list[int] = Field(..., min_length=1, description="本批追加的 token ID")

    @field_validator("token_ids")
    @classmethod
    def _check_range(cls, value: list[int]) -> list[int]:
        for tid in value:
            if tid < 0 or tid >= VOCAB_SIZE:
                raise ValueError(f"token id {tid} out of range [0, {VOCAB_SIZE - 1}]")
        return value


# --------------------------------------------------------------------- 响应模型


class BranchResponse(BaseModel):
    branch_id: str
    token_ids: list[int]
    length: int
    logits: list[float] | None
    full_recompute_logits: list[float] | None
    usage: dict


def _make_response(
    branch, logits, include_full_recompute: bool = True
) -> BranchResponse:
    full = None
    if include_full_recompute and branch.token_ids:
        full = forward_full(MODEL, branch.token_ids).tolist()
    return BranchResponse(
        branch_id=branch.id,
        token_ids=list(branch.token_ids),
        length=len(branch.token_ids),
        logits=None if logits is None else logits.tolist(),
        full_recompute_logits=full,
        usage=STORE.usage(),
    )


# --------------------------------------------------------------------- 应用


def create_app() -> FastAPI:
    app = FastAPI(
        title="NumPy 单头因果注意力解码 API",
        version="1.0.0",
        description=(
            "单进程解码服务：提交 token 建立上下文、任意位置分叉（继承 KV）、"
            "批量追加与关闭。缓存上限按实际保存的 token 位置数计。"
        ),
    )

    @app.exception_handler(CapacityExceeded)
    async def _capacity_handler(_request: Request, exc: CapacityExceeded):
        # 413：整批已被拒绝，分支与占用量均未改变。
        return JSONResponse(
            status_code=413,
            content={
                "error": "capacity_exceeded",
                "detail": str(exc),
                "usage": STORE.usage(),
            },
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health():
        return {"status": "ok", "vocab_size": VOCAB_SIZE, "d_model": D_MODEL}

    @app.get("/stats")
    async def stats():
        return STORE.usage()

    @app.get("/branches")
    async def list_branches():
        return STORE.debug_snapshot()

    @app.post("/branches", status_code=201, response_model=BranchResponse)
    async def create_branch(req: CreateBranchRequest):
        branch, logits = STORE.create_branch(req.token_ids)
        return _make_response(branch, logits)

    @app.post("/branches/{branch_id}/fork", response_model=BranchResponse)
    async def fork_branch(branch_id: str, req: ForkRequest):
        try:
            branch, logits = STORE.fork(branch_id, req.offset)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "invalid_offset", "detail": str(exc)},
            )
        return _make_response(branch, logits)

    @app.post("/branches/{branch_id}/tokens", response_model=BranchResponse)
    async def append_tokens(branch_id: str, req: AppendTokensRequest):
        try:
            branch, logits = STORE.append(branch_id, req.token_ids)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        return _make_response(branch, logits)

    @app.delete("/branches/{branch_id}")
    async def close_branch(branch_id: str):
        try:
            STORE.close(branch_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": "branch_not_found", "branch_id": branch_id},
            )
        return {"branch_id": branch_id, "closed": True, "usage": STORE.usage()}

    return app


app = create_app()
