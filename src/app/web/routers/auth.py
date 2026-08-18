"""Telefon orqali Telegram auth — HTTPS API (rejaning 4.1-bandi).

    POST /api/auth/phone     {phone}              → {flow_id, status, code_type, ...}
    POST /api/auth/code      {flow_id, code}      → {status: needs_2fa | done}
    POST /api/auth/password  {flow_id, password}  → {status: done}
    POST /api/auth/cancel    {flow_id}
    POST /api/auth/logout                          — web cookie o'chadi (session saqlanadi)
    GET  /api/me                                   — user + akkauntlar

`done` bo'lganda javob cookie o'rnatadi. Xatolar: 400 {"code": "auth.err.<x>"}.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import User
from app.logging import get_logger
from app.services.accounts import link_account, list_accounts
from app.services.auth_flow import AuthError, AuthFlowStore, FlowStatus
from app.web.security import (
    RateLimiter,
    WebIdentity,
    clear_session_cookie,
    client_ip,
    identity_of,
    require_csrf,
    require_identity,
    set_session_cookie,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["auth"])


def flows(request: Request) -> AuthFlowStore:
    return request.app.state.auth_flows  # type: ignore[no-any-return]


def limiter(request: Request) -> RateLimiter:
    return request.app.state.auth_limiter  # type: ignore[no-any-return]


class PhoneIn(BaseModel):
    phone: str = Field(min_length=6, max_length=32)


class CodeIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)
    code: str = Field(min_length=3, max_length=12)


class PasswordIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)
    password: str = Field(min_length=1, max_length=512)


class FlowIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)


def _flow_out(flow: Any) -> dict[str, Any]:
    return {
        "flow_id": flow.id,
        "status": str(flow.status),
        "code_type": flow.code_type,
        "code_length": flow.code_length,
        "timeout": flow.timeout,
    }


def _auth_http(exc: AuthError) -> HTTPException:
    return HTTPException(
        status_code=429 if exc.code == "flood" else 400,
        detail={"code": f"auth.err.{exc.code}", "retry_after": exc.retry_after},
    )


@router.post("/auth/phone", dependencies=[Depends(require_csrf)])
async def auth_phone(body: PhoneIn, request: Request) -> dict[str, Any]:
    if not limiter(request).check(client_ip(request)):
        raise HTTPException(status_code=429, detail={"code": "auth.err.rate_limited"})
    ident = identity_of(request)
    try:
        flow = await flows(request).start(
            body.phone, owner_user_id=ident.user_id if ident else None
        )
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return _flow_out(flow)


async def _finish(request: Request, response: Response, flow_id: str) -> dict[str, Any]:
    """DONE oqimni DB'ga tushiradi va cookie beradi."""
    store = flows(request)
    flow = store.get(flow_id)
    if flow.status is not FlowStatus.DONE:
        return _flow_out(flow)
    result = store.take_result(flow_id)
    try:
        async with session_scope() as db:
            linked = await link_account(db, result, owner_user_id=flow.owner_user_id)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    finally:
        del result  # session string RAM'da qolmasin
    set_session_cookie(request, response, linked.user_id)
    return {"flow_id": flow_id, "status": "done", "account_id": linked.account_id}


@router.post("/auth/code", dependencies=[Depends(require_csrf)])
async def auth_code(body: CodeIn, request: Request, response: Response) -> dict[str, Any]:
    try:
        flow = await flows(request).submit_code(body.flow_id, body.code)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    if flow.status is FlowStatus.NEEDS_2FA:
        return _flow_out(flow)
    return await _finish(request, response, body.flow_id)


@router.post("/auth/password", dependencies=[Depends(require_csrf)])
async def auth_password(body: PasswordIn, request: Request, response: Response) -> dict[str, Any]:
    try:
        await flows(request).submit_password(body.flow_id, body.password)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return await _finish(request, response, body.flow_id)


@router.post("/auth/cancel", dependencies=[Depends(require_csrf)])
async def auth_cancel(body: FlowIn, request: Request) -> dict[str, str]:
    await flows(request).cancel(body.flow_id)
    return {"status": "cancelled"}


@router.post("/auth/logout", dependencies=[Depends(require_csrf)])
async def auth_logout(response: Response) -> dict[str, str]:
    clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/me")
async def me(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        user = await db.get(User, ident.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "unauthorized"})
        accounts = await list_accounts(db, user.id)
        return {
            "user": {"id": user.id, "tg_user_id": user.tg_user_id, "username": user.username},
            "accounts": [
                {
                    "id": a.id,
                    "label": a.label,
                    "status": a.status,
                    "tg_account_id": a.tg_account_id,
                    "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
                }
                for a in accounts
            ],
            "max_accounts": get_settings().max_accounts,
        }
