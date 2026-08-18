"""Yozish amallari API (6-bosqich).

    GET   /api/actions?status=proposed&run_id=      ro'yxat (foydalanuvchiniki)
    POST  /api/actions/{id}/confirm                 tasdiqlash → bajarish
    POST  /api/actions/{id}/reject
    PATCH /api/accounts/{id}/chats/{chat_id}/write_mode   {mode}

Hamma o'zgartiruvchi so'rovlar CSRF header + cookie. Bajarish `services/actions`
orqali — guard, rate limit, TTL o'sha yerda.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.base import session_scope
from app.db.models import Account
from app.services import actions as ACT
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api", tags=["actions"])


class WriteModeIn(BaseModel):
    mode: str = Field(pattern="^(read_only|write_with_confirm|autonomous)$")


def _http(exc: ACT.ActionError) -> HTTPException:
    status = {
        "not_found": 404,
        "wrong_status": 409,
        "expired": 410,
        "rate_limited": 429,
        "blocked": 403,
        "read_only": 403,
        "autonomous_limit": 409,
        "not_writable": 403,
        "bad_mode": 400,
        "telegram": 502,
    }.get(exc.code, 400)
    return HTTPException(
        status_code=status, detail={"code": f"action.err.{exc.code}", "detail": exc.detail}
    )


@router.get("/actions")
async def list_actions(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    status: Annotated[
        str | None, Query(pattern="^(proposed|confirmed|executed|rejected|failed|blocked)$")
    ] = None,
    run_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    async with session_scope() as db:
        rows = await ACT.list_actions(db, ident.user_id, status=status, run_id=run_id, limit=limit)
        return {"items": [r.to_dict() for r in rows]}


@router.post("/actions/{action_id}/confirm", dependencies=[Depends(require_csrf)])
async def confirm(
    action_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        try:
            view = await ACT.confirm_action(db, ident.user_id, action_id)
        except ACT.ActionError as exc:
            raise _http(exc) from exc
        return view.to_dict()


@router.post("/actions/{action_id}/reject", dependencies=[Depends(require_csrf)])
async def reject(
    action_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        try:
            view = await ACT.reject_action(db, ident.user_id, action_id)
        except ACT.ActionError as exc:
            raise _http(exc) from exc
        return view.to_dict()


@router.patch(
    "/accounts/{account_id}/chats/{chat_id}/write_mode", dependencies=[Depends(require_csrf)]
)
async def set_write_mode(
    account_id: int,
    chat_id: int,
    body: WriteModeIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    async with session_scope() as db:
        acc = await db.get(Account, account_id)
        if acc is None or acc.user_id != ident.user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        try:
            chat = await ACT.set_chat_write_mode(
                db, user_id=ident.user_id, chat_id=chat_id, mode=body.mode
            )
        except ACT.ActionError as exc:
            raise _http(exc) from exc
        if chat.account_id != account_id:
            raise HTTPException(status_code=404, detail={"code": "pool.err.no_dialog"})
        return {"chat_id": chat.id, "write_mode": chat.write_mode}
