"""Ulangan akkauntlar bo'yicha faqat-o'qish Telegram ma'lumotlari.

GET /api/accounts/{id}/dialogs?limit=100&refresh=0
GET /api/accounts/{id}/dialogs/{peer_id}/messages?limit=50
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account
from app.mtproto.pool import PoolError, pool
from app.web.security import WebIdentity, require_identity

router = APIRouter(prefix="/api/accounts", tags=["telegram"])


async def _own_account(user_id: int, account_id: int) -> Account:
    async with session_scope() as db:
        acc = await db.get(Account, account_id)
        if acc is None or acc.user_id != user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        return acc


def _pool_http(exc: PoolError) -> HTTPException:
    status = 429 if exc.code == "flood" else 409 if exc.code == "session_revoked" else 502
    return HTTPException(status_code=status, detail={"code": f"pool.err.{exc.code}"})


@router.get("/{account_id}/dialogs")
async def dialogs(
    account_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=1, le=300)] = 100,
    refresh: bool = False,
) -> dict[str, Any]:
    await _own_account(ident.user_id, account_id)
    try:
        items = await pool.dialogs(account_id, limit=limit, force=refresh)
    except PoolError as exc:
        raise _pool_http(exc) from exc
    return {
        "items": [
            {
                "peer_id": d.peer_id,
                "title": d.title,
                "kind": d.kind,
                "username": d.username,
                "unread": d.unread,
                "last_message_at": d.last_message_at.isoformat() if d.last_message_at else None,
                "last_message_text": d.last_message_text,
            }
            for d in items
        ]
    }


@router.get("/{account_id}/dialogs/{peer_id}/messages")
async def messages(
    account_id: int,
    peer_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> dict[str, Any]:
    await _own_account(ident.user_id, account_id)
    limit = min(limit, get_settings().web_context_max_messages)
    try:
        title, items = await pool.recent_messages(account_id, peer_id, limit=limit)
    except PoolError as exc:
        raise _pool_http(exc) from exc
    return {
        "title": title,
        "items": [
            {
                "msg_id": m.msg_id,
                "date": m.date.isoformat() if m.date else None,
                "sender": m.sender,
                "text": m.text,
                "media_type": m.media_type,
                "views": m.views,
            }
            for m in items
        ],
    }
