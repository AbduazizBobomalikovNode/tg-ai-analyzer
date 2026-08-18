"""Ulangan akkauntlar bo'yicha faqat-o'qish Telegram ma'lumotlari.

GET  /api/accounts/{id}/dialogs?limit=100&refresh=0        jonli (MTProto)
GET  /api/accounts/{id}/dialogs/{peer_id}/messages?limit=50
GET  /api/accounts/{id}/chats                               DB registry + sync progress
POST /api/accounts/{id}/sync                                to'liq ingestion boshlash
POST /api/accounts/{id}/chats/{chat_id}/sync                bitta chat
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, Chat
from app.mtproto.pool import PoolError, pool
from app.web.security import WebIdentity, require_csrf, require_identity

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
    limit = min(limit, get_settings().web_context_max_messages, 200)  # jonli — 200 dan oshmasin
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


def _chat_out(c: Chat) -> dict[str, Any]:
    est = c.total_estimate or 0
    if est:
        pct = min(100, round(100 * c.synced_total / est))
    else:
        pct = 100 if c.sync_state == "done" else 0
    return {
        "id": c.id,
        "peer_id": c.tg_peer_id,
        "title": c.title,
        "type": c.type,
        "username": c.username,
        "is_admin": c.is_admin,
        "participants_count": c.participants_count,
        "sync_state": c.sync_state,
        "synced_total": c.synced_total,
        "total_estimate": c.total_estimate,
        "progress": pct,
        "sync_error": c.sync_error,
        "last_sync_at": c.last_sync_at.isoformat() if c.last_sync_at else None,
        "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
    }


@router.get("/{account_id}/chats")
async def chats(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    await _own_account(ident.user_id, account_id)
    async with session_scope() as db:
        rows = await db.execute(
            select(Chat)
            .where(Chat.account_id == account_id)
            .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
        )
        items = [_chat_out(c) for c in rows.scalars().all()]
    running = sum(1 for c in items if c["sync_state"] == "running")
    return {"items": items, "running": running}


async def _enqueue(name: str, *args: Any, job_id: str) -> bool:
    try:
        from app.worker.queue import enqueue

        return (await enqueue(name, *args, job_id=job_id)) is not None
    except Exception as exc:  # Redis/worker yo'q
        raise HTTPException(status_code=503, detail={"code": "sync.err.queue"}) from exc


@router.post("/{account_id}/sync", dependencies=[Depends(require_csrf)])
async def start_sync(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    from app.worker.tasks import job_id_account

    await _own_account(ident.user_id, account_id)
    queued = await _enqueue("sync_account", account_id, True, job_id=job_id_account(account_id))
    return {"status": "queued" if queued else "already_running"}


@router.post("/{account_id}/chats/{chat_id}/sync", dependencies=[Depends(require_csrf)])
async def start_chat_sync(
    account_id: int, chat_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    from app.worker.tasks import job_id_chat

    await _own_account(ident.user_id, account_id)
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None or chat.account_id != account_id:
            raise HTTPException(status_code=404, detail={"code": "pool.err.no_dialog"})
    queued = await _enqueue("sync_chat", chat_id, job_id=job_id_chat(chat_id))
    return {"status": "queued" if queued else "already_running"}
