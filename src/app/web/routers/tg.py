"""Ulangan akkauntlar bo'yicha faqat-o'qish Telegram ma'lumotlari.

GET  /api/accounts/{id}/dialogs?limit=100&refresh=0        jonli (MTProto)
GET  /api/accounts/{id}/dialogs/{peer_id}/messages?limit=50
GET  /api/accounts/{id}/chats                               DB registry + sync progress
POST /api/accounts/{id}/sync                                to'liq ingestion boshlash
POST /api/accounts/{id}/chats/{chat_id}/sync                bitta chat
GET  /api/accounts/{id}/chats/{chat_id}/avatar              profil rasmi (fayl)

Qolgan chat ma'lumotlari (`chats/search`, `messages/search`, `chats/{id}`,
`.../stats`, `.../digests`) — `routers/chatdata.py`, shu router ostida.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, Chat
from app.mtproto.pool import PoolError, pool
from app.services import avatars as AV
from app.web.routers import chatdata as chatdata_router
from app.web.routers.chatdata import chat_out
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api/accounts", tags=["telegram"])
# ui/R1a: chat profili / qidiruv / statistika endpoint'lari alohida modulda,
# lekin **shu** router ostiga ulanadi — `web/main.py` (R1b) tegilmaydi.
# `chatdata.router` prefikssiz: yakuniy yo'l `/api/accounts/...` bo'ladi.
router.include_router(chatdata_router.router)


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


#: Chat qatorining JSON ko'rinishi — yagona manba `routers/chatdata.py`
#: (u yerda `chats/search`, `chats/{id}` ham shu shaklni qaytaradi).
_chat_out = chat_out


@router.get("/{account_id}/chats/{chat_id}/avatar")
async def chat_avatar(
    account_id: int, chat_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> Any:
    """Chat profil rasmi — faqat egasiga (`services/avatars.py` yozadi).

    ETag = Telegram `photo_id`, ya'ni rasm almashsa URL ham, ETag ham o'zgaradi.
    """
    await _own_account(ident.user_id, account_id)
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None or chat.account_id != account_id:
            raise HTTPException(status_code=404, detail={"code": "pool.err.no_dialog"})
        mime = chat.avatar_mime
        photo_id = chat.avatar_photo_id
    if not mime or not photo_id:
        raise HTTPException(status_code=404, detail={"code": "chat.err.no_avatar"})
    path = AV.path_for(chat_id, mime)
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "chat.err.no_avatar"})
    return FileResponse(
        str(path),
        media_type=mime,
        headers={"Cache-Control": "private, max-age=86400", "ETag": f'"{photo_id}"'},
    )


@router.get("/{account_id}/chats")
async def chats(
    account_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=0, le=1000)] = 0,
) -> dict[str, Any]:
    """Akkauntning chat registri. `limit` — UI select'lari uchun (0 = hammasi)."""
    await _own_account(ident.user_id, account_id)
    async with session_scope() as db:
        stmt = (
            select(Chat)
            .where(Chat.account_id == account_id)
            .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = await db.execute(stmt)
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
