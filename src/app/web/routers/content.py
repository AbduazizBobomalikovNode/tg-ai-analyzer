"""7-bosqich API: rasmlar, auto-reply qoidalari, Telegram rejalashtirilgan postlar.

GET   /api/images/{id}                                   egasiga rasm fayli
GET   /api/accounts/{id}/chats/{chat_id}/autoreply       qoida (yo'q bo'lsa {rule: null})
PUT   /api/accounts/{id}/chats/{chat_id}/autoreply       {enabled, trigger, keywords,
                                                          instructions, max_per_hour, quiet_*}
GET   /api/accounts/{id}/chats/{chat_id}/scheduled       Telegram-side scheduled (READ)
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.db.base import session_scope
from app.db.models import Account, Chat
from app.mtproto.pool import PoolError, pool
from app.services import autoreply as AR
from app.services import images as IMG
from app.services.search import compact_text
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api", tags=["content"])


class AutoReplyIn(BaseModel):
    enabled: bool = False
    trigger: str = Field(default="questions", pattern="^(all|mentions|keywords|questions)$")
    keywords: str = Field(default="", max_length=1000)
    instructions: str = Field(default="", max_length=4000)
    max_per_hour: int = Field(default=5, ge=1, le=60)
    quiet_from: int | None = Field(default=None, ge=0, le=23)
    quiet_to: int | None = Field(default=None, ge=0, le=23)


async def _own_chat(user_id: int, account_id: int, chat_id: int) -> Chat:
    async with session_scope() as db:
        acc = await db.get(Account, account_id)
        if acc is None or acc.user_id != user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        chat = await db.get(Chat, chat_id)
        if chat is None or chat.account_id != account_id:
            raise HTTPException(status_code=404, detail={"code": "pool.err.no_dialog"})
        return chat


@router.get("/images/{image_id}")
async def get_image(image_id: str, ident: Annotated[WebIdentity, Depends(require_identity)]) -> Any:
    if len(image_id) > 40 or "/" in image_id or "\\" in image_id:
        raise HTTPException(status_code=404, detail={"code": "image.err.not_found"})
    async with session_scope() as db:
        try:
            row, path = await IMG.get_owned(db, user_id=ident.user_id, image_id=image_id)
        except IMG.ImageError as exc:
            raise HTTPException(status_code=404, detail={"code": f"image.err.{exc.code}"}) from exc
        return FileResponse(
            str(path), media_type=row.mime, headers={"Cache-Control": "private, max-age=86400"}
        )


@router.get("/accounts/{account_id}/chats/{chat_id}/autoreply")
async def get_autoreply(
    account_id: int, chat_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    await _own_chat(ident.user_id, account_id, chat_id)
    async with session_scope() as db:
        rule = await AR.get_rule(db, user_id=ident.user_id, chat_id=chat_id)
        return {"rule": rule.to_dict() if rule else None}


@router.put(
    "/accounts/{account_id}/chats/{chat_id}/autoreply", dependencies=[Depends(require_csrf)]
)
async def put_autoreply(
    account_id: int,
    chat_id: int,
    body: AutoReplyIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    await _own_chat(ident.user_id, account_id, chat_id)
    async with session_scope() as db:
        try:
            rule = await AR.upsert_rule(
                db, user_id=ident.user_id, chat_id=chat_id, data=body.model_dump()
            )
        except AR.AutoReplyError as exc:
            status = {"not_found": 404, "read_only": 403, "not_writable": 403}.get(exc.code, 400)
            raise HTTPException(
                status_code=status, detail={"code": f"autoreply.err.{exc.code}"}
            ) from exc
        return {"rule": rule.to_dict()}


@router.get("/accounts/{account_id}/chats/{chat_id}/scheduled")
async def scheduled(
    account_id: int, chat_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    chat = await _own_chat(ident.user_id, account_id, chat_id)
    try:
        client = await pool.get(account_id)
        peer = await pool.input_peer(account_id, chat.tg_peer_id)
        msgs = await client.get_messages(peer, limit=50, scheduled=True)
    except PoolError as exc:
        raise HTTPException(status_code=502, detail={"code": f"pool.err.{exc.code}"}) from exc
    items = [
        {
            "msg_id": int(m.id),
            "date": m.date.isoformat() if getattr(m, "date", None) else None,
            "text": compact_text(getattr(m, "message", "") or "", max_chars=300),
            "has_media": getattr(m, "media", None) is not None,
        }
        for m in (msgs or [])
    ]
    return {"items": items}
