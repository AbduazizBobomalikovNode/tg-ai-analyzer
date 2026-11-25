"""Javob eksporti.

GET /api/conversations/{id}/messages/{msg_id}/export?format=csv|md|json

Faqat egasi (imzolangan cookie, `chat_service.get_conversation` — begona
suhbat → 404). GET — o'zgartirmaydi, shuning uchun `X-Requested-With`
talab qilinmaydi (oddiy `<a download>` havolasi ishlasin). Nima chiqishi va
telefon qo'riqchisi — `services/export.py`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Account, Chat, ConversationMessage
from app.services import chat_service as cs
from app.services import export as EXP
from app.services.analytics import msg_link
from app.web.security import WebIdentity, require_identity

router = APIRouter(prefix="/api/conversations", tags=["export"])

_NOT_FOUND = HTTPException(status_code=404, detail={"code": "chat.err.not_found"})


async def _token_links(
    db: Any, user_id: int, tokens: list[tuple[str, int, int]]
) -> dict[str, dict[str, Any]]:
    """Jadval tokenlari → `{chat_title, link}` — faqat shu foydalanuvchining chatlari."""
    ids = sorted({chat for _, chat, _ in tokens})
    if not ids:
        return {}
    rows = (
        await db.execute(
            select(Chat)
            .join(Account, Account.id == Chat.account_id)
            .where(Chat.id.in_(ids), Account.user_id == user_id)
        )
    ).scalars()
    chats = {c.id: c for c in rows}
    out: dict[str, dict[str, Any]] = {}
    for token, chat_id, msg in tokens:
        chat = chats.get(chat_id)
        if chat is not None:
            out[token] = {"chat_title": chat.title, "link": msg_link(chat, msg) or ""}
    return out


@router.get("/{conversation_id}/messages/{message_id}/export")
async def export_message(
    conversation_id: int,
    message_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    format: Annotated[str, Query(pattern="^(csv|md|json)$")] = "csv",
) -> Response:
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, conversation_id)
        except cs.ChatError as exc:
            raise _NOT_FOUND from exc
        row = (
            await db.execute(
                select(ConversationMessage).where(
                    ConversationMessage.id == message_id,
                    ConversationMessage.conversation_id == conv.id,
                    ConversationMessage.role == "assistant",
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise _NOT_FOUND
        content = row.content or ""
        tokens = EXP.table_tokens(EXP.parse_tables(content))
        links = await _token_links(db, ident.user_id, tokens)
        data = EXP.build(
            conversation_id=conv.id,
            message_id=row.id,
            content=content,
            context=row.context,
            created_at=row.created_at.isoformat() if row.created_at else "",
            extra_links=links,
        )
    body = EXP.render(data, format)
    name = EXP.filename(conv.id, row.id, format)
    return Response(
        content=body.encode("utf-8"),
        media_type=EXP.MEDIA_TYPES[format],
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
