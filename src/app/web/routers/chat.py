"""AI chat API.

GET    /api/conversations
POST   /api/conversations                 {account_id?, title?}
GET    /api/conversations/{id}/messages
DELETE /api/conversations/{id}
POST   /api/conversations/{id}/messages   {text, context?: {account_id, peer_id, limit}, deep?}
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, ConversationMessage
from app.services import chat_service as cs
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api/conversations", tags=["chat"])


class ConversationIn(BaseModel):
    account_id: int | None = None
    title: str = Field(default="", max_length=128)


class ContextIn(BaseModel):
    account_id: int
    peer_id: int
    limit: int = Field(default=0, ge=0, le=1000)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    context: ContextIn | None = None
    deep: bool = False


def _chat_http(exc: cs.ChatError) -> HTTPException:
    status = {"not_found": 404, "empty": 400, "too_long": 400, "context": 502, "llm": 502}.get(
        exc.code, 400
    )
    return HTTPException(
        status_code=status, detail={"code": f"chat.err.{exc.code}", "detail": exc.detail}
    )


def _msg_out(m: ConversationMessage) -> dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "model": m.model,
        "provider": m.provider,
        "tokens_in": m.tokens_in,
        "tokens_out": m.tokens_out,
        "context": m.context,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


@router.get("")
async def list_conversations(
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    async with session_scope() as db:
        rows = await cs.list_conversations(db, ident.user_id)
        return {
            "items": [
                {
                    "id": c.id,
                    "title": c.title,
                    "account_id": c.account_id,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in rows
            ]
        }


@router.post("", dependencies=[Depends(require_csrf)])
async def create_conversation(
    body: ConversationIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        if body.account_id is not None:
            acc = await db.get(Account, body.account_id)
            if acc is None or acc.user_id != ident.user_id:
                raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        conv = await cs.create_conversation(
            db, ident.user_id, account_id=body.account_id, title=body.title
        )
        return {"id": conv.id, "title": conv.title, "account_id": conv.account_id}


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, conversation_id)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc
        rows = await cs.list_messages(db, conv.id)
        return {
            "conversation": {"id": conv.id, "title": conv.title, "account_id": conv.account_id},
            "items": [_msg_out(m) for m in rows],
        }


@router.delete("/{conversation_id}", dependencies=[Depends(require_csrf)])
async def delete_conversation(
    conversation_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, str]:
    async with session_scope() as db:
        try:
            await cs.delete_conversation(db, ident.user_id, conversation_id)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc
    return {"status": "deleted"}


@router.post("/{conversation_id}/messages", dependencies=[Depends(require_csrf)])
async def send_message(
    conversation_id: int,
    body: MessageIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    s = get_settings()
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, conversation_id)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc

        context: cs.ContextSpec | None = None
        if body.context is not None:
            acc = await db.get(Account, body.context.account_id)
            if acc is None or acc.user_id != ident.user_id:
                raise HTTPException(status_code=404, detail={"code": "account.not_found"})
            context = cs.ContextSpec(
                account_id=body.context.account_id,
                peer_id=body.context.peer_id,
                limit=body.context.limit or s.web_context_default_messages,
            )

        try:
            reply = await cs.send_message(
                db,
                user_id=ident.user_id,
                conversation=conv,
                text=body.text,
                context=context,
                deep=body.deep,
            )
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc

    return {
        "conversation_id": reply.conversation_id,
        "user_message_id": reply.user_message_id,
        "assistant_message_id": reply.assistant_message_id,
        "text": reply.text,
        "model": reply.model,
        "provider": reply.provider,
        "tokens_in": reply.tokens_in,
        "tokens_out": reply.tokens_out,
        "context": reply.context_used,
    }
