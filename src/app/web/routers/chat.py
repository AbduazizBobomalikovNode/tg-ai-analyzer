"""AI chat API.

GET    /api/conversations?limit=&before=
POST   /api/conversations                 {account_id?, title?}
PATCH  /api/conversations/{id}            {title}
GET    /api/conversations/{id}/messages?limit=&before=
DELETE /api/conversations/{id}
POST   /api/conversations/{id}/messages   {text, context?: {account_id, peer_id, limit}, deep?}
POST   /api/conversations/{id}/messages/stream            — SSE (ui/R1b)
POST   /api/conversations/{id}/messages/{mid}/regenerate  — oxirgi savolni qayta yugurtirish
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Coroutine
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, ConversationMessage
from app.logging import get_logger
from app.services import actions as ACT
from app.services import chat_service as cs
from app.services import evaluation as ev
from app.web.security import WebIdentity, locale_of, require_csrf, require_identity

log = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["chat"])


class ConversationIn(BaseModel):
    account_id: int | None = None
    title: str = Field(default="", max_length=128)


class ContextIn(BaseModel):
    account_id: int
    peer_id: int
    limit: int = Field(default=0, ge=0, le=5000)
    strategy: str = Field(default="auto", pattern="^(auto|recent|search|window)$")


class TitleIn(BaseModel):
    # 120 — `conversations.title` ustuni 128 belgi; zararsizlantirishdan keyin
    # ham sig'sin (`chat_service.rename_conversation`).
    title: str = Field(default="", max_length=120)


class RatingIn(BaseModel):
    rating: int = Field(ge=-1, le=1)  # 1 👍 · -1 👎 · 0 bekor
    comment: str = Field(default="", max_length=512)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    context: ContextIn | None = None
    deep: bool = False
    # auto — agent (tool'lar) yoki direct'ni o'zi tanlaydi; agent — majburan tool'lar;
    # direct — faqat oldindan berilgan kontekst; deep — `deep_search` quvuri
    # (oqimsiz yo'l; SSE varianti `routers/deep.py` da)
    mode: str = Field(default="auto", pattern="^(auto|agent|direct|deep)$")
    account_id: int | None = None  # agent rejimi uchun (chat tanlanmagan bo'lsa ham)


_STATUS = {
    "not_found": 404,
    "empty": 400,
    "too_long": 400,
    "context": 502,
    "llm": 502,
    "budget": 429,
    # H1: deep rejim qopqoqlari `chat_service.assert_deep_allowed` dan
    # keladi — `routers/deep.py::_http` bilan bir xil kodlar.
    "rate_limited": 429,
    "disabled": 403,
    "not_last": 409,  # ui/R1b: faqat oxirgi savolni qayta yugurtirish mumkin
}


def _chat_http(exc: cs.ChatError) -> HTTPException:
    return HTTPException(
        status_code=_STATUS.get(exc.code, 400),
        detail={"code": f"chat.err.{exc.code}", "detail": exc.detail},
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
        "task": m.task,
        "latency_ms": m.latency_ms,
        "cost_usd": m.cost_usd,
        "rating": m.rating,
        "auto": (
            {
                "relevance": m.auto_relevance,
                "usefulness": m.auto_usefulness,
                "grounded": m.auto_grounded,
                "note": m.auto_note,
            }
            if m.auto_relevance is not None
            else None
        ),
    }


def _parse_before(raw: str) -> datetime | None:
    """ISO-8601 kursor → datetime. Buzuq qiymat — kursorsiz (birinchi sahifa)."""
    value = raw.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "chat.err.bad_cursor", "detail": "before"}
        ) from exc


@router.get("")
async def list_conversations(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    before: Annotated[str, Query(max_length=64)] = "",
) -> dict[str, Any]:
    """Suhbatlar ro'yxati, yangisidan eskisiga.

    Sahifalash **kursor** bilan (`before` = oxirgi elementning `updated_at`),
    OFFSET emas: ro'yxat ochiq turganda yangi suhbat qo'shilsa OFFSET butun
    oynani siljitadi. To'liq sahifa qaytsa javobda `next_before` bo'ladi.
    """
    cursor = _parse_before(before)
    async with session_scope() as db:
        rows = await cs.list_conversations(db, ident.user_id, limit=limit, before=cursor)
        items = [
            {
                "id": c.id,
                "title": c.title,
                "account_id": c.account_id,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in rows
        ]
    next_before = items[-1]["updated_at"] if len(items) == limit and items else None
    return {"items": items, "next_before": next_before}


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


@router.patch("/{conversation_id}", dependencies=[Depends(require_csrf)])
async def rename_conversation(
    conversation_id: int,
    body: TitleIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    async with session_scope() as db:
        try:
            conv = await cs.rename_conversation(db, ident.user_id, conversation_id, body.title)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc
        return {"id": conv.id, "title": conv.title, "account_id": conv.account_id}


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    before: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Suhbat xabarlari — **eng yangisi oxirida**.

    `before` — kursor (xabar `id`): shundan eski `limit` ta qator. To'liq
    sahifa qaytsa `next_before` (eng eski qator `id`) beriladi; u bo'lmasa
    tarix boshiga yetilgan.
    """
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, conversation_id)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc
        rows = await cs.list_messages(db, conv.id, limit=limit, before_id=before or None)
        run_ids = [
            int(m.context["run_id"])
            for m in rows
            if m.role == "assistant" and m.context and m.context.get("run_id")
        ]
        actions = await ACT.actions_for_runs(db, run_ids)
        items = []
        for m in rows:
            d = _msg_out(m)
            rid = m.context.get("run_id") if m.context else None
            if rid and actions.get(int(rid)):
                d["actions"] = actions[int(rid)]
            items.append(d)
        return {
            "conversation": {"id": conv.id, "title": conv.title, "account_id": conv.account_id},
            "items": items,
            "next_before": items[0]["id"] if len(items) == limit and items else None,
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


async def _resolve_context(
    db: Any, ident: WebIdentity, body: MessageIn
) -> tuple[cs.ContextSpec | None, int | None]:
    """So'rovdagi chat/akkaunt havolalarini tekshiradi (egalik) va spec quradi.

    Oddiy, oqimli va `regenerate` yo'llari **bitta** shu funksiyadan o'tadi —
    egalik tekshiruvi uch joyda ajralib ketmasin.
    """
    s = get_settings()
    context: cs.ContextSpec | None = None
    if body.context is not None:
        acc = await db.get(Account, body.context.account_id)
        if acc is None or acc.user_id != ident.user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        context = cs.ContextSpec(
            account_id=body.context.account_id,
            peer_id=body.context.peer_id,
            limit=body.context.limit or s.web_context_default_messages,
            strategy=body.context.strategy,
        )

    acc_id = body.account_id
    if acc_id is not None:
        acc = await db.get(Account, acc_id)
        if acc is None or acc.user_id != ident.user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
    return context, acc_id


async def _collect_actions(reply: cs.Reply) -> list[dict[str, Any]]:
    if not (reply.context_used and reply.context_used.get("run_id")):
        return []
    run_id = int(reply.context_used["run_id"])
    async with session_scope() as db:
        got = await ACT.actions_for_runs(db, [run_id])
        return got.get(run_id, [])


def _reply_out(reply: cs.Reply, actions: list[dict[str, Any]]) -> dict[str, Any]:
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
        "latency_ms": reply.latency_ms,
        "cost_usd": reply.cost_usd,
        "actions": actions,
    }


@router.post("/{conversation_id}/messages", dependencies=[Depends(require_csrf)])
async def send_message(
    conversation_id: int,
    body: MessageIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    background: BackgroundTasks,
    request: Request,
) -> dict[str, Any]:
    from app.services import limits

    if not limits.check_chat_rate(ident.user_id):
        raise HTTPException(status_code=429, detail={"code": "chat.err.rate_limited"})
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, conversation_id)
            await limits.assert_within_budget(db, ident.user_id)
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc

        context, acc_id = await _resolve_context(db, ident, body)
        try:
            reply = await cs.send_message(
                db,
                user_id=ident.user_id,
                conversation=conv,
                text=body.text,
                context=context,
                deep=body.deep,
                mode=body.mode,
                account_id=acc_id,
                locale=locale_of(request),
            )
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc

    actions = await _collect_actions(reply)

    # javob yuborilgach — arzon model bilan fon baholash (foydalanuvchi kutmaydi)
    background.add_task(
        ev.auto_evaluate,
        reply.assistant_message_id,
        body.text,
        reply.text,
        had_context=reply.context_used is not None,
        **ev.run_flags(reply.context_used),
    )
    return _reply_out(reply, actions)


# ─── ui/R1b: oqimli javob (SSE) ──────────────────────────────────────────────
#
# Ramka va parser shartnomasi `routers/deep.py` bilan **bir xil**: `event: <nom>`
# + `data: <json>` + bo'sh satr, `POST` (shuning uchun `EventSource` emas,
# `fetch` + `ReadableStream`; CSRF odatdagidek `X-Requested-With`).
#
#   status {stage: "thinking" | "tool" | "writing", tool?: "<nom>"}
#   token  {text: "<bo'lak>"}
#   done   {assistant_message_id, model, provider, tokens_in, tokens_out,
#           cost_usd, latency_ms, actions: [...], + `_reply_out` maydonlari}
#   error  {code: "chat.err.<x>", detail: "<qisqa>"}
#
# Byudjet/shift xatolari **LLM chaqirilishidan oldin** `error` bo'lib chiqadi,
# HTTP status emas: javob sarlavhasi allaqachon 200 bilan ketgan bo'ladi.
# Auth (401) va CSRF (403) esa oqim boshlanishidan oldin, odatdagi status
# bilan — ular dependency darajasida.


def _sse(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


#: Fon vazifalariga havola: `asyncio` faqat zaif havola ushlaydi, ya'ni
#: havolasiz task o'rtasida yig'ib yuborilishi mumkin.
_BG_TASKS: set[asyncio.Task[Any]] = set()


def _spawn(coro: Coroutine[Any, Any, Any]) -> None:
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


async def _stream_turn(
    ident: WebIdentity,
    conversation_id: int,
    body: MessageIn,
    locale: str,
    *,
    regenerate_from: int | None = None,
) -> AsyncIterator[str]:
    """Bitta turn'ni bajaradi va hodisalarni SSE ramkasi bo'lib beradi.

    Ish alohida task'da yuriladi, hodisalar esa navbat orqali o'tadi: aks holda
    `send_message` tugagunicha generator hech narsa bermas edi va oqimning
    ma'nosi qolmasdi.
    """
    from app.services import limits

    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    async def events(kind: str, data: dict[str, Any]) -> None:
        await queue.put((kind, data))

    async def work() -> None:
        try:
            # "Ish boshlandi" — birinchi ramka darhol ketsin (proxy bufferini
            # ham ochadi). Byudjet/shift tekshiruvi shundan keyin.
            await events("status", {"stage": "thinking"})
            if not limits.check_chat_rate(ident.user_id):
                await queue.put(("error", {"code": "chat.err.rate_limited", "detail": ""}))
                return
            async with session_scope() as db:
                conv = await cs.get_conversation(db, ident.user_id, conversation_id)
                await limits.assert_within_budget(db, ident.user_id)
                text = body.text
                if regenerate_from is not None:
                    conv, text, _ctx = await cs.take_for_regenerate(
                        db, ident.user_id, conversation_id, regenerate_from
                    )
                context, acc_id = await _resolve_context(db, ident, body)
                reply = await cs.send_message(
                    db,
                    user_id=ident.user_id,
                    conversation=conv,
                    text=text,
                    context=context,
                    deep=body.deep,
                    mode=body.mode,
                    account_id=acc_id,
                    locale=locale,
                    events=events,
                )
            actions = await _collect_actions(reply)
            _spawn(
                ev.auto_evaluate(
                    reply.assistant_message_id,
                    text,
                    reply.text,
                    had_context=reply.context_used is not None,
                    **ev.run_flags(reply.context_used),
                )
            )
            await queue.put(("done", _reply_out(reply, actions)))
        except cs.ChatError as exc:
            await queue.put(("error", {"code": f"chat.err.{exc.code}", "detail": exc.detail}))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"code": "chat.err.not_found"}
            await queue.put(("error", detail))
        except Exception as exc:
            # L2: xom istisno matni klientga chiqmaydi (SQL/yo'l/sir bo'lishi
            # mumkin) — log'da to'liq, tashqariga faqat sinf nomi.
            log.exception("chat.stream_error", conversation_id=conversation_id)
            await queue.put(("error", {"code": "chat.err.internal", "detail": type(exc).__name__}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(work())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(*item)
    finally:
        if not task.done():
            task.cancel()


def _stream_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{conversation_id}/messages/stream", dependencies=[Depends(require_csrf)])
async def send_message_stream(
    conversation_id: int,
    body: MessageIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    request: Request,
) -> StreamingResponse:
    """Oddiy rejimlar uchun oqimli javob. Saqlash — oddiy yo'l bilan bir xil."""
    return _stream_response(_stream_turn(ident, conversation_id, body, locale_of(request)))


class RegenerateIn(BaseModel):
    """Qayta yugurtirish parametrlari. Matn **saqlangan** savoldan olinadi."""

    context: ContextIn | None = None
    deep: bool = False
    mode: str = Field(default="auto", pattern="^(auto|agent|direct|deep)$")
    account_id: int | None = None
    stream: bool = False


@router.post(
    "/{conversation_id}/messages/{message_id}/regenerate", dependencies=[Depends(require_csrf)]
)
async def regenerate(
    conversation_id: int,
    message_id: int,
    body: RegenerateIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    background: BackgroundTasks,
    request: Request,
) -> Any:
    """Oxirgi user turn'ini qayta yugurtiradi (eski javob o'chadi).

    `message_id` — user qatori yoki uning javobi. Faqat **oxirgi** turn:
    aks holda `409 chat.err.not_last`. `stream: true` bo'lsa javob
    `/messages/stream` bilan bir xil SSE ramkasida keladi.
    """
    from app.services import limits

    # `MessageIn` matni majburiy — bu yerda matn saqlangan qatordan olinadi,
    # shuning uchun joy egallovchi qo'yiladi (`_stream_turn` uni ishlatmaydi).
    turn = MessageIn(
        text="…",
        context=body.context,
        deep=body.deep,
        mode=body.mode,
        account_id=body.account_id,
    )
    if body.stream:
        return _stream_response(
            _stream_turn(
                ident, conversation_id, turn, locale_of(request), regenerate_from=message_id
            )
        )

    if not limits.check_chat_rate(ident.user_id):
        raise HTTPException(status_code=429, detail={"code": "chat.err.rate_limited"})
    async with session_scope() as db:
        try:
            await limits.assert_within_budget(db, ident.user_id)
            conv, text, _ctx = await cs.take_for_regenerate(
                db, ident.user_id, conversation_id, message_id
            )
            context, acc_id = await _resolve_context(db, ident, turn)
            reply = await cs.send_message(
                db,
                user_id=ident.user_id,
                conversation=conv,
                text=text,
                context=context,
                deep=body.deep,
                mode=body.mode,
                account_id=acc_id,
                locale=locale_of(request),
            )
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc

    actions = await _collect_actions(reply)
    background.add_task(
        ev.auto_evaluate,
        reply.assistant_message_id,
        text,
        reply.text,
        had_context=reply.context_used is not None,
        **ev.run_flags(reply.context_used),
    )
    return _reply_out(reply, actions)


@router.post("/{conversation_id}/messages/{message_id}/rate", dependencies=[Depends(require_csrf)])
async def rate(
    conversation_id: int,
    message_id: int,
    body: RatingIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    async with session_scope() as db:
        try:
            await cs.get_conversation(db, ident.user_id, conversation_id)
            row = await ev.rate_message(
                db,
                user_id=ident.user_id,
                message_id=message_id,
                rating=body.rating,
                comment=body.comment,
            )
        except cs.ChatError as exc:
            raise _chat_http(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail={"code": "chat.err.not_found"}) from exc
        return {"id": row.id, "rating": row.rating}
