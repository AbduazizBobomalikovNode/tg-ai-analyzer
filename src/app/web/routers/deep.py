"""Deep search API (2-bosqich, arxitektura v2, 7-bo'lim).

POST /api/deep/stream   {conversation_id, question, deep?} → text/event-stream
POST /api/deep          {conversation_id, question, deep?} → DeepResult JSON

Ikkala yo'l ham **bitta** quvurni (`services/deep_search.run_deep_search`)
chaqiradi va yozuvni o'sha quvur bajaradi (yagona yozuvchi) — oqimli va
oqimsiz javob bir-biridan farq qilmaydi, faqat yetkazib berish usuli boshqa.

Nega SSE va nega `POST`: hodisalar ketma-ket va bir tomonlama (`plan` →
`search` → `cluster` → `rerank` → `token` → `coverage` → `done`), lekin
`EventSource` POST qila olmaydi — frontend `fetch` + `ReadableStream` bilan
o'qiydi (`web/static/chat.js`). Shu sababli CSRF himoyasi ham odatdagidek
(`X-Requested-With`) ishlaydi.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db.base import session_scope
from app.db.models import Account
from app.logging import get_logger
from app.services import chat_service as cs
from app.services import deep_search as DS
from app.web.security import WebIdentity, locale_of, require_csrf, require_identity

log = get_logger(__name__)

router = APIRouter(prefix="/api/deep", tags=["deep"])


class DeepIn(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1, max_length=20_000)
    deep: bool = True


def _http(code: str, detail: str = "") -> HTTPException:
    status = {
        "not_found": 404,
        "empty": 400,
        "budget": 429,
        "rate_limited": 429,
        "disabled": 403,
        "llm": 502,
        "taint": 500,
    }.get(code, 400)
    return HTTPException(status_code=status, detail={"code": f"chat.err.{code}", "detail": detail})


async def _prepare(ident: WebIdentity, body: DeepIn) -> int:
    """Umumiy tekshiruvlar: rejim yoqilganmi, shift, egalik, akkaunt. → account_id."""
    # Rejim/shift qopqog'i **yagona** joyda (`chat_service.assert_deep_allowed`) —
    # chat API'dagi `mode="deep"` ham xuddi shu funksiyadan o'tadi (H1).
    try:
        cs.assert_deep_allowed(ident.user_id)
    except cs.ChatError as exc:
        raise _http(exc.code, exc.detail) from exc
    async with session_scope() as db:
        try:
            conv = await cs.get_conversation(db, ident.user_id, body.conversation_id)
        except cs.ChatError as exc:
            raise _http(exc.code, exc.detail) from exc
        if conv.account_id is None:
            raise _http("context", "no_account")
        acc = await db.get(Account, conv.account_id)
        if acc is None or acc.user_id != ident.user_id:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        return int(conv.account_id)


def _sse(event: DS.DeepEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.kind}\ndata: {payload}\n\n"


@router.post("/stream", dependencies=[Depends(require_csrf)])
async def deep_stream(
    body: DeepIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    request: Request,
) -> StreamingResponse:
    account_id = await _prepare(ident, body)
    locale = locale_of(request)

    async def gen() -> AsyncIterator[str]:
        try:
            async for event in DS.stream_deep_search(
                session_scope,
                account_id,
                body.conversation_id,
                body.question,
                locale=locale,
                deep=body.deep,
            ):
                yield _sse(event)
        except Exception as exc:
            # L2: xom istisno matni klientga chiqmaydi (SQL, yo'l, sir bo'lishi
            # mumkin) — log'da to'liq turadi, tashqariga faqat sinf nomi.
            log.exception("deep.stream_error", conversation_id=body.conversation_id)
            yield _sse(DS.DeepEvent("error", {"code": "internal", "detail": type(exc).__name__}))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("", dependencies=[Depends(require_csrf)])
async def deep_once(
    body: DeepIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    request: Request,
) -> dict[str, Any]:
    """Oqimsiz variant (testlar va oddiy klientlar uchun) — o'sha natija."""
    account_id = await _prepare(ident, body)
    try:
        result = await DS.run_deep_search(
            session_scope,
            account_id,
            body.conversation_id,
            body.question,
            locale=locale_of(request),
            deep=body.deep,
        )
    except DS.DeepSearchError as exc:
        raise _http(exc.code, exc.detail) from exc
    return result.to_dict()
