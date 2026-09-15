"""Sozlamalar va akkaunt boshqaruvi API (ui/R1b).

    GET  /api/settings/prefs               → {prefs: {...}}
    PUT  /api/settings/prefs   {…}         → {prefs: {...}}   (faqat tanilgan kalitlar)
    DELETE /api/accounts/{id}              → akkauntni uzish (ma'lumot qoladi)
    GET  /api/accounts/{id}/sync/status    → yengil holat (5 s poll uchun)
    GET  /api/accounts/{id}/sync/stream    → o'sha holat SSE bo'lib (2 s)

Nega alohida modul (`routers/tg.py` ham `/api/accounts` prefiksida): `tg.py`
Telegram bilan jonli ishlaydigan o'qish yo'llari, bu yerda esa faqat DB va
sozlama. FastAPI bir prefiksda bir nechta routerni bemalol ko'taradi.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.db.base import session_scope
from app.db.models import Account, AccountStatus, Chat, SyncState, User
from app.logging import get_logger
from app.services import prefs as P
from app.web.security import WebIdentity, require_csrf, require_identity

log = get_logger(__name__)

router = APIRouter(tags=["settings"])

#: Poll oralig'i (sekund) — REST yo'li uchun frontend maslahati, SSE yo'lida
#: hodisalar orasidagi kutish.
SYNC_POLL_SECONDS = 2.0
#: SSE oqimining umumiy chegarasi: sync soatlab ketishi mumkin, lekin ochiq
#: ulanish abadiy turmasin (proxy timeout'lari, worker restart'lari).
SYNC_STREAM_MAX_SECONDS = 600.0
#: Bitta javobda ko'rsatiladigan chat soni — polling yengil qolsin.
SYNC_CHATS_LIMIT = 50


# ─── prefs ───────────────────────────────────────────────────────────────────


class PrefsIn(BaseModel):
    """Erkin JSON: validatsiya `services/prefs.validate` da (noma'lum kalit tashlanadi).

    Pydantic bu yerda faqat "obyekt keldimi" degan tekshiruv; kalitlarning
    ro'yxati **bitta** joyda (`services/prefs.ALLOWED`) tursin, ikki nusxada
    emas — aks holda ular ajralib ketadi.
    """

    model_config = ConfigDict(extra="allow")


async def _get_user(db: Any, user_id: int) -> User:
    user: User | None = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail={"code": "unauthorized"})
    return user


@router.get("/api/settings/prefs")
async def get_prefs(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        user = await _get_user(db, ident.user_id)
        return {"prefs": P.normalize(user.prefs)}


@router.put("/api/settings/prefs", dependencies=[Depends(require_csrf)])
async def put_prefs(
    body: PrefsIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    """Qisman yangilash (merge). Noma'lum/buzuq kalit jimgina tashlanadi."""
    async with session_scope() as db:
        user = await _get_user(db, ident.user_id)
        merged = P.merge(user.prefs, body.model_dump())
        user.prefs = merged
        # JSONB — o'rniga yangi lug'at qo'yiladi (mutatsiya ORM'ga ko'rinmaydi).
        await db.flush()
        return {"prefs": merged}


# ─── akkauntni uzish ─────────────────────────────────────────────────────────


async def _own_account(db: Any, user_id: int, account_id: int) -> Account:
    acc: Account | None = await db.get(Account, account_id)
    if acc is None or acc.user_id != user_id:
        raise HTTPException(status_code=404, detail={"code": "account.not_found"})
    return acc


async def _abort_jobs(account_id: int) -> None:
    """Akkauntning sync joblarini bekor qiladi (Redis yo'q bo'lsa — jim o'tadi).

    Job'ni to'xtatish **kafolat emas**: ARQ `abort` faqat navbatdagi yoki
    abort'ni tekshiradigan ishga ta'sir qiladi. Asosiy himoya boshqa joyda —
    session envelope o'chirilgan, ya'ni ingestion MTProto'ga umuman ulana
    olmaydi va birinchi urinishda to'xtaydi.
    """
    try:
        from arq.jobs import Job

        from app.worker.queue import get_queue
        from app.worker.tasks import job_id_account

        q = await get_queue()
        await Job(job_id_account(account_id), redis=q).abort(timeout=0)
    except Exception as exc:
        log.info("account.abort_skipped", account_id=account_id, error=str(exc)[:200])


@router.delete("/api/accounts/{account_id}", dependencies=[Depends(require_csrf)])
async def unlink_account(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    """Akkauntni uzadi: saqlangan session o'chadi, status `revoked`.

    **`auth.LogOut` chaqirilmaydi** — u allowlist'da `DENIED` (destruktiv:
    foydalanuvchining Telegram sessiyasini haqiqatan o'ldiradi va uni biz
    hech qachon qilmaymiz, invariant 1). Bu yerda faqat *bizdagi* nusxa
    o'chadi; egasi sessiyani Telegram ilovasidan ham uzishi mumkin.

    Ma'lumot (chatlar, xabarlar, suhbatlar) **qoladi** — uzish ma'lumot
    o'chirish emas. Qayta ulash (QR login) o'sha akkaunt qatorini tiklaydi.
    """
    from app.services.session_store import revoke_session

    async with session_scope() as db:
        acc = await _own_account(db, ident.user_id, account_id)
        await revoke_session(db, acc, status=AccountStatus.REVOKED)
    await _abort_jobs(account_id)
    log.info("account.unlinked", user_id=ident.user_id, account_id=account_id)
    return {"status": "revoked", "id": account_id}


# ─── sync holati ─────────────────────────────────────────────────────────────


def _progress(c: Chat) -> int:
    est = c.total_estimate or 0
    if est:
        return min(100, round(100 * c.synced_total / est))
    return 100 if c.sync_state == SyncState.DONE else 0


async def _sync_status(db: Any, account_id: int) -> dict[str, Any]:
    """Yengil holat: **dialoglar RPC'si yo'q**, faqat DB.

    `chats` — hozir yugurayotgan va xato bergan chatlar (5 sekundda bir
    so'raladi, shuning uchun butun registr qaytarilmaydi; to'liq ro'yxat
    `GET /api/accounts/{id}/chats` da).
    """
    rows = (
        (
            await db.execute(
                select(Chat)
                .where(
                    Chat.account_id == account_id,
                    Chat.sync_state.in_((SyncState.RUNNING, SyncState.FAILED)),
                )
                .order_by(Chat.sync_state, Chat.id)
                .limit(SYNC_CHATS_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    counts_rows = await db.execute(
        select(Chat.sync_state, func.count(Chat.id))
        .where(Chat.account_id == account_id)
        .group_by(Chat.sync_state)
    )
    counts = {str(state): int(n) for state, n in counts_rows.all()}
    return {
        "running": counts.get(str(SyncState.RUNNING), 0) > 0,
        "chats": [
            {
                "chat_id": c.id,
                "state": c.sync_state,
                "progress": _progress(c),
                "synced_total": c.synced_total,
                "total_estimate": c.total_estimate,
                "error": c.sync_error,
            }
            for c in rows
        ],
        "counts": counts,
        "poll_seconds": SYNC_POLL_SECONDS,
    }


@router.get("/api/accounts/{account_id}/sync/status")
async def sync_status(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        await _own_account(db, ident.user_id, account_id)
        return await _sync_status(db, account_id)


@router.get("/api/accounts/{account_id}/sync/stream")
async def sync_stream(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)], request: Request
) -> StreamingResponse:
    """O'sha holat SSE bo'lib (`status` har 2 s), sync tugagach `done`.

    Ramka `routers/deep.py` bilan bir xil. `GET` — o'zgartirmaydi, shuning
    uchun CSRF header talab qilinmaydi va oddiy `EventSource` ham ishlaydi.
    """
    async with session_scope() as db:
        await _own_account(db, ident.user_id, account_id)

    async def gen() -> AsyncIterator[str]:
        deadline = time.monotonic() + SYNC_STREAM_MAX_SECONDS
        try:
            while True:
                async with session_scope() as db:
                    payload = await _sync_status(db, account_id)
                yield f"event: status\ndata: {json.dumps(payload, default=str)}\n\n"
                if not payload["running"] or time.monotonic() > deadline:
                    yield f"event: done\ndata: {json.dumps({'running': payload['running']})}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(SYNC_POLL_SECONDS)
        except asyncio.CancelledError:  # klient uzildi
            raise
        except Exception as exc:
            log.exception("sync.stream_error", account_id=account_id)
            detail = json.dumps({"code": "sync.err.internal", "detail": type(exc).__name__})
            yield f"event: error\ndata: {detail}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
