"""Sozlamalar va akkaunt boshqaruvi API (ui/R1b).

    GET  /api/settings/prefs               → {prefs: {...}}
    PUT  /api/settings/prefs   {…}         → {prefs: {...}}   (faqat tanilgan kalitlar)
    GET/PUT /api/settings/budget, POST/DELETE /api/settings/budget/boost
                                           → kunlik byudjet (pastda)
    GET/DELETE /api/settings/memory, PATCH/DELETE /api/settings/memory/{id},
    PUT  /api/settings/memory/toggle       → foydalanuvchi xotirasi (0026, pastda)
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
import math
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import case, func, select

from app.db.base import session_scope
from app.db.models import Account, AccountStatus, Chat, SyncState, User
from app.logging import get_logger
from app.services import ingestion as ing
from app.services import limits as LIM
from app.services import prefs as P
from app.services import sync_progress as SP
from app.services import user_memory as UM
from app.web.routers.chatdata import avatar_url
from app.web.security import WebIdentity, require_csrf, require_identity

log = get_logger(__name__)

router = APIRouter(tags=["settings"])

#: Poll oralig'i (sekund) — REST yo'li uchun frontend maslahati, SSE yo'lida
#: hodisalar orasidagi kutish.
SYNC_POLL_SECONDS = 2.0
#: Sync markazi: yugurayotganda jonli (1 s), bo'sh turganda arzon (5 s).
SYNC_POLL_RUNNING = 1.0
SYNC_POLL_IDLE = 5.0
#: SSE oqimining umumiy chegarasi: sync soatlab ketishi mumkin, lekin ochiq
#: ulanish abadiy turmasin (proxy timeout'lari, worker restart'lari).
SYNC_STREAM_MAX_SECONDS = 600.0
#: Bitta javobda ko'rsatiladigan chat soni — polling yengil qolsin.
SYNC_CHATS_LIMIT = 50
#: Navbat va "yaqinda tayyor" ro'yxatlarining uzunligi (UI ko'rsatadigan qadar).
SYNC_QUEUE_LIMIT = 8
SYNC_RECENT_LIMIT = 5
#: `worker/settings.py::WorkerSettings.max_jobs` nusxasi — api `arq` sozlamalarini
#: import qilmasligi uchun (u import paytida Redis DSN'ni tahlil qiladi).
#: Ikkisining tengligini `tests/test_sync_center.py` qo'riqlaydi.
SYNC_MAX_JOBS = 4


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


def _iso(v: Any) -> str | None:
    return v.isoformat() if hasattr(v, "isoformat") else (str(v) if v else None)


def _paused_until(current: dict[str, Any] | None, job: dict[str, Any] | None) -> str | None:
    """Pauza faqat **kelajakdagi** `paused_until` bo'lsa rost (eski kalit yolg'on
    "pauzada" ko'rsatmasin — TTL 10 daqiqa, FloodWait esa 30 soniya bo'lishi mumkin)."""
    raw = (current or {}).get("paused_until")
    if not raw:
        return None
    try:
        until = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    if until <= datetime.now(UTC):
        return None
    if job is not None and job.get("state") not in (None, SP.PAUSED, SP.RUNNING):
        return None
    return until.isoformat()


async def _counts(db: Any, account_id: int) -> tuple[dict[str, int], int, int]:
    """Bitta guruhlangan so'rov: holat bo'yicha son + `synced_total`/`total_estimate` yig'indisi.

    Yig'indi faqat **taxmini bor** chatlar bo'yicha — aks holda maxraj
    yolg'on kichik chiqadi va foiz 100 dan oshib ketadi.
    """
    rows = (
        await db.execute(
            select(
                Chat.sync_state,
                func.count(Chat.id),
                func.sum(case((Chat.total_estimate.is_(None), 0), else_=Chat.synced_total)),
                func.sum(func.coalesce(Chat.total_estimate, 0)),
            )
            .where(Chat.account_id == account_id)
            .group_by(Chat.sync_state)
        )
    ).all()
    counts: dict[str, int] = {}
    synced_sum = est_sum = 0
    for state, n, synced, est in rows:
        counts[str(state)] = int(n or 0)
        synced_sum += int(synced or 0)
        est_sum += int(est or 0)
    return counts, synced_sum, est_sum


def _pending_where(account_id: int) -> Any:
    """ "Navbatda" mezoni: hali ishlanadigan chat.

    `synced_min_id == 1` — tarix boshiga yetilgan sentinel (CLAUDE.md), ya'ni
    orqaga to'ldiradigan narsa qolmagan. `done` holati ham navbatda emas.
    """
    return (
        Chat.account_id == account_id,
        Chat.sync_state.in_((SyncState.IDLE, SyncState.RUNNING)),
        func.coalesce(Chat.synced_min_id, 0) != 1,
    )


def _queue_row(c: Chat) -> dict[str, Any]:
    return {
        "chat_id": c.id,
        "title": c.title,
        "avatar_url": avatar_url(c),
        "total_estimate": c.total_estimate,
        "synced_total": c.synced_total,
        "state": c.sync_state,
    }


async def _sync_status(db: Any, account_id: int, user_id: int | None = None) -> dict[str, Any]:
    """Sync markazi uchun to'liq holat: **dialoglar RPC'si yo'q**, faqat DB + Redis.

    Eski kalitlar (`running`, `chats`, `counts`, `poll_seconds`) o'z joyida —
    rebuild paytida eski frontend ham ishlayveradi.

    Rostgo'ylik: bir akkauntda chatlar **birma-bir** sinxronlanadi
    (`worker/tasks.sync_account` sikli), akkauntlar esa parallel
    (`max_jobs`). Shuning uchun `current` bitta chat, `parallel` esa
    akkauntlar soni haqida.
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
    counts, synced_sum, est_sum = await _counts(db, account_id)
    pending = int(
        (await db.execute(select(func.count(Chat.id)).where(*_pending_where(account_id))))
        .scalars()
        .first()
        or 0
    )
    queue_rows = (
        (
            await db.execute(
                select(Chat)
                .where(*_pending_where(account_id))
                .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
                .limit(SYNC_QUEUE_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    recent_rows = (
        (
            await db.execute(
                select(Chat)
                .where(Chat.account_id == account_id, Chat.sync_state == SyncState.DONE)
                .order_by(Chat.last_sync_at.desc().nulls_last(), Chat.id.desc())
                .limit(SYNC_RECENT_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    job = await SP.read_job(account_id)
    current = await SP.read_progress(account_id)
    paused = _paused_until(current, job)
    if current is not None:
        current = {**current, "paused_until": paused}

    running_chats = counts.get(str(SyncState.RUNNING), 0)
    job_state = (job or {}).get("state")
    running = bool(running_chats) or job_state in (SP.RUNNING, SP.QUEUED)
    if paused:
        running = True

    done = counts.get(str(SyncState.DONE), 0)
    failed = counts.get(str(SyncState.FAILED), 0)
    total = sum(counts.values())
    if est_sum:
        percent = min(100, round(100 * synced_sum / est_sum))
    else:
        percent = round(100 * done / total) if total else 0

    account = await db.get(Account, account_id)
    limits = ing.limits_for(getattr(account, "created_at", None))

    return {
        # ── eski shakl (buzilmaydi) ──
        "running": running,
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
        "counts": {
            **counts,
            "done": done,
            "running": running_chats,
            "pending": pending,
            "failed": failed,
            "paused": 1 if paused else 0,
            "total": total,
        },
        "poll_seconds": SYNC_POLL_RUNNING if running else SYNC_POLL_IDLE,
        # ── sync markazi ──
        "account_id": account_id,
        "job": job
        or {
            "state": SP.RUNNING if running else SP.DONE if done else SP.QUEUED,
            "started_at": None,
            "finished_at": None,
            "chats_done": done,
            "chats_total": total,
            "error": None,
        },
        "current": current,
        "percent_overall": percent,
        "queue": [_queue_row(c) for c in queue_rows],
        "recent_done": [
            {
                "chat_id": c.id,
                "title": c.title,
                "synced_total": c.synced_total,
                "finished_at": _iso(c.last_sync_at),
            }
            for c in recent_rows
        ],
        "failed": [
            {"chat_id": c.id, "title": c.title, "error": c.sync_error}
            for c in rows
            if c.sync_state == SyncState.FAILED
        ],
        "parallel": {
            "accounts_running": await _accounts_running(db, user_id),
            "max_jobs": SYNC_MAX_JOBS,
        },
        "ramp": {
            "limited": bool(limits.ramp_up),
            "note": "web.sync.ramp_note" if limits.ramp_up else "",
            "max_messages": limits.max_messages,
            "max_chats": limits.max_chats,
        },
        "last_sync_at": _iso(
            max((c.last_sync_at for c in recent_rows if c.last_sync_at), default=None)
        ),
    }


async def _account_ids(db: Any, user_id: int | None) -> list[int]:
    if user_id is None:
        return []
    rows = await db.execute(select(Account.id).where(Account.user_id == user_id))
    return [int(r[0]) for r in rows.all()]


async def _accounts_running(db: Any, user_id: int | None) -> int:
    """Parallel indikator: nechta akkaunt **ayni damda** sync qilyapti.

    Manba — Redis job kalitlari (`sync:job:<id>`), chunki DB'dagi
    `sync_state` job to'xtab qolsa ham `running` bo'lib osilib qolishi mumkin.
    """
    try:
        ids = await _account_ids(db, user_id)
    except Exception as exc:  # holat sahifasi bitta so'rov tufayli yiqilmasin
        log.debug("sync.accounts_running_failed", error=str(exc)[:160])
        return 0
    jobs = await SP.read_jobs(ids)
    return sum(1 for j in jobs.values() if j.get("state") in (SP.RUNNING, SP.PAUSED))


@router.get("/api/accounts/{account_id}/sync/status")
async def sync_status(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        await _own_account(db, ident.user_id, account_id)
        return await _sync_status(db, account_id, ident.user_id)


@router.get("/api/sync/overview")
async def sync_overview(
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    """Barcha akkauntlar — parallel indikatori uchun (DB'ga bitta so'rov + Redis).

    Bu yerda chat ro'yxati **yo'q**: har akkauntning to'liq holati
    `/api/accounts/{id}/sync/status` da.
    """
    async with session_scope() as db:
        rows = (
            (await db.execute(select(Account).where(Account.user_id == ident.user_id)))
            .scalars()
            .all()
        )
    jobs = await SP.read_jobs([int(a.id) for a in rows])
    items = []
    for a in rows:
        job = jobs.get(int(a.id))
        cur = await SP.read_progress(int(a.id)) if job and job.get("state") == SP.RUNNING else None
        items.append(
            {
                "account_id": int(a.id),
                "label": a.label,
                "status": a.status,
                "state": (job or {}).get("state"),
                "chats_done": (job or {}).get("chats_done", 0),
                "chats_total": (job or {}).get("chats_total", 0),
                "current_title": (cur or {}).get("title"),
                "paused_until": _paused_until(cur, job),
            }
        )
    running = sum(1 for it in items if it["state"] in (SP.RUNNING, SP.PAUSED))
    return {
        "accounts": items,
        "parallel": {"accounts_running": running, "max_jobs": SYNC_MAX_JOBS},
        "poll_seconds": SYNC_POLL_IDLE,
    }


@router.get("/api/accounts/{account_id}/sync/stream")
async def sync_stream(
    account_id: int, ident: Annotated[WebIdentity, Depends(require_identity)], request: Request
) -> StreamingResponse:
    """O'sha holat SSE bo'lib: yugurayotganda har 1 s, bo'sh turganda 5 s,
    sync tugagach `done`.

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
                    payload = await _sync_status(db, account_id, ident.user_id)
                yield f"event: status\ndata: {json.dumps(payload, default=str)}\n\n"
                if not payload["running"] or time.monotonic() > deadline:
                    yield f"event: done\ndata: {json.dumps({'running': payload['running']})}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(float(payload.get("poll_seconds") or SYNC_POLL_RUNNING))
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


# ─── kunlik byudjet (foydalanuvchi bo'yicha override + bugungi qo'shimcha) ───
#
#   GET    /api/settings/budget[?account_id=]      → holat (sarf, limit, boost)
#   PUT    /api/settings/budget        {token_budget?, cost_budget_usd?}
#          (berilmagan maydon o'zgarmaydi, `null` — global standartga qaytadi)
#   POST   /api/settings/budget/boost  {tokens?, usd?, hours?, account_id?}
#          (standart muddat — akkaunt `tz` sidagi keyingi yarim tun)
#   DELETE /api/settings/budget/boost
#
# Byudjet **foydalanuvchi** bo'yicha (sarf ham shunday sanaladi) — UI uni
# har akkaunt sozlamalarida ko'rsatadi, lekin qiymat hammasi uchun umumiy.
# `account_id` faqat vaqt zonasi uchun (egalik tekshiriladi).


class BudgetIn(BaseModel):
    token_budget: int | None = None
    cost_budget_usd: float | None = None


class BoostIn(BaseModel):
    tokens: int | None = None
    usd: float | None = None
    hours: float | None = None
    account_id: int | None = None


def _bad(code: str = "web.budget.err_invalid") -> HTTPException:
    return HTTPException(status_code=400, detail={"code": code})


def _check_tokens(v: int | None) -> None:
    if v is not None and not (0 <= v <= LIM.MAX_TOKENS):
        raise _bad()


def _check_usd(v: float | None) -> None:
    if v is not None and not (math.isfinite(v) and 0 <= v <= LIM.MAX_USD):
        raise _bad()


async def _budget_tz(db: Any, user_id: int, account_id: int | None) -> str:
    """Vaqt zonasi: berilgan (o'ziniki) akkaunt → birinchi akkaunt → standart."""
    if account_id is not None:
        acc = await _own_account(db, user_id, account_id)
        return str(acc.tz or LIM.DEFAULT_TZ)
    tz = (
        await db.execute(
            select(Account.tz).where(Account.user_id == user_id).order_by(Account.id).limit(1)
        )
    ).scalar_one_or_none()
    return str(tz or LIM.DEFAULT_TZ)


@router.get("/api/settings/budget")
async def get_budget(
    ident: Annotated[WebIdentity, Depends(require_identity)], account_id: int | None = None
) -> dict[str, Any]:
    async with session_scope() as db:
        tz = await _budget_tz(db, ident.user_id, account_id)
        return await LIM.budget_view(db, ident.user_id, tz=tz)


@router.put("/api/settings/budget", dependencies=[Depends(require_csrf)])
async def put_budget(
    body: BudgetIn,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    account_id: int | None = None,
) -> dict[str, Any]:
    fields = body.model_fields_set
    if not fields & {"token_budget", "cost_budget_usd"}:
        raise _bad()
    _check_tokens(body.token_budget)
    _check_usd(body.cost_budget_usd)
    kwargs: dict[str, Any] = {}
    if "token_budget" in fields:
        kwargs["token_budget"] = body.token_budget
    if "cost_budget_usd" in fields:
        kwargs["cost_budget_usd"] = body.cost_budget_usd
    async with session_scope() as db:
        tz = await _budget_tz(db, ident.user_id, account_id)
        await LIM.set_user_budget(db, ident.user_id, **kwargs)
        out = await LIM.budget_view(db, ident.user_id, tz=tz)
    await LIM.invalidate(ident.user_id)  # commit'dan keyin — kesh faqat yozilganni ko'rsin
    return out


@router.post("/api/settings/budget/boost", dependencies=[Depends(require_csrf)])
async def post_boost(
    body: BoostIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    tokens = int(body.tokens or 0)
    usd = float(body.usd or 0.0)
    _check_tokens(tokens)
    _check_usd(usd)
    if tokens <= 0 and usd <= 0:
        raise _bad("web.budget.err_boost_empty")
    if body.hours is not None and not (0 < body.hours <= LIM.MAX_BOOST_HOURS):
        raise _bad()
    async with session_scope() as db:
        tz = await _budget_tz(db, ident.user_id, body.account_id)
        until = (
            datetime.now(UTC) + timedelta(hours=float(body.hours))
            if body.hours is not None
            else LIM.next_local_midnight(tz)
        )
        await LIM.add_boost(db, ident.user_id, tokens=tokens, usd=usd, until=until)
        out = await LIM.budget_view(db, ident.user_id, tz=tz)
    await LIM.invalidate(ident.user_id)
    return out


@router.delete("/api/settings/budget/boost", dependencies=[Depends(require_csrf)])
async def delete_boost(
    ident: Annotated[WebIdentity, Depends(require_identity)], account_id: int | None = None
) -> dict[str, Any]:
    async with session_scope() as db:
        tz = await _budget_tz(db, ident.user_id, account_id)
        await LIM.clear_boost(db, ident.user_id)
        out = await LIM.budget_view(db, ident.user_id, tz=tz)
    await LIM.invalidate(ident.user_id)
    return out


# ─── foydalanuvchi xotirasi (0026) ───────────────────────────────────────────
#
#     GET    /api/settings/memory              → {enabled, items: [...], max}
#     DELETE /api/settings/memory              → hammasini o'chirish
#     PATCH  /api/settings/memory/{id}         {enabled} → bitta yozuvni yoqish/o'chirish
#     DELETE /api/settings/memory/{id}         → bitta yozuvni o'chirish
#     PUT    /api/settings/memory/toggle       {enabled} → global kalit
#
# Yagona yozuvchi — `services/user_memory.py`. O'zgartiruvchilar CSRF talab
# qiladi; yozuv faqat egasiniki (`user_id` bo'yicha filtr SQL'da).


class MemoryItemIn(BaseModel):
    enabled: bool


class MemoryToggleIn(BaseModel):
    enabled: bool


async def _memory_view(db: Any, user_id: int) -> dict[str, Any]:
    return {
        "enabled": await UM.get_enabled(db, user_id),
        "items": [it.to_dict() for it in await UM.list_all(db, user_id)],
        "max": UM.MAX_ITEMS,
    }


@router.get("/api/settings/memory")
async def get_memory(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        return await _memory_view(db, ident.user_id)


@router.delete("/api/settings/memory", dependencies=[Depends(require_csrf)])
async def delete_memory(
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    async with session_scope() as db:
        removed = await UM.delete_all(db, ident.user_id)
        out = await _memory_view(db, ident.user_id)
    log.info("memory.cleared", user_id=ident.user_id, removed=removed)
    return {**out, "removed": removed}


@router.put("/api/settings/memory/toggle", dependencies=[Depends(require_csrf)])
async def toggle_memory(
    body: MemoryToggleIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        if not await UM.set_enabled(db, ident.user_id, body.enabled):
            raise HTTPException(status_code=401, detail={"code": "unauthorized"})
        return await _memory_view(db, ident.user_id)


@router.patch("/api/settings/memory/{item_id}", dependencies=[Depends(require_csrf)])
async def patch_memory_item(
    item_id: int, body: MemoryItemIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        item = await UM.set_item_enabled(db, ident.user_id, item_id, body.enabled)
        if item is None:
            raise HTTPException(status_code=404, detail={"code": "web.memory.err_not_found"})
        return {"item": item.to_dict()}


@router.delete("/api/settings/memory/{item_id}", dependencies=[Depends(require_csrf)])
async def delete_memory_item(
    item_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        if not await UM.delete_item(db, ident.user_id, item_id):
            raise HTTPException(status_code=404, detail={"code": "web.memory.err_not_found"})
    return {"ok": True, "id": item_id}
