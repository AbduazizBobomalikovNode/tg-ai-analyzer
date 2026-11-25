"""Sync progressi — **o'tkinchi** holat, Redis'da (jadval emas).

Nega DB emas: har batchda (100 xabar) bitta yozuv chiqadi, bu soatiga
minglab UPDATE degani va uning hech biri tarix emas — sahifa yopilgach
qiymat keraksiz. Shuning uchun ikkita kalit, 10 daqiqalik TTL:

    sync:progress:<account_id>   hozir qaysi chat, qancha, qanday tezlikda
    sync:job:<account_id>        akkaunt ishining holati (queued…failed)

`chats.synced_total` / `sync_state` **o'z joyida qoladi** — ular haqiqat
manbai; bu yerdagi hamma narsa faqat ko'rsatkich. Shuning uchun Redis
yo'q bo'lsa ingestion **to'xtamaydi**: har funksiya xatoni yutadi va
log'ga yozadi (`log.debug`), chaqiruvchi hech qachon xato ko'rmaydi.

Tezlik (`rate_per_s`) — oxirgi 5 batch bo'yicha EMA. Oddiy
"fetched/elapsed" o'rtachasi FloodWait pauzasidan keyin soatlab yolg'on
past qiymat ko'rsatardi; EMA esa oxirgi batchlarga og'irlik beradi.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

#: Kalit muddati — sync tugab qolsa ham ko'rsatkich abadiy osilib qolmasin.
PROGRESS_TTL = 600
#: EMA oynasi (batch) va undan kelib chiqadigan silliqlash koeffitsienti.
EMA_WINDOW = 5
EMA_ALPHA = 2.0 / (EMA_WINDOW + 1)

#: `job.state` qiymatlari — frontend shu to'plamga tayanadi.
QUEUED, RUNNING, PAUSED, DONE, FAILED = "queued", "running", "paused", "done", "failed"

_client: Any = None


def progress_key(account_id: int) -> str:
    return f"sync:progress:{account_id}"


def job_key(account_id: int) -> str:
    return f"sync:job:{account_id}"


def now_iso(ts: float | None = None) -> str:
    """UTC ISO-8601 — barcha vaqt maydonlari shu formatda (frontend `Date` o'qiydi)."""
    return datetime.fromtimestamp(ts if ts is not None else time.time(), UTC).isoformat()


_iso = now_iso


# ─── Redis (jarayon bo'yicha bitta ulanish, xatoda qayta yaratiladi) ─────────


async def _redis() -> Any:
    global _client
    if _client is None:
        import redis.asyncio as aioredis

        _client = aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]
    return _client


def _drop_client() -> None:
    """Ulanish buzilgan — keyingi chaqiruv yangisini yasaydi."""
    global _client
    _client = None


async def _set(key: str, payload: dict[str, Any]) -> None:
    try:
        r = await _redis()
        await r.set(key, json.dumps(payload, default=str), ex=PROGRESS_TTL)
    except Exception as exc:  # ko'rsatkich hech qachon sync'ni to'xtatmaydi
        _drop_client()
        log.debug("sync_progress.write_failed", key=key, error=str(exc)[:120])


async def _get(key: str) -> dict[str, Any] | None:
    try:
        r = await _redis()
        raw = await r.get(key)
    except Exception as exc:
        _drop_client()
        log.debug("sync_progress.read_failed", key=key, error=str(exc)[:120])
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


async def _delete(key: str) -> None:
    try:
        r = await _redis()
        await r.delete(key)
    except Exception as exc:
        _drop_client()
        log.debug("sync_progress.delete_failed", key=key, error=str(exc)[:120])


async def close() -> None:
    """Jarayon tugaganda ulanishni yopadi (test va worker shutdown uchun)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as exc:  # pragma: no cover — yopilayotgan jarayon
            log.debug("sync_progress.close_failed", error=str(exc)[:120])
        _client = None


# ─── progress ────────────────────────────────────────────────────────────────


async def publish_progress(account_id: int, payload: dict[str, Any]) -> None:
    await _set(progress_key(account_id), payload)


async def read_progress(account_id: int) -> dict[str, Any] | None:
    return await _get(progress_key(account_id))


async def clear_progress(account_id: int) -> None:
    await _delete(progress_key(account_id))


# ─── job ─────────────────────────────────────────────────────────────────────


async def publish_job(account_id: int, **fields: Any) -> dict[str, Any]:
    """Job holatini **birlashtirib** yozadi (`started_at` qayta yozilmasin).

    Qaytaradi: yozilgan to'liq lug'at (test va chaqiruvchi uchun).
    """
    current = await _get(job_key(account_id)) or {}
    merged: dict[str, Any] = {
        "account_id": account_id,
        "state": QUEUED,
        "started_at": None,
        "finished_at": None,
        "chats_done": 0,
        "chats_total": 0,
        "error": None,
        **current,
        **fields,
        "updated_at": _iso(),
    }
    await _set(job_key(account_id), merged)
    return merged


async def read_job(account_id: int) -> dict[str, Any] | None:
    return await _get(job_key(account_id))


async def read_jobs(account_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Bir nechta akkaunt — parallel indikator uchun (`/api/sync/overview`)."""
    out: dict[int, dict[str, Any]] = {}
    for aid in account_ids:
        job = await read_job(aid)
        if job is not None:
            out[aid] = job
    return out


# ─── batch tracker ───────────────────────────────────────────────────────────


class ChatTracker:
    """Bitta chat sync'i davomidagi hisoblagich (EMA tezlik + ETA).

    Bir `sync_chat` chaqiruvi uchun bitta nusxa. `publish()` har batchda
    chaqiriladi; hisob-kitob sof Python, Redis'ga bitta `SET`.
    """

    __slots__ = (
        "_last",
        "_last_fetched",
        "_last_t",
        "_rate",
        "account_id",
        "budget",
        "chat_id",
        "phase",
        "started_at",
        "title",
    )

    def __init__(
        self,
        account_id: int,
        chat_id: int,
        title: str,
        *,
        budget: int,
        started_at: float | None = None,
    ) -> None:
        self.account_id = int(account_id)
        self.chat_id = int(chat_id)
        self.title = title or ""
        self.budget = int(budget)
        self.started_at = started_at if started_at is not None else time.time()
        self.phase = "new"
        self._rate: float | None = None
        self._last_t = self.started_at
        self._last_fetched = 0
        self._last: dict[str, Any] | None = None

    def _observe(self, fetched: int, now: float) -> None:
        dt = now - self._last_t
        delta = int(fetched) - self._last_fetched
        self._last_t = now
        self._last_fetched = int(fetched)
        if dt <= 0 or delta <= 0:
            return
        inst = delta / dt
        self._rate = inst if self._rate is None else self._rate + EMA_ALPHA * (inst - self._rate)

    def payload(
        self,
        *,
        fetched: int,
        synced_total: int,
        total_estimate: int | None = None,
        phase: str | None = None,
        paused_until: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        self._observe(fetched, now)
        if phase:
            self.phase = phase
        budget_left = max(0, self.budget - int(fetched))
        remaining = budget_left
        if total_estimate:
            remaining = min(remaining, max(0, int(total_estimate) - int(synced_total)))
        eta = round(remaining / self._rate, 1) if self._rate and self._rate > 0 else None
        self._last = {
            "chat_id": self.chat_id,
            "title": self.title,
            "phase": self.phase,
            "fetched": int(fetched),
            "synced_total": int(synced_total),
            "total_estimate": int(total_estimate) if total_estimate else None,
            "batch_at": _iso(now),
            "started_at": _iso(self.started_at),
            "rate_per_s": round(self._rate, 2) if self._rate else None,
            "eta_s": eta,
            "paused_until": paused_until,
            "job_budget_left": budget_left,
        }
        return self._last

    async def publish(self, **kw: Any) -> None:
        await publish_progress(self.account_id, self.payload(**kw))

    async def publish_paused(self, retry_after: int) -> None:
        """FloodWait — oxirgi ko'rsatkichni `paused_until` bilan qayta yozadi."""
        last = dict(self._last or {"chat_id": self.chat_id, "title": self.title})
        last["paused_until"] = _iso(time.time() + max(0, int(retry_after)))
        last["rate_per_s"] = None
        last["eta_s"] = None
        self._last = last
        await publish_progress(self.account_id, last)
