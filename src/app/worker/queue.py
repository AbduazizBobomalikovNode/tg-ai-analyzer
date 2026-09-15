"""ARQ navbatiga ish qo'yish — api/bot shu orqali worker'ga topshiriq beradi."""

from __future__ import annotations

from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

_redis: ArqRedis | None = None


async def get_queue() -> ArqRedis:
    global _redis
    if _redis is None:
        _redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _redis


async def _clear_stale(q: Any, job_id: str) -> bool:
    """`job_id` band, lekin ish **yugurmayapti** bo'lsa — izini tozalaydi.

    ARQ deterministik `job_id` ni ikki maqsadda ishlatadi: parallel yugurishni
    to'sish (kerak) va natijani saqlash (`arq:result:<job_id>`, `keep_result`
    muddatiga). Ikkinchisi tufayli tugagan sync natijasi turgan paytda yangi
    `enqueue` `None` qaytaradi va UI "already_running" deb yolg'on ko'rsatardi,
    holbuki hech narsa yugurmayotgan edi.

    Shuning uchun holat tekshiriladi: `deferred`/`queued`/`in_progress` —
    rostdan band, tegilmaydi. `complete`/`not_found` — eski iz, natija kaliti
    o'chiriladi va chaqiruvchi qayta `enqueue` qiladi. `job.abort()` bu yerda
    noto'g'ri: u tugagan ishni bekor qilmaydi, faqat yana bir kalit qo'shadi.
    """
    from arq.constants import result_key_prefix
    from arq.jobs import Job, JobStatus

    try:
        status = await Job(job_id, redis=q).status()
    except Exception as exc:  # holatni bilib bo'lmasa — dublikat deb qoldiramiz
        log.warning("queue.status_failed", job_id=job_id, error=str(exc)[:200])
        return False
    if status not in (JobStatus.complete, JobStatus.not_found):
        return False
    await q.delete(result_key_prefix + job_id)
    log.info("queue.stale_cleared", job_id=job_id, status=str(status))
    return True


async def enqueue(
    function: str,
    *args: Any,
    job_id: str | None = None,
    defer_by: float | None = None,
    **kwargs: Any,
) -> str | None:
    """Ishni navbatga qo'yadi. `job_id` bir xil bo'lsa — dublikat qo'shilmaydi (ARQ).

    `None` qaytishi ikki xil bo'lishi mumkin: ish rostdan yugurayapti yoki
    `job_id` dan faqat **eski natija** qolgan. Ikkinchisida iz tozalanadi va
    bir marta qayta urinamiz (`_clear_stale`).
    """
    q = await get_queue()
    job = await q.enqueue_job(function, *args, _job_id=job_id, _defer_by=defer_by, **kwargs)
    if job is None and job_id and await _clear_stale(q, job_id):
        job = await q.enqueue_job(function, *args, _job_id=job_id, _defer_by=defer_by, **kwargs)
    if job is None:
        log.debug("queue.duplicate", function=function, job_id=job_id)
        return None
    log.info("queue.enqueued", function=function, job_id=job.job_id, defer_by=defer_by)
    return job.job_id


async def close_queue() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
