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


async def enqueue(
    function: str,
    *args: Any,
    job_id: str | None = None,
    defer_by: float | None = None,
    **kwargs: Any,
) -> str | None:
    """Ishni navbatga qo'yadi. `job_id` bir xil bo'lsa — dublikat qo'shilmaydi (ARQ)."""
    q = await get_queue()
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
