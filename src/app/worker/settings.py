"""ARQ worker.

2-bosqich (hozir): `sync_account`, `sync_chat`, `refresh_chats`,
                   `snapshot_metrics` (4.6 — vaqt qatori, har soat),
                   `incremental_sync_all` (10 daqiqa)
3-bosqich: `embed_messages`
4-bosqich: `build_daily_rollups`
7-bosqich: `run_scheduled_job`
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings
from arq.cron import cron

from app.config import get_settings
from app.logging import get_logger, setup_logging
from app.worker import tasks

log = get_logger("worker")


async def startup(ctx: dict[str, Any]) -> None:
    s = get_settings()
    setup_logging(s.log_level, json_output=s.is_prod)
    await _reset_stuck_syncs()
    log.info("worker.start", max_jobs=WorkerSettings.max_jobs)


async def _reset_stuck_syncs() -> None:
    """Worker o'lganda `running` qolib ketgan chatlar — qayta navbatga tushishi uchun idle."""
    from sqlalchemy import update

    from app.db.base import session_scope
    from app.db.models import Chat, SyncState

    try:
        async with session_scope() as db:
            res = await db.execute(
                update(Chat)
                .where(Chat.sync_state == SyncState.RUNNING)
                .values(sync_state=SyncState.IDLE, sync_error="worker restarted")
            )
            if res.rowcount:
                log.warning("worker.reset_stuck_syncs", count=res.rowcount)
    except Exception as exc:  # DB hali tayyor bo'lmasa — worker baribir ko'tarilsin
        log.warning("worker.reset_stuck_syncs_failed", error=str(exc)[:200])


async def shutdown(ctx: dict[str, Any]) -> None:
    from app.mtproto.pool import pool
    from app.worker.queue import close_queue

    await pool.close_all()
    await close_queue()
    log.info("worker.stop")


async def heartbeat(ctx: dict[str, Any]) -> None:
    log.debug("worker.heartbeat")


class WorkerSettings:
    # ARQ bu atributlarni klass darajasida o'qiydi — RUF012 bu yerda o'rinsiz.
    functions: ClassVar[list[Any]] = [
        tasks.sync_account,
        tasks.sync_chat,
        tasks.refresh_chats,
        tasks.snapshot_metrics,
        tasks.incremental_sync_all,
        tasks.embed_messages,
    ]
    cron_jobs: ClassVar[list[Any]] = [
        cron(heartbeat, minute={0, 30}, run_at_startup=False),
        # ⚠️ Snapshot cron'ini o'chirmang — views/reactions vaqt qatori shu yerdan (4.6)
        cron(tasks.snapshot_metrics, minute={5}, run_at_startup=False, timeout=3000),
        cron(tasks.incremental_sync_all, minute={0, 10, 20, 30, 40, 50}, run_at_startup=True),
        cron(tasks.embed_messages, minute={7, 22, 37, 52}, run_at_startup=False, timeout=1200),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4  # 3 akkaunt — parallellik kerak emas, flood-wait riski kamayadi
    job_timeout = 3600
