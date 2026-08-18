"""ARQ worker.

0-bosqichda faqat skeleton + cron slot'lari. Keyingi bosqichlarda to'ladi:
  2-bosqich: `sync_chat_history`, `snapshot_metrics` (4.6 — vaqt qatori)
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

log = get_logger("worker")


async def startup(ctx: dict[str, Any]) -> None:
    s = get_settings()
    setup_logging(s.log_level, json_output=s.is_prod)
    log.info("worker.start")


async def shutdown(ctx: dict[str, Any]) -> None:
    log.info("worker.stop")


async def heartbeat(ctx: dict[str, Any]) -> None:
    log.debug("worker.heartbeat")


class WorkerSettings:
    # ARQ bu atributlarni klass darajasida o'qiydi — RUF012 bu yerda o'rinsiz.
    functions: ClassVar[list[Any]] = []
    cron_jobs: ClassVar[list[Any]] = [cron(heartbeat, minute={0, 30}, run_at_startup=False)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4  # 3 akkaunt — parallellik kerak emas, flood-wait riski kamayadi
    job_timeout = 3600
