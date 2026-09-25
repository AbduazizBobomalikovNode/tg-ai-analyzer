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

from app.config import embed_dims_mismatch, get_settings
from app.logging import get_logger, setup_logging
from app.worker import tasks

log = get_logger("worker")


async def startup(ctx: dict[str, Any]) -> None:
    s = get_settings()
    setup_logging(s.log_level, json_output=s.is_prod)
    from app.observability import init_sentry, start_metrics_server

    init_sentry("worker")
    try:
        start_metrics_server()  # WORKER_METRICS_PORT=0 → hech narsa qilmaydi
    except Exception as exc:  # port band bo'lsa worker baribir ko'tarilsin
        log.warning("worker.metrics_server_failed", error=str(exc)[:200])
    # L10: `EMBED_DIMS` DB ustuni bilan mos kelmasa embedding yozuvi butunlay
    # yiqiladi (har INSERT). Worker — asosiy yozuvchi, shuning uchun startupda.
    dims_error = embed_dims_mismatch(s)
    if dims_error:
        log.warning("worker.embed_dims_mismatch", error=dims_error)
    if s.llm_flex_background:
        # Fon LLM ishlari Flex'da — SDK uni bilmasa bu yerda (startupda) bir
        # marta `llm.flex_unsupported` ogohlantirishi, keyin standart tier.
        try:
            from app.llm.gemini import flex_supported

            flex_supported()
        except Exception as exc:  # SDK import xatosi worker'ni to'xtatmasin
            log.warning("worker.flex_check_failed", error=str(exc)[:200])
    await _reset_stuck_syncs()
    await _start_updates_listener()
    log.info("worker.start", max_jobs=WorkerSettings.max_jobs)


async def _start_updates_listener() -> None:
    """Real-time updates listener — **faqat shu jarayonda** (api'da hech qachon).

    `TG_UPDATES_LISTENER` o'chiq bo'lsa hech narsa qilmaydi. Xato bo'lsa ham
    worker ko'tariladi: 10 daqiqalik `incremental_sync_all` zaxira yo'l.
    """
    try:
        from app.worker.updates import start_listeners

        started = await start_listeners()
        if started:
            log.info("worker.updates_listener", accounts=started)
    except Exception as exc:
        log.warning("worker.updates_listener_failed", error=str(exc)[:200])


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
            n = getattr(res, "rowcount", 0)
            if n:
                log.warning("worker.reset_stuck_syncs", count=n)
    except Exception as exc:  # DB hali tayyor bo'lmasa — worker baribir ko'tarilsin
        log.warning("worker.reset_stuck_syncs_failed", error=str(exc)[:200])


async def shutdown(ctx: dict[str, Any]) -> None:
    from app.mtproto.pool import pool
    from app.worker.queue import close_queue
    from app.worker.updates import stop_listeners

    # Handler'lar klient uzilishidan OLDIN olib tashlanadi (task'lar bekor qilinadi).
    await stop_listeners()
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
        tasks.build_daily_digests,
        tasks.process_autoreplies,
        *tasks.PHASE2_FUNCTIONS,  # phase2/A: epizod/tugun indeksi
        *tasks.PHASE3C_FUNCTIONS,  # phase3/C: alert skaneri va yetkazish
        *tasks.PHASE3B_FUNCTIONS,  # phase3/B: briefing qurish va yetkazish
        *tasks.PHASE4B_FUNCTIONS,  # phase4/B: reembed_all (cron yo'q, qo'lda)
        *tasks.PHASE4A_FUNCTIONS,  # phase4/A: entity backfill (kursorli)
        *tasks.PHASE4C_FUNCTIONS,  # phase4/C: batch embedding joblarini o'qish
        *tasks.UI_R1A_FUNCTIONS,  # ui/R1a: chat profillari (avatar + about)
        *tasks.SENDERS_FUNCTIONS,  # jo'natuvchi nomlari (0019 dan oldingi tarix)
    ]
    cron_jobs: ClassVar[list[Any]] = [
        *tasks.PHASE2_CRONS,  # phase2/A: embed_nodes (15 daq), week_nodes (du 03:30)
        *tasks.PHASE3C_CRONS,  # phase3/C: scan_alerts (5 daq), deliver_alerts (5 daq)
        *tasks.PHASE3B_CRONS,  # phase3/B: build_briefings (soatiga), deliver_briefings (15 daq)
        *tasks.PHASE4A_CRONS,  # phase4/A: entity backfill (soatiga, ish bo'lmasa darhol chiqadi)
        *tasks.PHASE4C_CRONS,  # phase4/C: poll_embed_batches (5 daq)
        *tasks.UI_R1A_CRONS,  # ui/R1a: refresh_all_chat_profiles (soatiga, :40)
        *tasks.SENDERS_CRONS,  # backfill_senders_all (soatiga, :25)
        cron(heartbeat, minute={0, 30}, run_at_startup=False),
        # ⚠️ Snapshot cron'ini o'chirmang — views/reactions vaqt qatori shu yerdan (4.6)
        cron(tasks.snapshot_metrics, minute={5}, run_at_startup=False, timeout=3000),
        cron(tasks.incremental_sync_all, minute={0, 10, 20, 30, 40, 50}, run_at_startup=True),
        cron(tasks.embed_messages, minute={7, 22, 37, 52}, run_at_startup=False, timeout=1200),
        cron(tasks.build_daily_digests, hour={2}, minute={30}, run_at_startup=False, timeout=3000),
        cron(
            tasks.process_autoreplies,
            minute=set(range(2, 60, 5)),
            run_at_startup=False,
            timeout=600,
        ),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4  # 3 akkaunt — parallellik kerak emas, flood-wait riski kamayadi
    job_timeout = 3600
