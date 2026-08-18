"""ARQ ishlari — 2-bosqich (ingestion).

  sync_account(account_id, full)   chat registry + har chat tarixi (ketma-ket, 1 akkaunt = 1 ish)
  sync_chat(chat_id)               bitta chat (UI'dan "Sync" tugmasi)
  refresh_chats(account_id)        faqat dialog ro'yxati
  snapshot_metrics()               cron: kanal postlari views/reactions snapshot (har soat)
  incremental_sync_all()           cron: barcha faol akkauntlarda yangi xabarlar (10 daqiqa)
  embed_messages()                 cron: embedding'siz xabarlar → pgvector (15 daqiqa)
  build_daily_digests()            cron: kechagi kun digest'lari (02:30)
  process_autoreplies()            cron: auto-reply qoidalari → javob takliflari (5 daqiqa)

FloodWait siyosati: `SyncPaused` → ish shu yerda tugaydi va o'zini `retry_after`
soniyadan keyin qayta navbatga qo'yadi. Retry-storm yo'q, `max_jobs=4` oshirilmaydi.
`job_id` deterministik (`sync:acc:<id>`) — bir akkaunt uchun parallel sync yo'q.
"""

from __future__ import annotations

from typing import Any

from app.logging import get_logger
from app.observability import heartbeat
from app.services import ingestion as ing
from app.worker.queue import enqueue

log = get_logger("worker.tasks")


EMBED_BATCH = 100
EMBED_MAX_CHARS = 4000
EMBED_MIN_CHARS = 20


def job_id_account(account_id: int) -> str:
    return f"sync:acc:{account_id}"


def job_id_chat(chat_id: int) -> str:
    return f"sync:chat:{chat_id}"


async def sync_account(ctx: dict[str, Any], account_id: int, full: bool = False) -> dict[str, Any]:
    """Akkauntning chatlarini birma-bir sinxronlaydi. `full` — birinchi to'liq sync."""
    from app.db.base import session_scope
    from app.db.models import Account

    async with session_scope() as db:
        account = await db.get(Account, account_id)
        created_at = account.created_at if account else None
    limits = ing.limits_for(created_at)

    try:
        n_chats = await ing.refresh_chats(account_id)
        chat_ids = await ing.chats_to_sync(account_id, limits=limits)
    except ing.SyncPaused as exc:
        await _requeue_account(account_id, full, exc.retry_after)
        return {"paused": exc.retry_after}

    fetched = 0
    for chat_id in chat_ids:
        try:
            report = await ing.sync_chat(chat_id, max_messages=limits.max_messages)
            fetched += report.fetched
        except ing.SyncPaused as exc:
            await _requeue_account(account_id, full, exc.retry_after)
            return {"chats": n_chats, "fetched": fetched, "paused": exc.retry_after}

    log.info(
        "worker.sync_account.done",
        account_id=account_id,
        chats=len(chat_ids),
        fetched=fetched,
        ramp_up=limits.ramp_up,
    )
    await heartbeat("sync_account", extra={"account_id": account_id, "fetched": fetched})
    return {"chats": len(chat_ids), "fetched": fetched, "ramp_up": limits.ramp_up}


async def _requeue_account(account_id: int, full: bool, retry_after: int) -> None:
    delay = min(max(retry_after, 30), 3600) + 5
    log.warning("worker.sync_account.paused", account_id=account_id, retry_in=delay)
    await enqueue(
        "sync_account", account_id, full, job_id=job_id_account(account_id), defer_by=delay
    )


async def sync_chat(ctx: dict[str, Any], chat_id: int) -> dict[str, Any]:
    try:
        report = await ing.sync_chat(chat_id)
    except ing.SyncPaused as exc:
        delay = min(max(exc.retry_after, 30), 3600) + 5
        await enqueue("sync_chat", chat_id, job_id=job_id_chat(chat_id), defer_by=delay)
        return {"paused": exc.retry_after}
    return {"fetched": report.fetched, "done": report.done, "error": report.error}


async def refresh_chats(ctx: dict[str, Any], account_id: int) -> dict[str, Any]:
    try:
        n = await ing.refresh_chats(account_id)
    except ing.SyncPaused as exc:
        return {"paused": exc.retry_after}
    return {"chats": n}


async def snapshot_metrics(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har soat: barcha sinxronlangan kanallar bo'yicha snapshot (4.6-band)."""
    total = 0
    paused = 0
    for chat_id in await ing.channel_ids():
        try:
            total += await ing.snapshot_chat_metrics(chat_id)
        except ing.SyncPaused as exc:
            paused += 1
            log.warning("worker.snapshot.paused", chat_id=chat_id, retry_after=exc.retry_after)
            # keyingi soatda baribir qayta keladi — alohida retry shart emas
        except Exception as exc:  # bitta kanal xatosi qolganlarini to'xtatmasin
            log.warning("worker.snapshot.failed", chat_id=chat_id, error=str(exc)[:200])
    log.info("worker.snapshot.done", posts=total, paused=paused)
    await heartbeat("snapshot_metrics", extra={"posts": total, "paused": paused})
    return {"posts": total, "paused": paused}


async def incremental_sync_all(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har 10 daqiqa: faol akkauntlarda yangi xabarlar (real-time listener o'rniga)."""
    n = 0
    for account_id in await ing.active_account_ids():
        if await enqueue("sync_account", account_id, False, job_id=job_id_account(account_id)):
            n += 1
    await heartbeat("incremental_sync_all", extra={"enqueued": n})
    return {"enqueued": n}


async def embed_messages(ctx: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Embedding'siz xabarlarni batch'lab vektorlaydi (3-bosqich, hybrid qidiruv uchun).

    Faqat sinxronlangan chatlar, matni bo'sh bo'lmaganlar. Har xabar `EMBED_MAX_CHARS`
    gacha kesiladi. Provider — router (`Task.EMBED` → Gemini).
    """
    from sqlalchemy import func, select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Chat, Message, MessageEmbedding
    from app.llm import LLM, LLMError

    s = get_settings()
    if not s.embed_enabled:
        return {"skipped": "disabled"}
    limit = limit or s.embed_batch_per_run
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(Message.id, Message.text)
                .join(Chat, Chat.id == Message.chat_id)
                .outerjoin(MessageEmbedding, MessageEmbedding.message_id == Message.id)
                .where(
                    MessageEmbedding.message_id.is_(None),
                    Chat.synced_total > 0,
                    func.length(Message.text) >= EMBED_MIN_CHARS,  # "ok", "👍" — bekor xarajat
                )
                .order_by(Message.id.desc())
                .limit(limit)
            )
        ).all()
    if not rows:
        return {"embedded": 0}

    llm = LLM()
    done = 0
    for i in range(0, len(rows), EMBED_BATCH):
        chunk = rows[i : i + EMBED_BATCH]
        texts = [(t or "")[:EMBED_MAX_CHARS] for _, t in chunk]
        try:
            res = await llm.embed(texts)
        except LLMError as exc:
            log.warning("worker.embed_failed", error=str(exc)[:200], batch=i // EMBED_BATCH)
            break
        async with session_scope() as db:
            db.add_all(
                [
                    MessageEmbedding(message_id=int(mid), model=res.model, vector=vec)
                    for (mid, _), vec in zip(chunk, res.vectors, strict=False)
                ]
            )
        done += len(chunk)
    log.info("worker.embed.done", embedded=done, pending=len(rows) - done)
    await heartbeat("embed_messages", extra={"embedded": done})
    return {"embedded": done}


async def build_daily_digests(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har kecha: kechagi kun uchun digest yo'q chatlarni oldindan hisoblaydi (kesh issiq)."""
    from datetime import UTC, datetime, timedelta

    from app.db.base import session_scope
    from app.db.models import Chat
    from app.services.digests import chats_needing_digest, digest_for_day

    yesterday = datetime.now(UTC) - timedelta(days=1)
    built = 0
    tokens = 0
    async with session_scope() as db:
        chat_ids = await chats_needing_digest(db, yesterday)
    for chat_id in chat_ids:
        async with session_scope() as db:
            chat = await db.get(Chat, chat_id)
            if chat is None:
                continue
            try:
                d = await digest_for_day(db, chat, yesterday)
            except Exception as exc:  # bitta chat xatosi qolganini to'xtatmasin
                log.warning("worker.digest_failed", chat_id=chat_id, error=str(exc)[:200])
                continue
            if d is not None and not d.cached:
                built += 1
                tokens += d.tokens_in + d.tokens_out
    log.info("worker.digests.done", built=built, tokens=tokens)
    await heartbeat("build_daily_digests", extra={"built": built, "tokens": tokens})
    return {"built": built, "tokens": tokens}


async def process_autoreplies(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har 5 daqiqa: yoqilgan qoidalar → yangi xabarlarga javob takliflari (7-bosqich)."""
    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import AutoReplyRule
    from app.services import autoreply as AR

    if not get_settings().autoreply_enabled:
        return {"skipped": "disabled"}
    async with session_scope() as db:
        ids = await AR.enabled_rule_ids(db)
    proposed = 0
    for rid in ids:
        async with session_scope() as db:
            rule = await db.get(AutoReplyRule, rid)
            if rule is None:
                continue
            try:
                rep = await AR.process_rule(db, rule)
                proposed += int(rep.get("proposed") or 0)
            except Exception as exc:  # bitta qoida xatosi qolganini to'xtatmasin
                log.warning("worker.autoreply_failed", rule_id=rid, error=str(exc)[:200])
    log.info("worker.autoreply.done", rules=len(ids), proposed=proposed)
    await heartbeat("process_autoreplies", extra={"rules": len(ids), "proposed": proposed})
    return {"rules": len(ids), "proposed": proposed}
