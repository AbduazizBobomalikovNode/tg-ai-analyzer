"""ARQ ishlari — 2-bosqich (ingestion).

  sync_account(account_id, full)   chat registry + har chat tarixi (ketma-ket, 1 akkaunt = 1 ish)
  sync_chat(chat_id)               bitta chat (UI'dan "Sync" tugmasi)
  refresh_chats(account_id)        faqat dialog ro'yxati
  snapshot_metrics()               cron: kanal postlari views/reactions snapshot (har soat)
  incremental_sync_all()           cron: barcha faol akkauntlarda yangi xabarlar (10 daqiqa)
  embed_messages()                 cron: embedding'siz xabarlar → pgvector (15 daqiqa)
  build_daily_digests()            cron: kechagi kun digest'lari (02:30)
  process_autoreplies()            cron: auto-reply qoidalari → javob takliflari (5 daqiqa)
  build_episodes_account(acc)      ierarxik indeks: epizodlar (sync oxirida ham chaqiriladi)
  embed_nodes_task()               cron: vektori yo'q retrieval tugunlari (15 daqiqa)
  build_week_nodes_task()          cron: hafta tugunlari, faqat kun tugunlaridan (du 03:30)

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
    changed: list[int] = []
    for chat_id in chat_ids:
        try:
            report = await ing.sync_chat(chat_id, max_messages=limits.max_messages)
            fetched += report.fetched
            if report.fetched:
                changed.append(chat_id)
        except ing.SyncPaused as exc:
            await _build_episodes_for(changed)
            await _requeue_account(account_id, full, exc.retry_after)
            return {"chats": n_chats, "fetched": fetched, "paused": exc.retry_after}

    await _build_episodes_for(changed)
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
        if len(res.vectors) != len(chunk):
            # Moslik buzilsa vektorlar boshqa xabarga yozilib qolardi — batch tashlanadi.
            log.error(
                "worker.embed_misaligned",
                batch=i // EMBED_BATCH,
                expected=len(chunk),
                got=len(res.vectors),
                model=res.model,
            )
            continue
        async with session_scope() as db:
            db.add_all(
                [
                    MessageEmbedding(message_id=int(mid), model=res.model, vector=vec)
                    for (mid, _), vec in zip(chunk, res.vectors, strict=True)
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


# ─── 2-bosqich (phase2/A): ierarxik retrieval indeksi ────────────────────────
#
# Uch ish: epizodlar (har sync'dan keyin), tugun embeddinglari (15 daqiqa),
# hafta tugunlari (dushanba 03:30). Cron'lar `PHASE2_CRONS` da e'lon qilinadi —
# `worker/settings.py` ni bir nechta agent bir vaqtda tahrirlamasligi uchun.


async def _build_episodes_for(chat_ids: list[int]) -> int:
    """Sync'dan keyin o'zgargan chatlarning epizodlarini qayta quradi.

    Xato bitta chatni yo'qotadi, sync'ni emas: indeks keyingi yugurishda
    baribir tutib oladi (`build_episodes` idempotent).
    """
    if not chat_ids:
        return 0
    from app.db.base import session_scope
    from app.services.episodes import build_episodes

    done = 0
    for chat_id in chat_ids:
        try:
            async with session_scope() as db:
                await build_episodes(db, chat_id)
            done += 1
        except Exception as exc:  # bitta chat xatosi sync'ni yiqitmasin
            log.warning("worker.episodes_failed", chat_id=chat_id, error=str(exc)[:200])
    log.info("worker.episodes.done", chats=done)
    return done


async def build_episodes_account(
    ctx: dict[str, Any], account_id: int, full: bool = False
) -> dict[str, Any]:
    """Akkauntning barcha sinxronlangan chatlarida epizodlarni quradi/yangilaydi."""
    from sqlalchemy import select

    from app.db.base import session_scope
    from app.db.models import Chat
    from app.services.episodes import build_episodes

    async with session_scope() as db:
        chat_ids = [
            int(r[0])
            for r in (
                await db.execute(
                    select(Chat.id).where(Chat.account_id == account_id, Chat.synced_total > 0)
                )
            ).all()
        ]
    created = updated = deleted = 0
    for chat_id in chat_ids:
        try:
            async with session_scope() as db:
                rep = await build_episodes(db, chat_id, full=full)
            created += rep.created
            updated += rep.updated
            deleted += rep.deleted
        except Exception as exc:  # bitta chat xatosi qolganini to'xtatmasin
            log.warning("worker.episodes_failed", chat_id=chat_id, error=str(exc)[:200])
    log.info(
        "worker.episodes_account.done",
        account_id=account_id,
        chats=len(chat_ids),
        created=created,
        updated=updated,
    )
    await heartbeat(
        "build_episodes_account", extra={"account_id": account_id, "chats": len(chat_ids)}
    )
    return {"chats": len(chat_ids), "created": created, "updated": updated, "deleted": deleted}


async def embed_nodes_task(ctx: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Har 15 daqiqa: vektori yo'q retrieval tugunlarini embed qiladi."""
    from app.config import get_settings
    from app.db.base import session_scope
    from app.services.nodes import embed_nodes

    s = get_settings()
    if not s.node_index_enabled or not s.embed_enabled:
        return {"skipped": "disabled"}
    async with session_scope() as db:
        rep = await embed_nodes(db, limit=limit or s.node_embed_batch_per_run or None)
    await heartbeat("embed_nodes", extra={"embedded": rep.embedded})
    return {"embedded": rep.embedded, "pending": rep.pending, "error": rep.error}


async def build_week_nodes_task(ctx: dict[str, Any]) -> dict[str, Any]:
    """Dushanba 03:30: o'tgan hafta uchun level-3 tugunlar (faqat kun tugunlaridan)."""
    from datetime import UTC, datetime, timedelta

    from app.config import get_settings
    from app.db.base import session_scope
    from app.services.nodes import build_week_nodes, chats_needing_week_node, week_start

    s = get_settings()
    if not s.node_index_enabled or not s.week_nodes_enabled:
        return {"skipped": "disabled"}
    week = week_start(datetime.now(UTC) - timedelta(days=1))
    async with session_scope() as db:
        chat_ids = await chats_needing_week_node(db, week)
    built = 0
    for chat_id in chat_ids:
        try:
            async with session_scope() as db:
                node = await build_week_nodes(db, chat_id, week)
            if node is not None:
                built += 1
        except Exception as exc:  # bitta chat xatosi qolganini to'xtatmasin
            log.warning("worker.week_node_failed", chat_id=chat_id, error=str(exc)[:200])
    log.info("worker.week_nodes.done", week=str(week.date()), chats=len(chat_ids), built=built)
    await heartbeat("build_week_nodes", extra={"chats": len(chat_ids), "built": built})
    return {"week": str(week.date()), "chats": len(chat_ids), "built": built}


#: phase2/A cron'lari — `worker/settings.py::WorkerSettings.cron_jobs` ga
#: qo'shiladi (`*tasks.PHASE2_CRONS`). Ro'yxat shu yerda, chunki `settings.py`
#: ni parallel ishlayotgan boshqa agentlar ham tahrirlaydi.
PHASE2_FUNCTIONS: list[Any] = [build_episodes_account, embed_nodes_task, build_week_nodes_task]


def _phase2_crons() -> list[Any]:
    from arq.cron import cron

    return [
        cron(embed_nodes_task, minute={3, 18, 33, 48}, run_at_startup=False, timeout=1200),
        cron(
            build_week_nodes_task,
            weekday=0,  # dushanba
            hour={3},
            minute={30},
            run_at_startup=False,
            timeout=3000,
        ),
    ]


PHASE2_CRONS: list[Any] = _phase2_crons()
