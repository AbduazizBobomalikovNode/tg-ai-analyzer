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
    from app.services import sync_progress as SP

    async with session_scope() as db:
        account = await db.get(Account, account_id)
        created_at = account.created_at if account else None
    limits = ing.limits_for(created_at)

    # UI ko'rsatkichi: job boshlandi. Redis yo'q bo'lsa jim o'tadi (SP xatoni yutadi).
    await SP.publish_job(
        account_id,
        state=SP.RUNNING,
        started_at=SP.now_iso(),
        finished_at=None,
        chats_done=0,
        chats_total=0,
        error=None,
    )

    try:
        n_chats = await ing.refresh_chats(account_id)
        chat_ids = await ing.chats_to_sync(account_id, limits=limits)
    except ing.SyncPaused as exc:
        await SP.publish_job(account_id, state=SP.PAUSED, error=f"flood_wait:{exc.retry_after}")
        await _requeue_account(account_id, full, exc.retry_after)
        return {"paused": exc.retry_after}

    await SP.publish_job(account_id, chats_total=len(chat_ids))

    fetched = 0
    changed: list[int] = []
    done_chats = 0
    for chat_id in chat_ids:
        try:
            report = await ing.sync_chat(chat_id, max_messages=limits.max_messages)
            fetched += report.fetched
            if report.fetched:
                changed.append(chat_id)
        except ing.SyncPaused as exc:
            await SP.publish_job(
                account_id,
                state=SP.PAUSED,
                chats_done=done_chats,
                error=f"flood_wait:{exc.retry_after}",
            )
            await _build_episodes_for(changed)
            await _requeue_account(account_id, full, exc.retry_after)
            return {"chats": n_chats, "fetched": fetched, "paused": exc.retry_after}
        except Exception as exc:
            await SP.publish_job(
                account_id,
                state=SP.FAILED,
                chats_done=done_chats,
                finished_at=SP.now_iso(),
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            raise
        done_chats += 1
        # Har chatdan keyin — "N / M tayyor" hisoblagichi (bitta Redis SET).
        await SP.publish_job(account_id, chats_done=done_chats)

    await SP.publish_job(
        account_id, state=SP.DONE, finished_at=SP.now_iso(), chats_done=done_chats, error=None
    )
    await SP.clear_progress(account_id)
    await _build_episodes_for(changed)
    log.info(
        "worker.sync_account.done",
        account_id=account_id,
        chats=len(chat_ids),
        fetched=fetched,
        ramp_up=limits.ramp_up,
    )
    await heartbeat("sync_account", extra={"account_id": account_id, "fetched": fetched})
    # phase3/C: yangi xabarlar keldi — alert skanerini darhol navbatga qo'yamiz,
    # 5 daqiqalik cron'ni kutmasdan (kalit so'z alerti kechikkani sezilarli).
    await _enqueue_scan_alerts(account_id, fetched)
    # ui/R1a: yangi chatlar ro'yxatga tushdi — avatar/about ularga soatlik
    # cron'ni kutmasdan keladi (ish o'zi byudjetlangan, `job_id` deterministik).
    await _enqueue_chat_profiles(account_id)
    # Jo'natuvchi nomlari: 0019 dan oldingi tarix nomsiz qolgan. Ish o'zi
    # byudjetlangan va nomlar to'liq bo'lsa darhol chiqadi (`job_id` bitta).
    await _enqueue_senders_backfill(account_id)
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


async def _budget_blocked_accounts() -> set[int]:
    """M3: kunlik byudjeti tugagan egalarning akkauntlari.

    Fon ishlari chat so'rovi emas, lekin ular ham LLM pulini yeydi: bitta
    `reembed_all` yugurishi kunlik byudjetni jimgina oshirib yuborishi mumkin.
    Tekshiruv **xato ko'tarmaydi** — log va o'tkazib yuborish (cron qayta
    keladi). Xatoning o'zi ham ishni to'xtatmaydi (byudjet — tejash, to'siq emas).
    """
    from app.db.base import session_scope
    from app.services.limits import blocked_account_ids

    try:
        async with session_scope() as db:
            return await blocked_account_ids(db)
    except Exception as exc:  # DB yo'q/sekin — indekslash to'xtamasin
        log.warning("worker.budget_check_failed", error=str(exc)[:200])
        return set()


async def _upsert_message_vectors(db: Any, pairs: list[tuple[int, Any]], model: str) -> None:
    """`message_embeddings` ga yozadi — PK `(message_id, model)`, konflikt jim.

    `db.add_all()` emas: PK 0016 dan beri ikki ustunli, ya'ni bitta xabarda
    bir nechta model qatori bo'lishi **normal**. Oraliqda (batch yo'li,
    parallel cron) o'sha `(id, model)` yozilib qolgan bo'lsa ustidan
    yozilmaydi — birinchi yozilgani qoladi, ikkinchisi jimgina tushadi.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.db.models import MessageEmbedding

    if not pairs:
        return
    stmt = pg_insert(MessageEmbedding).values(
        [{"message_id": mid, "model": model, "vector": vec} for mid, vec in pairs]
    )
    await db.execute(stmt.on_conflict_do_nothing(index_elements=["message_id", "model"]))


async def embed_messages(ctx: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Embedding'siz xabarlarni batch'lab vektorlaydi (3-bosqich, hybrid qidiruv uchun).

    Faqat sinxronlangan chatlar, matni bo'sh bo'lmaganlar. Har xabar `EMBED_MAX_CHARS`
    gacha kesiladi. Provider — router (`Task.EMBED` → Gemini).
    """
    from sqlalchemy import and_, func, select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Chat, Message, MessageEmbedding
    from app.llm import LLM, LLMError, embed_model_name

    s = get_settings()
    if not s.embed_enabled:
        return {"skipped": "disabled"}
    # phase4/C: batch rejimda bu yugurish faqat **yuboradi**; vektorlarni
    # `poll_embed_batches` keyingi soatlarda yozadi. Batch mavjud bo'lmasa
    # (eski SDK, boshqa provider) sinxron yo'lga tushamiz — indekslash
    # to'xtab qolmasin.
    # M3: byudjeti tugagan egalarning akkauntlari chetlab o'tiladi (global ish
    # — shuning uchun to'xtatilmaydi, faqat o'sha chatlar tanlanmaydi).
    blocked = await _budget_blocked_accounts()
    if blocked:
        log.info("worker.embed.budget_skip", accounts=len(blocked))
    if s.embed_mode == "batch":
        # M3: byudjet qopqog'i batch yo'lida ham amal qiladi — batch **oldindan
        # to'lanadi**, ya'ni bu yerda o'tkazib yuborilmagan chat kelasi oyda
        # hisobda paydo bo'lardi.
        result = await _submit_embed_batch(KIND_MESSAGES, limit, skip_accounts=blocked)
        if result is not None:
            return result
    # Sinxron yo'lning bir yugurishdagi qopqog'i. `EMBED_SYNC_PER_RUN` 0 bo'lsa
    # eski xulq (`EMBED_BATCH_PER_RUN`) — ikkita raqam bir-biriga zid bo'lmasin.
    limit = limit or s.embed_sync_per_run or s.embed_batch_per_run
    # phase4: embedding **joriy model** bo'yicha yetishmayotganini qidiradi.
    # Ilgari "umuman embedding yo'q" edi — model almashtirilganda eski vektorli
    # xabarlar cron'da hech qachon to'ldirilmasdi (faqat qo'lda `reembed_all`).
    want_model = embed_model_name()
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(Message.id, Message.text)
                .join(Chat, Chat.id == Message.chat_id)
                .outerjoin(
                    MessageEmbedding,
                    and_(
                        MessageEmbedding.message_id == Message.id,
                        MessageEmbedding.model == want_model,
                    ),
                )
                .where(
                    MessageEmbedding.message_id.is_(None),
                    Chat.synced_total > 0,
                    func.length(Message.text) >= EMBED_MIN_CHARS,  # "ok", "👍" — bekor xarajat
                    *([Chat.account_id.not_in(sorted(blocked))] if blocked else []),
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
            await _upsert_message_vectors(
                db,
                [(int(mid), vec) for (mid, _), vec in zip(chunk, res.vectors, strict=True)],
                res.model,
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
    from app.llm.router import background_tier
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
                d = await digest_for_day(db, chat, yesterday, tier=background_tier())
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
    blocked = await _budget_blocked_accounts()  # M3
    if s.embed_mode == "batch":  # phase4/C — quyidagi izohga qarang
        result = await _submit_embed_batch(KIND_NODES, limit, skip_accounts=blocked)
        if result is not None:
            return result
    async with session_scope() as db:
        rep = await embed_nodes(
            db, limit=limit or s.node_embed_batch_per_run or None, skip_accounts=blocked
        )
    await heartbeat("embed_nodes", extra={"embedded": rep.embedded})
    return {"embedded": rep.embedded, "pending": rep.pending, "error": rep.error}


async def build_week_nodes_task(ctx: dict[str, Any]) -> dict[str, Any]:
    """Dushanba 03:30: o'tgan hafta uchun level-3 tugunlar (faqat kun tugunlaridan)."""
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.services.nodes import build_week_nodes, chats_needing_week_node, week_start

    s = get_settings()
    if not s.node_index_enabled or not s.week_nodes_enabled:
        return {"skipped": "disabled"}
    week = week_start(datetime.now(UTC) - timedelta(days=1))
    async with session_scope() as db:
        chat_ids = await chats_needing_week_node(db, week)
    # M3: byudjeti tugagan egalarning chatlari o'tkazib yuboriladi
    # (`build_week_nodes` ni ham ikkinchi qopqoq sifatida tekshiradi).
    blocked = await _budget_blocked_accounts()
    if blocked:
        from app.db.models import Chat as _Chat

        async with session_scope() as db:
            keep = (
                await db.execute(
                    select(_Chat.id).where(
                        _Chat.id.in_(chat_ids), _Chat.account_id.not_in(sorted(blocked))
                    )
                )
            ).scalars()
            chat_ids = [int(x) for x in keep]
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


# ─── 3-bosqich/C: alertlar ───────────────────────────────────────────────────
#
# Ikki ish, ikkalasi ham 5 daqiqada: `scan_alerts` (yoqilgan qoidalarni yangi
# xabarlarga qo'llaydi) va `deliver_alerts_task` (yetkazilmagan hodisalarni
# qoida bo'yicha guruhlab push qiladi). Ajratilgan, chunki jim soatlar
# yetkazishni to'xtatadi, skanerni esa **yo'q** — kursor oldinga siljib
# turishi kerak, aks holda jim soatlar tugagach bir yugurishda butun tun
# qayta skanerlanardi.


async def _enqueue_scan_alerts(account_id: int, fetched: int) -> None:
    """`sync_account` oxiridagi ilgak. Xato alert uchun — sync uchun emas."""
    if not fetched:
        return
    try:
        from app.config import get_settings

        if not get_settings().alerts_enabled:
            return
        await enqueue("scan_alerts", account_id, job_id=f"alerts:scan:{account_id}")
    except Exception as exc:  # navbat yo'q bo'lsa ham sync muvaffaqiyatli qoladi
        log.warning("worker.alerts_enqueue_failed", account_id=account_id, error=str(exc)[:200])


async def scan_alerts(ctx: dict[str, Any], account_id: int | None = None) -> dict[str, Any]:
    """Yoqilgan qoidalarni yangi xabarlarga qo'llaydi (`account_id` — faqat bittasi)."""
    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import AlertRule
    from app.services import alerts as AL

    s = get_settings()
    if not s.alerts_enabled:
        return {"skipped": "disabled"}
    async with session_scope() as db:
        ids = await AL.enabled_rule_ids(db, account_id)
    events = 0
    for rid in ids:
        try:
            async with session_scope() as db:
                rule = await db.get(AlertRule, rid)
                if rule is None:
                    continue
                events += len(await AL.scan_rule(db, rule, limit=s.alerts_scan_limit))
        except Exception as exc:  # bitta qoida xatosi qolganlarini to'xtatmasin
            log.warning("worker.alert_scan_failed", rule_id=rid, error=str(exc)[:200])
    log.info("worker.alerts.scanned", rules=len(ids), events=events)
    await heartbeat("scan_alerts", extra={"rules": len(ids), "events": events})
    return {"rules": len(ids), "events": events}


async def deliver_alerts_task(ctx: dict[str, Any]) -> dict[str, Any]:
    """Yetkazilmagan hodisalar → boshqaruv botiga push (jim soatlar hisobga olinadi)."""
    from datetime import UTC, datetime

    from app.config import get_settings
    from app.db.base import session_scope
    from app.services import alerts as AL

    if not get_settings().alerts_enabled:
        return {"skipped": "disabled"}
    now = datetime.now(UTC)
    async with session_scope() as db:
        account_ids = await AL.accounts_with_rules(db)
    pushes = 0
    for acc_id in account_ids:
        try:
            async with session_scope() as db:
                rep = await AL.deliver_alerts(db, acc_id, now)
                pushes += int(rep.get("pushes") or 0)
        except Exception as exc:  # bitta akkaunt xatosi qolganlarini to'xtatmasin
            log.warning("worker.alert_deliver_failed", account_id=acc_id, error=str(exc)[:200])
    log.info("worker.alerts.delivered", accounts=len(account_ids), pushes=pushes)
    await heartbeat("deliver_alerts", extra={"accounts": len(account_ids), "pushes": pushes})
    return {"accounts": len(account_ids), "pushes": pushes}


#: phase3/C ishlari va cron'lari — `worker/settings.py` ga `*tasks.PHASE3C_*`
#: bo'lib qo'shiladi (parallel agentlar bir xil qatorni tahrirlamasin).
PHASE3C_FUNCTIONS: list[Any] = [scan_alerts, deliver_alerts_task]


def _phase3c_crons() -> list[Any]:
    from arq.cron import cron

    return [
        # Skaner va yetkazish bir daqiqada tushmasin — skaner avval yugursin.
        cron(scan_alerts, minute=set(range(1, 60, 5)), run_at_startup=False, timeout=900),
        cron(deliver_alerts_task, minute=set(range(4, 60, 5)), run_at_startup=False, timeout=600),
    ]


PHASE3C_CRONS: list[Any] = _phase3c_crons()


# ─── phase3/B: shaxsiy briefing (qurish + yetkazish) ─────────────────────────


async def build_briefings(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har soat: lokal vaqti 02:00 bo'lgan akkauntlar uchun kechagi briefing.

    "02:45 lokal" o'rniga soatiga bir marta yugurib, akkauntning **o'z** vaqt
    zonasida soat 2 bo'lganini tekshiradi — bitta cron bilan barcha zonalar
    qamraladi. Idempotent: `uq_briefing_account_day` va mavjud qatorni
    qaytaradigan `build_briefing` tufayli takroriy yugurish qayta
    hisoblamaydi.
    """
    from app.config import get_settings
    from app.db.base import session_scope
    from app.services import briefing as BR

    s = get_settings()
    if not s.briefing_enabled:
        return {"skipped": "disabled"}
    async with session_scope() as db:
        accounts = [(a.id, a.tz) for a in await BR.enabled_accounts(db)]
    built = 0
    checked = 0
    for account_id, tz in accounts:
        local = BR.local_now(tz)
        if local.hour != 2:
            continue
        checked += 1
        day = BR.edition_day(local)
        try:
            async with session_scope() as db:
                row = await BR.build_briefing(db, account_id, day)
            if row is not None:
                built += 1
        except Exception as exc:  # bitta akkaunt xatosi qolganlarini to'xtatmasin
            log.warning("worker.briefing_failed", account_id=account_id, error=str(exc)[:200])
    log.info("worker.briefings.built", accounts=checked, built=built)
    await heartbeat("build_briefings", extra={"accounts": checked, "built": built})
    return {"accounts": checked, "built": built}


async def deliver_briefings(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har 15 daqiqada: tayyor, hali yetkazilmagan briefinglarni botga push qiladi.

    Soat, jim oyna va `delivered_at` tekshiruvi `services/briefing` da —
    bu yerda faqat sessiya boshqaruvi va xatolarni ajratish.
    """
    from datetime import UTC, datetime

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Account
    from app.services import briefing as BR

    s = get_settings()
    if not s.briefing_enabled:
        return {"skipped": "disabled"}
    now = datetime.now(UTC)
    async with session_scope() as db:
        account_ids = [a.id for a in await BR.enabled_accounts(db)]
    sent = 0
    errors = 0
    for account_id in account_ids:
        try:
            async with session_scope() as db:
                account = await db.get(Account, account_id)
                if account is None:
                    continue
                res = await BR.deliver_account(db, account, now=now)
            if res.status == "sent":
                sent += 1
            elif res.status == "error":
                errors += 1
        except Exception as exc:
            errors += 1
            log.warning(
                "worker.briefing_deliver_failed", account_id=account_id, error=str(exc)[:200]
            )
    log.info("worker.briefings.delivered", accounts=len(account_ids), sent=sent, errors=errors)
    await heartbeat("deliver_briefings", extra={"sent": sent, "errors": errors})
    return {"accounts": len(account_ids), "sent": sent, "errors": errors}


#: phase3/B ishlari va cron'lari — `worker/settings.py` ga `*tasks.PHASE3B_*`
#: bo'lib qo'shiladi (parallel agentlar bir xil qatorni tahrirlamasin).
PHASE3B_FUNCTIONS: list[Any] = [build_briefings, deliver_briefings]


def _phase3b_crons() -> list[Any]:
    from arq.cron import cron

    return [
        # Soatiga bir marta (45-daqiqa) — ichida "lokal soat 2 mi?" tekshiriladi.
        cron(build_briefings, minute={45}, run_at_startup=False, timeout=3000),
        cron(deliver_briefings, minute={6, 21, 36, 51}, run_at_startup=False, timeout=600),
    ]


PHASE3B_CRONS: list[Any] = _phase3b_crons()


# ─── eslatmalar (skills_ext `remind`, `services/scheduled.py`) ───────────────


async def deliver_reminders(ctx: dict[str, Any]) -> dict[str, Any]:
    """Har daqiqa: muddati kelgan `scheduled_jobs(type='reminder')` → egasining botiga.

    Har eslatma **o'z** tranzaksiyasida: band qilish (`scheduled.claim`) →
    push → yakun. Bitta xato qolganlarini to'xtatmaydi. Jim soatlar
    qo'llanmaydi — vaqtni foydalanuvchi o'zi aytgan.
    """
    from datetime import UTC, datetime

    from app.db.base import session_scope
    from app.services import scheduled as SCH

    now = datetime.now(UTC)
    async with session_scope() as db:
        ids = await SCH.due_ids(db, now)
    counts: dict[str, int] = {}
    for job_id in ids:
        try:
            async with session_scope() as db:
                out = await SCH.deliver_one(db, job_id, now=now)
            counts[out.status] = counts.get(out.status, 0) + 1
        except Exception as exc:  # bitta eslatma xatosi qolganlarini to'xtatmasin
            counts["error"] = counts.get("error", 0) + 1
            log.warning("worker.reminder_failed", job_id=job_id, error=str(exc)[:200])
    if ids:
        log.info("worker.reminders.delivered", due=len(ids), **counts)
    await heartbeat("deliver_reminders", extra={"due": len(ids), **counts})
    return {"due": len(ids), **counts}


#: Eslatma ishi va cron'i — `worker/settings.py` ga `*tasks.REMINDER_*` bo'lib qo'shiladi.
REMINDER_FUNCTIONS: list[Any] = [deliver_reminders]


def _reminder_crons() -> list[Any]:
    from arq.cron import cron

    # Har daqiqa: eslatma aniq vaqtga aytilgan, ≤60 s kechikish qabul qilinadi.
    return [cron(deliver_reminders, minute=set(range(60)), run_at_startup=False, timeout=120)]


REMINDER_CRONS: list[Any] = _reminder_crons()


# ─── 4-bosqich/B: model almashtirgandan keyin qayta embedding ────────────────
#
# Embedding modeli o'zgarsa (`EMBED_MODEL` / `LLM_MODEL_EMBED`) eski vektorlar
# **o'chirilmaydi**: ikkala fazo yonma-yon yashaydi va qidiruv joriy model
# bo'yicha filtrlaydi (`message_embeddings.model`). Bu ish shunchaki yangi
# model qatorlari yo'q xabarlarni to'ldiradi — to'xtatib, keyin davom ettirsa
# ham hech narsa yo'qolmaydi.
#
# Nega alohida ish: cron'dagi `embed_messages` bir yugurishda
# `EMBED_BATCH_PER_RUN` bilan cheklangan; re-embed esa million qatorni
# bosqichma-bosqich, o'zini qayta navbatga qo'yib bosib o'tadi.

REEMBED_MAX_ROUNDS = 20  # bitta ish ichidagi batch'lar; keyin o'zini requeue qiladi
REEMBED_JOB_ID = "reembed:all"


def job_id_reembed(account_id: int | None) -> str:
    return REEMBED_JOB_ID if account_id is None else f"reembed:acc:{account_id}"


async def reembed_all(
    ctx: dict[str, Any], account_id: int | None = None, model: str = ""
) -> dict[str, Any]:
    """Joriy embedding modeli uchun yetishmayotgan vektorlarni to'ldiradi.

    Hech narsa **o'chirmaydi**: `model` ustuni bo'yicha yangi qatorlar
    qo'shiladi, eskisi joyida qoladi (orqaga qaytish bepul).

    Args:
        ctx: ARQ konteksti.
        account_id: faqat shu akkaunt chatlari; `None` — hammasi.
        model: kutilayotgan model nomi. Bo'sh = konfiguratsiyadagi joriy model.
            Router boshqa modelni qaytarsa ish **to'xtaydi** (aralash fazo
            yaratmaslik uchun) va xatoni qaytaradi.

    Returns:
        `{"model", "embedded", "pending", "done"}`; `done=False` bo'lsa ish
        o'zini qayta navbatga qo'ygan.
    """
    from sqlalchemy import func, select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Chat, Message, MessageEmbedding
    from app.llm import LLM, LLMError, embed_model_name

    s = get_settings()
    if not s.embed_enabled:
        return {"skipped": "disabled"}
    # M3: qayta embed — bir yugurishda eng qimmat ish. Byudjet tugagan bo'lsa
    # boshlanmaydi ham (`account_id` berilgan bo'lsa shu akkaunt bo'yicha,
    # aks holda global: byudjeti tugagan egalarning chatlari chetlab o'tiladi).
    blocked = await _budget_blocked_accounts()
    if account_id is not None and int(account_id) in blocked:
        log.info("worker.reembed.budget_skip", account_id=account_id)
        return {"skipped": "budget", "account_id": int(account_id)}
    skip = sorted(blocked) if account_id is None else []
    want = (model or "").strip() or embed_model_name()

    llm = LLM()
    embedded = 0
    pending = 0
    for _round in range(REEMBED_MAX_ROUNDS):
        async with session_scope() as db:
            missing = (
                ~select(MessageEmbedding.message_id)
                .where(
                    MessageEmbedding.message_id == Message.id,
                    MessageEmbedding.model == want,
                )
                .exists()
            )
            rows = (
                await db.execute(
                    select(Message.id, Message.text)
                    .join(Chat, Chat.id == Message.chat_id)
                    .where(
                        missing,
                        Chat.synced_total > 0,
                        func.length(Message.text) >= EMBED_MIN_CHARS,
                        *([Chat.account_id == int(account_id)] if account_id is not None else []),
                        *([Chat.account_id.not_in(skip)] if skip else []),
                    )
                    .order_by(Message.id.desc())
                    .limit(EMBED_BATCH)
                )
            ).all()
        if not rows:
            log.info("worker.reembed.done", model=want, embedded=embedded, account_id=account_id)
            await heartbeat("reembed_all", extra={"model": want, "embedded": embedded})
            return {"model": want, "embedded": embedded, "pending": 0, "done": True}

        texts = [(t or "")[:EMBED_MAX_CHARS] for _, t in rows]
        try:
            res = await llm.embed(texts)
        except LLMError as exc:
            log.warning("worker.reembed_failed", model=want, error=str(exc)[:200])
            return {"model": want, "embedded": embedded, "error": str(exc)[:200], "done": False}
        if res.model != want:
            # Aralash fazo eng yomon holat: qidiruv filtri ikkalasini ham
            # o'tkazmay qoladi. Jimgina davom etmaymiz.
            err = f"model mismatch: want {want}, got {res.model} — EMBED_MODEL/LLM_MODEL_EMBED?"
            log.error("worker.reembed_model_mismatch", want=want, got=res.model)
            return {"model": want, "embedded": embedded, "error": err, "done": False}
        if len(res.vectors) != len(rows):
            log.error(
                "worker.reembed_misaligned", expected=len(rows), got=len(res.vectors), model=want
            )
            return {"model": want, "embedded": embedded, "error": "misaligned", "done": False}
        async with session_scope() as db:
            # Eski model qatorlariga **tegilmaydi**: PK `(message_id, model)`
            # (migratsiya 0016), ya'ni bu INSERT yangi fazani yonma-yon qo'yadi.
            await _upsert_message_vectors(
                db,
                [(int(mid), vec) for (mid, _), vec in zip(rows, res.vectors, strict=True)],
                res.model,
            )
        embedded += len(rows)
        pending = len(rows)
        log.info("worker.reembed.progress", model=want, embedded=embedded, batch=len(rows))

    # Hali tugamadi — o'zini qayta navbatga qo'yamiz (bitta ish uzoq ushlanmasin).
    await enqueue(
        "reembed_all", account_id, model=want, job_id=job_id_reembed(account_id), defer_by=5
    )
    return {"model": want, "embedded": embedded, "pending": pending, "done": False}


#: phase4/B — `worker/settings.py` ga `*tasks.PHASE4B_FUNCTIONS` bo'lib qo'shiladi.
#: Cron yo'q: re-embed faqat qo'lda (`python -m app.main reembed`) boshlanadi.
PHASE4B_FUNCTIONS: list[Any] = [reembed_all]


# ─── phase4/A: entity indeksi backfill (`node_entities`) ─────────────────────
#
# Yangi indeks eski xabarlarga ham kerak, lekin 4 million qatorni bitta ishda
# aylantirib bo'lmaydi. Shuning uchun kursorli backfill: `messages.id` bo'yicha
# oldinga yuriladi, kursor **Redis'da** (`kv` jadvali yo'q — yangi jadval
# qo'shish shu ish uchun ortiqcha). Cron soatiga bir marta yuguradi va qiladigan
# ishi bo'lmasa **darhol** chiqadi.

ENTITIES_CURSOR_KEY = "entities:backfill:{account_id}"


async def _entities_cursor(ctx: dict[str, Any], account_id: int) -> tuple[Any, int]:
    """`(redis, last_message_id)` — Redis yo'q bo'lsa `(None, 0)`."""
    redis = ctx.get("redis") if isinstance(ctx, dict) else None
    if redis is None:
        from app.worker.queue import get_queue

        redis = await get_queue()
    raw = await redis.get(ENTITIES_CURSOR_KEY.format(account_id=account_id))
    if raw is None:
        return redis, 0
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return redis, int(raw)
    except (TypeError, ValueError):
        return redis, 0


async def backfill_entities(
    ctx: dict[str, Any], account_id: int, limit: int | None = None
) -> dict[str, Any]:
    """Entity indeksi bo'sh bo'lgan eski xabarlarni to'ldiradi (kursorli).

    `mention_name` bu yerda **chiqmaydi**: Telegram entity'lari saqlanmaydi,
    ism esa matndan taxmin qilinmaydi. Qolgan turlar (username, hashtag,
    cashtag, domen, `t.me`) matndan olinadi.
    """
    from sqlalchemy import select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Chat, Message
    from app.services.entities import EntityRef, index_messages_entities

    s = get_settings()
    if not s.entities_enabled:
        return {"skipped": "disabled"}
    budget = int(limit or s.entities_backfill_limit)
    batch = s.entities_backfill_batch
    redis, cursor = await _entities_cursor(ctx, account_id)
    seen = 0
    indexed = 0
    done = False
    while seen < budget:
        async with session_scope() as db:
            rows = (
                await db.execute(
                    select(Message.id, Message.chat_id, Message.text, Message.published_at)
                    .join(Chat, Chat.id == Message.chat_id)
                    .where(
                        Chat.account_id == account_id,
                        Message.id > cursor,
                        Message.deleted_at.is_(None),
                    )
                    .order_by(Message.id)
                    .limit(min(batch, budget - seen))
                )
            ).all()
            if not rows:
                done = True
                break
            refs = [
                EntityRef(
                    message_id=int(r[0]),
                    account_id=account_id,
                    chat_id=int(r[1]),
                    text=r[2] or "",
                    seen_at=r[3],
                )
                for r in rows
            ]
            indexed += await index_messages_entities(db, refs)
        seen += len(rows)
        cursor = int(rows[-1][0])
        if redis is not None:
            await redis.set(ENTITIES_CURSOR_KEY.format(account_id=account_id), str(cursor))
    log.info(
        "worker.entities.backfilled",
        account_id=account_id,
        messages=seen,
        entities=indexed,
        cursor=cursor,
        done=done,
    )
    await heartbeat("backfill_entities", extra={"account_id": account_id, "messages": seen})
    return {"messages": seen, "entities": indexed, "cursor": cursor, "done": done}


async def backfill_entities_all(ctx: dict[str, Any]) -> dict[str, Any]:
    """Soatiga bir marta (`minute={17}`): qoldig'i bor akkauntlar uchun backfill navbatga qo'yiladi.

    Cron "darhol chiqmaydi": har **faol akkaunt** uchun bitta
    `max(messages.id)` agregati va bitta Redis kursor o'qishi baribir
    bajariladi. Tejaladigan narsa — **ish**: kursor eng katta `messages.id`
    ga yetgan akkaunt uchun hech narsa navbatga qo'yilmaydi (`queued` o'smaydi),
    ya'ni `backfill_entities` umuman yugurmaydi. Bitta akkaunt xatosi
    qolganlarini to'xtatmaydi — log'ga tushadi va tsikl davom etadi.
    """
    from sqlalchemy import func, select

    from app.config import get_settings
    from app.db.base import session_scope
    from app.db.models import Chat, Message

    if not get_settings().entities_enabled:
        return {"skipped": "disabled"}
    queued = 0
    account_ids = await ing.active_account_ids()
    for account_id in account_ids:
        try:
            async with session_scope() as db:
                max_id = (
                    await db.execute(
                        select(func.max(Message.id))
                        .join(Chat, Chat.id == Message.chat_id)
                        .where(Chat.account_id == account_id)
                    )
                ).scalar()
            _, cursor = await _entities_cursor(ctx, account_id)
            if not max_id or int(max_id) <= cursor:
                continue
            await enqueue("backfill_entities", account_id, job_id=f"entities:backfill:{account_id}")
            queued += 1
        except Exception as exc:  # bitta akkaunt xatosi qolganlarini to'xtatmasin
            log.warning(
                "worker.entities_backfill_failed", account_id=account_id, error=str(exc)[:200]
            )
    if queued:
        log.info("worker.entities.backfill_queued", accounts=queued)
    return {"accounts": len(account_ids), "queued": queued}


#: phase4/A ishlari va cron'i — `worker/settings.py` ga `*tasks.PHASE4A_*` bo'lib
#: qo'shiladi (parallel agentlar bir xil qatorni tahrirlamasin).
PHASE4A_FUNCTIONS: list[Any] = [backfill_entities, backfill_entities_all]


def _phase4a_crons() -> list[Any]:
    from arq.cron import cron

    return [
        cron(backfill_entities_all, minute={17}, run_at_startup=False, timeout=600),
    ]


PHASE4A_CRONS: list[Any] = _phase4a_crons()


# ─── 4-bosqich/C: batch embedding ────────────────────────────────────────────
#
# `EMBED_MODE=batch` da `embed_messages` / `embed_nodes_task` vektor **yozmaydi**
# — ular faqat Gemini Batch API'ga job yuboradi (narx yarmi), vektorlarni esa
# `poll_embed_batches` (har 5 daqiqa) tugagan jobdan o'qib yozadi.
#
# Nega "yuborib qo'yib yuborish" xavfsiz: job nomi va **tartiblangan** element
# id'lari `embed_batches` jadvalida (migratsiya 0013). Worker qayta ishga
# tushsa ham poll o'sha qatordan davom etadi.
#
# Fallback: batch yo'li mavjud bo'lmasa yoki submit xato bersa, `_submit_embed_batch`
# `None` qaytaradi va chaqiruvchi **sinxron** yo'lni davom ettiradi. Ya'ni eng
# yomon holatda narx eskisicha bo'ladi, indekslash esa to'xtamaydi.

KIND_MESSAGES = "messages"
KIND_NODES = "nodes"


async def _submit_embed_batch(
    kind: str, limit: int | None, *, skip_accounts: set[int] | None = None
) -> dict[str, Any] | None:
    """Batch job(lar) yuboradi. `None` = batch ishlamadi, sinxron yo'l davom etsin.

    `skip_accounts` — kunlik byudjeti tugagan egalarning akkauntlari (M3).
    Sinxron yo'ldagi bilan **bir xil** ro'yxat: batch arzonroq bo'lsa ham
    pulni o'sha byudjetdan yeydi.

    `EMBED_BATCH_JOBS_PER_RUN` > 1 bo'lsa bir yugurishda bir nechta job
    yuboriladi (2.6M xabarlik backfill bitta job/15 daqiqa bilan oylar olardi).
    Xavfsiz, chunki tanlov `_inflight` konteynmentidan o'tadi va har submit
    `flush` bilan yoziladi — **bitta** sessiya ichida keyingi tanlov oldingi
    jobga tushgan id'larni ko'rmaydi, ya'ni bir element ikki marta to'lanmaydi.

    Birinchi submit `None` qaytarsa (batch yo'li yo'q / xato) — sinxron yo'lga
    tushamiz. Keyingilari xato bersa **tushmaymiz**: bir qismi allaqachon
    yuborilgan, sinxron yo'l o'sha elementlarni ikkinchi marta to'lardi.
    """
    from app.config import get_settings
    from app.db.base import session_scope
    from app.services import embed_batches as eb

    if not eb.batch_embedding_available():
        log.warning("worker.embed_batch.unavailable", kind=kind)
        return None
    skip = set(skip_accounts or ())
    jobs = max(1, get_settings().embed_batch_jobs_per_run)
    submitted = 0
    names: list[str] = []
    async with session_scope() as db:
        for _ in range(jobs):
            rep = (
                await eb.submit_messages(db, limit=limit, skip_accounts=skip)
                if kind == KIND_MESSAGES
                else await eb.submit_nodes(db, limit=limit, skip_accounts=skip)
            )
            if rep.error:
                if not names:
                    return None
                log.warning("worker.embed_batch.partial", kind=kind, error=rep.error[:200])
                break
            if not rep.submitted:  # embed qilinadigan element qolmadi
                break
            submitted += rep.submitted
            names.append(rep.job)
    await heartbeat(
        f"embed_{kind}", extra={"submitted": submitted, "jobs": len(names), "mode": "batch"}
    )
    return {
        "mode": "batch",
        "submitted": submitted,
        "jobs": len(names),
        "job": names[0] if names else "",
    }


async def poll_embed_batches(ctx: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Har 5 daqiqa: tugagan batch joblarni o'qiydi va vektorlarni yozadi.

    Bu cron **batch rejimidan qat'i nazar** ishlaydi: rejim `sync` ga
    qaytarilgan bo'lsa ham allaqachon yuborilgan (va to'langan) joblar
    o'qilishi kerak.

    M3: byudjet qopqog'i bu yerda **ataylab yo'q**. Pul allaqachon sarflangan
    (job yuborilgan) — o'qimaslik faqat vektorlarni yo'qotadi, tejamaydi.
    Byudjet faqat **yuborish** yo'llarida tekshiriladi (`embed_messages`,
    `embed_nodes_task`, `reembed_all`).
    """
    from app.config import get_settings
    from app.db.base import session_scope
    from app.services.embed_batches import poll_batches

    s = get_settings()
    if not s.embed_batch_poll_enabled:
        return {"skipped": "disabled"}
    async with session_scope() as db:
        rep = await poll_batches(db, limit=limit)
    if rep.checked:
        log.info(
            "worker.embed_batch.poll",
            checked=rep.checked,
            succeeded=rep.succeeded,
            written=rep.written,
            failed=rep.failed,
            permanent=rep.permanent,
            running=rep.still_running,
        )
    await heartbeat("poll_embed_batches", extra={"written": rep.written, "checked": rep.checked})
    return {
        "checked": rep.checked,
        "succeeded": rep.succeeded,
        "written": rep.written,
        "failed": rep.failed,
        "permanent": rep.permanent,
        "running": rep.still_running,
    }


#: phase4/C ishlari va cron'i — `worker/settings.py` ga `*tasks.PHASE4C_*` bo'lib
#: qo'shiladi (parallel agentlar bir xil qatorni tahrirlamasin).
PHASE4C_FUNCTIONS: list[Any] = [poll_embed_batches]


def _phase4c_crons() -> list[Any]:
    from arq.cron import cron

    return [
        cron(
            poll_embed_batches,
            minute={1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56},
            run_at_startup=False,
            timeout=900,
        ),
    ]


PHASE4C_CRONS: list[Any] = _phase4c_crons()


# ─── ui/R1a: chat profillari (avatar + about) ────────────────────────────────
#
# Nega alohida ish: `refresh_chats` faqat `GetDialogs` javobini yozadi (RPC
# qo'shmaydi). `about` va profil rasmi esa har chat uchun **alohida** serverga
# borishni talab qiladi, shuning uchun ular byudjetlangan, sekin yo'lda:
# bir yugurishda ≤ `limit` ta chat, har biri uchun ko'pi bilan bitta
# `GetFull*` va bitta fayl yuklash, oralarida `WAIT_BETWEEN_BATCHES`.
# FloodWait retry qilinmaydi — `services/avatars.py` uni `chats.avatar_error`
# ga matn qilib yozadi, keyingi soatlik yugurish qayta uradi.

PROFILE_LIMIT = 50


def job_id_profiles(account_id: int) -> str:
    return f"profiles:acc:{account_id}"


async def _stale_profile_chats(account_id: int, limit: int) -> list[int]:
    """Profil ma'lumoti eskirgan chatlar: `about` yo'q yoki avatar yuklanmagan."""
    from sqlalchemy import or_, select

    from app.db.base import session_scope
    from app.db.models import Chat

    async with session_scope() as db:
        rows = await db.execute(
            select(Chat.id)
            .where(
                Chat.account_id == account_id,
                or_(
                    Chat.about.is_(None),
                    (Chat.avatar_photo_id.isnot(None)) & (Chat.avatar_updated_at.is_(None)),
                ),
            )
            .order_by(Chat.avatar_updated_at.asc().nulls_first(), Chat.id)
            .limit(limit)
        )
        return [int(r) for r in rows.scalars().all()]


async def refresh_chat_profiles(
    ctx: dict[str, Any], account_id: int, limit: int = PROFILE_LIMIT
) -> dict[str, Any]:
    """Chat profillari: `about` + profil rasmi (byudjetlangan, soatiga bir marta)."""
    import asyncio

    from app.db.base import session_scope
    from app.db.models import Chat
    from app.services import avatars as AV

    limit = max(1, min(int(limit), 200))
    chat_ids = await _stale_profile_chats(account_id, limit)
    about_done = 0
    avatars_done = 0
    for i, chat_id in enumerate(chat_ids):
        if i:
            await asyncio.sleep(ing.WAIT_BETWEEN_BATCHES)
        try:
            async with session_scope() as db:
                row = await db.get(Chat, chat_id)
                needs_about = row is not None and row.about is None
                needs_avatar = row is not None and row.avatar_updated_at is None
            if needs_about and await AV.refresh_about(account_id, chat_id):
                about_done += 1
            if needs_avatar and await AV.refresh_avatar(account_id, chat_id):
                avatars_done += 1
        except Exception as exc:  # bitta chat qolganlarini to'xtatmasin
            log.warning("worker.profiles.failed", chat_id=chat_id, error=str(exc)[:200])
    if chat_ids:
        log.info(
            "worker.profiles.done",
            account_id=account_id,
            chats=len(chat_ids),
            about=about_done,
            avatars=avatars_done,
        )
    await heartbeat("refresh_chat_profiles", extra={"chats": len(chat_ids)})
    return {"chats": len(chat_ids), "about": about_done, "avatars": avatars_done}


async def refresh_all_chat_profiles(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cron (soatiga): har faol akkaunt uchun bitta byudjetlangan yugurish."""
    accounts = await ing.active_account_ids()
    for account_id in accounts:
        await refresh_chat_profiles(ctx, account_id)
    return {"accounts": len(accounts)}


async def _enqueue_chat_profiles(account_id: int) -> None:
    """`sync_account` oxiridagi ilgak — yangi chatlar darhol profil oladi."""
    try:
        await enqueue("refresh_chat_profiles", account_id, job_id=job_id_profiles(account_id))
    except Exception as exc:  # navbat yo'q bo'lsa ham sync muvaffaqiyatli qoladi
        log.warning("worker.profiles_enqueue_failed", account_id=account_id, error=str(exc)[:200])


#: ui/R1a ishlari va cron'i — `worker/settings.py` ga `*tasks.UI_R1A_*` bo'lib
#: qo'shiladi (parallel agentlar bir xil qatorni tahrirlamasin).
UI_R1A_FUNCTIONS: list[Any] = [refresh_chat_profiles, refresh_all_chat_profiles]


def _ui_r1a_crons() -> list[Any]:
    from arq.cron import cron

    return [cron(refresh_all_chat_profiles, minute={40}, run_at_startup=False, timeout=1800)]


UI_R1A_CRONS: list[Any] = _ui_r1a_crons()


# ─── jo'natuvchi nomlari: migratsiya 0019 dan oldingi tarix uchun ────────────
# `senders` jadvali kech qo'shilgani uchun eski xabarlarning muallifi nomsiz
# qolgan ("Foydalanuvchi 1164475416"). Bu ish tarixni **qayta sinxronlamaydi**:
# har chatdan oxirgi `per_chat` xabarni o'qiydi va `m.sender` keshidan nom
# yig'adi. Byudjet: bir yugurishda ≤ `SENDERS_CHATS_PER_RUN` chat,
# oralarida `WAIT_BETWEEN_BATCHES`, FloodWait'da **to'xtaydi** (retry yo'q).

SENDERS_CHATS_PER_RUN = 30


def job_id_senders(account_id: int) -> str:
    return f"senders:acc:{account_id}"


async def backfill_senders(
    ctx: dict[str, Any], account_id: int, per_chat: int = ing.SENDER_BACKFILL_PER_CHAT
) -> dict[str, Any]:
    """Nomlar yetishmayotgan chatlardan `senders` qatorlarini to'ldiradi."""
    import asyncio

    per_chat = max(10, min(int(per_chat), 500))
    chat_ids = await ing.chats_missing_senders(
        account_id, limit=SENDERS_CHATS_PER_RUN, per_chat=per_chat
    )
    done = 0
    senders = 0
    paused = 0
    for i, chat_id in enumerate(chat_ids):
        if i:
            await asyncio.sleep(ing.WAIT_BETWEEN_BATCHES)
        try:
            senders += await ing.harvest_senders(chat_id, per_chat=per_chat)
            done += 1
        except ing.SyncPaused as exc:  # FloodWait — shu yugurish to'xtaydi
            paused = int(getattr(exc, "retry_after", 0) or 0)
            log.warning("worker.senders.flood_wait", account_id=account_id, seconds=paused)
            break
        except Exception as exc:  # bitta chat qolganlarini to'xtatmasin
            log.warning("worker.senders.failed", chat_id=chat_id, error=str(exc)[:200])
    if chat_ids:
        log.info(
            "worker.senders.done",
            account_id=account_id,
            chats=done,
            candidates=len(chat_ids),
            senders=senders,
        )
    await heartbeat("backfill_senders", extra={"chats": done, "senders": senders})
    return {"chats": done, "senders": senders, "candidates": len(chat_ids), "flood_wait": paused}


async def backfill_senders_all(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cron (soatiga): har faol akkaunt uchun bitta byudjetlangan yugurish."""
    accounts = await ing.active_account_ids()
    for account_id in accounts:
        await backfill_senders(ctx, account_id)
    return {"accounts": len(accounts)}


async def _enqueue_senders_backfill(account_id: int) -> None:
    """`sync_account` oxiridagi ilgak — login/sync'dan keyin bir marta."""
    try:
        await enqueue("backfill_senders", account_id, job_id=job_id_senders(account_id))
    except Exception as exc:  # navbat yo'q bo'lsa ham sync muvaffaqiyatli qoladi
        log.warning("worker.senders_enqueue_failed", account_id=account_id, error=str(exc)[:200])


SENDERS_FUNCTIONS: list[Any] = [backfill_senders, backfill_senders_all]


def _senders_crons() -> list[Any]:
    from arq.cron import cron

    return [cron(backfill_senders_all, minute={25}, run_at_startup=False, timeout=1800)]


SENDERS_CRONS: list[Any] = _senders_crons()
