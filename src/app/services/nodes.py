"""Retrieval tugunlari — kontekstli embedding, kun ko'zgusi, hafta tuguni.

Uch ish shu yerda (arxitektura v2, 5.3-5.4):

  ``embed_nodes``          tugunlarni ``RETRIEVAL_DOCUMENT`` bilan vektorlaydi;
                           embed qilinadigan matn — ``ctx_prefix + "\\n" + text``,
                           ya'ni "ha, roziman" ham chat/sana/so'zlovchi
                           konteksti bilan birga vektorga tushadi.
  ``upsert_digest_nodes``  ``chat_digests`` L1 (kun) qatorini **level 2** tugun
                           sifatida ko'zgulaydi — digest va retrieval bitta
                           indeksdan oziqlanadi.
  ``build_week_nodes``     **faqat kun tugunlaridan** hafta tuguni (level 3).
                           Xom xabarlar qayta o'qilmaydi: xarajat ierarxiya
                           bo'ylab chiziqli o'sadi, kvadratik emas.

Invalidatsiya bitta qoida bilan: tugunning ``content_hash`` i manbadan
hisoblanadi (epizodda — xabarlar ro'yxati, haftada — bolalarning
hash'laridan md5). Manba o'zgarsa hash o'zgaradi, tugun qayta quriladi va
vektori o'chiriladi.

Xavfsizlik: hafta tuguni yagona LLM chaqiruvi bo'lgan joy va u begona matnni
ko'radi — shuning uchun kirish **har doim** ``<untrusted_data kind="digests">``
konvertida, ``untrusted.neutralize`` dan o'tgan holda (invariant 3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chat, ChatDigest, Message, RetrievalNode, RetrievalNodeEmbedding
from app.llm import EMBED_DOCUMENT, LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services import untrusted as U
from app.services.episodes import to_signed64
from app.services.prompts_retrieval import WEEK_DIGEST_PROMPT
from app.services.textnorm import simhash64

log = get_logger(__name__)

LEVEL_EPISODE = 1
LEVEL_DAY = 2
LEVEL_WEEK = 3
LEVEL_MONTH = 4

#: Embeddingga beriladigan matn qopqog'i (`worker/tasks.EMBED_MAX_CHARS` bilan bir xil).
EMBED_MAX_CHARS = 4_000
#: Bitta LLM embed chaqiruvidagi tugunlar soni.
EMBED_BATCH = 64
#: Hafta tuguniga kiradigan kun digestlari matni qopqog'i (narx).
WEEK_MAX_CHARS = 24_000
#: Hafta tuguni javobi (token) — L2 digest ~250 so'z.
WEEK_MAX_TOKENS = 700

__all__ = [
    "EMBED_BATCH",
    "EMBED_MAX_CHARS",
    "LEVEL_DAY",
    "LEVEL_EPISODE",
    "LEVEL_MONTH",
    "LEVEL_WEEK",
    "EmbedReport",
    "build_week_nodes",
    "chats_needing_week_node",
    "embed_nodes",
    "embed_payload",
    "to_signed64",
    "upsert_digest_nodes",
    "week_start",
]


@dataclass(slots=True)
class EmbedReport:
    embedded: int = 0
    pending: int = 0
    model: str = ""
    error: str = ""


def week_start(dt: datetime) -> datetime:
    """Dushanba 00:00 UTC — hafta tugunining `starts_at` kaliti."""
    d = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    d = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return d - timedelta(days=d.weekday())


def _day_start(dt: datetime) -> datetime:
    d = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def embed_payload(node: RetrievalNode) -> str:
    """Vektorlanadigan matn: kontekst prefiksi + tugun matni (kesilgan).

    Prefiks **kesilmaydi** — u eng zich signal (chat, sana, so'zlovchilar);
    qopqoq faqat tugun matniga tegadi.
    """
    return f"{node.ctx_prefix}\n{(node.text or '')[:EMBED_MAX_CHARS]}"


# ─── embedding ───────────────────────────────────────────────────────────────


async def embed_nodes(
    session: AsyncSession,
    llm: LLM | None = None,
    *,
    batch: int = EMBED_BATCH,
    limit: int | None = None,
    skip_accounts: set[int] | None = None,
) -> EmbedReport:
    """Joriy model uchun vektori yo'q tugunlarni batch'lab embed qiladi.

    `task_type=RETRIEVAL_DOCUMENT` — so'rov tomonida `RETRIEVAL_QUERY`
    ishlatiladi (`search.vector_search`). Asimmetrik retrieval: ikkalasi ham
    default bilan yuborilsa bepul aniqlik yo'qoladi (v2, 5.3).

    **Qat'iy uzunlik tekshiruvi** (0-bosqichdagi kabi): provider qaytargan
    vektorlar soni batch bilan mos kelmasa batch butunlay tashlanadi — aks
    holda vektor boshqa tugunga yozilib, indeks jimgina buzilardi.

    Args:
        session: DB sessiyasi (parallel ishlatilmaydi).
        llm: LLM fasadi; berilmasa yangisi yaratiladi.
        batch: bitta chaqiruvdagi tugunlar soni.
        limit: bir yugurishda ko'rib chiqiladigan tugunlar soni.

    Returns:
        `EmbedReport(embedded, pending, model, error)`.
    """
    s = get_settings()
    # phase4/B: joriy embedding modeli — `EMBED_MODEL`/`LLM_MODEL_EMBED` ni ham
    # hisobga oladi, aks holda model almashganda router boshqa model qaytarib
    # bu yerda "model mismatch" bo'lardi.
    from app.llm import embed_model_name

    want_model = embed_model_name()
    limit = limit or s.embed_batch_per_run
    missing = (
        ~select(RetrievalNodeEmbedding.node_id)
        .where(
            RetrievalNodeEmbedding.node_id == RetrievalNode.id,
            RetrievalNodeEmbedding.model == want_model,
        )
        .exists()
    )
    # M3: kunlik byudjeti tugagan egalarning tugunlari bu yugurishda embed
    # qilinmaydi (keyingi cron'da, byudjet yangilangach, o'z-o'zidan qaytadi).
    skip = sorted(skip_accounts or ())
    rows = list(
        (
            await session.execute(
                select(RetrievalNode)
                .where(
                    missing,
                    func.length(RetrievalNode.text) > 0,
                    *([RetrievalNode.account_id.not_in(skip)] if skip else []),
                )
                .order_by(RetrievalNode.id.desc())
                .limit(limit)
            )
        ).scalars()
    )
    if not rows:
        return EmbedReport(model=want_model)

    client = llm or LLM()
    report = EmbedReport(pending=len(rows), model=want_model)
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        try:
            res = await client.embed([embed_payload(n) for n in chunk], task_type=EMBED_DOCUMENT)
        except LLMError as exc:
            report.error = str(exc)[:200]
            log.warning("nodes.embed_failed", error=report.error, batch=i // batch)
            break
        if len(res.vectors) != len(chunk):
            # Moslik buzilsa vektor boshqa tugunga yozilardi — batch tashlanadi.
            log.error(
                "nodes.embed_misaligned",
                batch=i // batch,
                expected=len(chunk),
                got=len(res.vectors),
                model=res.model,
            )
            continue
        if res.model != want_model:
            # Aks holda bu tugunlar har yugurishda qayta embed qilinaverardi.
            report.error = f"model mismatch: want {want_model}, got {res.model}"
            log.error("nodes.embed_model_mismatch", want=want_model, got=res.model)
            break
        # PK `(node_id, model)` (migratsiya 0016): bitta tugunda bir nechta
        # model qatori normal, shuning uchun `add_all` emas — `ON CONFLICT DO
        # NOTHING`. Oraliqda batch yo'li o'sha `(node_id, model)` ni yozib
        # qo'ygan bo'lsa birinchisi qoladi.
        stmt = pg_insert(RetrievalNodeEmbedding).values(
            [
                {"node_id": int(n.id), "model": res.model, "vector": vec}
                for n, vec in zip(chunk, res.vectors, strict=True)
            ]
        )
        await session.execute(stmt.on_conflict_do_nothing(index_elements=["node_id", "model"]))
        await session.flush()
        report.embedded += len(chunk)
    report.pending -= report.embedded
    log.info("nodes.embed.done", embedded=report.embedded, pending=report.pending)
    return report


# ─── level 2: kun digesti ko'zgusi ───────────────────────────────────────────


async def upsert_digest_nodes(
    session: AsyncSession, chat_id: int, day: datetime
) -> RetrievalNode | None:
    """`chat_digests` L1 qatorini level-2 tugun sifatida ko'zgulaydi.

    Digest matnining o'zi tugun matni bo'ladi: hafta/oy tugunlari **faqat**
    shu darajadan oziqlanadi, xom xabarlar qayta o'qilmaydi.

    Idempotent: `(chat_id, level=2, starts_at=day)` kaliti bo'yicha; manba
    o'zgarmagan bo'lsa (`content_hash` bir xil) qatorga tegilmaydi.

    Returns:
        Yaratilgan/yangilangan (yoki o'zgarmagan) tugun; digest yo'q bo'lsa `None`.
    """
    day = _day_start(day)
    digest = (
        await session.execute(
            select(ChatDigest).where(ChatDigest.chat_id == chat_id, ChatDigest.day == day)
        )
    ).scalar_one_or_none()
    if digest is None or not (digest.digest or "").strip():
        return None
    chat = await session.get(Chat, chat_id)
    if chat is None:
        return None

    bounds = (
        await session.execute(
            select(func.min(Message.tg_msg_id), func.max(Message.tg_msg_id)).where(
                Message.chat_id == chat_id,
                Message.published_at >= day,
                Message.published_at < day + timedelta(days=1),
                # L7: o'chirilgan xabar oraliqni kengaytirib yubormasin —
                # `node_hits_to_messages` o'sha oraliqdan sitata izlaydi.
                Message.deleted_at.is_(None),
            )
        )
    ).one()
    lo = int(bounds[0] or 0)
    hi = int(bounds[1] or 0)

    text = digest.digest
    src = digest.src_fingerprint or hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    node = (
        await session.execute(
            select(RetrievalNode).where(
                RetrievalNode.chat_id == chat_id,
                RetrievalNode.level == LEVEL_DAY,
                RetrievalNode.starts_at == day,
            )
        )
    ).scalar_one_or_none()
    if node is not None and node.content_hash == src:
        return node
    if node is None:
        node = RetrievalNode(account_id=chat.account_id, chat_id=chat_id, level=LEVEL_DAY)
        session.add(node)
    else:
        await _drop_embedding(session, node.id)
    node.starts_at = day
    node.ends_at = day + timedelta(days=1)
    node.msg_lo = lo
    node.msg_hi = hi
    node.msg_count = int(digest.msg_count or 0)
    node.speakers = []
    node.text = text
    node.ctx_prefix = f"[chat: {chat.title} ({chat.type}) | {day.date().isoformat()} | digest]"
    node.content_hash = src
    node.simhash = to_signed64(simhash64(text))
    await session.flush()
    log.info("nodes.day_upserted", chat_id=chat_id, day=str(day.date()), node_id=node.id)
    return node


async def _drop_embedding(session: AsyncSession, node_id: int) -> None:
    from sqlalchemy import delete

    await session.execute(
        delete(RetrievalNodeEmbedding).where(RetrievalNodeEmbedding.node_id == node_id)
    )


# ─── level 3: hafta tuguni (faqat kun tugunlaridan) ──────────────────────────


def children_hash(children: Sequence[RetrievalNode]) -> str:
    """Bolalar `content_hash` lari ustidan md5 — bittasi o'zgarsa ota qayta quriladi."""
    return hashlib.md5(
        ",".join(f"{c.starts_at.date()}:{c.content_hash}" for c in children).encode(),
        usedforsecurity=False,
    ).hexdigest()


def _week_envelope(chat: Chat, children: Sequence[RetrievalNode]) -> str:
    """Kun digestlari — `<untrusted_data kind="digests">` konvertida (invariant 3)."""
    body: list[str] = []
    used = 0
    for c in children:
        line = f"[{c.starts_at.date()}] {c.text}"
        if used + len(line) > WEEK_MAX_CHARS:
            break
        body.append(line)
        used += len(line) + 1
    head = (
        f'<untrusted_data kind="digests" source="telegram" chat="{U.attr(chat.title)}" '
        f'week="{children[0].starts_at.date()}">'
    )
    return f"{head}\n{U.neutralize(chr(10).join(body))}\n</untrusted_data>"


async def build_week_nodes(
    session: AsyncSession,
    chat_id: int,
    week: datetime,
    *,
    llm: LLM | None = None,
    force: bool = False,
) -> RetrievalNode | None:
    """Hafta tuguni (level 3) — **faqat** level-2 tugunlardan, bitta `Task.ROUTE`.

    Xom xabarlar so'ralmaydi (`tests/test_nodes.py` buni `messages` so'rovi
    yo'qligi bilan qo'riqlaydi): ierarxiya butun ma'nosi shunda.

    Args:
        session: DB sessiyasi.
        chat_id: `chats.id`.
        week: hafta ichidagi istalgan vaqt — dushanbaga tekislanadi.
        llm: LLM fasadi (testda soxta).
        force: `content_hash` bir xil bo'lsa ham qayta hisoblash.

    Returns:
        Tugun; hafta ichida kun tugunlari bo'lmasa yoki LLM uzilsa `None`.
    """
    start = week_start(week)
    end = start + timedelta(days=7)
    children = list(
        (
            await session.execute(
                select(RetrievalNode)
                .where(
                    RetrievalNode.chat_id == chat_id,
                    RetrievalNode.level == LEVEL_DAY,
                    RetrievalNode.starts_at >= start,
                    RetrievalNode.starts_at < end,
                )
                .order_by(RetrievalNode.starts_at)
            )
        ).scalars()
    )
    if not children:
        return None
    digest_hash = children_hash(children)
    node = (
        await session.execute(
            select(RetrievalNode).where(
                RetrievalNode.chat_id == chat_id,
                RetrievalNode.level == LEVEL_WEEK,
                RetrievalNode.starts_at == start,
            )
        )
    ).scalar_one_or_none()
    if node is not None and node.content_hash == digest_hash and not force:
        return node
    chat = await session.get(Chat, chat_id)
    if chat is None:
        return None
    # M3: hafta tuguni bu modulning **yagona** LLM chaqirig'i — akkaunt egasining
    # kunlik byudjeti tugagan bo'lsa o'tkazib yuboriladi (xato emas, keyingi
    # dushanbada yoki byudjet yangilangach qayta quriladi).
    from app.services.limits import account_budget_blocked

    over = await account_budget_blocked(session, int(chat.account_id))
    if over:
        log.info("nodes.week_budget_skip", chat_id=chat_id, exceeded=over)
        return None

    try:
        res = await (llm or LLM()).chat(
            Task.ROUTE,
            [Msg.system(WEEK_DIGEST_PROMPT), Msg.user(_week_envelope(chat, children))],
            max_tokens=WEEK_MAX_TOKENS,
        )
    except LLMError as exc:
        log.warning("nodes.week_failed", chat_id=chat_id, error=str(exc)[:200])
        return None
    text = res.text.strip()
    if not text:
        return None

    if node is None:
        node = RetrievalNode(account_id=chat.account_id, chat_id=chat_id, level=LEVEL_WEEK)
        session.add(node)
    else:
        await _drop_embedding(session, node.id)
    node.starts_at = start
    node.ends_at = end
    node.msg_lo = min(int(c.msg_lo) for c in children)
    node.msg_hi = max(int(c.msg_hi) for c in children)
    node.msg_count = sum(int(c.msg_count or 0) for c in children)
    node.speakers = []
    node.text = text
    node.ctx_prefix = (
        f"[chat: {chat.title} ({chat.type}) | week {start.date().isoformat()} | weekly digest]"
    )
    node.content_hash = digest_hash
    node.simhash = to_signed64(simhash64(text))
    node.structured = {"days": len(children), "model": res.model}
    await session.flush()
    log.info(
        "nodes.week_built",
        chat_id=chat_id,
        week=str(start.date()),
        days=len(children),
        tokens=res.usage.tokens_in + res.usage.tokens_out,
    )
    return node


async def chats_needing_week_node(
    session: AsyncSession, week: datetime, *, limit: int = 200
) -> list[int]:
    """O'sha haftada kun tuguni bor chatlar (hafta tuguni bor-yo'qligidan qat'i nazar).

    Hafta tuguni `content_hash` bo'yicha o'zi invalidatsiya qilinadi, shuning
    uchun ro'yxat "kun tuguni bor" mezoni bilan tuziladi — qayta hisoblash
    kerakmi degan qarorni `build_week_nodes` beradi (LLM chaqirilmaydi).
    """
    start = week_start(week)
    rows = await session.execute(
        select(RetrievalNode.chat_id)
        .where(
            RetrievalNode.level == LEVEL_DAY,
            RetrievalNode.starts_at >= start,
            RetrievalNode.starts_at < start + timedelta(days=7),
        )
        .group_by(RetrievalNode.chat_id)
        .limit(limit)
    )
    return [int(r[0]) for r in rows.all()]


def node_citation(node: RetrievalNode | Any) -> dict[str, Any]:
    """Tugundan level-0 sitata oralig'i — javobda `#msg_id` lar shu yerdan."""
    return {
        "node_id": int(node.id),
        "chat_id": int(node.chat_id),
        "level": int(node.level),
        "msg_lo": int(node.msg_lo),
        "msg_hi": int(node.msg_hi),
    }
