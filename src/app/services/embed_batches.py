"""Batch embedding orkestratsiyasi — 4-bosqich/C.

Indekslash embeddinglari Gemini Batch API orqali **sinxron narxning yarmiga**
tushadi (`gemini-embedding-001`: $0.15/M → $0.075/M). Evaziga javob soatlar
ichida keladi. Bu almashuv indekslash uchun deyarli bepul: `embed_messages` /
`embed_nodes` fon cronlari, hech kim javob kutib turmaydi. Qidiruv tomoni
(`RETRIEVAL_QUERY`) esa **hech qachon** batch bo'lmaydi.

Uch bosqich, uchtasi ham boshqa cron yugurishida:

  ``submit_messages`` / ``submit_nodes``  vektori yo'q elementlarni tanlaydi,
      bitta job qilib yuboradi va `embed_batches` qatorini yozadi.
  ``poll_batches``                        tugagan joblarni o'qiydi, vektorlarni
      **qat'iy moslik** bilan yozadi, holatni belgilaydi.

Nega DB qatori: submit qaytarishi bilan worker o'lishi mumkin, job esa
serverda qoladi va to'lanadi. Job nomi va **tartiblangan** element id'lari
diskda bo'lmasa, o'sha job hech qachon o'qilmay qolardi.

Qat'iy moslik (0-bosqich qoidasi): yagona bog'lovchi — tartib. Gemini
javoblarni kirish tartibida qaytaradi, ya'ni `vectors[i]` ↔ `item_ids[i]`.
Uzunlik mos kelmasa **hech narsa yozilmaydi** va job xato deb belgilanadi;
yarim yozish indeksni jimgina buzardi.

Band elementlar: uchayotgan (`pending`/`running`) jobga tushgan id qayta
yuborilmaydi — `item_ids @> to_jsonb(id)` konteynmenti, `jsonb_path_ops`
GIN indeksi bilan (migratsiya 0013). Alohida `pending_embed_items` yon
jadvali **ataylab yo'q**: u ikkinchi haqiqat manbai bo'lardi va job qatori
bilan sinxronsizlanib qolishi mumkin edi; uchayotgan joblar soni esa
o'nlab, yuzlab emas — konteynment so'rovi indeks bilan arzon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    Chat,
    EmbedBatch,
    EmbedBatchStatus,
    Message,
    MessageEmbedding,
    RetrievalNode,
    RetrievalNodeEmbedding,
)
from app.llm import EMBED_DOCUMENT, BatchState, Capability, LLMError, Task, resolve
from app.llm.pricing import estimate_cost
from app.logging import get_logger
from app.services.nodes import embed_payload

log = get_logger(__name__)

KIND_MESSAGES = "messages"
KIND_NODES = "nodes"

#: `worker/tasks.EMBED_MAX_CHARS` / `nodes.EMBED_MAX_CHARS` bilan bir xil.
EMBED_MAX_CHARS = 4_000
#: "ok", "👍" — bekor xarajat (`worker/tasks.EMBED_MIN_CHARS`).
EMBED_MIN_CHARS = 20

__all__ = [
    "KIND_MESSAGES",
    "KIND_NODES",
    "PollReport",
    "SubmitReport",
    "batch_embedding_available",
    "poll_batches",
    "submit_messages",
    "submit_nodes",
]


@dataclass(slots=True)
class SubmitReport:
    submitted: int = 0
    job: str = ""
    model: str = ""
    error: str = ""


@dataclass(slots=True)
class PollReport:
    checked: int = 0
    succeeded: int = 0
    written: int = 0
    failed: int = 0
    permanent: int = 0
    still_running: int = 0


# ─── provider ────────────────────────────────────────────────────────────────


def _provider_for_batch() -> tuple[Any, str]:
    """`Task.EMBED` provideri, agar u batch ham qila olsa.

    Qila olmasa `LLMError` — chaqiruvchi sinxron yo'lga tushadi. Jimgina
    "hech narsa qilmaslik" yo'q: aks holda batch rejimi yoqilgan deployda
    indekslash to'xtab, hech kim sezmasdi.
    """
    provider, model = resolve(Task.EMBED)
    if Capability.BATCH_EMBED not in provider.capabilities(model):
        raise LLMError(
            "embed_batches",
            f"{provider.name}:{model} batch embeddingni qo'llamaydi — EMBED_MODE=sync qiling",
        )
    return provider, model


def batch_embedding_available() -> bool:
    try:
        _provider_for_batch()
    except LLMError:
        return False
    return True


# ─── band elementlar ─────────────────────────────────────────────────────────


_INFLIGHT = [EmbedBatchStatus.PENDING.value, EmbedBatchStatus.RUNNING.value]


def _inflight(kind: str, id_col: Any) -> Any:
    """`id_col` uchayotgan jobga tushganmi (EXISTS).

    `item_ids @> to_jsonb(id)` — `jsonb_path_ops` GIN indeksidan foydalanadi
    (migratsiya 0013), shuning uchun uchayotgan joblar soni o'ssa ham arzon.
    """
    return (
        select(EmbedBatch.id)
        .where(
            EmbedBatch.kind == kind,
            EmbedBatch.status.in_(_INFLIGHT),
            EmbedBatch.item_ids.op("@>")(func.to_jsonb(id_col)),
        )
        .exists()
    )


def _cap(items: list[tuple[int, str]], max_items: int, max_chars: int) -> list[tuple[int, str]]:
    """Ikki qopqoq: elementlar soni va **jami belgilar**.

    Ikkinchisi inline so'rov tanasi uchun: Gemini uni ~20 MB bilan cheklaydi,
    va bitta juda uzun to'plam qopqoqsiz so'rovni rad ettirardi.
    """
    out: list[tuple[int, str]] = []
    total = 0
    for item_id, payload in items[:max_items]:
        if out and total + len(payload) > max_chars:
            break
        out.append((item_id, payload))
        total += len(payload)
    return out


# ─── submit ──────────────────────────────────────────────────────────────────


async def _submit(
    session: AsyncSession,
    *,
    kind: str,
    items: list[tuple[int, str]],
    account_id: int | None = None,
) -> SubmitReport:
    if not items:
        return SubmitReport()

    s = get_settings()
    try:
        provider, model = _provider_for_batch()
        job = await provider.embed_batch_submit(
            [payload for _, payload in items],
            model=model,
            dim=s.embed_dims,
            task_type=EMBED_DOCUMENT,
            tag=f"tgai-{kind}",
        )
    except LLMError as exc:
        log.warning("embed_batch.submit_failed", kind=kind, error=str(exc)[:200])
        return SubmitReport(error=str(exc)[:200])

    session.add(
        EmbedBatch(
            provider=provider.name,
            job_name=job,
            kind=kind,
            account_id=account_id,
            # Tartib — yagona bog'lovchi. Bu ro'yxat hech qachon qayta
            # tartiblanmaydi va yangilanmaydi.
            item_ids=[item_id for item_id, _ in items],
            model=model,
            task_type=EMBED_DOCUMENT,
            status=EmbedBatchStatus.PENDING,
            attempts=1,
        )
    )
    await session.flush()
    log.info("embed_batch.submitted", kind=kind, job=job, items=len(items), model=model)
    return SubmitReport(submitted=len(items), job=job, model=model)


async def submit_messages(
    session: AsyncSession, *, limit: int | None = None, account_id: int | None = None
) -> SubmitReport:
    """Vektori yo'q xabarlarni bitta batch jobga yuboradi."""
    s = get_settings()
    q = (
        select(Message.id, Message.text)
        .join(Chat, Chat.id == Message.chat_id)
        .outerjoin(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .where(
            MessageEmbedding.message_id.is_(None),
            Chat.synced_total > 0,
            func.length(Message.text) >= EMBED_MIN_CHARS,
            ~_inflight(KIND_MESSAGES, Message.id),
        )
        .order_by(Message.id.desc())
        .limit(limit or s.embed_batch_max_items)
    )
    if account_id is not None:
        q = q.where(Chat.account_id == account_id)

    rows = (await session.execute(q)).all()
    items = _cap(
        [(int(mid), (txt or "")[:EMBED_MAX_CHARS]) for mid, txt in rows],
        s.embed_batch_max_items,
        s.embed_batch_max_chars,
    )
    return await _submit(session, kind=KIND_MESSAGES, items=items, account_id=account_id)


async def submit_nodes(session: AsyncSession, *, limit: int | None = None) -> SubmitReport:
    """Vektori yo'q retrieval tugunlarini bitta batch jobga yuboradi."""
    s = get_settings()
    want_model = s.embed_model
    missing = (
        ~select(RetrievalNodeEmbedding.node_id)
        .where(
            RetrievalNodeEmbedding.node_id == RetrievalNode.id,
            RetrievalNodeEmbedding.model == want_model,
        )
        .exists()
    )
    nodes = list(
        (
            await session.execute(
                select(RetrievalNode)
                .where(
                    missing,
                    func.length(RetrievalNode.text) > 0,
                    ~_inflight(KIND_NODES, RetrievalNode.id),
                )
                .order_by(RetrievalNode.id.desc())
                .limit(limit or s.embed_batch_max_items)
            )
        ).scalars()
    )
    items = _cap(
        [(int(n.id), embed_payload(n)) for n in nodes],
        s.embed_batch_max_items,
        s.embed_batch_max_chars,
    )
    return await _submit(session, kind=KIND_NODES, items=items)


# ─── poll ────────────────────────────────────────────────────────────────────


async def poll_batches(session: AsyncSession, *, limit: int | None = None) -> PollReport:
    """Uchayotgan joblarni tekshiradi; tugaganlarini yozadi yoki xato belgilaydi."""
    s = get_settings()
    rep = PollReport()
    rows = list(
        (
            await session.execute(
                select(EmbedBatch)
                .where(EmbedBatch.status.in_(_INFLIGHT))
                .order_by(EmbedBatch.submitted_at)
                .limit(limit or s.embed_batch_poll_limit)
            )
        ).scalars()
    )
    if not rows:
        return rep

    try:
        provider, _ = _provider_for_batch()
    except LLMError as exc:
        log.warning("embed_batch.poll_unavailable", error=str(exc)[:200])
        return rep

    for row in rows:
        rep.checked += 1
        try:
            status = await provider.embed_batch_poll(row.job_name)
        except LLMError as exc:
            log.warning("embed_batch.poll_failed", job=row.job_name, error=str(exc)[:200])
            continue

        if not status.done:
            # PENDING → RUNNING o'tishi ham yoziladi: dashboardda job
            # qotib qolganini ko'rish uchun.
            if status.state is BatchState.RUNNING and row.status != EmbedBatchStatus.RUNNING:
                row.status = EmbedBatchStatus.RUNNING
            rep.still_running += 1
            continue

        if status.state is not BatchState.SUCCEEDED:
            _fail(
                row,
                status.state.value,
                status.error or status.raw_state,
                s.embed_batch_max_attempts,
            )
            rep.failed += 1
            rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
            continue

        try:
            result = await provider.embed_batch_fetch(row.job_name)
        except LLMError as exc:
            _fail(row, EmbedBatchStatus.FAILED, str(exc)[:200], s.embed_batch_max_attempts)
            rep.failed += 1
            rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
            continue

        item_ids = list(row.item_ids or [])
        if len(result.vectors) != len(item_ids):
            # Qat'iy: yarim yozish vektorni boshqa elementga tushirardi.
            log.error(
                "embed_batch.misaligned",
                job=row.job_name,
                expected=len(item_ids),
                got=len(result.vectors),
                model=result.model,
            )
            _fail(
                row,
                EmbedBatchStatus.FAILED,
                f"misaligned: {len(item_ids)} != {len(result.vectors)}",
                s.embed_batch_max_attempts,
            )
            rep.failed += 1
            rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
            continue

        written = await _write_vectors(session, row.kind, item_ids, result.vectors, result.model)
        row.status = EmbedBatchStatus.SUCCEEDED
        row.completed_at = datetime.now(UTC)
        # Narx qoidasi: API **qaytargan** model nomi.
        row.model = result.model or row.model
        row.cost_usd = (
            estimate_cost(result.provider, row.model, result.usage.tokens_in, 0, batch=True) or 0.0
        )
        row.error = None
        rep.succeeded += 1
        rep.written += written
        log.info(
            "embed_batch.written",
            job=row.job_name,
            kind=row.kind,
            items=written,
            model=row.model,
            cost_usd=row.cost_usd,
        )

    await session.flush()
    return rep


def _fail(row: EmbedBatch, state: str, error: str, max_attempts: int) -> None:
    """Xato/eskirgan job. `max_attempts` tugasa — boshqa yuborilmaydi.

    `failed`/`expired` holatidagi qator elementlarni **band tutmaydi**, ya'ni
    ular keyingi yugurishda qaytadan (batch yoki sync) olinadi. Cheksiz qayta
    yuborish esa cheksiz to'lov — shuning uchun qopqoq.
    """
    row.attempts += 1
    row.completed_at = datetime.now(UTC)
    row.error = error[:500]
    if row.attempts > max_attempts:
        row.status = EmbedBatchStatus.FAILED_PERMANENT
        log.error(
            "embed_batch.failed_permanent", job=row.job_name, attempts=row.attempts, error=row.error
        )
    else:
        row.status = state
        log.warning("embed_batch.failed", job=row.job_name, state=state, error=row.error)


async def _write_vectors(
    session: AsyncSession,
    kind: str,
    item_ids: list[int],
    vectors: list[list[float]],
    model: str,
) -> int:
    """Vektorlarni yozadi. Oraliqda paydo bo'lgan qatorlar ustidan yozilmaydi.

    Batch uchayotganda o'sha element sinxron yo'l bilan ham embed qilinishi
    mumkin edi (rejim `sync` ga qaytarilgan bo'lsa). `ON CONFLICT DO NOTHING`
    — birinchi yozilgani qoladi, ikkinchisi jimgina tushadi.
    """
    if not item_ids:
        return 0
    model_cls: type[Any]
    if kind == KIND_MESSAGES:
        model_cls, key = MessageEmbedding, "message_id"
    else:
        model_cls, key = RetrievalNodeEmbedding, "node_id"

    stmt = pg_insert(model_cls).values(
        [
            {key: item_id, "model": model, "vector": vec}
            for item_id, vec in zip(item_ids, vectors, strict=True)
        ]
    )
    await session.execute(stmt.on_conflict_do_nothing(index_elements=[key]))
    return len(item_ids)
