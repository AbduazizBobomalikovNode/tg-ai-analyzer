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

from sqlalchemy import and_, func, select
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
from app.llm import (
    EMBED_DOCUMENT,
    BaseProvider,
    BatchState,
    Capability,
    LLMError,
    Task,
    embed_model_name,
    estimate_tokens_heuristic,
    resolve,
)
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


def _provider_for_batch() -> tuple[BaseProvider, str]:
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
    session: AsyncSession,
    *,
    limit: int | None = None,
    account_id: int | None = None,
    skip_accounts: set[int] | None = None,
) -> SubmitReport:
    """Vektori yo'q xabarlarni bitta batch jobga yuboradi.

    Tanlov sharti `worker/tasks.embed_messages` bilan **bir xil**: "**joriy
    model** bo'yicha vektori yo'q". Ilgari bu yerda "umuman vektori yo'q" edi
    — `EMBED_MODEL` almashtirilganda batch yo'li hech narsa topmasdi, sinxron
    yo'l esa topardi, ya'ni ikkala rejim boshqa javob berardi (F3/H1).

    `skip_accounts` — kunlik byudjeti tugagan egalarning akkauntlari (M3).
    """
    s = get_settings()
    want_model = embed_model_name()
    skip = sorted(skip_accounts or ())
    q = (
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
            func.length(Message.text) >= EMBED_MIN_CHARS,
            ~_inflight(KIND_MESSAGES, Message.id),
            *([Chat.account_id.not_in(skip)] if skip else []),
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


async def submit_nodes(
    session: AsyncSession, *, limit: int | None = None, skip_accounts: set[int] | None = None
) -> SubmitReport:
    """Vektori yo'q retrieval tugunlarini bitta batch jobga yuboradi.

    `embed_model_name()` — `nodes.embed_nodes` bilan **bitta** manba
    (`s.embed_model` `LLM_TASK_EMBED` / `LLM_MODEL_EMBED` override'ini
    ko'rmasdi). `skip_accounts` — byudjeti tugagan egalar (M3).
    """
    s = get_settings()
    want_model = embed_model_name()
    skip = sorted(skip_accounts or ())
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
                    *([RetrievalNode.account_id.not_in(skip)] if skip else []),
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
        # L11: har job **o'z savepoint'ida**. Bittasida kutilmagan istisno
        # bo'lsa (DB xatosi, SDK shakli o'zgargan) faqat o'sha jobning
        # o'zgarishlari qaytariladi — qolganlari yozilgan holicha qoladi.
        # Ilgari bitta istisno butun tranzaksiyani (va boshqa joblarning
        # yozilgan vektorlarini) uchirib yuborardi.
        try:
            async with session.begin_nested():
                await _poll_one(
                    session, provider, row, rep, max_attempts=s.embed_batch_max_attempts
                )
        except Exception as exc:  # bitta job qolganini to'xtatmasin
            log.warning(
                "embed_batch.poll_job_failed",
                job=row.job_name,
                kind=row.kind,
                error=str(exc)[:200],
            )

    await session.flush()
    return rep


def _base_name(model: str) -> str:
    """`models/gemini-embedding-001` → `gemini-embedding-001` (taqqoslash uchun)."""
    return model.split("/")[-1].strip().lower()


async def _original_texts(session: AsyncSession, kind: str, item_ids: list[int]) -> list[str]:
    """Job yuborilganda ishlatilgan matnlarni `item_ids` tartibida qayta tiklaydi.

    Faqat `_poll_one` narx taxminini tiklashda chaqiradi — javobda
    `usage.tokens_in` 0 bo'lib qolganda (batch javobi ko'pincha usage
    bermaydi — H2/M). `submit_messages`/`submit_nodes` bilan **bir xil**
    kesish (`EMBED_MAX_CHARS` / `embed_payload`) — aks holda taxmin haqiqiy
    yuborilgan matndan farq qilib, noto'g'ri tomonga siljiydi.

    Oraliqda o'chirilgan/topilmagan element — bo'sh satr (0 token; narx bir oz
    kam baholanadi, lekin yozish/qat'iy moslikka ta'sir qilmaydi — bu funksiya
    faqat narx uchun matn yig'adi, vektor yozishga aralashmaydi).
    """
    if not item_ids:
        return []
    if kind == KIND_MESSAGES:
        rows = (
            await session.execute(select(Message.id, Message.text).where(Message.id.in_(item_ids)))
        ).all()
        by_id = {int(mid): (txt or "")[:EMBED_MAX_CHARS] for mid, txt in rows}
    else:
        nodes = (
            await session.execute(select(RetrievalNode).where(RetrievalNode.id.in_(item_ids)))
        ).scalars()
        by_id = {int(n.id): embed_payload(n) for n in nodes}
    return [by_id.get(i, "") for i in item_ids]


async def _poll_one(
    session: AsyncSession,
    provider: BaseProvider,
    row: EmbedBatch,
    rep: PollReport,
    *,
    max_attempts: int,
) -> None:
    """Bitta jobning holatini o'qiydi va tugagan bo'lsa vektorlarini yozadi."""
    try:
        status = await provider.embed_batch_poll(row.job_name)
    except LLMError as exc:
        log.warning("embed_batch.poll_failed", job=row.job_name, error=str(exc)[:200])
        return

    if not status.done:
        # PENDING → RUNNING o'tishi ham yoziladi: dashboardda job
        # qotib qolganini ko'rish uchun.
        if status.state is BatchState.RUNNING and row.status != EmbedBatchStatus.RUNNING:
            row.status = EmbedBatchStatus.RUNNING
        rep.still_running += 1
        return

    if status.state is not BatchState.SUCCEEDED:
        _fail(row, status.state.value, status.error or status.raw_state, max_attempts)
        rep.failed += 1
        rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
        return

    try:
        # H2: **so'ralgan** model provider'ga uzatiladi — javobda model nomi
        # bo'lmasa u shu nomga qaytadi (aks holda `normalize_vectors("")`
        # normallashtirishni jimgina o'tkazib yuborardi).
        result = await provider.embed_batch_fetch(row.job_name, model=row.model)
    except LLMError as exc:
        _fail(row, EmbedBatchStatus.FAILED, str(exc)[:200], max_attempts)
        rep.failed += 1
        rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
        return

    # H2: model **bir marta** hal qilinadi. Bo'sh javob → yuborilgan model
    # (qator hech qachon `model=''` bilan yozilmaydi: bunday qator qidiruv
    # filtridan hech qachon o'tmaydi va abadiy "yetishmayotgan" bo'lib qolardi).
    model = result.model or row.model
    if result.model and _base_name(result.model) != _base_name(row.model):
        # Boshqa model = boshqa vektor fazasi. Yozish o'rniga xato: aralash
        # faza cosine ostonalarini jimgina buzardi.
        log.error("embed_batch.model_mismatch", job=row.job_name, want=row.model, got=result.model)
        _fail(
            row,
            EmbedBatchStatus.FAILED,
            f"model mismatch: want {row.model}, got {result.model}",
            max_attempts,
        )
        rep.failed += 1
        rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
        return

    item_ids = list(row.item_ids or [])
    if len(result.vectors) != len(item_ids):
        # Qat'iy: yarim yozish vektorni boshqa elementga tushirardi.
        log.error(
            "embed_batch.misaligned",
            job=row.job_name,
            expected=len(item_ids),
            got=len(result.vectors),
            model=model,
        )
        _fail(
            row,
            EmbedBatchStatus.FAILED,
            f"misaligned: {len(item_ids)} != {len(result.vectors)}",
            max_attempts,
        )
        rep.failed += 1
        rep.permanent += int(row.status == EmbedBatchStatus.FAILED_PERMANENT)
        return

    written = await _write_vectors(session, row.kind, item_ids, result.vectors, model)
    row.status = EmbedBatchStatus.SUCCEEDED
    row.completed_at = datetime.now(UTC)
    # Narx qoidasi: API **qaytargan** model nomi (bo'sh bo'lsa yuborilgani).
    row.model = model
    tokens_in = result.usage.tokens_in
    if not tokens_in:
        # Gemini batch javobi ko'pincha usage/token_count bermaydi
        # (2026-09-16 da tekshirilgan) — 0 qoldirilsa narx doim $0 bo'lib,
        # kunlik xarajat limiti bu sarfni umuman ko'rmaydi. Original matnlar
        # (`_original_texts`) faqat shu holatda — kerak bo'lmaganda qo'shimcha
        # SELECT yo'q.
        texts = await _original_texts(session, row.kind, item_ids)
        tokens_in = estimate_tokens_heuristic(texts)
    row.cost_usd = estimate_cost(result.provider, row.model, tokens_in, 0, batch=True) or 0.0
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

    Konflikt kaliti `(id, model)` — PK bilan bir xil (migratsiya 0016).
    Faqat `id` bo'lganda boshqa modeldagi mavjud qator bu batchni jimgina
    yeb qo'yardi: vektor **hech qachon** yozilmasdi va job baribir
    `SUCCEEDED` bo'lardi.
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
    await session.execute(stmt.on_conflict_do_nothing(index_elements=[key, "model"]))
    return len(item_ids)
