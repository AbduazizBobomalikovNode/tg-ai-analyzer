"""phase4/C: batch embedding — adapter, submit/poll/fetch va qat'iy moslik.

Haqiqiy API chaqiruvi yo'q: `genai.Client` soxta obyekt bilan almashtiriladi
(`batches.create_embeddings` / `batches.get`). Testlar SDK **shakliga** qarab
yozilgan (`google-genai` 2.23: `BatchJob.state`, `dest.inlined_embed_content_responses`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from app.llm.base import BatchState, Capability, LLMError
from app.llm.gemini import GeminiProvider, _batch_status
from app.llm.pricing import estimate_cost

EMBED_MODEL = "gemini-embedding-001"


# ─── soxta SDK ───────────────────────────────────────────────────────────────


@dataclass
class FakeEmbedding:
    values: list[float]


@dataclass
class FakeSingle:
    embedding: FakeEmbedding | None
    token_count: int = 7


@dataclass
class FakeInlined:
    response: FakeSingle | None = None
    error: Any = None


@dataclass
class FakeDest:
    inlined_embed_content_responses: list[FakeInlined] | None = None
    file_name: str | None = None


@dataclass
class FakeJob:
    name: str
    state: str
    model: str = f"models/{EMBED_MODEL}"
    dest: FakeDest | None = None
    error: Any = None


class FakeBatches:
    """`client.aio.batches` o'rnini bosadi."""

    def __init__(self, jobs: dict[str, FakeJob] | None = None) -> None:
        self.jobs = jobs or {}
        self.created: list[dict[str, Any]] = []
        self.counter = 0

    async def create_embeddings(self, *, model: str, src: Any, config: Any = None) -> FakeJob:
        self.counter += 1
        name = f"batches/fake-{self.counter}"
        requests = src.inlined_requests
        self.created.append(
            {
                "model": model,
                "contents": list(requests.contents or []),
                "task_type": requests.config.task_type,
                "dim": requests.config.output_dimensionality,
                "display_name": getattr(config, "display_name", None),
            }
        )
        job = FakeJob(name=name, state="JOB_STATE_PENDING", model=f"models/{model}")
        self.jobs[name] = job
        return job

    async def get(self, *, name: str) -> FakeJob:
        return self.jobs[name]


@dataclass
class FakeAio:
    batches: FakeBatches


@dataclass
class FakeClient:
    aio: FakeAio = field(default_factory=lambda: FakeAio(FakeBatches()))


def make_provider(batches: FakeBatches | None = None) -> tuple[GeminiProvider, FakeBatches]:
    p = GeminiProvider.__new__(GeminiProvider)  # __init__ real klient yaratardi
    fake = batches or FakeBatches()
    p._client = FakeClient(aio=FakeAio(fake))  # type: ignore[assignment]
    return p, fake


def succeeded(name: str, vectors: list[list[float]]) -> FakeJob:
    return FakeJob(
        name=name,
        state="JOB_STATE_SUCCEEDED",
        dest=FakeDest(
            inlined_embed_content_responses=[
                FakeInlined(response=FakeSingle(embedding=FakeEmbedding(v))) for v in vectors
            ]
        ),
    )


# ─── imkoniyat ───────────────────────────────────────────────────────────────


def test_embedding_model_declares_batch_capability() -> None:
    p, _ = make_provider()
    caps = p.capabilities(EMBED_MODEL)
    assert Capability.EMBED in caps
    assert Capability.BATCH_EMBED in caps


def test_chat_model_has_no_batch_embed() -> None:
    p, _ = make_provider()
    assert Capability.BATCH_EMBED not in p.capabilities("gemini-2.5-flash")


async def test_other_providers_raise_not_supported() -> None:
    from app.llm.deepseek import DeepSeekProvider

    p = DeepSeekProvider(api_key="k")
    with pytest.raises(LLMError):
        await p.embed_batch_submit(["a"], model="deepseek-v4-flash", dim=768)
    with pytest.raises(LLMError):
        await p.embed_batch_poll("batches/x")
    with pytest.raises(LLMError):
        await p.embed_batch_fetch("batches/x")


async def test_missing_sdk_method_is_explicit() -> None:
    """Eski SDK — jimgina ishlamay qolish emas, aniq xato."""

    class NoBatch:
        pass

    p, _ = make_provider()
    p._client = FakeClient(aio=FakeAio(NoBatch()))  # type: ignore[arg-type,assignment]
    with pytest.raises(LLMError, match=r"2\.23"):
        await p.embed_batch_submit(["a"], model=EMBED_MODEL, dim=768)


# ─── submit ──────────────────────────────────────────────────────────────────


async def test_submit_sends_all_texts_with_one_config() -> None:
    p, fake = make_provider()
    job = await p.embed_batch_submit(
        ["bir", "ikki", "uch"], model=EMBED_MODEL, dim=768, tag="tgai-messages"
    )
    assert job == "batches/fake-1"
    (call,) = fake.created
    assert call["contents"] == ["bir", "ikki", "uch"]
    assert call["task_type"] == "RETRIEVAL_DOCUMENT"
    assert call["dim"] == 768
    assert call["display_name"] == "tgai-messages"


async def test_submit_rejects_empty() -> None:
    p, _ = make_provider()
    with pytest.raises(LLMError):
        await p.embed_batch_submit([], model=EMBED_MODEL, dim=768)


# ─── poll ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected", "done"),
    [
        ("JOB_STATE_PENDING", BatchState.PENDING, False),
        ("JOB_STATE_QUEUED", BatchState.PENDING, False),
        ("JOB_STATE_RUNNING", BatchState.RUNNING, False),
        ("JOB_STATE_SUCCEEDED", BatchState.SUCCEEDED, True),
        ("JOB_STATE_FAILED", BatchState.FAILED, True),
        ("JOB_STATE_EXPIRED", BatchState.EXPIRED, True),
        ("JOB_STATE_CANCELLED", BatchState.CANCELLED, True),
        # Yarim muvaffaqiyat — biz uchun failed: qaysi element tushib
        # qolgani tartibdan bilinmaydi.
        ("JOB_STATE_PARTIALLY_SUCCEEDED", BatchState.FAILED, True),
        # Noma'lum holat → RUNNING (tugadi deb hisoblash xavfliroq)
        ("JOB_STATE_SOMETHING_NEW", BatchState.RUNNING, False),
    ],
)
async def test_poll_maps_states(raw: str, expected: BatchState, done: bool) -> None:
    fake = FakeBatches({"batches/j": FakeJob(name="batches/j", state=raw)})
    p, _ = make_provider(fake)
    status = await p.embed_batch_poll("batches/j")
    assert status.state is expected
    assert status.done is done
    assert status.raw_state == raw


async def test_poll_normalizes_model_name() -> None:
    fake = FakeBatches({"batches/j": FakeJob(name="batches/j", state="JOB_STATE_RUNNING")})
    p, _ = make_provider(fake)
    # `models/` prefiksi olib tashlanadi — `pricing.PRICES` kaliti shu.
    assert (await p.embed_batch_poll("batches/j")).model == EMBED_MODEL


# ─── fetch ───────────────────────────────────────────────────────────────────


async def test_fetch_returns_vectors_in_order() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
    fake = FakeBatches({"batches/j": succeeded("batches/j", vectors)})
    p, _ = make_provider(fake)
    res = await p.embed_batch_fetch("batches/j")
    assert len(res.vectors) == 3
    # `gemini-embedding-001` — mijoz tomonida normallashtiriladi (phase4/B),
    # shuning uchun tartibni yo'nalish bo'yicha tekshiramiz.
    assert res.vectors[0][0] == pytest.approx(1.0)
    assert res.vectors[1][1] == pytest.approx(1.0)
    assert res.model == EMBED_MODEL
    assert res.usage.tokens_in == 21  # uchta javob, har biri 7 token


async def test_fetch_refuses_unfinished_job() -> None:
    fake = FakeBatches({"batches/j": FakeJob(name="batches/j", state="JOB_STATE_RUNNING")})
    p, _ = make_provider(fake)
    with pytest.raises(LLMError):
        await p.embed_batch_fetch("batches/j")


async def test_fetch_raises_on_item_error() -> None:
    job = succeeded("batches/j", [[1.0, 0.0]])
    assert job.dest is not None
    job.dest.inlined_embed_content_responses = [FakeInlined(error=Exception("quota"))]
    p, _ = make_provider(FakeBatches({"batches/j": job}))
    with pytest.raises(LLMError):
        await p.embed_batch_fetch("batches/j")


async def test_fetch_raises_when_output_is_a_file() -> None:
    """Biz doim inline yuboramiz — faylli chiqish = shakl o'zgardi."""
    job = FakeJob(
        name="batches/j",
        state="JOB_STATE_SUCCEEDED",
        dest=FakeDest(inlined_embed_content_responses=None, file_name="files/out"),
    )
    p, _ = make_provider(FakeBatches({"batches/j": job}))
    with pytest.raises(LLMError, match="files/out"):
        await p.embed_batch_fetch("batches/j")


def test_batch_status_reads_error_message() -> None:
    @dataclass
    class Err:
        message: str

    st = _batch_status(
        FakeJob(name="batches/j", state="JOB_STATE_FAILED", error=Err("nima'dir")), "batches/j"
    )
    assert st.state is BatchState.FAILED
    assert st.error == "nima'dir"


# ─── narx ────────────────────────────────────────────────────────────────────


def test_batch_price_is_half_of_sync() -> None:
    live = estimate_cost("gemini", EMBED_MODEL, 1_000_000, 0)
    batch = estimate_cost("gemini", EMBED_MODEL, 1_000_000, 0, batch=True)
    assert live == pytest.approx(0.15)
    assert batch == pytest.approx(0.075)
    assert batch is not None and live is not None and batch < live


def test_batch_price_falls_back_to_live_for_unknown_model() -> None:
    """Jadvalda batch narxi yo'q model — jonli narxda hisoblanadi (ortiq baho)."""
    assert estimate_cost("gemini", "gemini-2.5-flash", 1_000_000, 0, batch=True) == pytest.approx(
        0.30
    )


async def test_fetch_normalizes_vectors_like_the_sync_path() -> None:
    """F1: batch yo'li ham `normalize_vectors` dan o'tadi (aralash masshtab yo'q).

    Sinxron va batch qatorlari bitta `message_embeddings` jadvalida
    **bir xil `(message_id, model)` kaliti** ostida raqobatlashadi (PK
    migratsiya 0016 dan beri ikki ustunli, ya'ni har model o'z qatoriga ega) —
    biri normallashtirilmagan bo'lsa cosine ostonalari (alert `threshold`,
    rerank) jimgina siljib ketardi.
    """
    raw = [[3.0, 4.0]]  # uzunligi 5 — normallashtirilgach 1 bo'lishi kerak
    fake = FakeBatches({"batches/j": succeeded("batches/j", raw)})
    p, _ = make_provider(fake)

    res = await p.embed_batch_fetch("batches/j")

    got = res.vectors[0]
    assert sum(x * x for x in got) == pytest.approx(1.0)
    assert got[0] == pytest.approx(0.6) and got[1] == pytest.approx(0.8)


# ─── F3/H2: javobda model nomi bo'lmasa `model=""` yozilmasin ────────────────


def test_batch_status_ignores_a_non_string_model_field() -> None:
    """H2: SDK `model` o'rniga obyekt bersa — bo'sh satr, `str(obj)` axlati emas.

    `str(...)` bilan majburlash `<google.genai.types.Model object …>` ni
    `message_embeddings.model` ga yozib qo'yardi; bunday qator qidiruv
    filtridan **hech qachon** o'tmaydi.
    """
    job = FakeJob(name="batches/j", state="JOB_STATE_SUCCEEDED")
    job.model = object()  # type: ignore[assignment]
    assert _batch_status(job, "batches/j").model == ""


def test_batch_status_reports_no_model_when_the_field_is_missing() -> None:
    job = FakeJob(name="batches/j", state="JOB_STATE_SUCCEEDED", model="")
    assert _batch_status(job, "batches/j").model == ""


async def test_fetch_falls_back_to_the_requested_model() -> None:
    """H2: javobda model yo'q → so'ralgan nom (va normallashtirish **baribir**).

    Ilgari `model=""` qaytardi: `normalize_vectors("")` normallashtirishni
    o'tkazib yuborardi va `poll_batches` qatorni `model=''` bilan yozardi.
    """
    job = succeeded("batches/j", [[3.0, 4.0]])
    job.model = ""
    p, _ = make_provider(FakeBatches({"batches/j": job}))

    res = await p.embed_batch_fetch("batches/j", model=f"models/{EMBED_MODEL}")

    assert res.model == EMBED_MODEL
    assert sum(x * x for x in res.vectors[0]) == pytest.approx(1.0)


# ─── F3: `poll_batches` — model, izolyatsiya, konflikt kaliti ────────────────


class _Rows:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> list[Any]:
        return self._rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """`AsyncSession` ning `poll_batches` ishlatadigan qismi.

    `fail_on_write` — shu `job_name` uchun vektor yozishda istisno (L11:
    bitta jobning qulashi qolganlarini olib ketmasligi kerak).

    `text_rows` — `_original_texts` SELECT'i qaytaradigan qatorlar (faqat
    `_poll_one` narx taxminini tiklashda, ya'ni `usage.tokens_in == 0`
    bo'lganda chaqiriladi — boshqa testlarda bu yo'l umuman bosilmaydi).
    """

    def __init__(
        self, rows: list[Any], *, fail_on_write: str = "", text_rows: list[Any] | None = None
    ) -> None:
        self.rows = rows
        self.fail_on_write = fail_on_write
        self.text_rows = text_rows or []
        self.writes: list[Any] = []
        self.nested = 0
        self.rolled_back: list[str] = []
        self._selected = False
        self.current_job = ""

    async def execute(self, stmt: Any) -> Any:
        from sqlalchemy import Select

        if not self._selected:
            self._selected = True
            return _Rows(self.rows)
        if isinstance(stmt, Select):
            return _Rows(self.text_rows)
        if self.current_job and self.current_job == self.fail_on_write:
            raise RuntimeError("savepoint ichidagi DB xatosi")
        self.writes.append(stmt)
        return None

    async def flush(self) -> None:
        return None

    def begin_nested(self) -> Any:
        session = self

        class _Nested:
            async def __aenter__(self) -> Any:
                session.nested += 1
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False  # istisnoni yutmaydi — chaqiruvchi tutadi

        return _Nested()


def _batch_row(job: str, *, model: str = EMBED_MODEL, item_ids: tuple[int, ...] = (1, 2)) -> Any:
    from app.db.models import EmbedBatch, EmbedBatchStatus

    return EmbedBatch(
        provider="gemini",
        job_name=job,
        kind="messages",
        account_id=None,
        item_ids=list(item_ids),
        model=model,
        task_type="RETRIEVAL_DOCUMENT",
        status=EmbedBatchStatus.PENDING,
        attempts=1,
    )


class _StubProvider:
    """`BaseProvider` o'rnini bosadi; har job uchun oldindan berilgan natija."""

    name = "gemini"

    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results
        self.fetched: list[tuple[str, str]] = []

    async def embed_batch_poll(self, job: str) -> Any:
        from app.llm.base import BatchState, BatchStatus

        return BatchStatus(job=job, state=BatchState.SUCCEEDED, raw_state="JOB_STATE_SUCCEEDED")

    async def embed_batch_fetch(self, job: str, *, model: str = "") -> Any:
        self.fetched.append((job, model))
        return self.results[job]


def _result(model: str, n: int = 2, *, tokens_in: int = 10) -> Any:
    from app.llm.base import EmbedResult, Usage

    return EmbedResult(
        vectors=[[0.1] * 4 for _ in range(n)],
        model=model,
        provider="gemini",
        usage=Usage(tokens_in=tokens_in),
    )


async def _poll(session: Any, provider: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.services import embed_batches as EB

    monkeypatch.setattr(EB, "_provider_for_batch", lambda: (provider, EMBED_MODEL))
    # Har job qaysi savepoint ichida yozayotganini soxta sessiya bilishi uchun.
    original = EB._poll_one

    async def _wrapped(sess: Any, prov: Any, row: Any, rep: Any, **kw: Any) -> None:
        sess.current_job = row.job_name
        await original(sess, prov, row, rep, **kw)

    monkeypatch.setattr(EB, "_poll_one", _wrapped)
    return await EB.poll_batches(session)


async def test_poll_uses_the_requested_model_when_the_api_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H2: `model=''` hech qachon yozilmaydi — qator qidiruvdan yo'qolardi."""
    from app.db.models import EmbedBatchStatus

    row = _batch_row("batches/a")
    provider = _StubProvider({"batches/a": _result("")})

    rep = await _poll(_FakeSession([row]), provider, monkeypatch)

    assert provider.fetched == [("batches/a", EMBED_MODEL)]  # so'ralgan model uzatildi
    assert row.model == EMBED_MODEL
    assert row.status == EmbedBatchStatus.SUCCEEDED
    assert rep.written == 2


async def test_poll_estimates_tokens_when_usage_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-09-16: Gemini batch javobi ko'pincha `token_count` bermaydi.

    `_poll_one` `usage.tokens_in == 0` ko'rsa original matnlarni DB'dan
    o'qib (`_original_texts`) `estimate_tokens_heuristic` bilan narxni
    tiklaydi — aks holda kunlik xarajat limiti bu sarfni umuman ko'rmasdi.
    """
    from app.db.models import EmbedBatchStatus

    row = _batch_row("batches/a")  # item_ids=(1, 2), kind="messages"
    provider = _StubProvider({"batches/a": _result(EMBED_MODEL, tokens_in=0)})
    session = _FakeSession([row], text_rows=[(1, "a" * 40), (2, "b" * 8)])

    rep = await _poll(session, provider, monkeypatch)

    assert row.status == EmbedBatchStatus.SUCCEEDED
    assert rep.written == 2
    # ceil(40/4) + ceil(8/4) = 10 + 2 = 12 taxminiy token
    expected = estimate_cost("gemini", EMBED_MODEL, 12, 0, batch=True)
    assert row.cost_usd == expected
    assert row.cost_usd is not None and row.cost_usd > 0


async def test_poll_fails_the_job_when_the_api_returns_another_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H2: boshqa model = boshqa vektor fazasi → yozilmaydi, job xato bo'ladi."""
    from app.db.models import EmbedBatchStatus

    row = _batch_row("batches/a")
    session = _FakeSession([row])
    provider = _StubProvider({"batches/a": _result("gemini-embedding-2")})

    rep = await _poll(session, provider, monkeypatch)

    assert rep.written == 0 and rep.failed == 1
    assert session.writes == []
    assert row.status == EmbedBatchStatus.FAILED
    assert row.error is not None and "mismatch" in row.error
    assert row.model == EMBED_MODEL  # eski nom o'zgarmaydi


async def test_poll_isolates_one_failing_job_from_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L11: har job o'z savepoint'ida — ikkinchisi qulasa 1 va 3 yozilgan qoladi."""
    from app.db.models import EmbedBatchStatus

    rows = [_batch_row(f"batches/{n}") for n in ("a", "b", "c")]
    session = _FakeSession(rows, fail_on_write="batches/b")
    provider = _StubProvider({r.job_name: _result(EMBED_MODEL) for r in rows})

    rep = await _poll(session, provider, monkeypatch)

    assert session.nested == 3  # har job uchun bitta savepoint
    assert rep.checked == 3
    assert rep.succeeded == 2 and rep.written == 4
    assert [r.status for r in rows] == [
        EmbedBatchStatus.SUCCEEDED,
        EmbedBatchStatus.PENDING,  # qulagani — o'zgarishlari savepoint bilan qaytadi
        EmbedBatchStatus.SUCCEEDED,
    ]


async def test_write_vectors_conflicts_on_id_and_model() -> None:
    """H1: konflikt kaliti `(message_id, model)` — PK bilan bir xil.

    Faqat `message_id` bo'lganda boshqa modeldagi mavjud qator bu INSERT'ni
    jimgina yeb qo'yardi: vektor yozilmasdi, job esa `SUCCEEDED` bo'lardi.
    """
    from sqlalchemy.dialects import postgresql

    from app.services import embed_batches as EB

    session = _FakeSession([])
    session._selected = True  # birinchi `execute` ham yozuv bo'lsin
    await EB._write_vectors(session, EB.KIND_MESSAGES, [1], [[0.1] * 4], EMBED_MODEL)

    (stmt,) = session.writes
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (message_id, model) DO NOTHING" in sql


# ─── 2026-09-18: bir cron yugurishida bir nechta job (backfill) ──────────────


class _SubmitScope:
    """`session_scope()` o'rnini bosadi — `_submit_embed_batch` uchun."""

    def __init__(self) -> None:
        self.session = object()

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _patch_submit(
    monkeypatch: pytest.MonkeyPatch, reports: list[Any], *, jobs_per_run: int
) -> list[Any]:
    """`submit_messages` ni ketma-ket `reports` bilan almashtiradi."""
    from app.services import embed_batches as EB
    from app.worker import tasks as WT

    calls: list[Any] = []
    queue = list(reports)

    async def fake_submit(session: Any, **kw: Any) -> Any:
        calls.append(kw)
        return queue.pop(0) if queue else EB.SubmitReport()

    monkeypatch.setattr(EB, "batch_embedding_available", lambda: True)
    monkeypatch.setattr(EB, "submit_messages", fake_submit)
    monkeypatch.setattr("app.db.base.session_scope", lambda: _SubmitScope())
    monkeypatch.setattr(
        "app.config.get_settings", lambda: SimpleNamespace(embed_batch_jobs_per_run=jobs_per_run)
    )

    async def _noop(name: str, **kw: Any) -> None: ...

    monkeypatch.setattr(WT, "heartbeat", _noop)
    return calls


async def test_submit_sends_many_jobs_in_one_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """20 x 2000 = 40k element/yugurish — 2.6M backfill bir kunga sig'adi."""
    from app.services.embed_batches import SubmitReport
    from app.worker import tasks as WT

    reports = [SubmitReport(submitted=2000, job=f"batches/{i}") for i in range(3)]
    calls = _patch_submit(monkeypatch, reports, jobs_per_run=3)

    out = await WT._submit_embed_batch(WT.KIND_MESSAGES, None)
    assert out == {"mode": "batch", "submitted": 6000, "jobs": 3, "job": "batches/0"}
    assert len(calls) == 3  # bitta sessiya ichida — `_inflight` takrorlanishni to'sadi


async def test_submit_stops_when_nothing_is_left(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bo'sh tanlov — qolgan joblar uchun so'rov qilinmaydi."""
    from app.services.embed_batches import SubmitReport
    from app.worker import tasks as WT

    reports = [SubmitReport(submitted=7, job="batches/0"), SubmitReport()]
    calls = _patch_submit(monkeypatch, reports, jobs_per_run=10)

    out = await WT._submit_embed_batch(WT.KIND_MESSAGES, None)
    assert out is not None and out["submitted"] == 7 and out["jobs"] == 1
    assert len(calls) == 2


async def test_first_submit_error_falls_back_to_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hech narsa yuborilmagan bo'lsa — sinxron yo'l (`None`)."""
    from app.services.embed_batches import SubmitReport
    from app.worker import tasks as WT

    _patch_submit(monkeypatch, [SubmitReport(error="429")], jobs_per_run=5)
    assert await WT._submit_embed_batch(WT.KIND_MESSAGES, None) is None


async def test_later_submit_error_keeps_what_was_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bir qismi yuborilgan — sinxron yo'lga tushsak o'sha elementlar ikki marta to'lanardi."""
    from app.services.embed_batches import SubmitReport
    from app.worker import tasks as WT

    reports = [SubmitReport(submitted=2000, job="batches/0"), SubmitReport(error="429")]
    _patch_submit(monkeypatch, reports, jobs_per_run=5)

    out = await WT._submit_embed_batch(WT.KIND_MESSAGES, None)
    assert out is not None and out["submitted"] == 2000 and out["jobs"] == 1


def test_poll_limit_default_matches_submit_throughput() -> None:
    """Poll submit'dan sekin bo'lsa tugagan joblar navbatda to'planib qolardi."""
    from app.config import Settings

    assert Settings.model_fields["embed_batch_poll_limit"].default == 20
    assert Settings.model_fields["embed_batch_jobs_per_run"].default == 1
