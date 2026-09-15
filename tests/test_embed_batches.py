"""phase4/C: batch embedding — adapter, submit/poll/fetch va qat'iy moslik.

Haqiqiy API chaqiruvi yo'q: `genai.Client` soxta obyekt bilan almashtiriladi
(`batches.create_embeddings` / `batches.get`). Testlar SDK **shakliga** qarab
yozilgan (`google-genai` 2.23: `BatchJob.state`, `dest.inlined_embed_content_responses`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    Sinxron va batch qatorlari bitta `message_embeddings` jadvalida yonma-yon
    turadi — biri normallashtirilmagan bo'lsa cosine ostonalari (alert
    `threshold`, rerank) jimgina siljib ketardi.
    """
    raw = [[3.0, 4.0]]  # uzunligi 5 — normallashtirilgach 1 bo'lishi kerak
    fake = FakeBatches({"batches/j": succeeded("batches/j", raw)})
    p, _ = make_provider(fake)

    res = await p.embed_batch_fetch("batches/j")

    got = res.vectors[0]
    assert sum(x * x for x in got) == pytest.approx(1.0)
    assert got[0] == pytest.approx(0.6) and got[1] == pytest.approx(0.8)
