"""Qidiruv/kontekst optimizatsiyasi + sifat baholash — DB/tarmoqsiz qismlar.

Qo'riqlaydi:
  * matn siqish (URL, whitespace, kesish) va token bahosi
  * kalit so'z ajratish (stop-so'zlar), RRF birlashtirish, byudjetga sig'dirish
  * strategiya tanlash: vaqt oynasi / xulosa / qidiruv aniqlash
  * map-reduce: bo'laklash, arzon Task.ROUTE, digest'lar konvertda
  * judge JSON parse (fence, clamp), pricing, tarix qisqartirish
  * Claude adapterda uzun system → cache_control
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import ChatResult, Msg, Task, Usage
from app.llm import claude as C
from app.llm.pricing import estimate_cost
from app.mtproto.pool import MessageInfo
from app.services import chat_service as cs
from app.services import evaluation as ev
from app.services import search as S
from app.services.textnorm import uz_fold

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def _mi(i: int, text: str, when: datetime | None = None) -> MessageInfo:
    return MessageInfo(i, when or NOW, "A", text, None, None)


# ─── siqish ──────────────────────────────────────────────────────────────────


def test_compact_text_shortens_urls_and_whitespace() -> None:
    raw = "Qarang   https://example.com/very/long/path?x=1&y=2 \n\n\n\n keyin   davomi"
    out = S.compact_text(raw)
    assert "<example.com>" in out and "very/long" not in out
    assert "   " not in out and "\n\n\n" not in out


def test_compact_text_truncates() -> None:
    out = S.compact_text("x" * 5000, max_chars=100)
    assert len(out) == 100 and out.endswith("…")


def test_est_tokens_and_fit_budget_keeps_newest() -> None:
    infos = [_mi(i, "a" * 350) for i in range(10)]  # har biri ~100 tok + 12
    kept, trunc = S.fit_budget(infos, 350)
    assert trunc and [m.msg_id for m in kept] == [7, 8, 9]  # eng yangilar (ro'yxat oxiri)
    kept2, trunc2 = S.fit_budget(infos[:2], 10_000)
    assert not trunc2 and len(kept2) == 2


# ─── kalit so'zlar / RRF ─────────────────────────────────────────────────────


def test_keywords_drop_stopwords_and_dupes() -> None:
    kw = S.keywords("Bu hafta kanalda crypto va Bitcoin haqida nima yozildi? crypto")
    assert "crypto" in kw and "bitcoin" in kw
    assert "haqida" not in kw and "nima" not in kw and "bu" not in kw
    assert kw.count("crypto") == 1
    assert S.keywords("и в на") == []


def test_rrf_merges_and_marks_both() -> None:
    fts = [S.Hit(1, 101, 5.0, "fts"), S.Hit(2, 102, 4.0, "fts")]
    vec = [S.Hit(2, 102, 0.9, "vector"), S.Hit(3, 103, 0.8, "vector")]
    fused = S.rrf(fts, vec)
    assert fused[0].message_id == 2 and fused[0].source == "both"
    assert {h.message_id for h in fused} == {1, 2, 3}


# ─── 1-bosqich: fold, og'irlikli RRF, ustuvor byudjet ────────────────────────


def test_fold_terms_matches_index_fold() -> None:
    """So'rov varianti indeks bilan bir xil fold'da bo'lmasa — hech narsa topilmaydi."""
    assert S.fold_terms(["O'zbek", "G`ALABA"]) == [uz_fold("o'zbek"), uz_fold("g`alaba")]
    assert S.fold_terms(["тўлов"]) == [uz_fold("тўлов")]  # kirill → lotin
    assert S.fold_terms(["a", "  "]) == []  # 2 belgidan qisqa tushib qoladi
    assert len(S.fold_terms([f"term{i}" for i in range(50)])) == 12  # qopqoq


def test_keywords_use_shared_stopword_list() -> None:
    kw = S.keywords("Bu hafta kanalda crypto haqida nima yozildi?")
    assert "crypto" in kw and "haqida" not in kw and "nima" not in kw


def test_rrf_weights_change_the_winner() -> None:
    a = [S.Hit(1, 101, 1.0, "tsv_uz")]
    b = [S.Hit(2, 102, 1.0, "vector")]
    equal = S.rrf(a, b)
    assert equal[0].message_id == 1  # teng og'irlikda kirish tartibi barqaror
    weighted = S.rrf(a, b, weights=(0.5, 2.0))
    assert weighted[0].message_id == 2
    assert weighted[0].score > weighted[1].score


def test_rrf_accepts_list_of_lists() -> None:
    a = [S.Hit(1, 101, 1.0, "tsv_uz")]
    b = [S.Hit(2, 102, 1.0, "vector")]
    assert [h.message_id for h in S.rrf([a, b])] == [h.message_id for h in S.rrf(a, b)]


def test_rrf_default_weights_are_the_documented_four() -> None:
    assert S.DEFAULT_RRF_WEIGHTS == (1.0, 1.0, 0.6, 1.2)


def test_rrf_ignores_empty_channels() -> None:
    fts = [S.Hit(1, 101, 5.0, "tsv_uz"), S.Hit(2, 102, 4.0, "tsv_uz")]
    fused = S.rrf(fts, [], [], [], weights=S.DEFAULT_RRF_WEIGHTS)
    assert [h.message_id for h in fused] == [1, 2]


def test_rrf_carries_hydrated_fields() -> None:
    h = S.Hit(1, 101, 1.0, "tsv_uz", chat_id=9, chat_title="T", sender="u", text="matn")
    out = S.rrf([h])[0]
    assert out.text == "matn" and out.chat_title == "T" and out.chat_id == 9


def test_fit_by_priority_keeps_hits_over_recency_tail() -> None:
    """Regressiya: eski `fit_budget` qidiruv rejimida dumni hit'dan ustun qo'yardi."""
    hit = _mi(5, "javob shu yerda " * 20)  # ~100 token
    tail = [_mi(90 + i, "kundalik gap " * 20) for i in range(5)]
    ordered = [(5, hit), *[(m.msg_id, m) for m in tail]]
    kept, trunc = S.fit_by_priority(ordered, 260)
    assert trunc and kept[0].msg_id == 5  # hit saqlandi
    assert 1 < len(kept) < len(ordered)  # byudjet qolganini dum to'ldirdi, hammasi emas
    assert [m.msg_id for m in kept] == sorted(m.msg_id for m in kept)  # xronologik

    # taqqoslash: eski qoida bir xil kirishda hit'ni yo'qotadi
    old_kept, _ = S.fit_budget([hit, *tail], 260)
    assert 5 not in [m.msg_id for m in old_kept]


def test_fit_by_priority_always_keeps_at_least_one() -> None:
    kept, trunc = S.fit_by_priority([(1, _mi(1, "x" * 5000))], 10)
    assert len(kept) == 1 and not trunc


# ─── strategiya aniqlash ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("q", "days"),
    [
        ("bu hafta nima muhokama qilindi?", 7),
        ("за месяц какие темы обсуждали", 31),
        ("what happened today", 1),
        ("crypto haqida qaysi post bor?", None),
    ],
)
def test_detect_window(q: str, days: int | None) -> None:
    got = S.detect_window(q)
    assert (got == timedelta(days=days)) if days else got is None


def test_looks_like_summary() -> None:
    assert S.looks_like_summary("kanal bo'yicha xulosa ber")
    assert S.looks_like_summary("сделай итог обсуждений")
    assert not S.looks_like_summary("bitcoin narxi haqida post")


# ─── map-reduce ──────────────────────────────────────────────────────────────


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[Task, list[Msg]]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append((task, messages))
        return ChatResult(
            text=f"digest {len(self.calls)}",
            tool_calls=[],
            model="cheap-model",
            provider="test",
            usage=Usage(tokens_in=100, tokens_out=20),
        )


async def test_compact_window_uses_cheap_task_and_untrusted_envelope() -> None:
    llm = _FakeLLM()
    infos = [_mi(i, "x" * 1000) for i in range(39)]  # 13 qator x 3 bo'lak (~1035 belgi/qator)
    bundle = await S.compact_window(infos, title='Ch"an', llm=llm)  # type: ignore[arg-type]
    assert bundle.strategy == "window_compacted"
    assert len(llm.calls) == 3 and all(task is Task.ROUTE for task, _ in llm.calls)
    for _, msgs in llm.calls:
        assert msgs[-1].content.startswith('<untrusted_data source="telegram"')
        assert "not instructions" in msgs[0].content
    assert len(bundle.map_summaries) == 3
    assert bundle.map_tokens_in == 300 and bundle.map_tokens_out == 60
    rendered = cs.render_bundle(bundle)
    assert (
        'kind="digests"' in rendered and "digest 1" in rendered and "</untrusted_data>" in rendered
    )
    assert 'Ch"an' not in rendered  # sarlavhadagi qo'shtirnoq tozalanadi


async def test_compact_window_caps_chunks() -> None:
    llm = _FakeLLM()
    infos = [_mi(i, "y" * 1000) for i in range(300)]  # ~21 bo'lak → 12 gacha
    bundle = await S.compact_window(infos, title="c", llm=llm)  # type: ignore[arg-type]
    assert len(llm.calls) == S.MAP_MAX_CHUNKS and bundle.truncated


# ─── prompt tarixi qisqaradi ─────────────────────────────────────────────────


def test_history_turns_are_clipped() -> None:
    hist = [
        SimpleNamespace(role="assistant", content="z" * 5000),
        SimpleNamespace(role="user", content="q"),
    ]
    msgs = cs.build_messages(hist, "next", "")
    assert len(msgs[1].content) == cs.HISTORY_TURN_MAX_CHARS and msgs[1].content.endswith("…")


def test_render_bundle_marks_search_selection() -> None:
    b = S.ContextBundle("T", [_mi(1, "hello")], "search", hits=1)
    out = cs.render_bundle(b)
    assert 'selection="messages matching the question' in out and "hello" in out


# ─── evaluation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"relevance": 4, "usefulness": 5, "grounded": true, "note": "ok"}',
        '```json\n{"relevance": 4, "usefulness": 5, "grounded": true, "note": "ok"}\n```',
        'Sure: {"relevance": "4", "usefulness": 9, "grounded": true, "note": "ok"} done',
    ],
)
def test_parse_judgement(raw: str) -> None:
    j = ev.parse_judgement(raw)
    assert j is not None
    # `grounded` endi 1-5 shkalada (eski `true` → 5): degradatsiyalangan
    # yugurishda "qisman asosli" ni bool bilan ifodalab bo'lmasdi.
    assert j.relevance == 4 and j.usefulness == 5
    assert j.grounded == 5 and j.note == "ok"


def test_parse_judgement_garbage() -> None:
    assert ev.parse_judgement("no json here") is None
    assert ev.parse_judgement("[1,2]") is None


async def test_judge_uses_cheap_task_and_json_mode() -> None:
    class _L:
        async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
            assert task is Task.ROUTE and kw.get("json_mode") is True
            assert "<question>" in messages[-1].content and "<answer>" in messages[-1].content
            return ChatResult(
                text='{"relevance":2,"usefulness":1,"grounded":false,"note":"invented"}',
                tool_calls=[],
                model="m",
                provider="p",
                usage=Usage(50, 10),
            )

    j = await ev.judge("q", "a", had_context=False, llm=_L())  # type: ignore[arg-type]
    assert j is not None and j.relevance == 2 and j.tokens_in == 50
    assert j.grounded == 1  # `false` → shkalaning eng pasti


# ─── pricing ─────────────────────────────────────────────────────────────────


def test_estimate_cost() -> None:
    assert estimate_cost("claude", "claude-opus-5", 1_000_000, 0) == 5.0
    assert estimate_cost("claude", "claude-haiku-4-5-20251001", 0, 1_000_000) == 5.0
    assert estimate_cost("gemini", "models/gemini-2.5-flash", 1_000_000, 1_000_000) == 2.8
    assert estimate_cost("claude_code", "claude-opus-5", 10**6, 10**6) == 0.0
    assert estimate_cost("x", "unknown-model", 1, 1) is None


def test_estimate_cost_embedding_models() -> None:
    """phase4/B — `gemini-embedding-2` tayyorligi (narx taxminiy, batch arzonroq)."""
    from app.llm.pricing import BATCH_PRICES, PRICES

    assert estimate_cost("gemini", "gemini-embedding-2", 1_000_000, 0) == 0.20
    # API alias qaytarsa ham `estimate_cost` None qaytarmasligi kerak
    assert estimate_cost("gemini", "models/gemini-embedding-2-preview", 1_000_000, 0) == 0.20
    # batch narxi — yarmi
    assert estimate_cost("gemini", "gemini-embedding-2", 1_000_000, 0, batch=True) == 0.10
    # batch jadvalida yo'q model jonli narxda qoladi
    assert estimate_cost("gemini", "gemini-2.5-flash", 1_000_000, 0, batch=True) == 0.30
    # har batch narxi jonli narx bilan birga e'lon qilingan bo'lishi shart
    assert set(BATCH_PRICES) <= set(PRICES)


# ─── Claude prompt caching ───────────────────────────────────────────────────


# ─── embedding task_type va qidiruv kanallari (fake session) ────────────────


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return list(self._rows)

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalars(self) -> Any:
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeSession:
    """Navbatdagi natijalarni qaytaradi va bajarilgan SQL matnini saqlaydi."""

    def __init__(self, *results: list[Any]) -> None:
        self.queue = list(results)
        self.sql: list[str] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        sql = str(stmt)
        # HNSW sozlamasi (`search.tune_hnsw`) — natija qaytarmaydigan bitta
        # `set_config`. U navbatdan hech narsa yemaydi va hisobga olinmaydi,
        # aks holda har bir vektor kanalli test bitta bo'sh natija qo'shishga
        # majbur bo'lardi (test niyati — kanallar soni, sozlama emas).
        if "set_config('hnsw" in sql or "set_config('work_mem" in sql:
            return _FakeResult([])
        self.sql.append(sql)
        return _FakeResult(self.queue.pop(0) if self.queue else [])


class _EmbedLLM:
    def __init__(self) -> None:
        self.task_types: list[str] = []

    async def embed(self, texts: list[str], **kw: Any) -> Any:
        self.task_types.append(kw.get("task_type", "default"))
        return SimpleNamespace(vectors=[[0.1, 0.2]], model="m", provider="p", usage=Usage())


async def test_vector_search_uses_retrieval_query_task_type() -> None:
    llm = _EmbedLLM()
    session = _FakeSession([(1,)], [(1, 101, 0.2)])
    hits = await S.vector_search(session, 1, "to'lov", llm=llm)  # type: ignore[arg-type]
    assert llm.task_types == ["RETRIEVAL_QUERY"]
    assert [h.message_id for h in hits] == [1] and hits[0].source == "vector"


async def test_vector_search_without_embeddings_is_silent() -> None:
    llm = _EmbedLLM()
    assert await S.vector_search(_FakeSession([]), 1, "q", llm=llm) == []  # type: ignore[arg-type]
    assert llm.task_types == []


def test_llm_embed_defaults_to_document_task_type() -> None:
    import inspect

    from app.llm import EMBED_DOCUMENT, LLM, BaseProvider

    for fn in (LLM.embed, BaseProvider.embed):
        assert inspect.signature(fn).parameters["task_type"].default == EMBED_DOCUMENT


async def test_fts_channels_are_skipped_without_terms() -> None:
    session = _FakeSession()
    assert await S.fts_uz_search(session, 1, []) == []  # type: ignore[arg-type]
    assert await S.fts_ru_search(session, 1, []) == []  # type: ignore[arg-type]
    assert await S.trigram_search(session, 1, []) == []  # type: ignore[arg-type]
    assert session.sql == []


async def test_lexical_sql_uses_generated_columns_and_excludes_deleted() -> None:
    session = _FakeSession([(1, 101, 0.5)])
    await S.fts_uz_search(session, 1, ["to'lov"])  # type: ignore[arg-type]
    sql = session.sql[0]
    assert "tsv_uz" in sql and "to_tsquery" in sql
    assert "to_tsvector" not in sql  # inline tsvector qurish yo'q
    assert "deleted_at IS NULL" in sql

    ru = _FakeSession([(2, 102, 0.5)])
    await S.fts_ru_search(ru, 1, ["оплата"])  # type: ignore[arg-type]
    assert "tsv_ru" in ru.sql[0] and "deleted_at IS NULL" in ru.sql[0]

    tri = _FakeSession([(3, 103, 0.7)])
    await S.trigram_search(tri, 1, ["tolov"])  # type: ignore[arg-type]
    assert "strict_word_similarity" in tri.sql[0] and "text_fold" in tri.sql[0]
    assert "similarity(messages.text," not in tri.sql[0]  # eski o'lik yo'l


async def test_fts_uz_falls_back_from_and_to_or() -> None:
    session = _FakeSession([], [(1, 101, 0.3)])
    hits = await S.fts_uz_search(session, 1, ["tolov", "muddat"])  # type: ignore[arg-type]
    assert len(session.sql) == 2  # avval AND, natija bo'sh → OR
    assert len(hits) == 1
    assert S._tsquery(["tolov", "muddat"], prefix=True, op=" & ") == "tolov:* & muddat:*"
    assert S._tsquery(["tolov", "muddat"], prefix=True, op=" | ") == "tolov:* | muddat:*"


async def test_trigram_falls_back_to_word_similarity() -> None:
    session = _FakeSession([], [(1, 101, 0.8)])
    hits = await S.trigram_search(session, 1, ["tolov"])  # type: ignore[arg-type]
    assert len(session.sql) == 2 and "word_similarity" in session.sql[1]
    assert len(hits) == 1


class _TuneSession(_FakeSession):
    """`_FakeSession` + `set_config` statementlarini **tartib bilan** yozadi."""

    def __init__(self, *results: list[Any]) -> None:
        super().__init__(*results)
        self.order: list[str] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        sql = str(stmt)
        self.order.append("work_mem" if "set_config('work_mem" in sql else sql[:40])
        return await super().execute(stmt)


async def test_trigram_channels_raise_work_mem_before_the_trigram_sql() -> None:
    """Sukut 4MB da oynali bitmap lossy bo'lib, recheck 101 s ga cho'zilgan (0925)."""
    assert S.TRGM_WORK_MEM == "32MB"
    for run, row in (
        (lambda s: S.trigram_search(s, 1, ["tolov"]), (1, 101, 0.8)),
        (lambda s: S._trigram_multi(s, ["tolov"], conds=[], limit=10), (1, 101, 5, 0.8)),
    ):
        session = _TuneSession([row])
        await run(session)
        i = session.order.index("work_mem")
        assert "strict_word_similarity" in session.sql[0]
        assert i < len(session.order) - 1  # tune — trigram so'rovidan **oldin**


async def test_trigram_multi_runs_under_a_savepoint_timeout_and_degrades_to_empty() -> None:
    """Planner `BitmapAnd` qilmasa trigram 179 s gacha cho'zilgan (0925) — shift bor.

    Shift SAVEPOINT ichida: bekor qilingan so'rov tashqi tranzaksiyani
    buzmaydi, kanal esa bo'sh ro'yxat qaytaradi (butun subquery yiqilmaydi).
    """
    from contextlib import asynccontextmanager

    class _Nested(_FakeSession):
        def __init__(self, fail: bool) -> None:
            super().__init__([(1, 101, 5, 0.9)])
            self.fail = fail
            self.savepoints = 0
            self.settings: list[str] = []

        @asynccontextmanager
        async def begin_nested(self) -> Any:
            self.savepoints += 1
            yield self

        async def execute(self, stmt: Any) -> _FakeResult:
            sql = str(stmt)
            if "set_config('statement_timeout'" in sql or "set_config('work_mem" in sql:
                self.settings.append(sql)
                return _FakeResult([])
            if self.fail and "trgm_cand" in sql:
                raise RuntimeError("canceling statement due to statement timeout")
            return await super().execute(stmt)

    ok = _Nested(fail=False)
    hits = await S._trigram_multi(ok, ["tolov"], conds=[], limit=10)  # type: ignore[arg-type]
    assert [h.message_id for h in hits] == [1] and ok.savepoints == 1
    assert any(str(S.TRGM_MULTI_TIMEOUT_MS) in x for x in ok.settings)
    bad = _Nested(fail=True)
    assert await S._trigram_multi(bad, ["tolov"], conds=[], limit=10) == []  # type: ignore[arg-type]


async def test_hybrid_multi_embeds_the_query_concurrently_with_lexical_channels(
    monkeypatch,
) -> None:
    """So'rov vektori leksik kanallar **bilan bir vaqtda** olinadi.

    Leksik kanal embedding boshlanishini kutadi: ketma-ket kod bo'lsa (vektor
    leksikadan keyin) bu kutish hech qachon tugamaydi va test timeout bilan
    yiqiladi.
    """
    import asyncio

    started = asyncio.Event()

    async def slow_vector(question: str, llm: Any) -> list[float]:
        started.set()
        return [0.1] * 4

    async def fake_multi(session: Any, col: Any, **kw: Any) -> list[S.Hit]:
        await asyncio.wait_for(started.wait(), timeout=2)
        return []

    monkeypatch.setattr(S, "_query_vector", slow_vector)
    monkeypatch.setattr(S, "_tsv_multi", fake_multi)
    session = _FakeSession([], [(7, 107, 1, 0.1)])
    hits = await S.hybrid_search_multi(session, 1, "tolov muddati", levels=())  # type: ignore[arg-type]
    assert [h.message_id for h in hits] == [7]


async def test_hybrid_search_runs_four_channels_without_expansion() -> None:
    llm = _EmbedLLM()
    session = _FakeSession([(1, 101, 0.5)], [], [], [])
    hits = await S.hybrid_search(session, 1, "crypto narxi", llm=llm)  # type: ignore[arg-type]
    assert [h.message_id for h in hits] == [1]
    assert len(session.sql) >= 3  # tsv_uz + trigram + vektor tekshiruvi


async def test_hybrid_search_expands_when_asked(monkeypatch) -> None:
    from app.services import query_expand as QX

    seen: list[str] = []

    async def fake_expand(query: str, **kw: Any) -> QX.Expansion:
        seen.append(query)
        return QX.Expansion(ru=["оплата"])

    monkeypatch.setattr(QX, "expand", fake_expand)
    session = _FakeSession([], [(2, 102, 0.4)], [], [])
    hits = await S.hybrid_search(session, 1, "to'lov qachon", expand=True)  # type: ignore[arg-type]
    assert seen == ["to'lov qachon"]
    assert any("tsv_ru" in q for q in session.sql)  # ru kanali endi ishlaydi
    assert [h.message_id for h in hits] == [2]


async def test_search_context_prioritizes_hits_over_tail(monkeypatch) -> None:
    """Tor byudjetda mos xabar qoladi, oxirgi 15 xabarlik dum kesiladi."""
    from app.db.models import Chat, Message

    chat = Chat(id=1, account_id=1, tg_peer_id=7, type="channel", title="T", synced_total=100)
    answer = Message(
        id=500, chat_id=1, tg_msg_id=500, text="javob shu yerda " * 20, published_at=NOW
    )
    tail_rows = [
        Message(
            id=900 + i, chat_id=1, tg_msg_id=900 + i, text="kundalik gap " * 20, published_at=NOW
        )
        for i in range(6)
    ]

    async def fake_hybrid(*_a: Any, **_kw: Any) -> list[S.Hit]:
        return [S.Hit(500, 500, 1.0, "tsv_uz")]

    async def fake_hydrate(_s: Any, _c: Any, hits: list[S.Hit]) -> list[S.Hit]:
        return hits

    monkeypatch.setattr(S, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(S, "hydrate_hits", fake_hydrate)
    session = _FakeSession(
        [(m.tg_msg_id,) for m in reversed(tail_rows)],  # dum id'lari
        [answer, *tail_rows],  # xabar qatorlari
    )
    bundle = await S.search_context(
        session,  # type: ignore[arg-type]
        chat,
        "javob qayerda",
        budget_tokens=280,
        rerank_enabled=False,
    )
    ids = [m.msg_id for m in bundle.messages]
    assert 500 in ids  # hit byudjetdan tushib qolmadi
    assert bundle.truncated and bundle.hits == 1
    assert ids == sorted(ids)  # chiqish xronologik


def test_claude_long_system_gets_cache_control() -> None:
    long_sys = "S" * (C.CACHE_SYSTEM_MIN_CHARS + 10)
    req = C.build_request(
        [Msg.system(long_sys), Msg.user("hi")],
        model="claude-opus-5",
        tools=None,
        temperature=0.5,
        max_tokens=None,
        json_mode=False,
    )
    assert isinstance(req["system"], list)
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    short = C.build_request(
        [Msg.system("S"), Msg.user("hi")],
        model="claude-opus-5",
        tools=None,
        temperature=0.5,
        max_tokens=None,
        json_mode=False,
    )
    assert short["system"] == "S"


# ─── 2-bosqich: akkaunt bo'ylab qidiruv + tugun kanali (phase2/A) ────────────


def _node_row(node_id: int, *, level: int = 1, chat_id: int = 1, lo: int = 10, hi: int = 12) -> Any:
    from app.db.models import RetrievalNode

    return RetrievalNode(
        id=node_id,
        account_id=7,
        chat_id=chat_id,
        level=level,
        starts_at=NOW,
        ends_at=NOW,
        msg_lo=lo,
        msg_hi=hi,
        msg_count=hi - lo + 1,
        text="epizod matni",
        ctx_prefix="[chat: T (supergroup) | 2026-08-18 | speakers: user:1]",
        content_hash="h",
        simhash=0,
    )


def test_hit_level_defaults_to_zero_and_survives_rrf() -> None:
    raw = S.Hit(1, 101, 1.0, "tsv_uz")
    node = S.Hit(2, 102, 1.0, "node", level=2)
    assert raw.level == 0
    fused = S.rrf([raw], [node])
    assert {h.message_id: h.level for h in fused} == {1: 0, 2: 2}


async def test_node_search_tsv_channel_filters_account_levels_and_window() -> None:
    session = _FakeSession([(_node_row(5), 0.4)])
    hits = await S.node_search(  # type: ignore[arg-type]
        session,
        7,
        "to'lov",
        levels=(1, 2),
        chat_ids=[1, 3],
        since=NOW - timedelta(days=7),
        until=NOW,
        limit=5,
    )
    sql = session.sql[0]
    assert "retrieval_nodes.tsv" in sql and "to_tsquery" in sql
    assert "retrieval_nodes.account_id" in sql and "retrieval_nodes.level IN" in sql
    assert "retrieval_nodes.chat_id IN" in sql
    assert "retrieval_nodes.ends_at >=" in sql and "retrieval_nodes.starts_at <" in sql
    assert len(session.sql) == 1  # vektor kanali yo'q (vektor ham, llm ham berilmagan)
    assert [h.node_id for h in hits] == [5]
    assert hits[0].msg_lo == 10 and hits[0].msg_hi == 12 and hits[0].level == 1
    assert hits[0].source == "node_tsv"


async def test_node_search_uses_supplied_query_vector() -> None:
    session = _FakeSession([], [(_node_row(9, level=2), 0.1)])
    hits = await S.node_search(session, 7, [0.1, 0.2], text="to'lov")  # type: ignore[arg-type]
    assert any("retrieval_node_embeddings" in q for q in session.sql)
    assert [h.node_id for h in hits] == [9] and hits[0].source == "node_vector"


async def test_node_search_without_terms_or_vector_is_empty() -> None:
    session = _FakeSession()
    assert await S.node_search(session, 7, "и в на") == []  # type: ignore[arg-type]
    assert session.sql == []


async def test_node_hits_to_messages_maps_back_to_level_zero_ids() -> None:
    # L5: endi har tugun uchun alohida so'rov — ustunlar `(id, tg_msg_id)`
    session = _FakeSession([(501, 10), (502, 11), (503, 12)])
    node_hits = [S.NodeHit(5, 1, 1, 0.9, "node_tsv", msg_lo=10, msg_hi=12)]
    hits = await S.node_hits_to_messages(session, node_hits, per_node=2)  # type: ignore[arg-type]
    assert [(h.message_id, h.tg_msg_id) for h in hits] == [(501, 10), (502, 11)]
    assert all(h.source == "node" and h.level == 1 for h in hits)
    assert "deleted_at IS NULL" in session.sql[0]


async def test_node_hits_to_messages_ignores_empty_ranges() -> None:
    session = _FakeSession()
    assert await S.node_hits_to_messages(session, []) == []  # type: ignore[arg-type]
    bad = [S.NodeHit(1, 1, 1, 0.5, "node_tsv", msg_lo=0, msg_hi=0)]
    assert await S.node_hits_to_messages(session, bad) == []  # type: ignore[arg-type]
    assert session.sql == []


async def test_hybrid_multi_scopes_by_account_chats_and_dates() -> None:
    llm = _EmbedLLM()
    session = _FakeSession(
        [(1, 101, 1, 0.5)],  # tsv_uz
        [],  # trigram
        [],  # vektor
        [],  # tugun tsv
        [],  # tugun vektor
    )
    hits = await S.hybrid_search_multi(  # type: ignore[arg-type]
        session,
        7,
        "crypto",
        chat_ids=[1, 2],
        since=NOW - timedelta(days=30),
        until=NOW,
        llm=llm,
    )
    sql = session.sql[0]
    assert "chats.account_id" in sql and "JOIN chats" in sql
    assert "messages.chat_id IN" in sql
    assert "messages.published_at >=" in sql and "messages.published_at <" in sql
    assert "deleted_at IS NULL" in sql
    assert llm.task_types == ["RETRIEVAL_QUERY"]  # bitta so'rov embeddingi
    assert [h.message_id for h in hits] == [1] and hits[0].chat_id == 1


async def test_hybrid_multi_folds_node_channel_into_rrf() -> None:
    llm = _EmbedLLM()
    session = _FakeSession(
        [],  # tsv_uz
        [],  # trigram
        [],  # vektor
        [(_node_row(5, lo=10, hi=11), 0.4)],  # tugun tsv
        [],  # tugun vektor
        [(501, 10)],  # tugun → level 0 xabarlar
    )
    hits = await S.hybrid_search_multi(session, 7, "crypto", llm=llm)  # type: ignore[arg-type]
    assert [h.message_id for h in hits] == [501]
    assert hits[0].level == 1  # epizod tugunidan kelgan
    assert S.DEFAULT_MULTI_RRF_WEIGHTS == (1.0, 1.0, 0.6, 1.2, 0.8)


async def test_hybrid_multi_without_terms_does_no_sql() -> None:
    session = _FakeSession()
    assert await S.hybrid_search_multi(session, 7, "и в на") == []  # type: ignore[arg-type]
    assert session.sql == []


# ─── entity kanali (phase4/A) ────────────────────────────────────────────────


async def test_entity_search_sql_shape() -> None:
    session = _FakeSession([(11, 101, 3), (12, 102, 3)])
    hits = await S.entity_search(  # type: ignore[arg-type]
        session,
        7,
        ["@Nurbek", "#deploy", "nima gap"],
        chat_ids=[3],
        since=NOW - timedelta(days=7),
        limit=20,
    )
    sql = session.sql[0]
    assert "JOIN node_entities" in sql and "JOIN chats" in sql
    assert "node_entities.account_id" in sql and "node_entities.kind" in sql
    assert "chats.account_id" in sql and "messages.chat_id IN" in sql
    assert "messages.published_at >=" in sql
    assert "deleted_at IS NULL" in sql
    assert "ORDER BY messages.published_at DESC" in sql
    assert [(h.message_id, h.source) for h in hits] == [(11, "entity"), (12, "entity")]
    assert hits[0].score > hits[1].score  # yangilik tartibi → kamayuvchi ball


async def test_entity_search_without_recognizable_shapes_does_no_sql() -> None:
    session = _FakeSession()
    assert await S.entity_search(session, 7, ["to'lov muddati", ""]) == []  # type: ignore[arg-type]
    assert await S.entity_search(session, 7, []) == []  # type: ignore[arg-type]
    assert session.sql == []


async def test_hybrid_multi_adds_the_entity_list_only_when_entities_are_given() -> None:
    llm = _EmbedLLM()
    without = _FakeSession([], [], [], [], [])
    await S.hybrid_search_multi(without, 7, "crypto", llm=llm)  # type: ignore[arg-type]
    assert all("node_entities" not in s for s in without.sql)

    llm2 = _EmbedLLM()
    with_ents = _FakeSession([], [], [], [], [], [(11, 101, 3)])
    hits = await S.hybrid_search_multi(  # type: ignore[arg-type]
        with_ents, 7, "crypto", llm=llm2, entities=["@nurbek"]
    )
    assert any("node_entities" in s for s in with_ents.sql)
    assert [h.message_id for h in hits] == [11]
    assert S.ENTITY_MULTI_RRF_WEIGHTS == (1.0, 1.0, 0.6, 1.2, 0.8, 1.1)
    assert S.W_ENTITY == 1.1


async def test_hybrid_multi_runs_for_entities_even_without_keywords() -> None:
    """Savolda faqat stop-so'z bo'lsa ham entity kanali ishlaydi.

    Bu holatda leksik/trigram kanallari umuman SQL bajarmaydi (termlar yo'q),
    shuning uchun navbatda faqat: vektor → tugun (vektor) → entity.
    """
    llm = _EmbedLLM()
    session = _FakeSession([], [], [(11, 101, 3)])
    hits = await S.hybrid_search_multi(  # type: ignore[arg-type]
        session, 7, "и в на", llm=llm, entities=["@nurbek"]
    )
    assert [h.message_id for h in hits] == [11]


# ─── F1: embedding modeli filtri va tugun adolati ────────────────────────────


async def test_node_hits_to_messages_gives_every_node_its_share() -> None:
    """L5: umumiy `LIMIT` o'rniga har tugun uchun alohida so'rov.

    Ilgari bitta `OR` so'rovi `chat_id, tg_msg_id` bo'yicha tartiblanib
    kesilardi — birinchi chatlar butun byudjetni yeb, ro'yxat oxiridagi
    tugunlarga hech narsa qolmasdi.
    """
    session = _FakeSession([(501, 10)], [(901, 10)], [(701, 10)])
    node_hits = [
        S.NodeHit(1, 1, 1, 0.9, "node_tsv", msg_lo=10, msg_hi=12),
        S.NodeHit(2, 9, 1, 0.8, "node_tsv", msg_lo=10, msg_hi=12),
        S.NodeHit(3, 5, 1, 0.7, "node_tsv", msg_lo=10, msg_hi=12),
    ]
    hits = await S.node_hits_to_messages(session, node_hits, per_node=1)  # type: ignore[arg-type]
    assert len(session.sql) == 3  # har tugun — o'z so'rovi
    assert [h.chat_id for h in hits] == [1, 9, 5]  # RRF tartibi saqlanadi
    assert [h.message_id for h in hits] == [501, 901, 701]


def _embed_model_name() -> str:
    from app.llm import embed_model_name

    return embed_model_name()


async def test_vector_channels_filter_by_the_current_embed_model() -> None:
    """phase4/B: turli modellarning vektorlari bitta cosine'da aralashmaydi."""
    want = _embed_model_name()
    llm = _EmbedLLM()

    session = _FakeSession([(1,)], [(1, 101, 0.2)])
    await S.vector_search(session, 1, "to'lov", llm=llm)  # type: ignore[arg-type]
    assert all("message_embeddings.model" in q for q in session.sql)
    assert want  # konfiguratsiyada model nomi bor — filtr haqiqatan qo'yiladi

    multi = _FakeSession([], [], [(1, 101, 1, 0.2)], [], [])
    await S.hybrid_search_multi(multi, 7, "crypto", llm=_EmbedLLM())  # type: ignore[arg-type]
    vector_sql = [q for q in multi.sql if "message_embeddings" in q]
    assert vector_sql and all("message_embeddings.model" in q for q in vector_sql)


async def test_node_vector_channel_filters_by_the_current_embed_model() -> None:
    session = _FakeSession([], [(_node_row(9, level=2), 0.1)])
    await S.node_search(session, 7, [0.1, 0.2], text="to'lov")  # type: ignore[arg-type]
    node_sql = [q for q in session.sql if "retrieval_node_embeddings" in q]
    assert node_sql and all("retrieval_node_embeddings.model" in q for q in node_sql)


async def test_fetch_vectors_filters_by_the_current_embed_model() -> None:
    session = _FakeSession([])
    await S.fetch_vectors(session, [1, 2, 3])  # type: ignore[arg-type]
    assert "message_embeddings.model" in session.sql[0]
