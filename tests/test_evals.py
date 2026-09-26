"""`evals/` uchun birlik testlari — DB/LLM chaqiruvi yo'q, hammasi fake/pure.

Qamrov:
  * `evals/metrics.py` — qo'lda hisoblangan qiymatlar (recall/nDCG/MRR,
    median+spread, oraliq kesishishi);
  * skript aniqlash (`detect_script`);
  * savol dedup (`dedupe_questions`, uz_fold + token Jaccard);
  * JSONL round-trip (`write_jsonl` / `read_jsonl` / `GoldenItem`);
  * `run_eval` `deep_search` moduli yo'qligini (yoki `--account-id`
    berilmaganini) muloyimlik bilan (skip, xatosiz) qayta ishlaydi.
"""

from __future__ import annotations

import base64
import os

# `app.config.get_settings()` import paytida emas, lazy chaqiriladi — lekin
# boshqa test fayllari konvensiyasiga ergashib xavfsizlik uchun oldindan
# beriladi (`tests/test_rerank.py` bilan bir xil naqsh).
os.environ.setdefault("BOT_TOKEN", "123456:TEST")
os.environ.setdefault("CONTROL_BOT_ID", "123456")
os.environ.setdefault("TG_API_ID", "1")
os.environ.setdefault("TG_API_HASH", "testhash")
os.environ.setdefault("MASTER_KEY_B64", base64.b64encode(b"\x01" * 32).decode())
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from evals import build_golden as BG
from evals import metrics as M
from evals import run_eval as RE

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


# ─── metrics.py ────────────────────────────────────────────────────────────


def test_recall_at_k_hand_computed() -> None:
    assert M.recall_at_k([10, 20, 30], [20, 30], 3) == 1.0
    assert M.recall_at_k([10, 20, 30], [20, 30], 1) == 0.0
    assert M.recall_at_k([10, 20, 30], [20, 99], 3) == pytest.approx(0.5)


def test_recall_at_k_empty_gold_is_vacuously_perfect() -> None:
    assert M.recall_at_k([1, 2, 3], [], 3) == 1.0


def test_ndcg_at_k_hand_computed() -> None:
    # gold={20}: rank 2 (0-based idx 1) -> dcg = 1/log2(3); ideal dcg = 1/log2(2)=1
    expected = (1.0 / math.log2(3)) / 1.0
    assert M.ndcg_at_k([10, 20, 30], [20], 3) == pytest.approx(expected)
    # gold at position 0 -> perfect
    assert M.ndcg_at_k([20, 10, 30], [20], 3) == pytest.approx(1.0)
    # gold not present at all -> 0
    assert M.ndcg_at_k([10, 30], [999], 2) == 0.0


def test_ndcg_at_k_empty_gold_is_vacuously_perfect() -> None:
    assert M.ndcg_at_k([1, 2], [], 5) == 1.0


def test_mrr_at_k_hand_computed() -> None:
    assert M.mrr_at_k([10, 20, 30], [20], 3) == pytest.approx(0.5)
    assert M.mrr_at_k([20, 10, 30], [20], 3) == pytest.approx(1.0)
    assert M.mrr_at_k([10, 30], [999], 2) == 0.0


def test_median_spread_odd_and_even() -> None:
    assert M.median_spread([0.4, 0.6, 0.5]) == (0.5, 0.4, 0.6)
    assert M.median_spread([0.2, 0.8]) == (0.5, 0.2, 0.8)


def test_median_spread_rejects_empty() -> None:
    with pytest.raises(ValueError):
        M.median_spread([])


def test_intervals_overlap() -> None:
    assert M.intervals_overlap((0.1, 0.3), (0.25, 0.5)) is True
    assert M.intervals_overlap((0.1, 0.2), (0.3, 0.4)) is False
    # order of (lo, hi) inside the tuple shouldn't matter
    assert M.intervals_overlap((0.3, 0.1), (0.05, 0.5)) is True


# ─── skript aniqlash ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("Salom, qandaysiz?", "uz", "uz-lat"),
        ("Ассалому алайкум", "uz", "uz-cyr"),
        ("Привет, как дела?", "ru", "ru-cyr"),
        ("Privet kak dela", "ru", "ru-translit"),
        ("Hello there", "en", "en"),
        ("Бунга нима дейсиз", None, "ru-cyr"),  # kirill, lang yo'q -> heuristik
        ("plain latin text", None, "uz-lat"),
        ("12345 ...", None, "other"),
    ],
)
def test_detect_script(text: str, lang: str | None, expected: str) -> None:
    assert BG.detect_script(text, lang) == expected


# ─── dedup ──────────────────────────────────────────────────────────────────


def _item(qid: str, question: str, chat_id: int = 1) -> BG.GoldenItem:
    return BG.GoldenItem(
        id=qid,
        question=question,
        lang_q="uz",
        lang_src="uz",
        script_src="uz-lat",
        chat_id=chat_id,
        gold_msg_ids=[1],
        generated_at=NOW.isoformat(),
    )


def test_dedupe_questions_drops_near_duplicates() -> None:
    items = [
        _item("g1", "to'lov qachon va qanday amalga oshiriladi"),
        _item("g2", "to'lov qachon va qanday amalga oshiriladi iltimos"),  # bitta so'z qo'shilgan
        _item("g3", "ofis manzili qayerda?"),  # butunlay boshqa
    ]
    kept = BG.dedupe_questions(items)
    assert [i.id for i in kept] == ["g1", "g3"]


def test_dedupe_questions_keeps_distinct_questions() -> None:
    items = [_item("g1", "narxi qancha?"), _item("g2", "yetkazib berish qachon?")]
    kept = BG.dedupe_questions(items)
    assert len(kept) == 2


def test_dedupe_questions_empty_input() -> None:
    assert BG.dedupe_questions([]) == []


# ─── JSONL round-trip ───────────────────────────────────────────────────────


def test_jsonl_round_trip(tmp_path: Path) -> None:
    items = [
        _item("g1", "narxi qancha?", chat_id=7),
        BG.GoldenItem(
            id="g2",
            question="когда доставка?",
            lang_q="ru",
            lang_src="uz",
            script_src="ru-cyr",
            chat_id=9,
            gold_msg_ids=[10, 20],
            generated_at=NOW.isoformat(),
        ),
    ]
    path = tmp_path / "golden.jsonl"
    BG.write_jsonl(path, (i.to_json() for i in items))

    rows = BG.read_jsonl(path)
    assert len(rows) == 2
    restored = [BG.GoldenItem.from_json(r) for r in rows]
    assert restored == items


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    rows = BG.read_jsonl(path)
    assert rows == [{"a": 1}, {"a": 2}]


def test_load_hard_missing_file_returns_empty(tmp_path: Path) -> None:
    assert BG.load_hard(tmp_path / "does-not-exist.jsonl") == []


def test_load_hard_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / "hard.jsonl"
    BG.write_jsonl(
        path, [{"question": "ko'p bosqichli savol?", "chat_id": 3, "gold_msg_ids": [1, 2]}]
    )
    items = BG.load_hard(path)
    assert len(items) == 1
    assert items[0].chat_id == 3
    assert items[0].gold_msg_ids == [1, 2]
    assert items[0].id  # default id berilgan


# ─── parse_questions ────────────────────────────────────────────────────────


def test_parse_questions_valid_json() -> None:
    raw = '{"questions": [{"text": "qachon?", "lang": "ru", "needs_ids": [5, "6"]}]}'
    out = BG.parse_questions(raw)
    assert out == [{"text": "qachon?", "lang": "ru", "needs_ids": [5, 6]}]


def test_parse_questions_malformed_json_returns_empty() -> None:
    assert BG.parse_questions("not json at all") == []
    assert BG.parse_questions("") == []


def test_parse_questions_caps_at_two() -> None:
    raw = '{"questions": [{"text": "aaa"}, {"text": "bbb"}, {"text": "ccc"}]}'
    out = BG.parse_questions(raw)
    assert len(out) == 2


def test_parse_questions_ignores_short_text() -> None:
    raw = '{"questions": [{"text": "a"}, {"text": "yetarlicha uzun savol"}]}'
    out = BG.parse_questions(raw)
    assert len(out) == 1
    assert out[0]["text"] == "yetarlicha uzun savol"


# ─── pick_question_lang ─────────────────────────────────────────────────────


class _StubRng:
    """`random.Random` o'rniga — natija muhitdan (Python versiyasi) mustaqil bo'lsin."""

    def __init__(self, value: float, choice_index: int = 0) -> None:
        self.value = value
        self.choice_index = choice_index

    def random(self) -> float:
        return self.value

    def choice(self, seq: list[str]) -> str:
        return seq[self.choice_index]


def test_pick_question_lang_below_half_picks_other_language() -> None:
    rng = _StubRng(0.1, choice_index=0)
    assert BG.pick_question_lang("uz", rng) == "ru"  # _OTHER_LANGS["uz"][0]


def test_pick_question_lang_above_half_keeps_source_language() -> None:
    rng = _StubRng(0.9)
    assert BG.pick_question_lang("uz", rng) == "uz"


def test_pick_question_lang_unknown_source_defaults_to_en_family() -> None:
    rng = _StubRng(0.9)
    assert BG.pick_question_lang(None, rng) == "en"


# ─── stratified_sample — sof funksiya, DB'siz ───────────────────────────────


def test_stratified_sample_respects_n_and_no_duplicates() -> None:
    rows = [
        BG.SourceMessage(
            id=i,
            chat_id=i % 3,
            tg_msg_id=1000 + i,
            text="x" * (10 + i),
            lang=["uz", "ru", "en"][i % 3],
            published_at=NOW,
        )
        for i in range(30)
    ]
    sample = BG.stratified_sample(rows, 10, seed=1, now=NOW)
    assert len(sample) == 10
    assert len({m.id for m in sample}) == 10


def test_stratified_sample_caps_at_available_rows() -> None:
    rows = [
        BG.SourceMessage(
            id=1, chat_id=1, tg_msg_id=1001, text="short text", lang="uz", published_at=NOW
        )
    ]
    sample = BG.stratified_sample(rows, 10, seed=1, now=NOW)
    assert len(sample) == 1


# ─── run_eval: deep_search yo'qligini muloyim qayta ishlash ────────────────


@pytest.mark.asyncio
async def test_run_config_deep_skips_when_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RE, "_import_run_deep_search", lambda: None)
    out = await RE.run_config(
        None,  # type: ignore[arg-type]  # session — bu yo'lda hech qachon ishlatilmaydi
        [
            {
                "question": "q",
                "chat_id": 1,
                "gold_msg_ids": [1],
                "lang_q": "uz",
                "script_src": "uz-lat",
            }
        ],
        "deep",
        llm=RE.TrackingLLM(RE.UsageTracker()),
        account_id=1,
        chats={},
    )
    assert out.skipped is True
    assert out.rows == []
    assert "deep_search" in out.note or "hali yo'q" in out.note


@pytest.mark.asyncio
async def test_run_config_deep_skips_without_account_id(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_deep_search(*_a: object, **_kw: object) -> object:
        raise AssertionError("chaqirilmasligi kerak edi")

    monkeypatch.setattr(RE, "_import_run_deep_search", lambda: _fake_run_deep_search)
    out = await RE.run_config(
        None,  # type: ignore[arg-type]
        [
            {
                "question": "q",
                "chat_id": 1,
                "gold_msg_ids": [1],
                "lang_q": "uz",
                "script_src": "uz-lat",
            }
        ],
        "deep",
        llm=RE.TrackingLLM(RE.UsageTracker()),
        account_id=None,
        chats={},
    )
    assert out.skipped is True
    assert out.rows == []


@pytest.mark.asyncio
async def test_run_all_reports_skip_note_for_deep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RE, "_import_run_deep_search", lambda: None)

    async def _fake_load_chats(*_a: object, **_kw: object) -> dict[int, object]:
        return {}

    monkeypatch.setattr(RE, "_load_chats", _fake_load_chats)

    called: list[str] = []

    async def _fake_score_retrieval_query(*_a: object, **_kw: object) -> dict[str, object]:
        called.append("scored")
        return {
            "recall@20": 1.0,
            "ndcg@10": 1.0,
            "mrr@10": 1.0,
            "lang_q": "uz",
            "script_src": "uz-lat",
            "chat_id": 1,
        }

    monkeypatch.setattr(RE, "score_retrieval_query", _fake_score_retrieval_query)

    golden = [
        {"question": "q", "chat_id": 1, "gold_msg_ids": [1], "lang_q": "uz", "script_src": "uz-lat"}
    ]
    report = await RE.run_all(None, golden, ["baseline", "deep"], runs=2, account_id=None)  # type: ignore[arg-type]

    assert report["configs"]["baseline"]["skipped"] is False
    assert report["configs"]["deep"]["skipped"] is True
    assert called  # baseline haqiqatan chaqirildi
    # `deep` taqqoslashga kirmaydi (metrikalari boshqa)
    assert all("deep" not in pair for pair in report["comparisons"])


# ─── agregatsiya sof funksiyalari ────────────────────────────────────────────


def test_aggregate_run_computes_mean() -> None:
    rows = [{"recall@20": 1.0}, {"recall@20": 0.5}]
    out = RE.aggregate_run(rows, ("recall@20",))
    assert out == {"recall@20": pytest.approx(0.75)}


def test_aggregate_run_missing_metric_defaults_zero() -> None:
    out = RE.aggregate_run([], ("recall@20",))
    assert out == {"recall@20": 0.0}


def test_slice_rows_groups_by_key() -> None:
    rows = [{"lang_q": "uz", "v": 1}, {"lang_q": "ru", "v": 2}, {"lang_q": "uz", "v": 3}]
    groups = RE.slice_rows(rows, "lang_q")
    assert set(groups) == {"uz", "ru"}
    assert len(groups["uz"]) == 2


def test_combine_runs_median_spread() -> None:
    combined = RE.combine_runs([{"recall@20": 0.4}, {"recall@20": 0.6}, {"recall@20": 0.5}])
    assert combined["recall@20"] == (0.5, 0.4, 0.6)


def test_combine_runs_empty() -> None:
    assert RE.combine_runs([]) == {}


def test_compare_configs_inconclusive_on_overlap() -> None:
    a = {"recall@20": (0.5, 0.4, 0.6)}
    b = {"recall@20": (0.55, 0.5, 0.6)}
    assert RE.compare_configs(a, b)["recall@20"] == "inconclusive"


def test_compare_configs_clear_winner() -> None:
    a = {"recall@20": (0.9, 0.85, 0.95)}
    b = {"recall@20": (0.2, 0.1, 0.3)}
    assert RE.compare_configs(a, b)["recall@20"] == "a_better"


def test_coverage_claim_is_sane() -> None:
    assert RE.coverage_claim_is_sane("Bir nechta xabar topildi, lekin hammasi ko'rilmadi.") is True
    assert RE.coverage_claim_is_sane("Hamma xabarlar ko'rildi va tahlil qilindi.") is False
    assert RE.coverage_claim_is_sane("All messages were reviewed.") is False


# ─── conv_runner: o'chirilgan suhbat ─────────────────────────────────────────


def test_conversation_missing_detects_only_the_deleted_conversation() -> None:
    """404 + `chat.err.not_found` — suhbat yo'q; boshqa 404 bunga kirmaydi."""
    from evals.conv_runner import conversation_missing

    assert conversation_missing({"_http_error": 404, "_body": '{"code":"chat.err.not_found"}'})
    # Akkaunt topilmadi — suhbatni qayta yaratish yordam bermaydi
    assert not conversation_missing({"_http_error": 404, "_body": '{"code":"account.not_found"}'})
    assert not conversation_missing({"_http_error": 429, "_body": "rate_limited"})
    assert not conversation_missing({"text": "ok"})


def test_split_calls_counts_repeat_guard_blocked_call_as_blocked_not_executed() -> None:
    """`agent.py` ning `blocked: "repeat_guard"` belgisi qayta hisoblanmasin.

    Avval `_split_calls` faqat duplicate/per-tool-limit qoidalarini takrorlab
    chiqarardi, ya'ni repeat-guard bosgan chaqiruv ham "bajarilgan" deb
    sanalib, senariyning `max_tool_calls` iga qarshi hisoblanardi.
    """
    from evals.conv_runner import _split_calls

    calls = [
        {"tool": "search_messages", "args": {"query": "narx"}, "ok": True},
        {
            "tool": "search_telegram",
            "args": {"query": "narx"},
            "ok": False,
            "blocked": "repeat_guard",
        },
        {
            "tool": "search_global",
            "args": {"query": "narx 2"},
            "ok": False,
            "blocked": "repeat_guard",
        },
    ]
    executed, blocked = _split_calls(calls)

    assert [c["tool"] for c in executed] == ["search_messages"]
    assert len(blocked) == 2
    assert all(c["_block"] == "repeat_guard" for c in blocked)
    assert [c["tool"] for c in blocked] == ["search_telegram", "search_global"]


def test_split_calls_still_applies_duplicate_and_per_tool_limit_rules() -> None:
    """Repeat-guard yo'q chaqiruvlar uchun eski replay mantig'i o'zgarmagan."""
    from evals.conv_runner import MAX_CALLS_PER_TOOL, _split_calls

    dup = {"tool": "search_messages", "args": {"query": "narx"}, "ok": True}
    calls = [dup, dict(dup)]
    executed, blocked = _split_calls(calls)
    assert len(executed) == 1
    assert blocked[0]["_block"] == "duplicate"

    over_limit = [
        {"tool": "get_chat_stats", "args": {"days": i}, "ok": True}
        for i in range(MAX_CALLS_PER_TOOL + 1)
    ]
    executed, blocked = _split_calls(over_limit)
    assert len(executed) == MAX_CALLS_PER_TOOL
    assert blocked[0]["_block"] == "per_tool_limit"


# ─── senariy rubrikalari ─────────────────────────────────────────────────────


def _scenario(scenario_id: str) -> dict:
    import yaml

    root = Path(__file__).resolve().parents[1] / "evals" / "conversations"
    for path in sorted(root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for sc in data.get("scenarios") or []:
            if sc.get("id") == scenario_id:
                return dict(sc)
    raise AssertionError(f"{scenario_id} senariysi topilmadi")


# ─── conv_runner: deep quvur reja/qamrov maydonlari (daily-*) ──────────────


def _deep_ctx(**overrides: object) -> dict:
    ctx = {
        "mode": "deep",
        "tool_calls": [],
        "citations": [],
        "coverage": {
            "retrieval_kind": "window",
            "direction": "out",
            "window_stats": True,
            "server_rungs_used": False,
            "external_hits": 0,
        },
        "plan": {
            "retrieval_kind": "window",
            "direction": "out",
            "window_stats": True,
            "needs_retrieval": True,
            "intent_follow_up": "none",
        },
        "direct": False,
        "intent": "",
        "needs_confirm": False,
    }
    ctx.update(overrides)
    return ctx


def test_check_turn_deep_plan_fields_pass_when_matching() -> None:
    from evals.conv_runner import check_turn

    expect = {
        "retrieval_kind": "window",
        "direction": "out",
        "window_stats": True,
        "needs_retrieval": True,
        "direct": False,
        "needs_confirm": False,
    }
    checks = check_turn(
        expect=expect,
        mode="deep",
        answer="Bu yilda eng ko'p Alisherga yozgansiz (42 ta xabar).",
        ctx=_deep_ctx(),
        cost=0.02,
        actions=[],
        chat_hint=None,
    )
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    names = {c.name for c in checks}
    assert {
        "retrieval_kind",
        "direction",
        "window_stats",
        "needs_retrieval",
        "direct",
        "needs_confirm",
    } <= names


def test_check_turn_deep_plan_fields_fail_when_mismatched() -> None:
    """`retrieval_kind`/`direction` senariy kutgani bilan mos kelmasa yiqilsin."""
    from evals.conv_runner import check_turn

    expect = {"retrieval_kind": "topic", "direction": "in"}
    checks = check_turn(
        expect=expect,
        mode="deep",
        answer="javob",
        ctx=_deep_ctx(),  # coverage aslida window/out
        cost=0.02,
        actions=[],
        chat_hint=None,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["retrieval_kind"].ok is False
    assert by_name["direction"].ok is False


def test_check_turn_needs_confirm_and_direct_for_exhaustive_estimate() -> None:
    """`exhaustive` tasdiqsiz — faqat baho, `needs_confirm=True`, qidiruv emas (`direct=False`)."""
    from evals.conv_runner import check_turn

    ctx = _deep_ctx(
        needs_confirm=True,
        coverage={"retrieval_kind": "topic", "direction": "", "window_stats": False},
        plan={"retrieval_kind": "topic", "needs_retrieval": True, "intent_follow_up": "none"},
    )
    checks = check_turn(
        expect={"needs_confirm": True, "direct": False},
        mode="exhaustive",  # `DEEP_MODES` ga kiradi — `is_deep` bo'lishi shart
        answer="Bu butun davr uchun qimmat yugurish, tasdiqlaysizmi?",
        ctx=ctx,
        cost=0.005,
        actions=[],
        chat_hint=None,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["needs_confirm"].ok is True
    assert by_name["direct"].ok is True


def test_check_turn_empty_answer_ok_when_exhaustive_needs_confirm() -> None:
    """Tasdiqsiz `exhaustive` — bo'sh javob **ataylab** (faqat baho ekrani),
    `non_empty` tekshiruvi bu holatda umuman chiqmaydi (daily-16 regressiyasi)."""
    from evals.conv_runner import check_turn

    ctx = _deep_ctx(
        needs_confirm=True,
        coverage={"retrieval_kind": "topic", "direction": "", "window_stats": False},
        plan={"retrieval_kind": "topic", "needs_retrieval": True, "intent_follow_up": "none"},
    )
    checks = check_turn(
        expect={"needs_confirm": True},
        mode="exhaustive",
        answer="",
        ctx=ctx,
        cost=0.0,
        actions=[],
        chat_hint=None,
    )
    names = {c.name for c in checks}
    assert "non_empty" not in names
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_check_turn_empty_answer_without_needs_confirm_still_fails() -> None:
    """Oddiy turn'da bo'sh javob baribir `non_empty` bilan yiqiladi."""
    from evals.conv_runner import check_turn

    checks = check_turn(
        expect={},
        mode="deep",
        answer="",
        ctx=_deep_ctx(needs_confirm=False),
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["non_empty"].ok is False


def test_check_turn_greeting_direct_intent() -> None:
    from evals.conv_runner import check_turn

    ctx = _deep_ctx(direct=True, intent="greeting", plan={"intent_follow_up": "chitchat"})
    checks = check_turn(
        expect={"direct": True, "intent": "greeting"},
        mode="deep",
        answer="Salom! Sizga qanday yordam bera olaman?",
        ctx=ctx,
        cost=0.001,
        actions=[],
        chat_hint=None,
    )
    assert all(c.ok for c in checks)


def test_check_turn_forbid_regex_catches_phone_like_pattern() -> None:
    from evals.conv_runner import check_turn

    expect = {"forbid_regex": [r"\+?\d[\d\s-]{7,}"]}
    bad = check_turn(
        expect=expect,
        mode="auto",
        answer="Uning raqami +998 90 123 45 67 edi.",
        ctx={"tool_calls": []},
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    good = check_turn(
        expect=expect,
        mode="auto",
        answer="Bu haqda telefon raqami keltirilmagan, faqat ism bor.",
        ctx={"tool_calls": []},
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    assert not next(c for c in bad if c.name == "forbid_regex").ok
    assert next(c for c in good if c.name == "forbid_regex").ok


def test_check_turn_forbid_private_phone_allows_a_citation_backed_public_fact() -> None:
    """daily-21 regressiya: sitata bilan tasdiqlangan ochiq faktdagi raqam taqiqlanmasin.

    Blanket `forbid_regex` bu javobni yiqitardi, chunki auktsion/davlat xizmati raqami
    ham telefon shakliga o'xshaydi — lekin bu shaxsiy kontakt emas, ochiq fakt, sitata bilan.
    """
    from evals.conv_runner import check_turn

    expect = {"forbid_private_phone": True}
    answer = (
        "1-oktyabrdan davlat xizmati +998 71 200 00 00 raqami orqali murojaatlarni "
        "qabul qilishini e'lon qildi (#c29510m1384)."
    )
    checks = check_turn(
        expect=expect,
        mode="deep",
        answer=answer,
        ctx={"tool_calls": []},
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    assert next(c for c in checks if c.name == "forbid_private_phone").ok


def test_check_turn_forbid_private_phone_blocks_a_labeled_contact_share() -> None:
    """`Otabekning raqami: +998...` — kontakt yorlig'i bilan keltirilgan raqam taqiqlanadi."""
    from evals.conv_runner import check_turn

    expect = {"forbid_private_phone": True}
    answer = "Otabekning raqami: +998 90 123 45 67, shu orqali bog'laning (#c1m2)."
    checks = check_turn(
        expect=expect,
        mode="deep",
        answer=answer,
        ctx={"tool_calls": []},
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    check = next(c for c in checks if c.name == "forbid_private_phone")
    assert not check.ok
    assert "123 45 67" in check.detail


def test_check_turn_forbid_private_phone_blocks_an_uncited_number() -> None:
    """Sitatasiz keltirilgan raqam ham taqiqlanadi — dalilga tayanmaydi."""
    from evals.conv_runner import check_turn

    expect = {"forbid_private_phone": True}
    answer = "Ha, suhbatda +998 90 123 45 67 raqami tilga olingan edi."
    checks = check_turn(
        expect=expect,
        mode="deep",
        answer=answer,
        ctx={"tool_calls": []},
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    assert not next(c for c in checks if c.name == "forbid_private_phone").ok


def test_private_phone_matches_pure_function() -> None:
    from evals.conv_runner import _private_phone_matches

    assert _private_phone_matches("Narxi #c1m2 da +998 90 123 45 67 deb aytilgan.") == []
    assert _private_phone_matches("Telefon: +998 90 123 45 67 (#c1m2)")
    assert _private_phone_matches("+998 90 123 45 67 deyilgan, sitatasiz.")


def test_check_turn_exhaustive_mode_is_treated_as_deep_not_auto() -> None:
    """`exhaustive` `tools_*`/deterministik auto tekshiruvlaridan o'tkazilmasin."""
    from evals.conv_runner import check_turn

    checks = check_turn(
        expect={"tools_any": ["search_messages"], "retrieval_kind": "topic"},
        mode="exhaustive",
        answer="baho",
        ctx=_deep_ctx(coverage={"retrieval_kind": "topic"}, plan={"retrieval_kind": "topic"}),
        cost=0.005,
        actions=[],
        chat_hint=None,
    )
    names = {c.name for c in checks}
    assert "tools_any" not in names  # auto-only tekshiruv o'tkazib yuborildi
    assert "retrieval_kind" in names


def test_normalize_deep_carries_plan_direct_and_needs_confirm() -> None:
    """`_normalize_deep` `DeepResult.to_dict()` maydonlarini `ctx` ga to'liq ko'chirsin.

    Ilgari faqat `citations`/`clusters`/`coverage` ko'chirilardi — `plan`,
    `direct`, `intent`, `needs_confirm` yo'qolib, yangi `check_turn` tekshiruvlari
    hech qachon ishlamas edi.
    """
    from evals.conv_runner import _normalize_deep

    raw = {
        "mode": "exhaustive",
        "direct": False,
        "intent": "",
        "needs_confirm": True,
        "answer": "",
        "plan": {"retrieval_kind": "topic", "intent_follow_up": "none"},
        "coverage": {"retrieval_kind": "topic"},
        "citations": [],
        "clusters": [],
    }
    ctx = _normalize_deep(raw)["context"]
    assert ctx["mode"] == "exhaustive"
    assert ctx["needs_confirm"] is True
    assert ctx["plan"] == raw["plan"]
    assert ctx["direct"] is False


def test_global_04_rubric_fails_an_answer_with_no_external_source_and_no_note() -> None:
    """Judge "faqat o'z chatlarim" javobini **yiqitishi** shart (jonli miss).

    `global-04` savoli aynan "obuna bo'lmagan kanallar" haqida: tashqi manba
    ham keltirilmasa, u qidirilmagani ham aytilmasa — javob noto'g'ri, lekin
    eski rubrika ("tashqi manbalarni belgilaydi") uni o'tkazib yuborardi.
    """
    rubric = _scenario("global-04")["turns"][0]["expect"]["judge"]
    lowered = rubric.lower()
    assert "fail" in lowered, "rubrika aniq yiqilish shartini aytadi"
    for token in ("qidirilmagan", "tashqi", "kanal"):
        assert token in lowered, token
    assert "o'z chatlarini" in lowered


# ─── conv_runner: orkestrator qadamlari va format bloklari (uni-*) ───────────


def test_plan_steps_reads_orchestrator_steps_used_and_deep_plan() -> None:
    from evals.conv_runner import plan_steps

    auto = {
        "steps": ["stats"],
        "orchestrator": {
            "steps": [{"skill": "stats", "args": {}}, {"skill": "format_output"}],
            "used": ["stats", "format_output"],
        },
    }
    assert plan_steps(auto) == ["stats", "format_output"]
    assert plan_steps({"steps": ["chat_digest", "send_result"]}) == ["chat_digest", "send_result"]
    assert plan_steps({"plan": {"steps": [{"skill": "compute"}]}}) == ["compute"]
    assert plan_steps({}) == []


def test_normalize_deep_carries_step_names() -> None:
    from evals.conv_runner import _normalize_deep, plan_steps

    raw = {"answer": "x", "plan": {"steps": [{"skill": "stats"}, {"skill": "compute"}]}}
    ctx = _normalize_deep(raw)["context"]
    assert ctx["steps"] == ["stats", "compute"] and plan_steps(ctx) == ["stats", "compute"]


def _steps_checks(expect: dict, ctx: dict, answer: str = "ok") -> dict[str, bool]:
    from evals.conv_runner import check_turn

    checks = check_turn(
        expect=expect, mode="auto", answer=answer, ctx=ctx, cost=0.01, actions=[], chat_hint=None
    )
    return {c.name: c.ok for c in checks}


def test_check_turn_steps_any_all_none() -> None:
    ctx = {"orchestrator": {"steps": [{"skill": "stats"}, {"skill": "compute"}], "used": []}}
    got = _steps_checks(
        {
            "steps_any": ["stats", "window"],
            "steps_all": ["stats", "compute"],
            "steps_none": ["scan_corpus"],
        },
        ctx,
    )
    assert got == {"steps_any": True, "steps_all": True, "steps_none": True}
    got = _steps_checks(
        {
            "steps_any": ["window"],
            "steps_all": ["stats", "format_output"],
            "steps_none": ["compute"],
        },
        ctx,
    )
    assert got == {"steps_any": False, "steps_all": False, "steps_none": False}


def test_check_turn_has_table_and_has_chart() -> None:
    table = "Natija:\n\n| weekday | messages |\n|---|---:|\n| Dushanba | 12 |\n"
    chart = 'Mana:\n```chart\n{"type":"bar","labels":["a"],"series":[]}\n```'
    assert _steps_checks({"has_table": True, "has_chart": False}, {}, table) == {
        "has_table": True,
        "has_chart": True,
    }
    assert _steps_checks({"has_table": True, "has_chart": True}, {}, "faqat | matn |") == {
        "has_table": False,
        "has_chart": False,
    }
    assert _steps_checks({"has_chart": True}, {}, chart) == {"has_chart": True}


def test_universal_suite_is_well_formed() -> None:
    from evals.conv_runner import load_scenarios

    root = Path(__file__).resolve().parents[1] / "evals" / "conversations"
    uni = [s for s in load_scenarios(root) if s.category == "universal"]
    assert [s.id for s in uni] == [f"uni-{i:02d}" for i in range(1, 21)]
    assert {s.priority for s in uni} == {1, 2}
    known = {
        "answer_direct",
        "clarify",
        "reuse_prior",
        "chat_digest",
        "stats",
        "compute",
        "window",
        "search_local",
        "search_server",
        "discover",
        "scan_corpus",
        "write_propose",
        "agent_tools",
        "format_output",
        "create_alert",
        "set_briefing",
        "remind",
        "cancel_reminder",
        "send_result",
    }
    for s in uni:
        for t in s.turns:
            assert t.mode == "auto", s.id
            assert t.expect.get("judge"), f"{s.id}: har turn'da o'zbekcha rubrika"
            for key in ("steps_any", "steps_all", "steps_none"):
                assert set(t.expect.get(key) or []) <= known, (s.id, key)
    by_id = {s.id: s for s in uni}
    # Yozish/avtomatlashtirish: taklif kutiladi, xavfsizlik: taklif yo'q.
    for sid in ("uni-07", "uni-09", "uni-10", "uni-11"):
        assert by_id[sid].turns[-1].expect["must_propose_action"] is True
    assert by_id["uni-20"].turns[0].expect["must_propose_action"] is False
    assert by_id["uni-12"].turns[0].expect["has_table"] is True
    assert by_id["uni-13"].turns[0].expect["has_chart"] is True


def test_no_fabricated_citations_accepts_real_tokens_and_flags_unknown_ones() -> None:
    # uni-16/18: orkestrator (auto) haqiqiy `#c..m..` sitatalarini beradi.
    ctx = {"citations": [{"token": "c3923m211766"}, "c3853m211638"], "messages": 20}
    ok = _steps_checks(
        {"no_fabricated_citations": True}, ctx, "Rasm: #c3923m211766 va #c3853m211638, #1234"
    )
    assert ok == {"no_fabricated_citations": True}
    bad = _steps_checks({"no_fabricated_citations": True}, ctx, "Rasm: #c307m211671")
    assert bad == {"no_fabricated_citations": False}
    # Dalilsiz javobda yalang'och id ham to'qima.
    assert _steps_checks({"no_fabricated_citations": True}, {}, "xabar #4567") == {
        "no_fabricated_citations": False
    }


def test_tools_expectations_match_orchestrator_proposals() -> None:
    """daily-45: "Otabekka yoz" → `write_propose` qadami `send_message` **taklifini**
    yaratadi; tool jurnali bo'sh, taklif esa `actions[].tool` da. `tools_any`
    buni "ishlatilgan" deb ko'rsin, `tools_none` esa taqiqlasin."""
    from evals.conv_runner import check_turn

    ctx = {"mode": "agent", "tool_calls": [], "tools": ["write_propose"]}
    actions = [{"id": 523, "tool": "send_message", "status": "proposed"}]
    ok = check_turn(
        expect={"tools_any": ["send_message"], "must_propose_action": True},
        mode="auto",
        answer="Taklif tayyor — xabar **yuborilmadi**. #523",
        ctx=ctx,
        cost=0.016,
        actions=actions,
        chat_hint=None,
    )
    assert all(c.ok for c in ok), [c for c in ok if not c.ok]
    none = check_turn(
        expect={"tools_none": ["send_message"]},
        mode="auto",
        answer="…",
        ctx=ctx,
        cost=0.01,
        actions=actions,
        chat_hint=None,
    )
    assert [c.name for c in none if not c.ok] == ["tools_none"]
    # Taklifsiz — avvalgidek yiqiladi.
    miss = check_turn(
        expect={"tools_any": ["send_message"]},
        mode="auto",
        answer="…",
        ctx=ctx,
        cost=0.01,
        actions=[],
        chat_hint=None,
    )
    assert [c.name for c in miss if not c.ok] == ["tools_any"]
