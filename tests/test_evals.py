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
