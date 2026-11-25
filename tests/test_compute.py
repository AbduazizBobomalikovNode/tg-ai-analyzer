"""`compute` ko'nikmasi — xavfsiz evaluator, savoldan hisob turi, raqam tekshiruvi.

Qo'riqlanadigan narsalar:

  * evaluator faqat arifmetika: `eval`/atribut/indeks/lambda/satr/kalit
    so'zli argument — `ComputeError`, hech qachon bajarilmaydi;
  * planner ifoda bermasa savol so'zlaridan (kuniga/haftasiga/foiz) hisob
    tanlanadi, mos kelmasa — hech narsa (uydirma o'rniga jimlik);
  * javobdagi raqamlar faktlardan kelganini `unbacked_numbers` tekshiradi.
"""

from __future__ import annotations

import pytest

from app.services import compute as CMP
from app.services.compute import ComputeError, Fact


def _facts(**values: float) -> dict[str, Fact]:
    return {k: Fact(k, v) for k, v in values.items()}


def test_safe_eval_arithmetic_names_and_helpers() -> None:
    names = {"total": 1234.0, "days": 25.0, "mine": 300.0}
    assert CMP.safe_eval("total / days", names) == pytest.approx(49.36)
    assert CMP.safe_eval("pct(mine, total)", names) == pytest.approx(24.311, rel=1e-3)
    assert CMP.safe_eval("avg(1, 2, 3)", {}) == 2
    assert CMP.safe_eval("round(total / 7, 1)", names) == pytest.approx(176.3)
    assert CMP.safe_eval("-(total - mine) + 2 ** 3", names) == -926
    assert CMP.safe_eval("max(total, mine) - min(total, mine)", names) == 934


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id')",
        "total.__class__",
        "[1, 2][0]",
        "(lambda: 1)()",
        "'abc'",
        "round(total, ndigits=2)",
        "open('/etc/passwd')",
        "2 ** 100",
        "unknown / 2",
        "total / 0",
        "pct(total, 0)",
        "x" * 300,
        "1 if total else 2",
        "[x for x in (1, 2)]",
        "True + 1",
        "",
    ],
)
def test_safe_eval_rejects_everything_but_arithmetic(expr: str) -> None:
    with pytest.raises(ComputeError):
        CMP.safe_eval(expr, {"total": 10.0})


def test_heuristic_picks_the_obvious_formula_from_the_question() -> None:
    facts = _facts(total=100, days=25, weeks=3.57, months=0.82, mine=40)
    got = CMP.heuristic_items("Bu oyda o'rtacha kuniga nechta xabar yozganman?", facts)
    assert got[0]["expr"] == "total / days"
    got = CMP.heuristic_items("haftasiga nechta xabar keladi", facts)
    assert got[0]["expr"] == "total / weeks"
    got = CMP.heuristic_items("xabarlarning qancha foizini men yozganman?", facts)
    assert {"label": "mine_pct", "expr": "pct(mine, total)"} in got
    got = CMP.heuristic_items("necha kunda bir xabar yozaman", facts)
    assert got[0]["expr"] == "days / total"
    # Mos formula yo'q — hech narsa hisoblanmaydi.
    assert CMP.heuristic_items("kim eng ko'p yozdi?", facts) == []
    assert CMP.heuristic_items("kuniga nechta", _facts(total=5)) == []


def test_parse_items_accepts_single_and_list_and_drops_garbage() -> None:
    assert CMP.parse_items({"expr": "total / days", "label": "Avg Per Day"}) == [
        {"label": "avg_per_day", "expr": "total / days"}
    ]
    items = CMP.parse_items(
        {"items": [{"expr": "a - b", "label": "diff"}, {"label": "no expr"}, "junk", {"expr": 5}]}
    )
    assert items == [{"label": "diff", "expr": "a - b"}]
    assert CMP.parse_items({"expr": "1+1", "label": "9bad label!"})[0]["label"] == "result_1"


def test_run_items_chains_results_and_shows_the_inputs() -> None:
    made, errors = CMP.run_items(
        [
            {"label": "avg_per_day", "expr": "total / days"},
            {"label": "per_week", "expr": "avg_per_day * 7"},
            {"label": "bad", "expr": "total / nothing"},
        ],
        _facts(total=1234, days=25),
    )
    assert [f.name for f in made] == ["avg_per_day", "per_week"]
    assert made[0].value == pytest.approx(49.36)
    assert made[1].value == pytest.approx(345.52)
    assert "1234 / 25" in made[0].expr
    assert errors and errors[0].startswith("bad:")
    line = made[0].render()
    assert line.startswith("- avg_per_day = total / days (1234 / 25) = 49.36")


def test_fact_render_integer_and_float() -> None:
    assert Fact("total", 42, label="messages").render() == "- total = 42 (messages)"
    assert Fact("x", 3.0).render() == "- x = 3"
    assert Fact("y", 1 / 3).render() == "- y = 0.33"


def test_unbacked_numbers_only_flags_invented_numbers() -> None:
    facts = [Fact("total", 1234), Fact("days", 25), Fact("avg_per_day", 49.36)]
    ok = (
        "Bu oyda 1 234 ta xabar yozgansiz: 1234 / 25 kun ≈ 49.4 xabar kuniga "
        "(2026-09-01 → 2026-09-25, 14:05 holatiga) #c12m345."
    )
    assert CMP.unbacked_numbers(ok, facts) == []
    bad = "Bu oyda 1234 ta xabar, kuniga taxminan 61 ta."
    assert CMP.unbacked_numbers(bad, facts) == [61.0]
    # Kichik butun sonlar (ro'yxat raqami) va yillar hisob emas.
    assert CMP.unbacked_numbers("1. Birinchi 2. Ikkinchi, 2025 yilda", facts) == []
    # Qo'shimcha matn (sanoq bloki) ham manba sanaladi.
    assert CMP.unbacked_numbers("Ish chatida 812 ta", facts, extra_text="- Ish: 812") == []


def test_numbers_in_ignores_citations_dates_and_times() -> None:
    assert CMP.numbers_in("#c3819m7775 2026-09-21 08:12 da 17 ta, #x3") == [17.0]
    assert CMP.numbers_in("1,234 va 12.5 va 3,5") == [1234.0, 12.5, 3.5]
