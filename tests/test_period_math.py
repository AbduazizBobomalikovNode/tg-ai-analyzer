"""Sana va davr arifmetikasi: `compare_periods`, hafta kuni / soat taqsimoti, nisbiy sanalar.

Qo'riqlanadigan narsalar:

  * `timewin.comparison_kinds` / `period_window` — "bu oy o'tgan oyga
    nisbatan" → `(this_month, last_month)`, oynalar akkaunt mintaqasida;
  * `compute.compare_periods` — farq, foiz, hukm, kuniga o'rtacha; `b_n == 0`
    da foiz **yo'q** (uydirma o'rniga);
  * orkestrator: "bu oy o'tgan oyga nisbatan ko'proq yozdimmi?" →
    `[stats(a), stats(b), compute(compare_periods)]` (planner nima desa ham),
    javobdagi raqamlar faktdan (`unbacked_numbers == []`);
    "haftaning qaysi kuni ko'p yozaman" → `[stats(by=weekday)]`, eng faol kun
    fakt bo'lib;
  * haqiqiy Postgres'da taqsimot SQL'i mahalliy vaqtda (bo'lsa).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from test_orchestrator import FakeLLM, _plan, run, wired  # type: ignore[import-not-found]

from app.services import compute as CMP
from app.services import deep_search as DS
from app.services import orchestrator as OR
from app.services import scale_search as SC
from app.services import skills as SK
from app.services import timewin as TW
from app.services import window_search as WS
from app.services.prompts_deep import DEEP_SYNTH_PROMPT

__all__ = ["wired"]

NOW = datetime(2026, 9, 26, 9, 30, tzinfo=UTC)  # Toshkentda 14:30, shanba
RU_COMPARE = "в этом месяце по сравнению с прошлым я писал больше?"  # noqa: RUF001


# ─── timewin ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("bu oy o'tgan oyga nisbatan ko'proq yozdimmi?", ("this_month", "last_month")),
        ("bu hafta o'tgan haftaga qaraganda kamroq yozdimmi", ("this_week", "last_week")),
        ("this week vs last week", ("this_week", "last_week")),
        (RU_COMPARE, ("this_month", "last_month")),
        ("bu yil o'tgan yildan ko'proq xabar yozdimmi", ("this_year", "last_year")),
        ("bu oyda nechta xabar yozdim?", None),
        ("o'tgan oyda nima bo'ldi", None),
    ],
)
def test_comparison_kinds(question: str, expected: tuple[str, str] | None) -> None:
    assert TW.comparison_kinds(question) == expected


def test_period_window_uses_the_account_zone() -> None:
    this = TW.period_window("this_month", tz="Asia/Tashkent", now=NOW)
    last = TW.period_window("last_month", tz="Asia/Tashkent", now=NOW)
    assert this and last
    # 1-sentyabr 00:00 Toshkent = 31-avgust 19:00 UTC.
    assert this.since == datetime(2026, 8, 31, 19, 0, tzinfo=UTC) and this.until == NOW
    assert last.since == datetime(2026, 7, 31, 19, 0, tzinfo=UTC)
    assert last.until == this.since
    week = TW.period_window("last_week", tz="Asia/Tashkent", now=NOW)
    assert week and (week.until - week.since) == timedelta(days=7)
    assert TW.period_window("next_decade", now=NOW) is None


# ─── compute ─────────────────────────────────────────────────────────────────


def _facts(**kw: Any) -> dict[str, CMP.Fact]:
    return {k: CMP.Fact(k, v) for k, v in kw.items()}


def test_compare_periods_facts_and_zero_base() -> None:
    made = CMP.compare_periods(
        _facts(a_n=120, b_n=80, a_days=26, b_days=31, a_period="Sep", b_period="Aug")
    )
    got = {f.name: f.value for f in made}
    assert got["diff"] == 40 and got["pct_change"] == 50 and got["verdict"] == "more"
    assert got["a_per_day"] == 4.62 and got["b_per_day"] == 2.58
    assert got["pct_change_per_day"] == 78.85
    assert "Sep" in next(f.label for f in made if f.name == "pct_change")
    zero = {f.name: f.value for f in CMP.compare_periods(_facts(a_n=5, b_n=0))}
    assert zero == {"diff": 5, "verdict": "more"}  # 0 ga bo'lish — foiz yo'q
    assert CMP.run_op("compare_periods", _facts(total=3)) == (
        [],
        ["compare_periods: needs a_n and b_n (two stats steps with slot a / b)"],
    )
    assert CMP.run_op("rm -rf", {})[1] == ["unknown op 'rm -rf'"]
    # Ishora so'z bilan aytiladi — "33.3% kam" -33.33 faktiga mos.
    down = CMP.compare_periods(_facts(a_n=20, b_n=30))
    assert CMP.unbacked_numbers("30 dan 20 ga, 33.3% kam", down, extra_text="a 20 b 30") == []


def test_relative_date_facts_use_local_calendar_days() -> None:
    first = datetime(2026, 9, 1, 5, 0, tzinfo=UTC)
    last = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)  # Toshkentda 26-sentyabr 01:00
    got = {
        f.name: f.value
        for f in SK.relative_date_facts(first, last, tz="Asia/Tashkent", source="t", now=NOW)
    }
    assert got["days_since_last"] == 0 and got["last_at"] == "2026-09-26 01:00"
    assert got["days_since_first"] == 25


def test_distribution_facts_pick_the_busiest_bucket() -> None:
    agg = WS.WindowStats(by_weekday=[(1, 10), (3, 25), (5, 7)], by_hour=[(9, 4), (22, 11)])
    got = {f.name: f.value for f in SK.distribution_facts(agg)}
    assert got["top_weekday"] == "Wednesday" and got["top_weekday_n"] == 25
    assert got["wd_mon_n"] == 10 and got["wd_fri_n"] == 7
    assert got["top_hour"] == 22 and got["top_hour_n"] == 11
    assert got["hr_09_n"] == 4 and got["hr_22_n"] == 11 and "hr_10_n" not in got


def test_typed_gate_ignores_topical_questions() -> None:
    plan = DS.DeepPlan(standalone_question="q")
    topical = "ijara haqida bu oy o'tgan oyga nisbatan ko'p gapirildimi"
    assert OR.typed_stats_steps(plan, topical) is None
    assert OR.typed_stats_steps(plan, "Otabek to'lovni qaysi kuni yubordi?") is None
    steps = OR.typed_stats_steps(plan, "soat nechada eng ko'p yozaman?")
    assert steps and steps[0].args == {"by": "hour"}


# ─── orkestrator: uchdan-uchgacha ────────────────────────────────────────────


def fake_counts(monkeypatch: pytest.MonkeyPatch, totals: dict[str, int]) -> list[dict[str, Any]]:
    """`run_window` — davr nomiga qarab `count(*)`, taqsimot so'ralsa bucket'lar."""
    seen: list[dict[str, Any]] = []

    async def run_window(_f: Any, **kw: Any) -> SC.ScaleOutcome:
        plan = kw["plan"]
        seen.append({"plan": plan, "distributions": list(kw.get("distributions") or [])})
        total = totals.get(plan.window_kind, totals.get("", 0))
        until = plan.until or NOW
        spec = WS.WindowSpec(
            account_id=1, chats=None, since=plan.since, until=until, direction=plan.direction
        )
        out = SC.ScaleOutcome(stats=SC.ScanStats())
        out.window = spec
        out.window_count = WS.WindowCount(
            total=total, chats=2, first_at=until - timedelta(days=3), last_at=until
        )
        agg = WS.WindowStats(direction=plan.direction, mine_total=total)
        if "weekday" in (kw.get("distributions") or []):
            agg.by_weekday = [(1, 12), (3, 41), (6, 9)]
        out.window_agg = agg
        out.window_stats = f"window stats · {plan.window_kind or 'all'} · total {total}"
        return out

    monkeypatch.setattr(WS, "run_window", run_window)
    return seen


async def test_this_month_vs_last_month_is_stats_stats_compute(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    seen = fake_counts(monkeypatch, {"this_month": 120, "last_month": 80})
    llm = FakeLLM(
        # Planner adashsa ham (mavzu qidiruvi) — kod qadamlari.
        plan=_plan(steps=[{"skill": "search_local"}], expected_answer_kind="number"),
        synth="Ha: bu oy 120 ta, o'tgan oy 80 ta — 40 taga, ya'ni 50% ko'proq.",
    )
    events: list[DS.DeepEvent] = []
    result = await run(
        "bu oy o'tgan oyga nisbatan ko'proq yozdimmi?", llm, mode="deep", events=events
    )
    steps = [(s["skill"], s["args"]) for s in result.plan["steps"]]
    assert steps == [
        ("stats", {"period": "this_month", "slot": "a"}),
        ("stats", {"period": "last_month", "slot": "b"}),
        ("compute", {"op": "compare_periods"}),
    ]
    assert [x["plan"].window_kind for x in seen] == ["this_month", "last_month"]
    assert all(x["plan"].direction == "out" for x in seen)  # "yozdimmi" — mening xabarlarim
    meta = wired["persisted"]["context_meta"]
    facts = {f["name"]: f["value"] for f in meta["facts"]}
    assert facts["a_n"] == 120 and facts["b_n"] == 80
    assert facts["diff"] == 40 and facts["pct_change"] == 50 and facts["verdict"] == "more"
    assert "total" not in facts  # slot faktlari prefiksli — biri ikkinchisini bosmaydi
    assert meta["unbacked_numbers"] == []
    synth = llm.user_turn(DEEP_SYNTH_PROMPT)
    assert "pct_change = (a_n - b_n) / b_n * 100 = 50 %" in synth
    rail = [e.data["stages"] for e in events if e.kind == "stages"][-1]
    assert rail == ["plan", "stats", "compute", "write"]


async def test_weekday_distribution_gives_the_top_day_as_a_fact(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    seen = fake_counts(monkeypatch, {"": 62})
    llm = FakeLLM(
        plan=_plan(steps=[{"skill": "stats"}]),
        synth="Eng ko'p chorshanba kunlari yozasiz: 41 ta xabar.",
    )
    result = await run("haftaning qaysi kuni ko'p yozaman?", llm, mode="fast")
    assert [(s["skill"], s["args"]) for s in result.plan["steps"]] == [("stats", {"by": "weekday"})]
    assert seen and seen[0]["distributions"] == ["weekday"]
    facts = {f["name"]: f["value"] for f in wired["persisted"]["context_meta"]["facts"]}
    assert facts["top_weekday"] == "Wednesday" and facts["top_weekday_n"] == 41
    assert "days_since_last" in facts
    assert wired["persisted"]["context_meta"]["unbacked_numbers"] == []


async def test_question_window_beats_planner_period_for_a_single_stats(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    # Jonli uni-12: "oxirgi 7 kunda" + planner `period=last_week` — oyna o'tgan
    # kalendar haftasiga siljirdi. Savolning o'z oynasi (kodda) ustun bo'lsin.
    seen = fake_counts(monkeypatch, {"": 154, "last_n": 154})
    llm = FakeLLM(
        plan=_plan(steps=[{"skill": "stats", "args": {"period": "last_week"}}]),
        synth="Oxirgi 7 kunda 154 ta xabar yozgansiz.",
    )
    await run("oxirgi 7 kunda nechta xabar yozdim?", llm, mode="fast")
    assert seen and seen[0]["plan"].window_kind == "last_n"


# ─── haqiqiy Postgres: taqsimot SQL'i ────────────────────────────────────────


async def test_distribution_sql_uses_local_time(pg_engine: Any) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    # Dushanba 2026-09-21: Toshkentda 23:30 (UTC 18:30) va 00:30 (UTC yakshanba 19:30).
    stamps = [
        datetime(2026, 9, 21, 18, 30, tzinfo=UTC),  # Du 23:30
        datetime(2026, 9, 21, 19, 30, tzinfo=UTC),  # Se 00:30
        datetime(2026, 9, 23, 5, 0, tzinfo=UTC),  # Chor 10:00
        datetime(2026, 9, 23, 5, 40, tzinfo=UTC),  # Chor 10:40
    ]
    async with pg_engine.begin() as conn:
        uid = (
            await conn.execute(
                text(
                    "INSERT INTO users (tg_user_id, username) VALUES (900000551, 'd') RETURNING id"
                )
            )
        ).scalar_one()
        aid = (
            await conn.execute(
                text(
                    "INSERT INTO accounts (user_id, label, status, tz) "
                    "VALUES (:u, 'd', 'active', 'Asia/Tashkent') RETURNING id"
                ),
                {"u": uid},
            )
        ).scalar_one()
        chat_id = (
            await conn.execute(
                text(
                    "INSERT INTO chats (account_id, tg_peer_id, type, title, write_mode, "
                    "sync_state, synced_total, synced_min_id) VALUES (:a, "
                    "floor(random() * 1e12)::bigint, 'private', 'D', 'read_only', 'idle', 4, 1) "
                    "RETURNING id"
                ),
                {"a": aid},
            )
        ).scalar_one()
        for i, ts in enumerate(stamps):
            await conn.execute(
                text(
                    "INSERT INTO messages (chat_id, tg_msg_id, published_at, text, out) "
                    "VALUES (:c, :m, :d, 'x', true)"
                ),
                {"c": chat_id, "m": 10 + i, "d": ts},
            )
    try:
        spec = WS.WindowSpec(
            account_id=int(aid),
            chats=[int(chat_id)],
            since=None,
            until=datetime(2026, 9, 30, tzinfo=UTC),
            tz="Asia/Tashkent",
            direction="out",
        )
        async with maker() as session:
            agg = await WS.stats_window(session, spec, distributions=("hour", "weekday"))
        assert agg.by_weekday == [(1, 1), (2, 1), (3, 2)]
        assert agg.by_hour == [(0, 1), (10, 2), (23, 1)]
        block = WS._render_stats(
            WS.WindowCount(total=4, chats=1), agg, names={}, spec=spec, samples=0
        )
        assert "Wednesday 2" in block and "23h 1" in block
    finally:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
