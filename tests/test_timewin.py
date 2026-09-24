"""`services/timewin.py` — savoldagi nisbiy vaqt oynasi (uz / ru / en).

Jadval testi: har qator "savol → (kind, since, until)". Hisob **akkaunt
mintaqasida** bo'lishi kerak, shuning uchun `NOW` UTC'da berilgan va
kutilgan chegaralar Toshkent kunining boshidan hisoblanadi.

Bu modul sof: DB ham, LLM ham yo'q.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.services import timewin as TW

TZ = "Asia/Tashkent"
ZI = ZoneInfo(TZ)
#: 2026-09-24 14:05 Toshkent (payshanba) = 09:05 UTC.
NOW = datetime(2026, 9, 24, 9, 5, tzinfo=UTC)


def _local(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ZI).astimezone(UTC)


def parse(q: str, *, tz: str = TZ) -> TW.TimeWindow | None:
    return TW.parse_window(q, tz=tz, now=NOW)


# ─── (1) nisbiy nomlar ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "kind", "since", "until"),
    [
        # bugun — kun boshidan **hozirgacha** (kelajak emas)
        ("bugun menga kimlar yozgan?", "today", _local(2026, 9, 24), NOW),
        ("сегодня кто мне писал?", "today", _local(2026, 9, 24), NOW),
        ("who wrote to me today?", "today", _local(2026, 9, 24), NOW),
        # kecha — to'liq kun
        ("kecha qaysi chatlar faol edi?", "yesterday", _local(2026, 9, 23), _local(2026, 9, 24)),
        ("вчера что было?", "yesterday", _local(2026, 9, 23), _local(2026, 9, 24)),
        ("yesterday's messages", "yesterday", _local(2026, 9, 23), _local(2026, 9, 24)),
        # shu hafta — dushanbadan (2026-09-21 — dushanba)
        ("bu hafta nima bo'ldi?", "this_week", _local(2026, 9, 21), NOW),
        ("shu haftada", "this_week", _local(2026, 9, 21), NOW),
        ("на этой неделе", "this_week", _local(2026, 9, 21), NOW),
        ("this week", "this_week", _local(2026, 9, 21), NOW),
        # o'tgan hafta — to'liq oldingi hafta
        ("o'tgan hafta", "last_week", _local(2026, 9, 14), _local(2026, 9, 21)),
        ("прошлой неделе", "last_week", _local(2026, 9, 14), _local(2026, 9, 21)),
        ("last week", "last_week", _local(2026, 9, 14), _local(2026, 9, 21)),
        # oy
        ("shu oyda", "this_month", _local(2026, 9, 1), NOW),
        ("в этом месяце", "this_month", _local(2026, 9, 1), NOW),
        ("this month", "this_month", _local(2026, 9, 1), NOW),
        ("o'tgan oy", "last_month", _local(2026, 8, 1), _local(2026, 9, 1)),
        ("в прошлом месяце", "last_month", _local(2026, 8, 1), _local(2026, 9, 1)),
        ("last month", "last_month", _local(2026, 8, 1), _local(2026, 9, 1)),
        # yil
        ("bu yil", "this_year", _local(2026, 1, 1), NOW),
        ("shu yilda", "this_year", _local(2026, 1, 1), NOW),
        ("в этом году", "this_year", _local(2026, 1, 1), NOW),
        ("this year", "this_year", _local(2026, 1, 1), NOW),
        ("o'tgan yil", "last_year", _local(2025, 1, 1), _local(2026, 1, 1)),
        ("last year", "last_year", _local(2025, 1, 1), _local(2026, 1, 1)),
    ],
)
def test_relative_names(question: str, kind: str, since: datetime, until: datetime) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == kind
    assert win.since == since
    assert win.until == until
    assert win.tz == TZ


# ─── (2) "butun davr" — quyi chegara yo'q ────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "butun davr bo'yicha ijara narxlari qanday o'zgargan?",
        "butun tarix davomida",
        "hammasi",
        "barchasi bo'yicha",
        "eng boshidan boshlab ko'r",
        "за всё время",
        "всё время что обсуждали",
        "с самого начала",
        "all time",
        "all-time stats",
        "from the very beginning",
        "entire history",
    ],
)
def test_all_time_has_no_lower_bound(question: str) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == "all_time"
    assert win.since is None
    assert win.open_ended is True
    assert win.days == float("inf")
    assert win.until == NOW


# ─── (3) "oxirgi N <birlik>" — daqiqa · soat · kun · hafta · oy · yil ────────


@pytest.mark.parametrize(
    ("question", "since"),
    [
        ("oxirgi 30 daqiqa", NOW - timedelta(minutes=30)),
        ("oxirgi 30 daqiqada nima yozildi", NOW - timedelta(minutes=30)),
        ("последние 30 минут", NOW - timedelta(minutes=30)),
        ("last 30 minutes", NOW - timedelta(minutes=30)),
        ("oxirgi 5 soatda menga kimlar yozgan?", NOW - timedelta(hours=5)),
        ("за последние 5 часов", NOW - timedelta(hours=5)),
        ("last 5 hours", NOW - timedelta(hours=5)),
        # kun/hafta — **kun boshidan** (soat emas)
        ("oxirgi 4 kun", _local(2026, 9, 21)),
        ("oxirgi 3 kunda kommunal haqida nima deyishdi?", _local(2026, 9, 22)),
        ("последние 4 дня", _local(2026, 9, 21)),
        ("last 4 days", _local(2026, 9, 21)),
        ("oxirgi 2 hafta", _local(2026, 9, 10)),
        ("последние 2 недели", _local(2026, 9, 10)),
        ("last 2 weeks", _local(2026, 9, 10)),
        ("oxirgi 2 oy", _local(2026, 7, 1)),
        ("последние 2 месяца", _local(2026, 7, 1)),
        ("last 2 months", _local(2026, 7, 1)),
        ("oxirgi 2 yilda eng faol chatlar qaysi?", _local(2024, 9, 1)),
        ("за последние 2 года", _local(2024, 9, 1)),
        ("last 2 years", _local(2024, 9, 1)),
        # "3 kun ichida"
        ("3 kun ichida nima bo'ldi", _local(2026, 9, 22)),
    ],
)
def test_last_n_units(question: str, since: datetime) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == "last_n"
    assert win.since == since
    assert win.until == NOW


@pytest.mark.parametrize(
    ("question", "since"),
    [
        ("oxirgi ikki hafta", _local(2026, 9, 10)),
        ("so'nggi uch kun", _local(2026, 9, 22)),
        ("oxirgi besh soat", NOW - timedelta(hours=5)),
        ("последние две недели", _local(2026, 9, 10)),
        ("за последние три дня", _local(2026, 9, 22)),
        ("last two weeks", _local(2026, 9, 10)),
        ("last three days", _local(2026, 9, 22)),
    ],
)
def test_word_numerals(question: str, since: datetime) -> None:
    """1-10 oralig'idagi son so'z bilan ham yozilishi mumkin."""
    win = parse(question)
    assert win is not None, question
    assert win.since == since


# ─── (4) "…dan beri" ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "kind", "since"),
    [
        ("5 kundan beri", "since_n", _local(2026, 9, 20)),
        ("ikki oydan buyon", "since_n", _local(2026, 7, 1)),
        ("sentyabrdan beri", "since_month", _local(2026, 9, 1)),
        ("noyabrdan beri", "since_month", _local(2025, 11, 1)),
        ("с сентября", "since_month", _local(2026, 9, 1)),
        ("since september", "since_month", _local(2026, 9, 1)),
        ("2024 dan beri", "since_year", _local(2024, 1, 1)),
        ("2024 yildan boshlab", "since_year", _local(2024, 1, 1)),
        ("с 2024", "since_year", _local(2024, 1, 1)),
        ("since 2024", "since_year", _local(2024, 1, 1)),
    ],
)
def test_since_forms(question: str, kind: str, since: datetime) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == kind
    assert win.since == since
    assert win.until == NOW


def test_shu_haftadan_boshlab_is_this_week() -> None:
    win = parse("shu haftadan boshlab nima bo'ldi")
    assert win is not None
    assert win.since == _local(2026, 9, 21)
    assert win.until == NOW


# ─── (5) aniq sanalar va oraliqlar ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "since", "until"),
    [
        ("2026-09-21 kuni", _local(2026, 9, 21), _local(2026, 9, 22)),
        ("21.09.2026", _local(2026, 9, 21), _local(2026, 9, 22)),
        ("21 sen", _local(2026, 9, 21), _local(2026, 9, 22)),
        ("21 sentyabr", _local(2026, 9, 21), _local(2026, 9, 22)),
        ("21 сентября", _local(2026, 9, 21), _local(2026, 9, 22)),
        ("21 september", _local(2026, 9, 21), _local(2026, 9, 22)),
        # yil aytilmagan va sana kelajakda — o'tgan yil
        ("21 dekabr", _local(2025, 12, 21), _local(2025, 12, 22)),
    ],
)
def test_explicit_dates(question: str, since: datetime, until: datetime) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == "date"
    assert (win.since, win.until) == (since, until)


@pytest.mark.parametrize(
    "question",
    ["15-20 sentyabr", "15–20 sentyabr", "15 - 20 сентября", "15-20 september"],
)
def test_ranges(question: str) -> None:
    win = parse(question)
    assert win is not None, question
    assert win.kind == "range"
    assert win.since == _local(2026, 9, 15)
    assert win.until == _local(2026, 9, 21)  # 20-kun **ichida**


def test_iso_range() -> None:
    win = parse("2026-09-15 — 2026-09-20 oralig'ida")
    assert win is not None
    assert win.kind == "range"
    assert (win.since, win.until) == (_local(2026, 9, 15), _local(2026, 9, 21))


# ─── (6) mintaqa ─────────────────────────────────────────────────────────────


def test_today_is_computed_in_the_account_timezone() -> None:
    """Toshkentda 03:00 — UTC'da hali kechagi kun. "Bugun" lokal kun."""
    early = datetime(2026, 9, 23, 22, 30, tzinfo=UTC)  # 2026-09-24 03:30 Toshkent
    win = TW.parse_window("bugun", tz=TZ, now=early)
    assert win is not None
    assert win.since == _local(2026, 9, 24)
    utc_win = TW.parse_window("bugun", tz="UTC", now=early)
    assert utc_win is not None
    assert utc_win.since == datetime(2026, 9, 23, tzinfo=UTC)


def test_unknown_timezone_falls_back_without_raising() -> None:
    win = TW.parse_window("bugun", tz="Mars/Olympus", now=NOW)
    assert win is not None
    assert win.tz == TW.DEFAULT_TZ


# ─── (7) vaqt havolasi yo'q ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "Nurbek bilan to'lov haqida nima kelishgandik?",
        "кто отвечает за деплой?",
        "what did we decide about the invoice?",
        "salom",
        "",
    ],
)
def test_no_time_reference_returns_none(question: str) -> None:
    assert parse(question) is None


def test_apostrophe_variants_are_folded() -> None:
    for apo in ("'", "’", "ʻ", "`"):
        win = parse(f"o{apo}tgan hafta")
        assert win is not None, apo
        assert win.kind == "last_week"


def test_format_window_is_readable() -> None:
    win = parse("bugun")
    assert win is not None
    text = TW.format_window(win)
    assert "2026-09-24 00:00" in text
    assert "14:05" in text
    assert TZ in text
    all_time = parse("butun davr")
    assert all_time is not None
    assert TW.format_window(all_time).startswith("…")
