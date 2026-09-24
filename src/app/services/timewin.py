"""Savoldagi **nisbiy vaqt oynasini** deterministik tahlil qilish (uz / ru / en).

Nega alohida modul va nega LLM'siz: "bugun menga kimlar yozgan?" degan savolda
oyna savolning **o'zida** turadi. Planner uni ba'zan topadi, ba'zan yo'q —
jonli yugurishda u `window: null` qaytardi va `exhaustive` butun korpusni
haftalik bucket'lar bilan rejalashtirib, ustiga "vaqt oralig'ini savolda
ko'rsating" deb ogohlantirdi. Savolda oraliq **bor** edi.

Shuning uchun oyna kodda hisoblanadi (`parse_window`) va `parse_plan` uni
modelning javobi ustidan **qo'yadi**: model bo'sh qoldirgan yoki kengroq
bergan bo'lsa — kodniki, aniqrog'i (torroq) bergan bo'lsa — modelniki.

Barcha hisob **akkaunt mintaqasida** (`accounts.tz`, default `Asia/Tashkent`)
bajariladi va natija UTC'ga o'giriladi: "bugun" Toshkentda soat 03:00 da ham
o'sha kunning 00:00 idan boshlanadi, UTC kuni esa hali kechagi.

`since is None` — **"butun davr"**: quyi chegara yo'q, ya'ni sinxronlangan
tarixning boshidan. Bu "oyna yo'q" degani emas: `parse_window` `None`
qaytarganda savolda vaqt havolasi umuman yo'q, `TimeWindow(since=None)`
qaytarganda esa foydalanuvchi aynan "hammasi" deb so'ragan.

Modul sof: DB ham, LLM ham, tarmoq ham yo'q — shuning uchun uni jadval bilan
sinash mumkin (`tests/test_timewin.py`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

log = structlog.get_logger(__name__)

DEFAULT_TZ = "Asia/Tashkent"

#: O'zbek apostrofi har xil ko'chadi (to'g'ri tirnoq, modifikator, teskari).
_APO_RE = re.compile(r"[‘’ʻʼ`´]")

#: Oy nomlari — uz (to'liq + qisqa), ru (bosh/qaratqich + qisqa), en.
_MONTHS: dict[str, int] = {
    # uz
    "yanvar": 1, "yan": 1,
    "fevral": 2, "fev": 2,
    "mart": 3, "mar": 3,
    "aprel": 4, "apr": 4,
    "may": 5,
    "iyun": 6, "iyn": 6,
    "iyul": 7, "iyl": 7,
    "avgust": 8, "avg": 8,
    "sentyabr": 9, "sentabr": 9, "sen": 9, "sent": 9,
    "oktyabr": 10, "oktabr": 10, "okt": 10,
    "noyabr": 11, "noy": 11,
    "dekabr": 12, "dek": 12,
    # ru — bosh kelishik va qaratqich ("21 сентября")
    "январь": 1, "января": 1, "янв": 1,
    "февраль": 2, "февраля": 2, "фев": 2,
    "март": 3, "марта": 3,
    "апрель": 4, "апреля": 4,
    "мая": 5, "май": 5,
    "июнь": 6, "июня": 6, "июн": 6,
    "июль": 7, "июля": 7, "июл": 7,
    "август": 8, "августа": 8,
    "сентябрь": 9, "сентября": 9, "сент": 9,
    "октябрь": 10, "октября": 10,
    "ноябрь": 11, "ноября": 11, "ноя": 11,
    "декабрь": 12, "декабря": 12,
    # en
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,
    "april": 4,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}  # fmt: skip

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: 1..10 — so'z bilan yozilgan sonlar (uz / ru / en). "oxirgi ikki hafta".
_WORD_NUMS: dict[str, int] = {
    "bir": 1, "ikki": 2, "uch": 3, "to'rt": 4, "tort": 4, "besh": 5,
    "olti": 6, "yetti": 7, "sakkiz": 8, "to'qqiz": 9, "toqqiz": 9, "o'n": 10, "on": 10,
    "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
_NUM_ALT = r"\d{1,4}|" + "|".join(sorted(_WORD_NUMS, key=len, reverse=True))

#: Birliklar — `_unit_of` prefiks bo'yicha tanlaydi.
_UNIT_ALT = (
    r"daqiqa\w*|daqqa\w*|minut\w*|мин\w*|minutes|minute|mins|min"
    r"|soat\w*|час\w*|hours|hour|hrs|hr"
    r"|kun\w*|дн\w*|день|days|day"
    r"|hafta\w*|недел\w*|weeks|week"
    r"|oy\w*|месяц\w*|months|month"
    r"|yil\w*|год\w*|лет|years|year"
)

#: `kind` qiymatlari — qamrov va loglar shu nom bilan yuradi.
KINDS = (
    "all_time",
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
    "this_year",
    "last_year",
    "last_n",
    "since_n",
    "since_month",
    "since_year",
    "date",
    "range",
)


@dataclass(slots=True, frozen=True)
class TimeWindow:
    """Savoldan olingan oyna. `since`/`until` — **UTC**, yarim ochiq [since, until).

    `since is None` — quyi chegara yo'q ("butun davr"): sinxronlangan
    tarixning boshidan. Bu holatda `days` cheksiz va hech bir bosqich uni
    "tor oyna" deb hisoblamaydi.
    """

    since: datetime | None
    until: datetime
    kind: str
    #: Savoldagi aynan o'sha matn bo'lagi (log va tushuntirish uchun).
    phrase: str = ""
    #: Hisob bajarilgan mintaqa nomi — qamrov qatorida ko'rsatiladi.
    tz: str = DEFAULT_TZ

    @property
    def days(self) -> float:
        if self.since is None:
            return math.inf
        return (self.until - self.since).total_seconds() / 86400.0

    @property
    def open_ended(self) -> bool:
        return self.since is None

    def to_dict(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat(),
            "kind": self.kind,
            "phrase": self.phrase,
            "tz": self.tz,
            "days": None if self.since is None else round(self.days, 4),
        }


def zone(tz: str | None) -> ZoneInfo:
    """`accounts.tz` — noto'g'ri bo'lsa ham quvur yiqilmaydi (default mintaqa)."""
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("timewin.bad_tz", tz=tz)
        return ZoneInfo(DEFAULT_TZ)


def fold(text: str) -> str:
    """Apostrofni bir xillashtiradi va kichik harfga tushiradi."""
    return _APO_RE.sub("'", (text or "")).lower()


def _day_start(when: datetime) -> datetime:
    return when.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(when: datetime) -> datetime:
    return _day_start(when).replace(day=1)


def _shift_months(when: datetime, months: int) -> datetime:
    total = (when.year * 12 + when.month - 1) + months
    year, month = divmod(total, 12)
    return when.replace(year=year, month=month + 1, day=1)


def _utc(local: datetime) -> datetime:
    return local.astimezone(UTC)


def _num(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return _WORD_NUMS.get(raw, 0)


def _unit_of(raw: str) -> str:
    """Birlik prefiksi → kanonik nom. Tartib muhim: `min` va `мин` — daqiqa."""
    if raw.startswith(("daqiqa", "daqqa", "minut", "мин", "minute", "mins", "min")):
        return "minute"
    if raw.startswith(("soat", "час", "hour", "hrs", "hr")):
        return "hour"
    if raw.startswith(("kun", "дн", "день", "day")):
        return "day"
    if raw.startswith(("hafta", "недел", "week")):
        return "week"
    if raw.startswith(("oy", "месяц", "month")):
        return "month"
    return "year"


# ─── naqshlar ────────────────────────────────────────────────────────────────
# Tartib muhim: "o'tgan hafta" ichida "hafta" bor, shuning uchun aniqroq
# (uzunroq) naqshlar birinchi tekshiriladi.

_ALL_TIME_RE = re.compile(
    r"\bbutun\s+(?:davr\w*|vaqt\w*|tarix\w*|arxiv\w*)\b"
    r"|\bhamma\s+vaqt\w*\b|\bbarcha\s+vaqt\w*\b"
    r"|\bhammasi\b|\bbarchasi\b"
    r"|\b(?:eng\s+)?boshidan\b|\bboshlangani\w*\s+beri\b"
    r"|\bза\s+вс[её]\s+врем\w*\b|\bвс[её]\s+врем\w*\b|\bс\s+самого\s+начала\b"
    r"|\bза\s+вс[юе]\s+истори\w*\b"
    r"|\ball[\s-]time\b|\ball\s+the\s+time\b|\bentire\s+histor\w*\b"
    r"|\bfrom\s+the\s+(?:very\s+)?beginning\b|\bwhole\s+(?:archive|histor\w*|period)\b"
)

_TODAY_RE = re.compile(r"\b(bugun\w*|сегодня\w*|сегодняшн\w*|today)\b")
_YESTERDAY_RE = re.compile(r"\b(kecha|kechagi|kechada|вчера\w*|вчерашн\w*|yesterday)\b")
_LAST_WEEK_RE = re.compile(
    r"\b(?:o't(?:gan|kan)|utgan)\s+hafta\w*\b"
    r"|\b(?:на\s+|с\s+)?прошл\w+\s+недел\w+\b"
    r"|\blast\s+week\b"
)
_THIS_WEEK_RE = re.compile(
    r"\b(?:bu|shu|joriy)\s+hafta\w*\b"
    r"|\bhaftalik\b"
    r"|\b(?:на\s+|с\s+)?эт(?:ой|у)\s+недел\w+\b"
    r"|\bза\s+недел\w+\b"
    r"|\bthis\s+week\b"
)
_LAST_MONTH_RE = re.compile(
    r"\b(?:o't(?:gan|kan)|utgan)\s+oy\w*\b"
    r"|\b(?:в\s+|с\s+)?прошл\w+\s+месяц\w*\b"
    r"|\blast\s+month\b"
)
_THIS_MONTH_RE = re.compile(
    r"\b(?:bu|shu|joriy)\s+oy\w*\b"
    r"|\b(?:в\s+|с\s+)?эт\w+\s+месяц\w*\b"
    r"|\bза\s+месяц\b"
    r"|\bthis\s+month\b"
)
_LAST_YEAR_RE = re.compile(
    r"\b(?:o't(?:gan|kan)|utgan)\s+yil\w*\b"
    r"|\b(?:в\s+|с\s+)?прошл\w+\s+год\w*\b"
    r"|\blast\s+year\b"
)
_THIS_YEAR_RE = re.compile(
    r"\b(?:bu|shu|joriy)\s+yil\w*\b"
    r"|\b(?:в\s+|с\s+)?эт\w+\s+год\w*\b"
    r"|\bза\s+год\b"
    r"|\bthis\s+year\b"
)
#: "oxirgi 3 kun", "so'nggi ikki haftada", "последние 5 дней", "last 3 weeks",
#: "oxirgi 5 soat", "oxirgi 30 daqiqa".
_LAST_N_RE = re.compile(
    rf"\b(?:oxirgi|so'nggi|songgi|(?:за\s+)?последни\w+|past|last)\s+({_NUM_ALT})\s*({_UNIT_ALT})\b"
)
#: "3 kun ichida", "2 hafta ichida", "30 daqiqa ichida" — son oldinda.
_N_WITHIN_RE = re.compile(rf"\b({_NUM_ALT})\s+({_UNIT_ALT})\s+ichida\b")
#: "5 kundan beri", "ikki oydan buyon", "3 дня назад" emas — faqat "beri/buyon".
_SINCE_N_RE = re.compile(
    rf"\b({_NUM_ALT})\s*({_UNIT_ALT})dan\s+(?:beri|buyon|boshlab)\b"
    rf"|\b({_NUM_ALT})\s*({_UNIT_ALT})\s+(?:beri|buyon)\b"
)
_SINCE_MONTH_RE = re.compile(
    rf"\b({_MONTH_ALT})\w*dan\s+(?:beri|buyon|boshlab)\b"
    rf"|\bс\s+({_MONTH_ALT})\b"
    rf"|\bsince\s+({_MONTH_ALT})\b"
)
_SINCE_YEAR_RE = re.compile(
    r"\b(\d{4})\s*(?:-)?\s*(?:yil\w*)?\s*dan\s+(?:beri|buyon|boshlab)\b"
    r"|\bс\s+(\d{4})(?:\s*г\w*)?\b"
    r"|\bsince\s+(\d{4})\b"
)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOTTED_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})[-\s]\s*({_MONTH_ALT})\b(?:\s+(\d{{4}}))?")
#: "15–20 sentyabr", "15-20 сентября", "15 - 20 sep".
_RANGE_DM_RE = re.compile(
    rf"\b(\d{{1,2}})\s*[-–—]\s*(\d{{1,2}})\s*({_MONTH_ALT})\b(?:\s+(\d{{4}}))?"
)
_RANGE_ISO_RE = re.compile(
    r"\b(\d{4})-(\d{1,2})-(\d{1,2})\s*[-–—]{1,2}\s*(\d{4})-(\d{1,2})-(\d{1,2})\b"
)

_UNIT_DELTA = {
    "minute": lambda n: timedelta(minutes=n),
    "hour": lambda n: timedelta(hours=n),
    "day": lambda n: timedelta(days=n),
    "week": lambda n: timedelta(weeks=n),
}


def _safe_date(year: int, month: int, day: int, tz: ZoneInfo) -> datetime | None:
    try:
        return datetime(year, month, day, tzinfo=tz)
    except ValueError:
        return None


def _back(local_now: datetime, count: int, unit: str) -> datetime:
    """`count` birlik orqaga. Kun/hafta — **kun boshidan**, soat/daqiqa — aniq."""
    if unit == "minute" or unit == "hour":
        return local_now - _UNIT_DELTA[unit](count)
    if unit == "day":
        # "oxirgi 3 kun" — bugun ham kiradi: 2 to'liq kun + bugun.
        return _day_start(local_now) - timedelta(days=count - 1)
    if unit == "week":
        return _day_start(local_now) - timedelta(weeks=count)
    if unit == "month":
        return _shift_months(_day_start(local_now), -count)
    return _shift_months(_day_start(local_now), -12 * count)


def parse_window(
    question: str,
    *,
    tz: str | None = DEFAULT_TZ,
    now: datetime | None = None,
) -> TimeWindow | None:
    """Savoldagi vaqt oynasi yoki `None` (savolda vaqt havolasi yo'q).

    `now` — sinov uchun; berilmasa hozirgi vaqt. Natija har doim UTC'da,
    yarim ochiq oraliq `[since, until)`; `since is None` — "butun davr".

    Qoidalar ataylab **tor**: shubha bo'lsa `None` qaytadi va reja o'z
    oynasi bilan qoladi. Noto'g'ri "bugun" — yo'qolgan javob; topilmagan
    "bugun" — faqat kengroq skan.
    """
    zi = zone(tz)
    local_now = (now or datetime.now(UTC)).astimezone(zi)
    q = fold(question)
    if not q:
        return None

    def win(since: datetime | None, until: datetime, kind: str, phrase: str) -> TimeWindow:
        return TimeWindow(
            since=_utc(since) if since is not None else None,
            until=_utc(until),
            kind=kind,
            phrase=phrase,
            tz=str(zi),
        )

    # ── (0) "butun davr" — quyi chegara yo'q ────────────────────────────────
    m = _ALL_TIME_RE.search(q)
    if m:
        return win(None, local_now, "all_time", m[0])

    # ── (1) aniq oraliqlar — eng aniq, shuning uchun birinchi ───────────────
    m = _RANGE_ISO_RE.search(q)
    if m:
        a = _safe_date(int(m[1]), int(m[2]), int(m[3]), zi)
        b = _safe_date(int(m[4]), int(m[5]), int(m[6]), zi)
        if a and b:
            lo, hi = sorted((a, b))
            return win(lo, hi + timedelta(days=1), "range", m[0])

    m = _RANGE_DM_RE.search(q)
    if m:
        month = _MONTHS[m[3]]
        year = int(m[4]) if m[4] else local_now.year
        a = _safe_date(year, month, int(m[1]), zi)
        b = _safe_date(year, month, int(m[2]), zi)
        if a and b:
            lo, hi = sorted((a, b))
            return win(lo, hi + timedelta(days=1), "range", m[0])

    # ── (2) aniq sanalar ────────────────────────────────────────────────────
    m = _ISO_RE.search(q)
    if m:
        day = _safe_date(int(m[1]), int(m[2]), int(m[3]), zi)
        if day:
            return win(day, day + timedelta(days=1), "date", m[0])

    m = _DOTTED_RE.search(q)
    if m:
        day = _safe_date(int(m[3]), int(m[2]), int(m[1]), zi)
        if day:
            return win(day, day + timedelta(days=1), "date", m[0])

    # ── (3) "…dan beri" — ochiq o'ng chegara (hozirgacha) ───────────────────
    # Yil birinchi: "2024 yildan boshlab" da `2024` **sanoq emas, yil**
    # (aks holda u "oxirgi 2024 yil" bo'lib o'qilardi).
    m = _SINCE_YEAR_RE.search(q)
    if m:
        year = int(m[1] or m[2] or m[3])
        if 1990 <= year <= local_now.year:
            start = _safe_date(year, 1, 1, zi)
            if start:
                return win(start, local_now, "since_year", m[0])

    m = _SINCE_N_RE.search(q)
    if m:
        raw_n = m[1] or m[3] or ""
        raw_u = m[2] or m[4] or ""
        count = _num(raw_n)
        if count > 0:
            return win(
                _back(local_now, min(count, 3650), _unit_of(raw_u)), local_now, "since_n", m[0]
            )

    m = _SINCE_MONTH_RE.search(q)
    if m:
        month = _MONTHS[m[1] or m[2] or m[3]]
        year = local_now.year if month <= local_now.month else local_now.year - 1
        start = _safe_date(year, month, 1, zi)
        if start:
            return win(start, local_now, "since_month", m[0])

    m = _DAY_MONTH_RE.search(q)
    if m:
        month = _MONTHS[m[2]]
        year = int(m[3]) if m[3] else local_now.year
        day = _safe_date(year, month, int(m[1]), zi)
        if day:
            # Yil aytilmagan va sana kelajakda qolsa — o'tgan yil nazarda
            # tutilgan ("21 dekabr" 5-yanvarda so'ralsa).
            if not m[3] and day > local_now + timedelta(days=1):
                day = day.replace(year=year - 1)
            return win(day, day + timedelta(days=1), "date", m[0])

    # ── (4) "oxirgi N <birlik>" ─────────────────────────────────────────────
    m = _LAST_N_RE.search(q) or _N_WITHIN_RE.search(q)
    if m:
        count = _num(m[1])
        if count > 0:
            unit = _unit_of(m[2])
            return win(_back(local_now, min(count, 3650), unit), local_now, "last_n", m[0])

    # ── (5) nisbiy nomlar ───────────────────────────────────────────────────
    m = _YESTERDAY_RE.search(q)
    if m:
        start = _day_start(local_now) - timedelta(days=1)
        return win(start, start + timedelta(days=1), "yesterday", m[0])

    m = _TODAY_RE.search(q)
    if m:
        return win(_day_start(local_now), local_now, "today", m[0])

    m = _LAST_WEEK_RE.search(q)
    if m:
        this_week = _day_start(local_now) - timedelta(days=local_now.weekday())
        return win(this_week - timedelta(days=7), this_week, "last_week", m[0])

    m = _THIS_WEEK_RE.search(q)
    if m:
        this_week = _day_start(local_now) - timedelta(days=local_now.weekday())
        return win(this_week, local_now, "this_week", m[0])

    m = _LAST_MONTH_RE.search(q)
    if m:
        this_month = _month_start(local_now)
        return win(_shift_months(this_month, -1), this_month, "last_month", m[0])

    m = _THIS_MONTH_RE.search(q)
    if m:
        return win(_month_start(local_now), local_now, "this_month", m[0])

    m = _LAST_YEAR_RE.search(q)
    if m:
        this_year = _day_start(local_now).replace(month=1, day=1)
        return win(this_year.replace(year=this_year.year - 1), this_year, "last_year", m[0])

    m = _THIS_YEAR_RE.search(q)
    if m:
        return win(_day_start(local_now).replace(month=1, day=1), local_now, "this_year", m[0])

    return None


def format_window(win: TimeWindow, *, tz: str | None = None) -> str:
    """Qamrov qatoridagi ko'rinish: `2026-09-24 00:00 → 14:05 (Asia/Tashkent)`."""
    zi = zone(tz or win.tz)
    b = win.until.astimezone(zi)
    if win.since is None:
        return f"… → {b:%Y-%m-%d %H:%M} ({zi})"
    a = win.since.astimezone(zi)
    if (b - a) <= timedelta(days=1) and a.date() == b.date():
        return f"{a:%Y-%m-%d %H:%M} → {b:%H:%M} ({zi})"
    return f"{a:%Y-%m-%d %H:%M} → {b:%Y-%m-%d %H:%M} ({zi})"


__all__ = [
    "DEFAULT_TZ",
    "KINDS",
    "TimeWindow",
    "fold",
    "format_window",
    "parse_window",
    "zone",
]
