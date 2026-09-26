"""`format_output` ko'nikmasi — faktlardan jadval / diagramma / raqamli ro'yxat.

Nega kodda (LLM'siz): foydalanuvchi "jadval qil", "diagramma", "grafik",
"таблица", "chart" desa, model raqamlarni **qayta yozib** jadval yasardi va
har qayta yozishda xato (yoki o'ylab topilgan son) chiqish xavfi bor edi.
Bu modul faqat **kodda hisoblangan** faktlardan (`compute.Fact`) va
`count(*)` sanoqlaridan blok chizadi — raqam hech qachon o'ylab topilmaydi.

Chiqish — markdown bloki (GFM jadval, ```chart fence yoki raqamli ro'yxat).
U sintez kontekstiga `pre-rendered block` bo'lib qo'shiladi
(`wrap_block`), sintezator uni o'zgartirmasdan javobga ko'chirishi kerak;
`ensure_blocks` esa sintezdan keyin blok javobda yo'q bo'lsa uni oxiriga
qo'shadi (orkestrator ulagichi — model buyruqqa bo'ysunmasa ham blok
yo'qolmasin).

```chart spetsifikatsiyasi — `web/static/md.js::chartSvg` bilan **bir xil**:
`{"type": "bar"|"line", "title": str, "labels": [str], "series":
[{"name": str, "values": [number]}]}`; ≤60 yorliq, ≤6 seriya. `pie`
md.js'da bar bo'lib chiziladi — shuning uchun biz uni to'g'ridan-to'g'ri
`bar` deb yozamiz (va'da qilinmagan narsa chizilmasin).

Begona matn: chat sarlavhalari `untrusted.attr` dan o'tgan holda
keladi (`skills._stats` faktlari yorlig'ida); jadval katagida `|` va qator
uzilishi qo'shimcha zararsizlantiriladi. Xabar **matni** blokka hech qachon
tushmaydi.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.services import untrusted as U
from app.services.compute import Fact, fmt_number

# ─── qopqoqlar (md.js bilan bir xil) ─────────────────────────────────────────
CHART_MAX_LABELS = 60
CHART_MAX_SERIES = 6
TABLE_MAX_ROWS = 50
CELL_MAX_CHARS = 60
#: Oyna berilmasa kunlik seriya shuncha kunni oladi (izoh bilan).
DEFAULT_SERIES_DAYS = 14

FORMATS = ("table", "chart", "list")
CHART_TYPES = ("bar", "line")

#: Sintez kontekstidagi blok chegaralari (`wrap_block` / `extract_blocks`).
PRE_START = (
    "pre_rendered_block (format_output — built by code from exact numbers; copy it into the "
    "answer VERBATIM, do not edit, reorder or recompute it):"
)
PRE_END = "end_pre_rendered_block"

#: `SkillResult.notes` → sintez prompt'iga satr (orkestrator ulaganda).
FORMAT_NOTE = (
    "format_output: the context holds a pre-rendered block (table/chart/list). Include it in "
    "the answer exactly as given; you may add one short sentence before it. Do not restate "
    "its numbers differently."
)

# ─── so'z darvozalari (uz / ru / en), `TW.fold` qilingan matn ustida ─────────
_TABLE_RE = re.compile(r"\b(jadval\w*|таблиц\w*|table|tabular)\b")
_CHART_RE = re.compile(
    r"\b(diagramm\w*|grafi[kg]\w*|chart\w*|график\w*|диаграмм\w*|graph\w*|plot\w*|gistogramm\w*)\b"
)
_LIST_RE = re.compile(r"\b(ro'yxat\w*|sanab\w*|списк\w*|список|numbered\s+list|list)\b")
#: Vaqt bo'yicha seriya: kunlik dinamika, trend.
_TIME_SERIES_RE = re.compile(
    r"\b(kunlik|kun\s+bo'yicha|kunlar\s+bo'yicha|dinamika\w*|trend\w*|haftalik|"
    r"по\s+дням|динамик\w*|по\s+неделям|ежедневн\w*|daily|per\s+day|by\s+day|over\s+time|"
    r"timeline|line)\b"
)
_PIE_RE = re.compile(r"\b(doira\w*|pie|кругов\w*)\b")

_TOP_NAME_RE = re.compile(r"^top(\d+)_n$")
_TOP_LABEL_RE = re.compile(r'#\d+\s+"(.*)"\s*$')


def _fold(text: str) -> str:
    from app.services import timewin as TW

    return TW.fold(text or "")


@dataclass(frozen=True, slots=True)
class FormatChoice:
    fmt: str  # table | chart | list
    chart: str = "bar"  # bar | line
    time_series: bool = False
    pie_requested: bool = False


def detect_format(
    text: str, *, expected_kind: str = "", arg_format: str = "", arg_chart: str = ""
) -> FormatChoice | None:
    """Foydalanuvchi so'zlari (yoki planner argumenti) → format. Bilmasa `None`.

    Tartib: aniq argument > so'z (diagramma > jadval > ro'yxat) >
    `expected_answer_kind` (table/list).
    """
    folded = _fold(text)
    time_series = bool(_TIME_SERIES_RE.search(folded))
    pie = bool(_PIE_RE.search(folded)) or arg_chart == "pie"
    fmt = arg_format if arg_format in FORMATS else ""
    if not fmt:
        if _CHART_RE.search(folded) or pie:
            fmt = "chart"
        elif _TABLE_RE.search(folded):
            fmt = "table"
        elif _LIST_RE.search(folded):
            fmt = "list"
        elif expected_kind in ("table", "list"):
            fmt = expected_kind
    if not fmt:
        return None
    chart = arg_chart if arg_chart in CHART_TYPES else ("line" if time_series else "bar")
    if pie:
        chart = "bar"  # md.js pie chizmaydi — bar bo'lib chiqadi
    # "grafik" (uz) odatda vaqt bo'yicha chiziq: so'z bor-u seriya aytilmagan bo'lsa ham.
    if fmt == "chart" and re.search(r"\b(grafi[kg]\w*|график\w*)\b", folded) and not pie:
        if arg_chart not in CHART_TYPES:
            chart = "line"
        time_series = True
    return FormatChoice(fmt=fmt, chart=chart, time_series=time_series, pie_requested=pie)


# ─── faktlardan qatorlar ─────────────────────────────────────────────────────


def cell(value: Any, *, max_chars: int = CELL_MAX_CHARS) -> str:
    """GFM jadval katagi: bir qator, `|` qochirilgan, zararsizlantirilgan."""
    text = U.neutralize(str(value if value is not None else ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text.replace("\\", "\\\\").replace("|", "\\|")


def ranking_rows(facts: Sequence[Fact]) -> list[tuple[str, float | int]]:
    """`top{i}_n` faktlari (`skills._stats`) → `(nom, son)` tartibda."""
    found: list[tuple[int, str, float | int]] = []
    for f in facts:
        m = _TOP_NAME_RE.match(f.name)
        if not m or not f.numeric:
            continue
        lm = _TOP_LABEL_RE.search(f.label or "")
        name = lm.group(1) if lm else f.label or f.name
        found.append((int(m.group(1)), name, f.value))  # type: ignore[arg-type]
    found.sort(key=lambda x: x[0])
    return [(name, value) for _, name, value in found]


#: `stats(by=weekday|hour)` faktlari (`skills.distribution_facts`): `wd_<kun>_n`, `hr_<HH>_n`.
_WD_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WD_LABELS = {
    "uz": ("Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"),
    "ru": ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}
_HOUR_NAME_RE = re.compile(r"^hr_(\d{2})_n$")
_HOUR_WORD_RE = re.compile(r"\b(soat\w*|hour\w*|час\w*|время)\b")


def distribution_rows(
    facts: Sequence[Fact], *, locale: str = "uz", prefer: str = ""
) -> tuple[str, list[tuple[str, float | int]]]:
    """Hafta kuni / soat taqsimoti → `(tur, [(yorliq, son)])` (tur: weekday|hour|"").

    Ilgari `format_output` bu faktlarni ko'rmasdi: "haftaning qaysi kuni ko'p
    yozaman? jadval" da blok **chat reytingi** bo'lib chiqardi (jonli, conv 382).
    `prefer` — "weekday"/"hour" yoki savol matni (soat so'zi bo'lsa soat).
    """
    by_name = {f.name: f for f in facts if f.numeric}
    labels = _WD_LABELS.get(locale, _WD_LABELS["uz"])
    wd: list[tuple[str, float | int]] = [
        (labels[i], by_name[f"wd_{k}_n"].value)  # type: ignore[misc]
        for i, k in enumerate(_WD_KEYS)
        if f"wd_{k}_n" in by_name
    ]
    hours: list[tuple[int, float | int]] = []
    for name, f in by_name.items():
        m = _HOUR_NAME_RE.match(name)
        if m:
            hours.append((int(m.group(1)), f.value))  # type: ignore[arg-type]
    hours.sort()
    hr = [(f"{h:02d}:00", v) for h, v in hours]
    want_hour = prefer == "hour" or (
        prefer not in ("weekday", "hour") and bool(_HOUR_WORD_RE.search(_fold(prefer)))
    )
    if hr and (want_hour or not wd):
        return "hour", hr
    if wd:
        return "weekday", wd
    return "", []


def numeric_rows(facts: Sequence[Fact]) -> list[tuple[str, str, float | int]]:
    """Umumiy faktlar jadvali: `(nom, izoh, qiymat)` — faqat raqamlilar."""
    out: list[tuple[str, str, float | int]] = []
    for f in facts:
        if f.numeric:
            out.append((f.name, f.label or f.expr, f.value))  # type: ignore[arg-type]
    return out


# ─── renderlar ───────────────────────────────────────────────────────────────


def render_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], *, numeric_cols: Sequence[int] = ()
) -> str:
    """GFM jadval. Raqamli ustun o'ngga tekislanadi (`---:`)."""
    if not rows:
        return ""
    head = "| " + " | ".join(cell(h) for h in headers) + " |"
    sep = "|" + "|".join("---:" if i in numeric_cols else "---" for i in range(len(headers))) + "|"
    body = []
    for row in rows[:TABLE_MAX_ROWS]:
        vals = [
            fmt_number(v) if i in numeric_cols and not isinstance(v, str) else cell(v)
            for i, v in enumerate(row)
        ]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *body])


def chart_spec(
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float | int]]],
    *,
    kind: str = "bar",
    title: str = "",
) -> dict[str, Any] | None:
    """`md.js::chartSvg` qabul qiladigan spetsifikatsiya (yaroqsiz bo'lsa `None`)."""
    labs = [cell(x, max_chars=24).replace("\\|", "/") for x in labels][:CHART_MAX_LABELS]
    ser = []
    for name, values in list(series)[:CHART_MAX_SERIES]:
        vals = [v for v in list(values)[: len(labs)] if isinstance(v, (int, float))]
        if vals:
            ser.append({"name": cell(name, max_chars=32).replace("\\|", "/"), "values": vals})
    if not labs or not ser:
        return None
    spec: dict[str, Any] = {"type": kind if kind in CHART_TYPES else "bar", "labels": labs}
    if title:
        spec["title"] = cell(title, max_chars=80).replace("\\|", "/")
    spec["series"] = ser
    return spec


def render_chart(spec: dict[str, Any]) -> str:
    """```chart fence — bitta qatorli JSON (md.js `JSON.parse` qiladi)."""
    return "```chart\n" + json.dumps(spec, ensure_ascii=False, separators=(",", ":")) + "\n```"


def render_list(rows: Sequence[tuple[str, float | int]], *, unit: str = "") -> str:
    tail = f" {unit}" if unit else ""
    return "\n".join(
        f"{i}. {cell(name)} — {fmt_number(v)}{tail}" for i, (name, v) in enumerate(rows, 1)
    )


# ─── sintez konteksti bilan ulanish ──────────────────────────────────────────


def wrap_block(block: str) -> str:
    return f"{PRE_START}\n{block}\n{PRE_END}"


def extract_blocks(texts: Sequence[str]) -> list[str]:
    """`wrap_block` bilan o'ralgan bloklarni (kontekst bo'laklaridan) ajratadi."""
    out: list[str] = []
    for t in texts:
        start = 0
        while True:
            i = t.find(PRE_START, start)
            if i < 0:
                break
            j = t.find(PRE_END, i)
            if j < 0:
                break
            out.append(t[i + len(PRE_START) : j].strip("\n"))
            start = j + len(PRE_END)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def ensure_blocks(answer: str, context: Sequence[str]) -> str:
    """Sintezdan keyin: kontekstdagi har blok javobda (bo'sh joyni hisobga olmay)
    bo'lmasa, oxiriga qo'shiladi. Model buyruqni e'tiborsiz qoldirsa ham blok yo'qolmaydi."""
    have = _norm(answer)
    missing = [b for b in extract_blocks(context) if b and _norm(b) not in have]
    if not missing:
        return answer
    return (answer.rstrip() + "\n\n" + "\n\n".join(missing)).strip()


# ─── vaqt seriyasi (count(*) — kodda) ────────────────────────────────────────


def series_grain(days: int) -> str:
    """≤60 kun → kunlik; ≤60 hafta → haftalik; aks holda oylik (md.js 60 yorliq)."""
    if days <= CHART_MAX_LABELS:
        return "day"
    if days <= CHART_MAX_LABELS * 7:
        return "week"
    return "month"


def bucket_keys(since_local: date, until_local: date, grain: str) -> list[date]:
    """Oynadagi hamma bucket boshlari (bo'shlar 0 bo'lib to'ldiriladi)."""
    keys: list[date] = []
    if grain == "day":
        d = since_local
        while d <= until_local and len(keys) < CHART_MAX_LABELS:
            keys.append(d)
            d += timedelta(days=1)
    elif grain == "week":
        d = since_local - timedelta(days=since_local.weekday())
        while d <= until_local and len(keys) < CHART_MAX_LABELS:
            keys.append(d)
            d += timedelta(weeks=1)
    else:
        d = since_local.replace(day=1)
        while d <= until_local and len(keys) < CHART_MAX_LABELS:
            keys.append(d)
            d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return keys


def bucket_label(key: date, grain: str) -> str:
    if grain == "month":
        return key.strftime("%Y-%m")
    return key.strftime("%m-%d")


def fill_series(
    counts: dict[date, int], since_local: date, until_local: date, grain: str
) -> tuple[list[str], list[int]]:
    keys = bucket_keys(since_local, until_local, grain)
    return [bucket_label(k, grain) for k in keys], [int(counts.get(k, 0)) for k in keys]


async def count_series(
    session: Any,
    *,
    chat_ids: Sequence[int],
    since: datetime,
    until: datetime,
    tz: str,
    direction: str = "",
) -> tuple[str, dict[date, int]]:
    """Oynadagi xabarlar sanog'i lokal kun/hafta/oy bo'yicha — `count(*)`, LLM yo'q."""
    from sqlalchemy import func, select

    from app.db.models import Message
    from app.services import timewin as TW

    zone = TW.zone(tz)
    days = max(1, (until - since).days + 1)
    grain = series_grain(days)
    local = func.timezone(zone.key, Message.published_at)
    bucket = func.date_trunc(grain, local)
    stmt = (
        select(bucket.label("b"), func.count().label("n"))
        .where(
            Message.chat_id.in_([int(c) for c in chat_ids]),
            Message.published_at >= since,
            Message.published_at < until,
            Message.deleted_at.is_(None),
        )
        .group_by(bucket)
    )
    if direction == "out":
        stmt = stmt.where(Message.out.is_(True))
    elif direction == "in":
        stmt = stmt.where(Message.out.is_(False))
    rows = (await session.execute(stmt)).all()
    out: dict[date, int] = {}
    for b, n in rows:
        key = b.date() if isinstance(b, datetime) else b
        out[key] = out.get(key, 0) + int(n or 0)
    return grain, out


def local_bounds(since: datetime, until: datetime, tz: str) -> tuple[date, date]:
    from app.services import timewin as TW

    zone = TW.zone(tz)
    last = until - timedelta(microseconds=1)
    return since.astimezone(zone).date(), last.astimezone(zone).date()


def default_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    end = now or datetime.now(UTC)
    return end - timedelta(days=DEFAULT_SERIES_DAYS), end


__all__ = [
    "CHART_MAX_LABELS",
    "FORMAT_NOTE",
    "PRE_END",
    "PRE_START",
    "FormatChoice",
    "cell",
    "chart_spec",
    "count_series",
    "detect_format",
    "distribution_rows",
    "ensure_blocks",
    "extract_blocks",
    "fill_series",
    "numeric_rows",
    "ranking_rows",
    "render_chart",
    "render_list",
    "render_table",
    "series_grain",
    "wrap_block",
]
