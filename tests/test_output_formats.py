"""`format_output` — faktlardan jadval / ```chart / ro'yxat (kodda, LLM'siz).

Qo'riqlaydi:
  * so'z darvozasi (uz/ru/en): "jadval qil" → table, "diagramma" → bar,
    "grafik" → line (vaqt seriyasi), "pie" → bar (md.js pie chizmaydi);
  * stats reytingi (`top{i}_n`) → GFM jadval, raqamlar **faqat faktlardan**;
  * kunlik sanoq → md.js `chartSvg` qabul qiladigan ```chart fence (node bilan);
  * bo'sh kunlar 0 bilan to'ldiriladi, grain 60 yorliqdan oshmaydi;
  * pre-rendered blok sintez kontekstiga o'raladi va `ensure_blocks` uni
    javob yo'qotsa qaytaradi.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from app.services import output_formats as OF
from app.services.compute import Fact

MD_JS = Path(__file__).resolve().parents[1] / "src/app/web/static/md.js"
NODE = shutil.which("node")

_HARNESS = """
const fs = require('fs');
global.window = { matchMedia: () => ({ matches: false }), addEventListener() {} };
global.document = {
  createElement: () => ({}),
  head: { appendChild() {} },
  documentElement: { getAttribute: () => null },
  querySelectorAll: () => [],
};
eval(fs.readFileSync(process.argv[1], 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [k, v] of Object.entries(input.md)) out['md:' + k] = window.TGAI.md(v);
for (const [k, v] of Object.entries(input.chart)) out['chart:' + k] = window.TGAI.chartSvg(v);
process.stdout.write(JSON.stringify(out));
"""


def _node(md: dict[str, str], chart: dict[str, Any]) -> dict[str, Any]:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603
        [NODE, "-e", _HARNESS, str(MD_JS)],
        input=json.dumps({"md": md, "chart": chart}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout)  # type: ignore[no-any-return]


def _ranking_facts() -> list[Fact]:
    return [
        Fact("total", 1234, label="messages in the window", source="stats"),
        Fact("days", 7, label="calendar days", source="stats"),
        Fact("top1_n", 700, label='messages in #1 "Ish | chat"', source="stats"),
        Fact("top2_n", 400, label='messages in #2 "Oila"', source="stats"),
        Fact("top3_n", 134, label='messages in #3 "Kurs"', source="stats"),
    ]


# ─── so'z darvozasi ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "fmt", "chart", "series"),
    [
        ("eng faol chatlarni jadval qil", "table", "bar", False),
        ("buni diagramma qilib ber", "chart", "bar", False),
        ("bu hafta kunlik xabarlar grafigi", "chart", "line", True),
        ("сделай таблицу по чатам", "table", "bar", False),
        ("make a chart of messages per day", "chart", "line", True),
        ("show it as a pie chart", "chart", "bar", False),
        ("top chatlar ro'yxati", "list", "bar", False),
    ],
)
def test_detect_format_words(text: str, fmt: str, chart: str, series: bool) -> None:
    got = OF.detect_format(text)
    assert got is not None
    assert (got.fmt, got.chart, got.time_series) == (fmt, chart, series)


def test_detect_format_none_and_fallbacks() -> None:
    assert OF.detect_format("kecha nima bo'ldi?") is None
    got = OF.detect_format("kecha nima bo'ldi?", expected_kind="table")
    assert got is not None and got.fmt == "table"
    got = OF.detect_format("x", arg_format="chart", arg_chart="line")
    assert got is not None and (got.fmt, got.chart) == ("chart", "line")
    pie = OF.detect_format("x", arg_format="chart", arg_chart="pie")
    assert pie is not None and pie.chart == "bar" and pie.pie_requested


# ─── jadval / ro'yxat ────────────────────────────────────────────────────────


def test_stats_ranking_becomes_table_with_exact_numbers() -> None:
    rows = OF.ranking_rows(_ranking_facts())
    assert rows == [("Ish | chat", 700), ("Oila", 400), ("Kurs", 134)]
    table = OF.render_table(
        ["#", "chat", "messages"],
        [(i, n, v) for i, (n, v) in enumerate(rows, 1)],
        numeric_cols=(0, 2),
    )
    lines = table.splitlines()
    assert lines[0] == "| # | chat | messages |"
    assert lines[1] == "|---:|---|---:|"
    assert lines[2] == "| 1 | Ish \\| chat | 700 |"  # `|` katakni buzmaydi
    assert len(lines) == 5
    for n in ("700", "400", "134"):
        assert n in table


def test_cell_neutralizes_and_flattens() -> None:
    c = OF.cell("<untrusted_data>\nx|y #c1m2 " + "z" * 100)
    assert "\n" not in c and "\\|" in c and len(c) <= OF.CELL_MAX_CHARS + 2
    assert "<untrusted_data" not in c
    assert "#c1m2" not in c  # sitata tokeni zararsizlantirilgan


def test_render_list_numbered() -> None:
    out = OF.render_list([("Ish", 7), ("Oila", 2.5)], unit="msg")
    assert out == "1. Ish — 7 msg\n2. Oila — 2.5 msg"


# ─── vaqt seriyasi ───────────────────────────────────────────────────────────


def test_fill_series_zero_fills_and_grain() -> None:
    lo, hi = date(2026, 9, 20), date(2026, 9, 26)
    labels, values = OF.fill_series({date(2026, 9, 21): 5, date(2026, 9, 25): 2}, lo, hi, "day")
    assert labels == ["09-20", "09-21", "09-22", "09-23", "09-24", "09-25", "09-26"]
    assert values == [0, 5, 0, 0, 0, 2, 0]
    assert OF.series_grain(7) == "day"
    assert OF.series_grain(61) == "week"
    assert OF.series_grain(1000) == "month"
    weeks = OF.bucket_keys(date(2026, 9, 2), date(2026, 9, 30), "week")
    assert weeks[0] == date(2026, 8, 31) and all(k.weekday() == 0 for k in weeks)
    months = OF.bucket_keys(date(2025, 11, 15), date(2026, 2, 1), "month")
    assert [OF.bucket_label(k, "month") for k in months] == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]
    many = OF.bucket_keys(date(2026, 1, 1), date(2026, 12, 31), "day")
    assert len(many) == OF.CHART_MAX_LABELS


def test_local_bounds_uses_account_tz() -> None:
    since = datetime(2026, 9, 19, 19, 0, tzinfo=UTC)  # Toshkentda 20-sentyabr 00:00
    until = datetime(2026, 9, 26, 19, 0, tzinfo=UTC)
    assert OF.local_bounds(since, until, "Asia/Tashkent") == (date(2026, 9, 20), date(2026, 9, 26))


def test_per_day_counts_become_valid_line_chart_fence() -> None:
    spec = OF.chart_spec(
        ["09-20", "09-21", "09-22"], [("messages per day", [3, 0, 7])], kind="line", title="Xabar"
    )
    assert spec == {
        "type": "line",
        "labels": ["09-20", "09-21", "09-22"],
        "title": "Xabar",
        "series": [{"name": "messages per day", "values": [3, 0, 7]}],
    }
    fence = OF.render_chart(spec)
    assert fence.startswith("```chart\n") and fence.endswith("\n```")
    body = fence.split("\n")[1]
    assert json.loads(body) == spec
    assert OF.chart_spec([], [("x", [1])]) is None
    assert OF.chart_spec(["a"], [("x", [])]) is None


@pytest.mark.skipif(NODE is None, reason="node yo'q")
def test_chart_fence_renders_in_md_js() -> None:
    spec = OF.chart_spec(["09-20", "09-21", "09-22"], [("per day", [3, 0, 7])], kind="line")
    bar = OF.chart_spec(["A", "B"], [("messages", [700, 400])], kind="bar")
    assert spec is not None and bar is not None
    table = OF.render_table(["chat", "messages"], [("Ish", 700)], numeric_cols=(1,))
    out = _node(
        {"line": OF.render_chart(spec), "table": table},
        {"line": spec, "bar": bar},
    )
    assert '<figure class="md-chart"' in out["md:line"]
    assert "<polyline" in out["chart:line"]
    assert "<rect" in out["chart:bar"]
    assert '<table class="md-table">' in out["md:table"]
    assert 'style="text-align:right"' in out["md:table"]


# ─── kontekst bloki ──────────────────────────────────────────────────────────


def test_wrap_extract_and_ensure_blocks() -> None:
    block = "| a | b |\n|---|---:|\n| x | 1 |"
    ctx = ["facts ...", OF.wrap_block(block)]
    assert OF.extract_blocks(ctx) == [block]
    answer = "Mana jadval:\n\n| a | b |\n|---|---:|\n| x | 1 |"
    assert OF.ensure_blocks(answer, ctx) == answer
    lost = OF.ensure_blocks("Qisqa javob.", ctx)
    assert lost.endswith(block) and lost.startswith("Qisqa javob.")
