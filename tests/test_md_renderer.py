"""md.js — xavfsiz Markdown renderer (node bo'lsa ishga tushadi, bo'lmasa skip).

Qo'riqlaydi: **escape-first** (xom HTML hech qachon o'tmaydi), jadval, ro'yxat,
kod, callout, details, izoh, chart, mermaid kartasi, xavfsiz havolalar.

Nega node: `md.js` — klassik brauzer skripti, yagona renderer (CLAUDE.md,
"Frontend"). Python nusxasi **yo'q** (ikkinchi nusxa jimgina ajralib ketardi),
shuning uchun test haqiqiy faylni `eval()` bilan yuklaydi va chiqishni tekshiradi.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MD_JS = Path(__file__).resolve().parents[1] / "src/app/web/static/md.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node yo'q")

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
for (const [k, v] of Object.entries(input)) out[k] = window.TGAI.md(v);
process.stdout.write(JSON.stringify(out));
"""

_CHART_HARNESS = """
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
for (const [k, v] of Object.entries(input)) out[k] = window.TGAI.chartSvg(v);
process.stdout.write(JSON.stringify(out));
"""


def _run(harness: str, payload: object) -> dict:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603
        [NODE, "-e", harness, str(MD_JS)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout)  # type: ignore[no-any-return]


def render(cases: dict[str, str]) -> dict[str, str]:
    return _run(_HARNESS, cases)


def charts(cases: dict[str, object]) -> dict[str, str | None]:
    return _run(_CHART_HARNESS, cases)


# ─── asosiy GFM ──────────────────────────────────────────────────────────────


def test_markdown_features_and_xss() -> None:
    out = render(
        {
            "table": "| Post | Views |\n|---|---:|\n| #1 | 120 |\n| <b>x</b> | 5 |",
            "list": "- a\n- b\n  - b1\n- c\n\n1. one\n2. two",
            "inline": (
                "**b** *i* `x < y` [ok](https://ex.com/a?b=1) "
                "[bad](javascript:alert(1)) https://auto.link/x."
            ),
            "code": "```python\nprint('<hi>')\n```",
            "mermaid": '```mermaid\npie title V\n "A": 40\n```',
            "xss": "<img src=x onerror=alert(1)> **b**",
            "quote": "> q1\n> q2\n\n### H\n---\np",
        }
    )
    t = out["table"]
    assert '<table class="md-table">' in t and 'style="text-align:right"' in t
    assert "&lt;b&gt;x&lt;/b&gt;" in t
    # gorizontal scroll konteyneri — kichik ekranda jadval sahifani yormaydi
    assert 'class="md-table-wrap table-wrap"' in t
    li = out["list"]
    assert li.count('<ul class="md-ul"') == 2
    assert "><li>one</li><li>two</li></ol>" in li and '<ol class="md-ol"' in li
    inl = out["inline"]
    assert "<strong>b</strong>" in inl and "<em>i</em>" in inl
    assert '<code class="md-code-inline">x &lt; y</code>' in inl
    assert 'href="https://ex.com/a?b=1"' in inl and 'rel="noopener noreferrer"' in inl
    assert "javascript:" not in inl  # xavfsiz emas → matn
    assert 'href="https://auto.link/x"' in inl
    code = out["code"]
    assert '<figure class="md-code" data-lang="python"' in code
    assert '<span class="md-code-lang">python</span>' in code
    assert '<pre><code class="lang-python">print(&#39;&lt;hi&gt;&#39;)</code></pre>' in code
    assert 'class="md-mermaid-src"' in out["mermaid"] and "pie title V" in out["mermaid"]
    assert "<img" not in out["xss"] and "&lt;img src=x onerror=alert(1)&gt;" in out["xss"]
    assert "<strong>b</strong>" in out["xss"]
    q = out["quote"]
    assert '<blockquote class="md-quote"' in q and "<h5" in q and '<hr class="md-hr"' in q


def test_images_only_from_our_api() -> None:
    out = render(
        {
            "ok": "![cover](/api/images/0b1c2d3e-4f50-6a7b-8c9d-0e1f2a3b4c5d)",
            "bad": "![x](https://evil.example/pic.png) ![y](/api/images/../../etc/passwd)",
        }
    )
    assert (
        '<img class="gen-img" src="/api/images/0b1c2d3e-4f50-6a7b-8c9d-0e1f2a3b4c5d"' in out["ok"]
    )
    assert "<img" not in out["bad"]


# ─── escape-first: har bir yangi konstruksiya uchun ──────────────────────────


#: Har biri xom HTML olib kiruvchi urinish. Hech birida `<img`/`<script` chiqmasin.
_INJECTIONS = {
    "table_cell": "| a | b |\n|---|---|\n| <img onerror=alert(1) src=x> | ok |",
    "table_header": "| <script>alert(1)</script> | b |\n|---|---|\n| 1 | 2 |",
    "table_caption": "Table: <img src=x onerror=alert(1)>\n\n| a |\n|---|\n| 1 |",
    "link_text": "[<img onerror=alert(1) src=x>](https://ok.example/)",
    "link_href": "[t](javascript:alert(1)) [d](data:text/html,<script>1</script>)",
    "heading": "### <img src=x onerror=alert(1)>",
    "callout_title": "> [!WARN] <img src=x onerror=alert(1)>\n> tana",
    "callout_body": "> [!NOTE]\n> <script>alert(1)</script>",
    "details_title": ":::details <img src=x onerror=alert(1)>\ntana\n:::",
    "details_body": ":::details ok\n<script>alert(1)</script>\n:::",
    "footnote": "x[^a]\n\n[^a]: <img src=x onerror=alert(1)>",
    "task": "- [x] <img src=x onerror=alert(1)>",
    "mark": "==<script>alert(1)</script>==",
    "code_fence": "```<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n```",
    "mermaid": "```mermaid\nflowchart TD\n A[<img src=x onerror=1>]-->B\n```",
    "chart": '```chart\n{"type":"<img src=x>","labels":["<b>a</b>"],'
    '"series":[{"name":"<script>x</script>","values":[1,2]}]}\n```',
    "quote": "> <img src=x onerror=alert(1)>",
    "nested": "- a\n  - > [!TIP] <img src=x onerror=alert(1)>\n",
}


#: `md.js` chiqara oladigan teglar. Boshqasi paydo bo'lsa — renderer'da teshik.
_ALLOWED_TAGS = {
    "p",
    "br",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "strong",
    "em",
    "del",
    "mark",
    "code",
    "pre",
    "a",
    "img",
    "sup",
    "ul",
    "ol",
    "li",
    "input",
    "span",
    "div",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "caption",
    "blockquote",
    "aside",
    "details",
    "summary",
    "figure",
    "figcaption",
    "section",
    "button",
    "svg",
    "use",
}
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][\w-]*)([^>]*)>")


def assert_no_html_injection(html: str) -> None:
    """Chiqishdagi **har bir haqiqiy teg** bizniki va hodisa atributi yo'q.

    Escape qilingan matn (`&lt;img …&gt;`) tekshiruvdan o'tadi — u matn, teg
    emas; aynan shu escape-first tamoyilining natijasi.
    """
    for close, tag, attrs in _TAG_RE.findall(html):
        assert tag.lower() in _ALLOWED_TAGS, f"noma'lum teg: <{tag}>"
        if close:
            continue
        # Atribut QIYMATI ichidagi matn allaqachon escape qilingan (`&lt;img …`)
        # — u teg emas, matn; shuning uchun qiymatlar tekshiruvdan chiqariladi.
        bare = re.sub(r'"[^"]*"', '""', attrs)
        assert not re.search(r"\son[a-z]+\s*=", bare, re.I), f"hodisa atributi: <{tag}{attrs}>"
        assert "javascript:" not in attrs.lower()
        assert "data:text/html" not in attrs.lower()
        if tag.lower() == "img":
            assert 'src="/api/images/' in attrs, f"begona rasm: <img{attrs}>"


@pytest.mark.parametrize("name", sorted(_INJECTIONS))
def test_escape_first_for_every_construct(name: str) -> None:
    """Xom HTML hech bir konstruksiyadan o'tmaydi."""
    html = render({name: _INJECTIONS[name]})[name]
    assert_no_html_injection(html)
    assert "<script" not in html.lower()


def test_no_raw_html_passthrough() -> None:
    out = render(
        {
            "tags": "<b>x</b> <a href=#>y</a> <style>body{}</style>",
            "attr": '[t](https://ok.example/" onmouseover="alert(1))',
        }
    )
    assert "<b>" not in out["tags"] and "<style>" not in out["tags"]
    assert "&lt;b&gt;x&lt;/b&gt;" in out["tags"]
    for html in out.values():
        assert_no_html_injection(html)


# ─── yangi konstruksiyalar ───────────────────────────────────────────────────


def test_headings_get_stable_ids_for_toc() -> None:
    """Sarlavha darajasi ikki pog'ona pastga siljiydi (`#` → h3) va id oladi."""
    out = render({"h": "# Birinchi\n\n# Birinchi\n\n### Ички\n", "same": "# Birinchi"})
    h = out["h"]
    assert h.count("<h3 ") == 2 and "<h5 " in h  # `#` → h3, `###` → h5
    ids = [chunk.split('"')[0] for chunk in h.split(' id="')[1:]]
    assert len(ids) == 3 and len(set(ids)) == 3  # bir xil sarlavha → boshqa id
    assert any(i.endswith("-2") for i in ids)
    assert all(len(i) > 3 for i in ids)  # kirill sarlavha ham bo'sh id bermaydi
    # Bir xil manba → bir xil id (oqim paytida sakramaydi, TOC havolasi tirik)
    assert out["same"] == render({"same": "# Birinchi"})["same"]


def test_mark_strike_and_task_lists() -> None:
    out = render(
        {
            "mark": "asosiy: ==30% o'sish==",
            "task": "- [ ] ochiq\n- [x] bajarilgan",
        }
    )
    assert "<mark>30% o&#39;sish</mark>" in out["mark"]
    t = out["task"]
    assert 'class="md-ul md-tasklist"' in t
    assert '<input type="checkbox" class="md-task-box" disabled aria-hidden="true">' in t
    assert 'disabled checked aria-hidden="true">' in t


def test_callouts() -> None:
    out = render(
        {
            "note": "> [!NOTE]\n> oddiy izoh",
            "warn": "> [!WARN] Ma'lumot to'liq emas\n> 3 chat sync qilinmagan",
            "plain": "> oddiy iqtibos",
        }
    )
    assert 'class="md-callout md-callout--note"' in out["note"]
    assert "icons.svg#info" in out["note"]
    w = out["warn"]
    assert "md-callout--warn" in w and "Ma&#39;lumot to&#39;liq emas" in w
    assert "3 chat sync qilinmagan" in w
    assert "md-callout" not in out["plain"] and '<blockquote class="md-quote"' in out["plain"]


def test_details_blocks() -> None:
    out = render(
        {
            "d": ":::details Xom qatorlar\n- a\n- b\n:::",
            "unclosed": ":::details X\nmatn",
        }
    )
    d = out["d"]
    assert '<details class="md-details"' in d
    assert '<summary class="md-details-sum">Xom qatorlar</summary>' in d
    assert '<ul class="md-ul"><li>a</li><li>b</li></ul>' in d
    assert "<details" in out["unclosed"] and "matn" in out["unclosed"]


def test_footnotes() -> None:
    out = render(
        {
            "fn": "Birinchi[^1] va ikkinchi[^2].\n\n[^1]: manba A\n[^2]: manba B",
            "orphan": "matn[^9] davomi",
        }
    )
    f = out["fn"]
    assert '<sup class="md-fn-ref"' in f
    assert '<section class="md-footnotes">' in f
    assert "manba A" in f and "manba B" in f
    assert f.count("<li id=") == 2
    # Ta'rifsiz havola matn bo'lib qoladi — "1" degan sirli raqam chiqmaydi
    assert "md-fn-ref" not in out["orphan"] and "[^9]" in out["orphan"]


def test_tables_alignment_numeric_and_caption() -> None:
    out = render(
        {
            "cap": "Table: Eng ko'p ko'rilgan\n\n| Post | Views | Izoh |\n|---|---|---|\n"
            "| A | 1 200 | ha |\n| B | 980 | yo'q |",
            "explicit": "| a | b |\n|:--|:-:|\n| 1 | 2 |",
        }
    )
    c = out["cap"]
    assert '<caption class="md-table-cap">Eng ko&#39;p ko&#39;rilgan</caption>' in c
    assert "<p>Table:" not in c  # sarlavha paragraf bo'lib qolmaydi
    # `Views` raqamli → avtomat o'ngga, `Izoh` matnli → o'zgarmaydi
    assert '<th scope="col" style="text-align:right">Views</th>' in c
    assert '<th scope="col">Izoh</th>' in c
    e = out["explicit"]
    assert 'style="text-align:left"' in e and 'style="text-align:center"' in e


def test_links_external_icon_and_telegram_chip() -> None:
    out = render(
        {
            "ext": "[manba](https://example.com/a)",
            "tg": "[post](https://t.me/kanal/42)",
            "auto": "qarang https://t.me/kanal/7 va tugadi",
        }
    )
    assert 'class="md-link"' in out["ext"] and "icons.svg#external-link" in out["ext"]
    assert 'class="md-link md-tglink"' in out["tg"] and "icons.svg#send" in out["tg"]
    assert "md-tglink" in out["auto"]


def test_citation_tokens_stay_plain_text() -> None:
    """`#c1023m42` — TreeWalker linkifier'ining ishi; md.js unga tegmaydi."""
    out = render(
        {
            "cite": "Shunday bo'ldi #c1023m42 va #x3.",
            "heading_like": "#c1m2 boshida\n\n# Haqiqiy sarlavha",
        }
    )
    assert "#c1023m42" in out["cite"] and "#x3" in out["cite"]
    assert "<a" not in out["cite"] and "<h" not in out["cite"]
    assert ">#c1m2 boshida</p>" in out["heading_like"]
    assert "<h3" in out["heading_like"]


def test_mermaid_is_a_lazy_placeholder_not_an_autoload() -> None:
    """2.5MB vendor fayli o'z-o'zidan yuklanmaydi — kartada tugma turadi."""
    out = render({"m": "```mermaid\ntimeline\n 2026 : boshlandi\n```"})["m"]
    assert '<figure class="md-mermaid" data-hash="' in out
    assert '<pre class="md-mermaid-src">timeline' in out
    assert 'class="btn md-mermaid-draw"' in out
    assert "Diagrammani chizish" in out
    assert "mermaid.min.js" not in out  # skript tegi chizishda emas, bosilganda
    assert "onclick" not in out  # inline handler yo'q


def test_chart_fence_becomes_a_hydrated_placeholder() -> None:
    ok = (
        '```chart\n{"type":"bar","labels":["a","b","c"],'
        '"series":[{"name":"v","values":[1,2,3]}]}\n```'
    )
    out = render(
        {
            "ok": ok,
            "broken": "```chart\n{not json\n```",
            "empty": '```chart\n{"labels":[]}\n```',
        }
    )
    assert '<figure class="md-chart" data-hash="' in out["ok"]
    assert '<pre class="md-chart-src" hidden>' in out["ok"]
    # Yaroqsiz JSON — ma'lumot yo'qolmaydi, oddiy kod bloki bo'lib qoladi
    assert '<figure class="md-code"' in out["broken"] and "not json" in out["broken"]
    assert '<figure class="md-code"' in out["empty"]


def test_chart_svg_is_pure_and_escaped() -> None:
    out = charts(
        {
            "bar": {
                "type": "bar",
                "labels": ["a", "b", "c"],
                "series": [{"name": "v", "values": [1, 5, 3]}],
            },
            "line": {
                "type": "line",
                "title": "<script>x</script>",
                "labels": ["1", "2", "3"],
                "series": [{"name": "a", "values": [1, 2, 3]}, {"name": "b", "values": [3, 2, 1]}],
            },
            "pie": {
                "type": "pie",
                "labels": ["a", "b", "c"],
                "series": [{"name": "v", "values": [1, 2, 3]}],
            },
            "empty": {"labels": [], "series": []},
            "junk": "not an object",
        }
    )
    assert "<rect" in out["bar"] and "md-chart-svg" in out["bar"]
    assert "<polyline" in out["line"] and '<div class="md-chart-legend">' in out["line"]
    assert "<script" not in out["line"] and "&lt;script&gt;" in out["line"]
    # `pie` qo'llab-quvvatlanmaydi → bar bo'lib chiziladi (bo'sh emas)
    assert "<rect" in out["pie"]
    assert out["empty"] is None and out["junk"] is None


def test_block_source_ranges_for_copy() -> None:
    """Yuqori darajadagi har blok manba qatorlarini olib yuradi (`data-l`)."""
    out = render({"x": "birinchi\n\n| a |\n|---|\n| 1 |\n\n- b"})["x"]
    assert '<p data-l="0,1">' in out
    assert 'data-l="2,5"' in out
    assert 'data-l="6,7"' in out


def test_long_input_does_not_hang() -> None:
    """Katastrofik backtracking yo'q: uzun javob sekundlar emas, millisekundlar."""
    evil = (
        "[" * 400
        + "]("
        + "a" * 4000
        + "\n\n"
        + "*" * 2000
        + "\n\n"
        + "| "
        + " | ".join(["x"] * 40)
        + " |\n"
        + "|"
        + "---|" * 40
        + "\n"
        + ("| " + " | ".join(["1"] * 40) + " |\n") * 50
        + "\n"
        + ("> " * 50 + "chuqur\n")
    )
    out = render({"e": evil})["e"]  # timeout=30 — osilib qolsa test yiqiladi
    assert out
