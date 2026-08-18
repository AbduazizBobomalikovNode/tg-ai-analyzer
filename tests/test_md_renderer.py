"""md.js — xavfsiz Markdown renderer (node bo'lsa ishga tushadi, bo'lmasa skip).

Qo'riqlaydi: XSS escape, jadval, ro'yxat, kod, mermaid blok, xavfsiz havolalar.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MD_JS = Path(__file__).resolve().parents[1] / "src/app/web/static/md.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node yo'q")

_HARNESS = """
const fs = require('fs');
global.window = { matchMedia: () => ({ matches: false }) };
global.document = { createElement: () => ({}), head: { appendChild() {} } };
eval(fs.readFileSync(process.argv[1], 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [k, v] of Object.entries(input)) out[k] = window.TGAI.md(v);
process.stdout.write(JSON.stringify(out));
"""


def render(cases: dict[str, str]) -> dict[str, str]:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603
        [NODE, "-e", _HARNESS, str(MD_JS)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout)  # type: ignore[no-any-return]


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
    assert "<table>" in t and 'style="text-align:right"' in t and "&lt;b&gt;x&lt;/b&gt;" in t
    assert t.startswith('<div class="table-wrap">')  # gorizontal scroll — kichik ekran
    li = out["list"]
    assert li.count("<ul>") == 2 and "<ol><li>one</li><li>two</li></ol>" in li
    inl = out["inline"]
    assert "<strong>b</strong>" in inl and "<em>i</em>" in inl and "<code>x &lt; y</code>" in inl
    assert 'href="https://ex.com/a?b=1"' in inl and 'rel="noopener noreferrer"' in inl
    assert "javascript:" not in inl  # xavfsiz emas → matn
    assert 'href="https://auto.link/x"' in inl
    assert '<pre><code class="lang-python">print(&#39;&lt;hi&gt;&#39;)</code></pre>' == out["code"]
    assert 'class="mermaid-src"' in out["mermaid"] and "pie title V" in out["mermaid"]
    assert "<img" not in out["xss"] and "&lt;img src=x onerror=alert(1)&gt;" in out["xss"]
    assert "<strong>b</strong>" in out["xss"]
    assert (
        "<blockquote>" in out["quote"] and "<h5>H</h5>" in out["quote"] and "<hr>" in out["quote"]
    )
