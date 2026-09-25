"""Sozlamalar → Byudjet kartasi va Ask banneri (UI qismi, `tests/test_budget.py` juftligi).

Qo'riqlaydi: `/settings` da karta va sub-nav havolasi, `web.budget.*` kalitlari
uchala tilda (API xato kodlari ham), Ask sahifasidagi byudjet banneri
"Sozlamalar → Byudjet" ga olib boradi va sahifa moduli budget API'siga yozadi.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.web import main as web_main
from app.web import security as sec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROOT = Path(__file__).resolve().parents[1]


# ─── UI ──────────────────────────────────────────────────────────────────────


def test_settings_page_has_budget_card() -> None:
    with TestClient(web_main.app) as c:
        c.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        html = c.get("/settings?lang=uz").text
    for needle in (
        'id="sec-budget"',
        'href="#sec-budget"',
        'id="bud-usage"',
        'id="bud-form"',
        'id="bud-reset"',
        'id="bud-chips"',
        'id="bud-boost-form"',
        'id="bud-boost-active"',
    ):
        assert needle in html, needle
    content = html.split("window.TGAI", 1)[0]
    assert "Bugun uchun qo&#39;shimcha" in content or "Bugun uchun qo'shimcha" in content


def test_budget_strings_in_all_locales() -> None:
    loc = {
        n: json.loads((ROOT / f"src/app/i18n/locales/{n}.json").read_text(encoding="utf-8"))
        for n in ("uz", "ru", "en")
    }
    keys = {k for k in loc["uz"] if k.startswith("web.budget.")}
    assert len(keys) >= 30
    for name in ("ru", "en"):
        assert not sorted(k for k in keys if not loc[name].get(k)), name
    # API xato kodlari i18n kalitlari.
    for code in ("web.budget.err_invalid", "web.budget.err_boost_empty", "web.budget.goto"):
        assert all(loc[n].get(code) for n in loc)


def test_ask_budget_banner_links_to_settings() -> None:
    answer = (ROOT / "src/app/web/static/pages/ask/answer.js").read_text(encoding="utf-8")
    panels = (ROOT / "src/app/web/static/pages/ask/panels.js").read_text(encoding="utf-8")
    assert '"/settings#sec-budget"' in answer and "web.budget.goto" in answer
    assert "opts.link" in panels
    settings_js = (ROOT / "src/app/web/static/pages/settings.js").read_text(encoding="utf-8")
    for route in ("/api/settings/budget", "/api/settings/budget/boost"):
        assert route in settings_js
