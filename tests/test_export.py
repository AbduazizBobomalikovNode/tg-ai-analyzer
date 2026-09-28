"""Javob eksporti (`services/export.py`, `web/routers/export.py`).

Qo'riqlaydi:
  * jadvallar (GFM + ```chart), faktlar, sitatalar (+ jadvaldagi tokenlar), qamrov;
  * CSV: UTF-8 BOM, bo'limlar shakli, formula in'ektsiyasi qochiriladi;
  * telefon raqami hech bir formatda chiqmaydi;
  * router: login yo'q → 401, begona suhbat / xabar → 404, sarlavhalar;
  * UI: "Eksport" tugmasi va i18n kalitlari, JS sintaksisi.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import export as EXP

ROOT = Path(__file__).resolve().parents[1]

ANSWER = """Bu hafta eng ko'p ko'rilgan postlar:

**Top postlar** — 2

| # | chat | sana | o'sish | manba |
|---:|---|---|---:|---|
| 1 | Kanal \\| rasmiy | 2026-09-27 | 500 | #c5m77 |
| 2 | Kanal | 2026-09-26 | -3 | #c5m78 |

Aloqa uchun +998 90 123 45 67 [#c5m77].

```chart
{"type":"bar","title":"Kunlar","labels":["09-26","09-27"],"series":[{"name":"views","values":[10,20]}]}
```

---
**Qamrov** — chatlar: 1/3
"""

CONTEXT = {
    "facts": [
        {"name": "tr_total_delta", "value": 700, "unit": "", "label": "views gained"},
        {"name": "note", "value": '=HYPERLINK("x")', "unit": "", "label": "tel 901234567"},
    ],
    "citations": [
        {
            "token": "c5m77",
            "id": 77,
            "chat_id": 5,
            "chat_title": "Kanal",
            "date": "2026-09-27T10:00:00+00:00",
            "link": "https://t.me/kanal/77",
        }
    ],
    "coverage": {
        "chats_searched": 1,
        "chats_total": 3,
        "truncated": False,
        "gaps": ["Kanal"],
        "nested": {"x": 1},
    },
}


def _data() -> EXP.ExportData:
    return EXP.build(
        conversation_id=44,
        message_id=9,
        content=ANSWER,
        context=CONTEXT,
        extra_links={"c5m78": {"chat_title": "Kanal", "link": "https://t.me/kanal/78"}},
    )


def test_build_collects_tables_facts_citations_coverage() -> None:
    data = _data()
    [table, chart] = data.tables
    assert table.title == "Top postlar"
    assert table.headers == ["#", "chat", "sana", "o'sish", "manba"]
    assert table.rows[0] == ["1", "Kanal | rasmiy", "2026-09-27", "500", "#c5m77"]
    assert chart.title == "Kunlar" and chart.rows == [["09-26", "10"], ["09-27", "20"]]
    assert [c["token"] for c in data.citations] == ["c5m77", "c5m78"]
    assert (
        data.citations[1]["link"] == "https://t.me/kanal/78" and data.citations[1]["msg_id"] == 78
    )
    assert data.facts[0] == {
        "name": "tr_total_delta",
        "value": 700,
        "unit": "",
        "label": "views gained",
    }
    assert data.coverage == {
        "chats_searched": 1,
        "chats_total": 3,
        "truncated": False,
        "gaps": "Kanal",
    }


def test_csv_has_bom_sections_and_no_phones() -> None:
    body = EXP.render(_data(), "csv")
    assert body.startswith(EXP.BOM)
    rows = list(csv.reader(io.StringIO(body.removeprefix(EXP.BOM))))
    assert rows[0] == ["[Top postlar]"]
    assert rows[1] == ["#", "chat", "sana", "o'sish", "manba"]
    assert rows[3][3] == "-3"  # son — formula deb qochirilmaydi
    sections = [r[0] for r in rows if r and r[0].startswith("[")]
    assert sections == ["[Top postlar]", "[Kunlar]", "[facts]", "[citations]", "[coverage]"]
    fact = next(r for r in rows if r and r[0] == "note")
    assert fact[1].startswith("'=")  # formula in'ektsiyasi
    assert "901234567" not in body.replace(" ", "") and EXP.clean("tel 901234567") == "tel [tel]"


def test_md_and_json() -> None:
    md = EXP.render(_data(), "md")
    assert md.startswith("# tgai — conversation 44, message 9")
    assert "## Facts" in md and "## Citations" in md and "## Coverage" in md
    assert "Kanal \\| rasmiy" in md and "901234567" not in md
    data = json.loads(EXP.render(_data(), "json"))
    assert data["conversation_id"] == 44 and len(data["tables"]) == 2
    assert data["facts"][1]["label"] == "tel [tel]"
    assert EXP.filename(44, 9, "csv") == "tgai-44-9.csv"
    with pytest.raises(ValueError):
        EXP.render(_data(), "xlsx")


def test_empty_answer_exports_empty_sections() -> None:
    data = EXP.build(conversation_id=1, message_id=2, content="Salom!", context=None)
    assert data.tables == [] and data.facts == [] and data.citations == []
    assert EXP.render(data, "csv") == EXP.BOM


# ─── router ──────────────────────────────────────────────────────────────────


class _Result:
    def __init__(self, row: Any) -> None:
        self.row = row

    def scalar_one_or_none(self) -> Any:
        return self.row

    def scalars(self) -> Any:
        return iter(
            [
                SimpleNamespace(
                    id=5,
                    title="Kanal",
                    type="channel",
                    link_prefix="https://t.me/kanal/",
                    username="kanal",
                )
            ]
        )


class _Db:
    def __init__(self, row: Any) -> None:
        self.row = row

    async def execute(self, stmt: Any) -> _Result:
        return _Result(self.row)


def test_export_router_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from fastapi.testclient import TestClient

    from app.services import chat_service as cs
    from app.web import main as web_main
    from app.web import security as sec
    from app.web.routers import export as export_router

    row = SimpleNamespace(
        id=9,
        content=ANSWER,
        context=CONTEXT,
        created_at=datetime(2026, 9, 28, tzinfo=UTC),
    )
    state: dict[str, Any] = {"row": row}

    class _Scope:
        async def __aenter__(self) -> Any:
            return _Db(state["row"])

        async def __aexit__(self, *exc: Any) -> None:
            return None

    async def fake_conv(db: Any, user_id: int, conversation_id: int) -> Any:
        if user_id != 7:
            raise cs.ChatError("not_found")
        return SimpleNamespace(id=conversation_id)

    monkeypatch.setattr(export_router, "session_scope", lambda: _Scope())
    monkeypatch.setattr(export_router.cs, "get_conversation", fake_conv)
    url = "/api/conversations/44/messages/9/export"
    with TestClient(web_main.app) as client:
        assert client.get(url).status_code == 401
        client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        r = client.get(url)  # X-Requested-With shart emas (GET)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert r.headers["content-disposition"] == 'attachment; filename="tgai-44-9.csv"'
        assert r.content.startswith(b"\xef\xbb\xbf")
        assert b"901234567" not in r.content.replace(b" ", b"")
        assert "https://t.me/kanal/78" in r.content.decode("utf-8")  # jadval tokeni → havola
        r = client.get(url + "?format=json")
        assert r.status_code == 200 and r.json()["message_id"] == 9
        assert client.get(url + "?format=md").headers["content-type"].startswith("text/markdown")
        assert client.get(url + "?format=xlsx").status_code == 422
        state["row"] = None  # boshqa suhbatning xabari / user turn
        assert client.get(url).status_code == 404
        state["row"] = row
        client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(8))
        assert client.get(url).status_code == 404


# ─── UI ──────────────────────────────────────────────────────────────────────


def test_export_button_and_i18n() -> None:
    src = (ROOT / "src/app/web/static/pages/ask/answer.js").read_text(encoding="utf-8")
    assert "function exportButton" in src and "/export`" in src
    assert "exportButton(S.convId, m.id)" in src
    keys = {
        "web.export.title",
        "web.export.csv",
        "web.export.md",
        "web.export.json",
        "web.export.hint",
        "web.skill.trends",
        "web.skill.extract",
    }
    for lang in ("uz", "ru", "en"):
        data = json.loads((ROOT / f"src/app/i18n/locales/{lang}.json").read_text(encoding="utf-8"))
        missing = [k for k in keys if not data.get(k)]
        assert not missing, (lang, missing)


def test_answer_js_syntax() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node yo'q")
    src = (ROOT / "src/app/web/static/pages/ask/answer.js").read_text(encoding="utf-8")
    res = subprocess.run(  # noqa: S603 — argv qat'iy, shell yo'q
        [node, "--input-type=module", "--check"],
        input=src,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res.returncode == 0, res.stderr
