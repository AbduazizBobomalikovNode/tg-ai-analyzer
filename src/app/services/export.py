"""Javob eksporti — assistant javobining **tuzilmali** qismlari CSV / Markdown / JSON bo'lib.

Nima chiqadi (javob matnining o'zi emas):

* jadvallar — javobdagi GFM jadvallar (`format_output` / `trends` / `extract`
  bloklari ham shu shaklda) va ```chart fence'lari (yorliq x seriya jadvali);
* faktlar — `context.facts` (kodda hisoblangan raqamlar);
* sitatalar — `context.citations` (token, chat, sana, havola) + jadvallarda
  uchragan, lekin sitata ro'yxatida yo'q tokenlar (havola `chats.link_prefix`
  dan — `analytics.msg_link` bilan bir manba; chaqiruvchi beradi);
* qamrov — `context.coverage` ning skalyar maydonlari.

**Telefon raqami hech qachon chiqmaydi**: har satr `extract.strip_phones` dan
o'tadi (ko'nikmalar bilan bir qo'riqchi). CSV — UTF-8 **BOM** bilan (Excel
kirillni to'g'ri ochsin), formula in'ektsiyasiga qarshi `= + - @` bilan
boshlangan katak `'` bilan qochiriladi.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.services.extract import strip_phones

FORMATS = ("csv", "md", "json")
MEDIA_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
BOM = "﻿"
MAX_TABLES = 20
MAX_ROWS = 500
CELL_MAX = 2000

_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CHART_RE = re.compile(r"```chart\s*\n(.*?)\n```", re.S)
_TOKEN_RE = re.compile(r"#(c(\d{1,12})m(\d{1,12})|x\d{1,12})")
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_NUMBER_RE = re.compile(r"[-+]?\d[\d\s.,]*%?")


@dataclass(slots=True)
class Table:
    title: str
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "headers": self.headers, "rows": self.rows}


@dataclass(slots=True)
class ExportData:
    conversation_id: int
    message_id: int
    created_at: str = ""
    tables: list[Table] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "tables": [t.to_dict() for t in self.tables],
            "facts": self.facts,
            "citations": self.citations,
            "coverage": self.coverage,
        }


def clean(value: Any) -> str:
    """Bitta qiymat → matn: telefon tozalangan, markdown qochirishi olib tashlangan."""
    if value is None:
        return ""
    text = str(value).replace("\\|", "|").replace("​", "")
    text = re.sub(r"\s+", " ", text).strip()
    return strip_phones(text)[:CELL_MAX]


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells = re.split(r"(?<!\\)\|", body)
    return [clean(c) for c in cells]


def parse_tables(text: str) -> list[Table]:
    """Markdown matndagi GFM jadvallar (sarlavha + `---` ajratuvchi + qatorlar)."""
    lines = (text or "").splitlines()
    out: list[Table] = []
    title = ""
    i = 0
    while i < len(lines) and len(out) < MAX_TABLES:
        line = lines[i]
        if (
            "|" in line
            and i + 1 < len(lines)
            and _SEP_RE.match(lines[i + 1])
            and line.strip().startswith("|")
        ):
            headers = _split_row(line)
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                if len(rows) < MAX_ROWS:
                    rows.append(_split_row(lines[j]))
                j += 1
            out.append(Table(title=title or f"table {len(out) + 1}", headers=headers, rows=rows))
            title = ""
            i = j
            continue
        stripped = line.strip()
        if stripped:
            title = clean(re.sub(r"[*_#`]", "", stripped).split("—")[0])[:120]
        i += 1
    return out


def parse_charts(text: str) -> list[Table]:
    """```chart fence (`md.js::chartSvg` sxemasi) → yorliq x seriya jadvali."""
    out: list[Table] = []
    for m in _CHART_RE.finditer(text or ""):
        try:
            spec = json.loads(m.group(1))
        except ValueError:
            continue
        if not isinstance(spec, dict):
            continue
        labels = [clean(x) for x in spec.get("labels") or []][:MAX_ROWS]
        series = [s for s in spec.get("series") or [] if isinstance(s, dict)]
        if not labels or not series:
            continue
        headers = [
            "label",
            *[clean(s.get("name") or f"series {k + 1}") for k, s in enumerate(series)],
        ]
        rows = []
        for idx, label in enumerate(labels):
            row = [label]
            for s in series:
                vals = s.get("values") or []
                row.append(clean(vals[idx]) if idx < len(vals) else "")
            rows.append(row)
        out.append(Table(title=clean(spec.get("title") or "chart"), headers=headers, rows=rows))
    return out


def table_tokens(tables: Sequence[Table]) -> list[tuple[str, int, int]]:
    """Jadval kataklaridagi `#c<chat>m<msg>` tokenlari (tartibda, takrorsiz)."""
    seen: set[str] = set()
    out: list[tuple[str, int, int]] = []
    for t in tables:
        for row in t.rows:
            for c in row:
                for m in _TOKEN_RE.finditer(c):
                    if m.group(2) and m.group(1) not in seen:
                        seen.add(m.group(1))
                        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def build(
    *,
    conversation_id: int,
    message_id: int,
    content: str,
    context: dict[str, Any] | None,
    created_at: str = "",
    extra_links: dict[str, dict[str, Any]] | None = None,
) -> ExportData:
    """Assistant qatoridan eksport ma'lumoti. `extra_links` — token → `{chat_title, link}`."""
    ctx = context if isinstance(context, dict) else {}
    tables = parse_tables(content) + parse_charts(content)
    facts = []
    for f in ctx.get("facts") or []:
        if isinstance(f, dict) and f.get("name"):
            facts.append(
                {
                    "name": clean(f.get("name")),
                    "value": f.get("value")
                    if isinstance(f.get("value"), (int, float))
                    and not isinstance(f.get("value"), bool)
                    else clean(f.get("value")),
                    "unit": clean(f.get("unit")),
                    "label": clean(f.get("label")),
                }
            )
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in ctx.get("citations") or []:
        if not isinstance(c, dict):
            continue
        token = str(c.get("token") or "")
        if token in seen:
            continue
        seen.add(token)
        citations.append(
            {
                "token": clean(token),
                "chat_title": clean(c.get("chat_title")),
                "msg_id": c.get("id") if isinstance(c.get("id"), int) else None,
                "date": clean(c.get("date")),
                "link": clean(c.get("link")),
            }
        )
    for token, _chat, msg in table_tokens(tables):
        if token in seen:
            continue
        seen.add(token)
        info = (extra_links or {}).get(token) or {}
        citations.append(
            {
                "token": token,
                "chat_title": clean(info.get("chat_title")),
                "msg_id": msg,
                "date": "",
                "link": clean(info.get("link")),
            }
        )
    coverage: dict[str, Any] = {}
    cov = ctx.get("coverage")
    if isinstance(cov, dict):
        for k, v in cov.items():
            if isinstance(v, bool) or v is None:
                coverage[str(k)] = v
            elif isinstance(v, (int, float)):
                coverage[str(k)] = v
            elif isinstance(v, str):
                coverage[str(k)] = clean(v)
            elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
                coverage[str(k)] = ", ".join(clean(x) for x in v[:50])
    return ExportData(
        conversation_id=conversation_id,
        message_id=message_id,
        created_at=created_at,
        tables=tables,
        facts=facts,
        citations=citations,
        coverage=coverage,
    )


# ─── render ──────────────────────────────────────────────────────────────────


def _csv_cell(value: Any) -> str:
    text = clean(value) if not isinstance(value, (int, float)) else str(value)
    if (
        isinstance(value, str)
        and text.startswith(_FORMULA_START)
        and not _NUMBER_RE.fullmatch(text)
    ):
        return "'" + text  # formula in'ektsiyasi (Excel/Sheets)
    return text


def to_csv(data: ExportData) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")

    def section(title: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        w.writerow([f"[{title}]"])
        w.writerow([_csv_cell(h) for h in header])
        for r in rows:
            w.writerow([_csv_cell(c) for c in r])
        w.writerow([])

    for t in data.tables:
        section(t.title, t.headers, t.rows)
    if data.facts:
        section(
            "facts",
            ["name", "value", "unit", "label"],
            [[f["name"], f["value"], f["unit"], f["label"]] for f in data.facts],
        )
    if data.citations:
        section(
            "citations",
            ["token", "chat", "msg_id", "date", "link"],
            [
                [c["token"], c["chat_title"], c["msg_id"] or "", c["date"], c["link"]]
                for c in data.citations
            ],
        )
    if data.coverage:
        section("coverage", ["key", "value"], [[k, v] for k, v in data.coverage.items()])
    return BOM + buf.getvalue()


def _md_cell(value: Any) -> str:
    return (clean(value) if not isinstance(value, (int, float)) else str(value)).replace("|", "\\|")


def _md_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(_md_cell(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def to_md(data: ExportData) -> str:
    parts = [f"# tgai — conversation {data.conversation_id}, message {data.message_id}"]
    if data.created_at:
        parts.append(f"_{clean(data.created_at)}_")
    for t in data.tables:
        parts.append(f"## {_md_cell(t.title)}\n\n{_md_table(t.headers, t.rows)}")
    if data.facts:
        rows = [[f["name"], f["value"], f["unit"], f["label"]] for f in data.facts]
        parts.append("## Facts\n\n" + _md_table(["name", "value", "unit", "label"], rows))
    if data.citations:
        rows = [
            [c["token"], c["chat_title"], c["msg_id"] or "", c["date"], c["link"]]
            for c in data.citations
        ]
        parts.append(
            "## Citations\n\n" + _md_table(["token", "chat", "msg_id", "date", "link"], rows)
        )
    if data.coverage:
        rows = [[k, v] for k, v in data.coverage.items()]
        parts.append("## Coverage\n\n" + _md_table(["key", "value"], rows))
    return "\n\n".join(parts) + "\n"


def to_json(data: ExportData) -> str:
    return json.dumps(data.to_dict(), ensure_ascii=False, indent=2) + "\n"


def render(data: ExportData, fmt: str) -> str:
    if fmt == "csv":
        return to_csv(data)
    if fmt == "md":
        return to_md(data)
    if fmt == "json":
        return to_json(data)
    raise ValueError(f"unknown export format {fmt!r}")


def filename(conversation_id: int, message_id: int, fmt: str) -> str:
    return f"tgai-{int(conversation_id)}-{int(message_id)}.{fmt}"


__all__ = [
    "BOM",
    "FORMATS",
    "MEDIA_TYPES",
    "ExportData",
    "Table",
    "build",
    "clean",
    "filename",
    "parse_charts",
    "parse_tables",
    "render",
    "table_tokens",
    "to_csv",
    "to_json",
    "to_md",
]
