"""`compute` ko'nikmasi — allaqachon olingan raqamlar ustida **kodda** hisob.

Nega alohida modul: "bu oyda o'rtacha kuniga nechta xabar yozganman" ga javob
bitta bo'lish amali. Uni modelga qoldirish — uydirma raqam xavfi (model
sanoqni ham, kun sonini ham "taxmin qiladi"). Shuning uchun:

* raqamlar faqat **faktlardan** keladi (`Fact` — `stats`/`chat_digest`
  ko'nikmalari `count(*)` dan to'ldiradi);
* ifoda — kichik, xavfsiz evaluator (`safe_eval`): `ast` daraxti oq ro'yxat
  bo'yicha yuriladi, `eval`/`exec`/atribut/indeks/lambda **yo'q**;
* natija yana `Fact` bo'lib dalil konvertiga tushadi, sintez uni o'zgartirmay
  keltiradi, `unbacked_numbers` esa javobdagi raqamlar faktlardan kelganini
  tekshiradi (log + kontekst, javob qayta yozilmaydi).

Planner ifodani `{"expr": "total / days", "label": "avg_per_day"}` shaklida
beradi; bermasa yoki buzuq bersa `heuristic_items` savol so'zlaridan
(kuniga/haftasiga/oyiga/foiz, uz/ru/en) eng oddiy hisobni tanlaydi.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services import textnorm as TN

#: Ifoda uzunligi va daraja qopqoqlari — evaluator "kalkulyator", dastur emas.
MAX_EXPR_CHARS = 200
MAX_NODES = 60
MAX_POW = 6
MAX_ITEMS = 5
#: Javobdagi raqamni fakt bilan solishtirishda nisbiy chetlanish (yaxlitlash).
NUMBER_TOLERANCE = 0.011

_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


class ComputeError(ValueError):
    """Ifoda xavfli, buzuq yoki hisoblab bo'lmaydi (nolga bo'lish, noma'lum nom)."""


@dataclass(slots=True)
class Fact:
    """Bitta **kodda** hisoblangan raqam (yoki sana matni) — belgi bilan.

    `name` — ifodada ishlatiladigan identifikator (`total`, `days`),
    `label` — odam o'qiydigan izoh, `expr` — hisoblangan bo'lsa formulasi.
    """

    name: str
    value: float | int | str
    label: str = ""
    unit: str = ""
    expr: str = ""
    source: str = ""  # qaysi ko'nikma yozdi

    @property
    def numeric(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    def render(self) -> str:
        """`- avg_per_day = total / days = 49.36 (messages per day)` — konvert satri."""
        value = fmt_number(self.value) if self.numeric else str(self.value)
        head = f"- {self.name} = "
        if self.expr:
            head += f"{self.expr} = "
        tail = " ".join(x for x in (self.unit, f"({self.label})" if self.label else "") if x)
        return f"{head}{value}{(' ' + tail) if tail else ''}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "label": self.label,
            "unit": self.unit,
            "expr": self.expr,
            "source": self.source,
        }


def fmt_number(value: Any) -> str:
    """Butun son — butun; kasr — 2 xonagacha, oxirgi nollarsiz."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


# ─── xavfsiz evaluator ───────────────────────────────────────────────────────


def _pct(a: float, b: float) -> float:
    if b == 0:
        raise ComputeError("division by zero")
    return a / b * 100.0


def _avg(*xs: float) -> float:
    if not xs:
        raise ComputeError("avg() of nothing")
    return sum(xs) / len(xs)


def _round(x: float, n: float = 0) -> float:
    digits = int(n)
    if not 0 <= digits <= 6:
        raise ComputeError("round digits out of range")
    return round(x, digits)


_FUNCS: dict[str, Callable[..., float]] = {
    "round": _round,
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "abs": abs,
    "pct": _pct,
    "avg": _avg,
    "sum": lambda *xs: sum(xs),
}

_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def safe_eval(expr: str, names: Mapping[str, float]) -> float:
    """Arifmetik ifodani **oq ro'yxat** bo'yicha hisoblaydi.

    Ruxsat: sonlar, `names` dagi identifikatorlar, `+ - * / // % **`
    (daraja ≤ `MAX_POW`), unar `+ -`, qavslar va `_FUNCS` dagi funksiyalar
    (faqat pozitsion argument). Qolgan hamma narsa — atribut, indeks,
    lambda, comprehension, satr, kalit so'zli argument — `ComputeError`.

    Raises:
        ComputeError: ifoda ruxsat etilmagan, buzuq yoki hisoblanmaydi.
    """
    text = (expr or "").strip()
    if not text:
        raise ComputeError("empty expression")
    if len(text) > MAX_EXPR_CHARS:
        raise ComputeError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ComputeError(f"syntax: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ComputeError("expression too complex")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ComputeError("only numeric constants")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ComputeError(f"unknown name: {node.id}")
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
                raise ComputeError("division by zero")
            if isinstance(node.op, ast.Pow) and abs(b) > MAX_POW:
                raise ComputeError("exponent too large")
            return float(_BINOPS[type(node.op)](a, b))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ComputeError("function not allowed")
            if node.keywords:
                raise ComputeError("keyword arguments not allowed")
            if len(node.args) > 8:
                raise ComputeError("too many arguments")
            args = [walk(a) for a in node.args]
            try:
                return float(_FUNCS[node.func.id](*args))
            except ComputeError:
                raise
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise ComputeError(f"{node.func.id}: {exc}") from exc
        raise ComputeError(f"not allowed: {type(node).__name__}")

    try:
        value = walk(tree)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ComputeError(str(exc)) from exc
    if not math.isfinite(value):
        raise ComputeError("result is not finite")
    return value


# ─── savoldan hisob turi (planner bermasa) ───────────────────────────────────

_PER_DAY_RE = re.compile(
    r"\b(kuniga|kunda|har kuni|kunlik|v den|v sutki|ezhedn\w*|per day|a day|daily|each day)\b"
)
_PER_WEEK_RE = re.compile(
    r"\b(haftasiga|haftada|har hafta|haftalik|v nedel\w*|per week|a week|weekly)\b"
)
_PER_MONTH_RE = re.compile(r"\b(oyiga|har oy|oylik|v mesyac|per month|a month|monthly)\b")
_PCT_RE = re.compile(r"(foiz|%|protsent|procent|percent|ulush|dolya|share)")
_EVERY_N_RE = re.compile(r"\b(necha kunda bir|har necha kunda|how often|kak chasto)\b")


def heuristic_items(question: str, facts: Mapping[str, Fact]) -> list[dict[str, str]]:
    """Planner ifoda bermagan/buzgan bo'lsa — savol so'zlaridan **bitta** hisob.

    Faqat mavjud faktlar ustida: `total` + `days|weeks|months` yoki
    `mine` + `total` (foiz). Hech narsa mos kelmasa bo'sh ro'yxat — hisob
    o'tkazib yuboriladi va bu qamrovga yoziladi (uydirma o'rniga jimlik).
    """
    q = TN.uz_fold(question or "")
    have = {k for k, f in facts.items() if f.numeric}
    out: list[dict[str, str]] = []
    if "total" in have:
        if _EVERY_N_RE.search(q) and "days" in have:
            out.append({"label": "days_per_message", "expr": "days / total"})
        elif _PER_DAY_RE.search(q) and "days" in have:
            out.append({"label": "avg_per_day", "expr": "total / days"})
        elif _PER_WEEK_RE.search(q) and "weeks" in have:
            out.append({"label": "avg_per_week", "expr": "total / weeks"})
        elif _PER_MONTH_RE.search(q) and "months" in have:
            out.append({"label": "avg_per_month", "expr": "total / months"})
    if _PCT_RE.search(q) and {"mine", "total"} <= have:
        out.append({"label": "mine_pct", "expr": "pct(mine, total)"})
    return out[:MAX_ITEMS]


def parse_items(args: Mapping[str, Any]) -> list[dict[str, str]]:
    """Planner argumentlari → `[{label, expr}]` (≤`MAX_ITEMS`), buzuqlari tashlanadi."""
    raw: list[Any] = []
    if isinstance(args.get("items"), list):
        raw.extend(args["items"])
    if isinstance(args.get("expr"), str):
        raw.append({"expr": args["expr"], "label": args.get("label", "")})
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        expr = item.get("expr")
        if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR_CHARS:
            continue
        label = str(item.get("label") or "").strip().lower().replace(" ", "_")
        if not _NAME_RE.match(label):
            label = f"result_{len(out) + 1}"
        out.append({"label": label, "expr": expr.strip()})
        if len(out) >= MAX_ITEMS:
            break
    return out


def run_items(
    items: Sequence[Mapping[str, str]], facts: Mapping[str, Fact]
) -> tuple[list[Fact], list[str]]:
    """Har ifodani hisoblaydi → `(yangi faktlar, xato izohlari)`.

    Oldingi natija keyingisiga nom bo'lib o'tadi (`avg_per_day`, keyin
    `avg_per_day * 7`). Xato bitta elementni tashlaydi, qolganini emas.
    """
    names: dict[str, float] = {k: float(f.value) for k, f in facts.items() if f.numeric}
    made: list[Fact] = []
    errors: list[str] = []
    for item in items:
        label, expr = item["label"], item["expr"]
        try:
            value = safe_eval(expr, names)
        except ComputeError as exc:
            errors.append(f"{label}: {exc}")
            continue
        rounded = round(value, 2)
        fact = Fact(
            name=label,
            value=int(rounded) if rounded == int(rounded) else rounded,
            expr=_show_expr(expr, names),
            source="compute",
        )
        names[label] = value
        made.append(fact)
    return made, errors


def _show_expr(expr: str, names: Mapping[str, float]) -> str:
    """`total / days` → `total / days (1234 / 25)` — o'quvchi hisobni tekshira olsin."""
    subst = re.sub(
        r"\b[a-z_][a-z0-9_]*\b",
        lambda m: fmt_number(names[m.group(0)]) if m.group(0) in names else m.group(0),
        expr,
    )
    return expr if subst == expr else f"{expr} ({subst})"


# ─── javobdagi raqamlar faktlardanmi ─────────────────────────────────────────

_NUM_RE = re.compile(
    r"(?<![\w#.])-?\d{1,3}(?:[  ,]\d{3})+(?:[.,]\d+)?"  # 1 234 / 1,234
    r"|(?<![\w#.])-?\d+(?:[.,]\d+)?"
)
_CITE_RE = re.compile(r"#?[cx]\d+(?:m\d+)?")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?|\d{1,2}:\d{2}")


def _to_float(token: str) -> float | None:
    t = token.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    """Matndagi raqamlar — sitata tokeni, sana va soat olib tashlangan holda."""
    clean = _DATE_RE.sub(" ", _CITE_RE.sub(" ", text or ""))
    out: list[float] = []
    for m in _NUM_RE.finditer(clean):
        v = _to_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def unbacked_numbers(
    answer: str, facts: Sequence[Fact], *, extra_text: str = "", small: int = 10
) -> list[float]:
    """Javobdagi, faktlarda (yoki `extra_text` da) **yo'q** raqamlar.

    `small` dan kichik butun sonlar (ro'yxat raqami, "2 ta chat") va yillar
    tekshirilmaydi — ular hisob emas. Solishtirish nisbiy chetlanish bilan
    (`NUMBER_TOLERANCE`): `49.36` ↔ `49.4`, `33.33` ↔ `33.3%`.
    """
    known: list[float] = [float(f.value) for f in facts if f.numeric]
    known.extend(numbers_in(extra_text))
    bad: list[float] = []
    for v in numbers_in(answer):
        if abs(v) < small and v == int(v):
            continue
        if 1900 <= v <= 2100 and v == int(v):
            continue
        if any(abs(v - k) <= max(NUMBER_TOLERANCE * abs(k), 0.051) for k in known):
            continue
        bad.append(v)
    return bad


__all__ = [
    "MAX_ITEMS",
    "ComputeError",
    "Fact",
    "fmt_number",
    "heuristic_items",
    "numbers_in",
    "parse_items",
    "run_items",
    "safe_eval",
    "unbacked_numbers",
]
