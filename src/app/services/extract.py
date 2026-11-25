"""`extract` ko'nikmasi — dalildan tuzilmali ajratish, **deterministik** (regex, LLM yo'q).

Manba: joriy turn'da allaqachon olingan dalil (`ctx.ranked`) yoki — dalil
bo'lmasa — bitta (yoki tanlangan) chatning vaqt oynasi (`messages`, eng
yangisidan `EXTRACT_MAX_MSGS` ta).

Turlar: `links` (URL + domen), `addresses` ("manzil:" yorlig'i va ko'cha / uy /
mavze / tuman / mahalla naqshlari, ru/en ham), `prices` (so'm / $ / UZS … bilan
son), `dates` (sana va `HH:MM`), `files` (`media_type=document` sanog'i —
**fayl nomi saqlanmaydi**, `messages` da ustuni yo'q; matnda tilga olingan
fayl nomlari alohida), `mentions` (@username va #hashtag).

**Telefon raqami hech qachon chiqmaydi** (loyiha invarianti: `list_members`,
`get_user_info` bilan bir mezon). Har matn avval `strip_phones` dan o'tadi —
qolgan hamma naqsh **tozalangan** matnda qidiriladi, ya'ni raqam na URL
(`wa.me/998…`), na manzil, na narx ichida chiqib keta oladi. Raqamlar faqat
**sanaladi** (`ex_phone_n`). `export.py` ham shu `strip_phones` ni chaqiradi.

Chiqish: `output_formats` jadval bloki (`wrap_block`), har element yonida
sitata tokeni (birinchi uchragan xabar) va `ex_<kind>_n` faktlari. Ko'nikma
karantinda — begona matn ko'radi, planner faqat sanoqlarni oladi.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from app.services import output_formats as OF
from app.services import untrusted as U
from app.services.compute import Fact

if TYPE_CHECKING:
    from app.services.skills import RunState, SkillResult

KINDS = ("links", "addresses", "prices", "dates", "files", "mentions")
EXTRACT_MAX_MSGS = 2000
ROWS_PER_KIND = 20
DEFAULT_DAYS = 7
VALUE_MAX_CHARS = 120
PHONE_MASK = "[tel]"

# ─── telefon qo'riqchisi ─────────────────────────────────────────────────────

_UZ_CODES = "20|33|50|55|61|62|65|66|67|69|70|71|72|73|74|75|76|77|78|79|88|90|91|93|94|95|97|98|99"
_NOT_MONEY = r"(?!\s*(?:so[ʻ'‘’`]?m|сўм|сум|sum\b|uzs|usd|\$|mln|ming|млн|тыс))"
#: O'zbek raqami: `+998 90 123 45 67`, `(90) 123-45-67`, `901234567`, `998901234567`.
_PHONE_UZ = (
    rf"(?<![\d+])(?:\+?\s?998[\s\-.]?)?\(?(?:{_UZ_CODES})\)?[\s\-.]?\d{{3}}[\s\-.]?"
    rf"\d{{2}}[\s\-.]?\d{{2}}(?!\d){_NOT_MONEY}"
)
#: Xalqaro: `+` bilan boshlanadigan 8-15 raqam (`+7 912 345-67-89`, `+1 (555) 123 4567`).
_PHONE_INTL = r"(?<![\w+])\+\d{1,3}[\s\-.]?\(?\d{2,4}\)?(?:[\s\-.]?\d{2,4}){2,4}(?!\d)"
PHONE_RE = re.compile(rf"{_PHONE_UZ}|{_PHONE_INTL}", re.I)


def count_phones(text: str | None) -> int:
    return len(PHONE_RE.findall(text or ""))


def strip_phones(text: str | None) -> str:
    """Telefon raqamlarini `[tel]` bilan almashtiradi — **har** chiqishdan oldin."""
    return PHONE_RE.sub(PHONE_MASK, text or "")


# ─── naqshlar (tozalangan matn ustida) ───────────────────────────────────────

_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]{}\"'`«»]+|(?<![\w./])(?:t\.me|telegram\.me)/[\w+/\-]+",
    re.I,
)
_URL_TAIL = ".,;:!?)»”'\"*_"

_APOS = "[ʻ'‘’`]?"
_ADDR_LABEL_RE = re.compile(
    rf"(?:manzil\w*|mo{_APOS}ljal\w*|адрес\w*|ориентир\w*|address)\s*[:\-–—]\s*([^\n]{{3,160}})",
    re.I,
)
_ADDR_MARK_RE = re.compile(
    rf"(?:ko{_APOS}ch(?:a\w*|\.)|\bmavze\w*|\btuman\w*|\bmahalla\w*|\bmfy\b|\bmassiv\w*|"
    r"\bkvartal\w*|\b\d{1,4}\s*-?\s*uy\b|\buy\s*\d{1,4}\b|\b\d{1,3}\s*-?\s*(?:mavze|kvartal)\b|"
    r"\bулиц\w*|\bул\.\s|\bпроспект\w*|\bпр-т\b|\bдом\s*\d|\bд\.\s*\d|\bрайон\w*|\bр-н\b|"
    r"\bквартал\w*|\bмассив\w*|\bstreet\b|\bavenue\b|\bdistrict\b)",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[^\W\d]{3}[.!?;])\s+|\n+")
_ADDR_END_RE = re.compile(r"(?<=[^\W\d]{3})[.!?;](?:\s|$)|\[tel\]")

_NUM = r"\d{1,3}(?:[  .,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
_MULT = r"(?:\s?(?:mln|million|mlrd|млн|млрд|ming|минг|тыс\.?|тысяч\w*|k)\b\.?)"
_CUR_AFTER = (
    rf"(?:so{_APOS}m\b|сўм|сум\w*|\bsum\b|uzs\b|usd\b|dollar\w*|доллар\w*|\$|€|eur\b|евро|"
    r"rub\b|руб\w*|у\.\s?е\.)"
)
_PRICE_LABEL = r"(?:narx\w*|summa\w*|цена|стоимост\w*|price)\s*[:\-–]?\s*"
_PRICE_RE = re.compile(
    rf"(?:(?<![\w.,])(?:{_NUM}){_MULT}?\s?{_CUR_AFTER})"
    rf"|(?:(?:\$|€|usd|uzs)\s?(?:{_NUM}){_MULT}?)"
    rf"|(?:{_PRICE_LABEL}(?:{_NUM}){_MULT}?(?:\s?{_CUR_AFTER})?)",
    re.I,
)
_PRICE_LABEL_RE = re.compile(rf"^{_PRICE_LABEL}", re.I)

_MONTHS = (
    r"yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sent[ay]?abr|okt[ay]?abr|noyabr|dekabr|"
    r"январ\w*|феврал\w*|март\w*|апрел\w*|ма[яй]|июн\w*|июл\w*|август\w*|сентябр\w*|"
    r"октябр\w*|ноябр\w*|декабр\w*|january|february|march|april|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    r"(?<![\d.])(?:\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2})(?![\d.])"
    rf"|(?<!\d)\d{{1,2}}\s?-?\s?(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    r"|(?<![\d:.])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:])",
    re.I,
)

_FILE_RE = re.compile(
    r"(?<![/\w.])[\w\-]{1,60}\.(?:pdf|docx?|xlsx?|pptx?|csv|txt|zip|rar|7z|apk|mp3|mp4|mov)\b",
    re.I,
)
_MENTION_RE = re.compile(r"(?<![\w@.])@([A-Za-z][A-Za-z0-9_]{3,31})\b")
_HASHTAG_RE = re.compile(r"(?<![\w#&/])#([^\W\d_]\w{1,63})")
_CITE_LIKE_RE = re.compile(r"^(?:c\d+m\d+|x\d+)$", re.I)

#: Savol so'zlari → tur (planner `kind` bermasa); `TW.fold` qilingan matn ustida.
_KIND_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("links", re.compile(r"\b(havola|link|ssilka|ссылк|url|sayt|сайт)")),
    ("addresses", re.compile(r"\b(manzil|adres|адрес|address)")),
    ("prices", re.compile(r"\b(narx|summa|цен[аыу]|стоимост|price|cost)")),
    ("dates", re.compile(r"\b(sana(si|lar\w*)?\b|qachon|muddat|дат[аыу]|срок|date|deadline)")),
    ("files", re.compile(r"\b(fayl|hujjat|файл|документ|file|document|pdf)")),
    ("mentions", re.compile(r"\b(mention|heshteg|hashtag|хештег|хэштег|упомин)")),
)
_PHONE_ASK_RE = re.compile(r"(telefon|raqam|номер|телефон|phone)", re.I)


# ─── ma'lumot ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Row:
    """Bitta manba xabar (dalil `S.Hit` yoki `messages` qatori)."""

    message_id: int
    chat_id: int
    tg_msg_id: int
    date: datetime | None
    text: str
    media_type: str = ""


@dataclass(slots=True)
class Item:
    kind: str
    value: str
    extra: str = ""
    count: int = 0
    chat_id: int = 0
    tg_msg_id: int = 0
    message_id: int = 0


@dataclass(slots=True)
class Extraction:
    scanned: int = 0
    phones: int = 0
    documents: int = 0
    items: dict[str, list[Item]] = field(default_factory=dict)

    def n(self, kind: str) -> int:
        return len(self.items.get(kind, []))


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def normalize_kinds(arg: Any, question: str = "") -> list[str]:
    """`kind` argumenti (bitta, vergul bilan, `all`) > savol so'zlari > hammasi."""
    raw = str(arg or "").lower()
    picked = [k for k in KINDS if k in {x.strip() for x in raw.split(",")}]
    if picked:
        return picked
    if raw.strip() == "all":
        return list(KINDS)
    folded = OF._fold(question)
    found = [kind for kind, pattern in _KIND_WORDS if pattern.search(folded)]
    return found or list(KINDS)


def _clean_url(url: str) -> str:
    url = url.rstrip(_URL_TAIL)
    if not re.match(r"https?://", url, re.I):
        url = "https://" + url
    return url


def domain_of(url: str) -> str:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def _squash(text: str, limit: int = VALUE_MAX_CHARS) -> str:
    out = re.sub(r"\s+", " ", text).strip(" ,;:-–—")
    return out[: limit - 1].rstrip() + "…" if len(out) > limit else out


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def find_links(clean: str) -> list[tuple[str, str]]:
    out = []
    for m in _URL_RE.finditer(clean):
        url = _clean_url(m.group(0))
        dom = domain_of(url)
        if dom:
            out.append((url, dom))
    return out


def find_addresses(clean: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    labelled: set[str] = set()
    for m in _ADDR_LABEL_RE.finditer(clean):
        # Yorliqdan keyin — gap oxirigacha ("Manzil: X 5-mavze. Tel: …" → "X 5-mavze");
        # "ул." kabi 2 harfli qisqartma gapni bo'lmaydi.
        val = _squash(_ADDR_END_RE.split(m.group(1))[0])
        if len(val) >= 3:
            out.append((val, "label"))
            labelled.add(_norm_key(val))
    for sentence in _SENTENCE_SPLIT_RE.split(clean):
        if not _ADDR_MARK_RE.search(sentence) or _ADDR_LABEL_RE.search(sentence):
            continue
        val = _squash(sentence)
        key = _norm_key(val)
        if len(val) >= 5 and not any(key in known for known in labelled):
            out.append((val, "pattern"))
    return out


def _currency(value: str) -> str:
    low = value.lower()
    if "$" in low or "usd" in low or "dollar" in low or "доллар" in low or "у.е" in low:
        return "USD"
    if "€" in low or "eur" in low or "евро" in low:
        return "EUR"
    if "rub" in low or "руб" in low:
        return "RUB"
    if re.search(rf"so{_APOS}m|сўм|сум|\bsum\b|uzs", low):
        return "UZS"
    return ""


def find_prices(clean: str) -> list[tuple[str, str]]:
    out = []
    for m in _PRICE_RE.finditer(clean):
        value = _squash(_PRICE_LABEL_RE.sub("", m.group(0)), 60)
        if value:
            out.append((value, _currency(value)))
    return out


def find_dates(clean: str) -> list[tuple[str, str]]:
    out = []
    for m in _DATE_RE.finditer(clean):
        val = _squash(m.group(0), 40)
        out.append((val, "time" if re.fullmatch(r"\d{1,2}:\d{2}", val) else "date"))
    return out


def find_files(clean: str) -> list[tuple[str, str]]:
    no_urls = _URL_RE.sub(" ", clean)
    return [(m.group(0), m.group(0).rsplit(".", 1)[-1].lower()) for m in _FILE_RE.finditer(no_urls)]


def find_mentions(clean: str) -> list[tuple[str, str]]:
    no_urls = _URL_RE.sub(" ", clean)
    out = [(f"@{m.group(1)}", "mention") for m in _MENTION_RE.finditer(no_urls)]
    out.extend(
        (f"#{m.group(1)}", "hashtag")
        for m in _HASHTAG_RE.finditer(no_urls)
        if not _CITE_LIKE_RE.match(m.group(1))
    )
    return out


_FINDERS = {
    "links": find_links,
    "addresses": find_addresses,
    "prices": find_prices,
    "dates": find_dates,
    "files": find_files,
    "mentions": find_mentions,
}


def extract(rows: Iterable[Row], kinds: Sequence[str]) -> Extraction:
    """Hamma xabardan tanlangan turlar — telefon avval tozalanadi, keyin naqshlar."""
    res = Extraction(items={k: [] for k in kinds})
    index: dict[tuple[str, str], Item] = {}
    for row in rows:
        res.scanned += 1
        if (row.media_type or "") == "document":
            res.documents += 1
        raw = row.text or ""
        res.phones += count_phones(raw)
        clean = strip_phones(raw)
        for kind in kinds:
            for value, extra in _FINDERS[kind](clean):
                value = strip_phones(value)  # ikkinchi qavat — hech qachon raqam chiqmasin
                key = (kind, value.lower() if kind in ("links", "mentions") else _norm_key(value))
                item = index.get(key)
                if item is None:
                    item = Item(
                        kind=kind,
                        value=value,
                        extra=extra,
                        chat_id=row.chat_id,
                        tg_msg_id=row.tg_msg_id,
                        message_id=row.message_id,
                    )
                    index[key] = item
                    res.items[kind].append(item)
                item.count += 1
    for kind in kinds:
        res.items[kind].sort(key=lambda it: -it.count)  # barqaror: teng sanoqda — birinchi uchragan
    return res


# ─── render ──────────────────────────────────────────────────────────────────

_T: dict[str, dict[str, Any]] = {
    "uz": {
        "links": ("Havolalar", ("havola", "domen", "soni", "manba")),
        "addresses": ("Manzillar", ("manzil", "soni", "manba")),
        "prices": ("Narxlar / summalar", ("summa", "valyuta", "soni", "manba")),
        "dates": ("Sanalar / vaqtlar", ("sana/vaqt", "soni", "manba")),
        "files": ("Matnda tilga olingan fayllar", ("fayl", "turi", "soni", "manba")),
        "mentions": ("Mention va heshteglar", ("teg", "soni", "manba")),
        "docs": "Hujjat (document) yuborilgan xabarlar: {n} ta — fayl nomi saqlanmaydi.",
        "phones": "Telefon raqamlari: {n} ta topildi — maxfiylik uchun ko'rsatilmaydi.",
        "more": "… va yana {n} ta",
        "none": "Hech narsa topilmadi ({n} ta xabar ko'rildi).",
    },
    "ru": {
        "links": ("Ссылки", ("ссылка", "домен", "кол-во", "источник")),
        "addresses": ("Адреса", ("адрес", "кол-во", "источник")),
        "prices": ("Цены / суммы", ("сумма", "валюта", "кол-во", "источник")),
        "dates": ("Даты / время", ("дата/время", "кол-во", "источник")),
        "files": ("Файлы, упомянутые в тексте", ("файл", "тип", "кол-во", "источник")),
        "mentions": ("Упоминания и хештеги", ("тег", "кол-во", "источник")),
        "docs": "Сообщений с документами: {n} — имена файлов не сохраняются.",
        "phones": "Телефонных номеров: {n} — не показываются (приватность).",
        "more": "… и ещё {n}",
        "none": "Ничего не найдено (просмотрено сообщений: {n}).",
    },
    "en": {
        "links": ("Links", ("link", "domain", "count", "source")),
        "addresses": ("Addresses", ("address", "count", "source")),
        "prices": ("Prices / amounts", ("amount", "currency", "count", "source")),
        "dates": ("Dates / times", ("date/time", "count", "source")),
        "files": ("Files mentioned in text", ("file", "type", "count", "source")),
        "mentions": ("Mentions and hashtags", ("tag", "count", "source")),
        "docs": "Messages with a document: {n} — file names are not stored.",
        "phones": "Phone numbers: {n} found — not shown (privacy).",
        "more": "… and {n} more",
        "none": "Nothing found ({n} messages scanned).",
    },
}


def cite_token(item: Item) -> str:
    return f"#c{int(item.chat_id)}m{int(item.tg_msg_id)}" if item.chat_id else ""


def _row(item: Item) -> tuple[Any, ...]:
    tok = cite_token(item)
    if item.kind in ("links", "prices", "files"):
        return (item.value, item.extra, item.count, tok)
    return (item.value, item.count, tok)


def render_block(ex: Extraction, kinds: Sequence[str], locale: str = "uz") -> str:
    t = _T.get(locale, _T["uz"])
    parts: list[str] = []
    for kind in kinds:
        items = ex.items.get(kind, [])
        title, headers = t[kind]
        if kind == "files":
            parts.append(t["docs"].format(n=ex.documents))
        if not items:
            continue
        shown = items[:ROWS_PER_KIND]
        count_col = len(headers) - 2
        table = OF.render_table(
            headers,
            [_row(it) for it in shown],
            numeric_cols=(count_col,),
            raw_cols=(len(headers) - 1,),
        )
        parts.append(f"**{title}** — {len(items)}\n\n{table}")
        if len(items) > len(shown):
            parts.append(t["more"].format(n=len(items) - len(shown)))
    if ex.phones:
        parts.append(t["phones"].format(n=ex.phones))
    if not any(ex.items.get(k) for k in kinds) and "files" not in kinds:
        parts.append(t["none"].format(n=ex.scanned))
    # Oxirgi qavat: blokda telefon raqami qolmasin (masalan, sarlavha/yorliqdan).
    return strip_phones("\n\n".join(parts))


def facts_for(ex: Extraction, kinds: Sequence[str]) -> list[Fact]:
    src = "extract"
    facts = [
        Fact("ex_messages_n", ex.scanned, label="messages scanned by extract", source=src),
        Fact("ex_phone_n", ex.phones, label="phone numbers found (never shown)", source=src),
    ]
    for kind in kinds:
        facts.append(Fact(f"ex_{kind}_n", ex.n(kind), label=f"distinct {kind} found", source=src))
    if "files" in kinds:
        facts.append(
            Fact(
                "ex_documents_n",
                ex.documents,
                label="messages with a document attachment (file names are not stored)",
                source=src,
            )
        )
    return facts


# ─── ko'nikma ────────────────────────────────────────────────────────────────

EXTRACT_NOTE = (
    "extract: the context holds a code-built list of extracted items (ex_* facts). Copy the "
    "block; do not add items that are not in it. Phone numbers are never shown — only their "
    "count (ex_phone_n). File names are not stored — only document counts."
)


def _rows_from_hits(hits: Sequence[Any]) -> list[Row]:
    return [
        Row(
            message_id=int(h.message_id),
            chat_id=int(h.chat_id or 0),
            tg_msg_id=int(h.tg_msg_id),
            date=h.date,
            text=h.text or "",
            media_type=h.media_type or "",
        )
        for h in hits
    ]


def _scope(ctx: Any, args: dict[str, Any]) -> list[int]:
    known = ctx.chats_by_id
    cid = args.get("chat_id")
    if isinstance(cid, int) and cid in known:
        return [cid]
    for source in (list(ctx.plan.chats or []), list(ctx.selected or [])):
        ids = [int(c) for c in source if int(c) in known]
        if ids:
            return ids
    return sorted(known)


async def _window_rows(
    ctx: Any, chat_ids: Sequence[int], since: datetime, until: datetime
) -> list[Row]:
    from sqlalchemy import select

    from app.db.models import Message

    async with ctx.session_factory() as session:
        res = await session.execute(
            select(
                Message.id,
                Message.chat_id,
                Message.tg_msg_id,
                Message.published_at,
                Message.text,
                Message.media_type,
            )
            .where(
                Message.chat_id.in_([int(c) for c in chat_ids]),
                Message.published_at >= since,
                Message.published_at < until,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.published_at.desc(), Message.id.desc())
            .limit(EXTRACT_MAX_MSGS)
        )
        return [
            Row(
                message_id=int(r[0]),
                chat_id=int(r[1]),
                tg_msg_id=int(r[2]),
                date=r[3],
                text=str(r[4] or ""),
                media_type=str(r[5] or ""),
            )
            for r in res.all()
        ]


def _cited_hits(ctx: Any, ex: Extraction, rows: Sequence[Row]) -> list[Any]:
    """Jadvaldagi sitata tokenlari dalilda bo'lsin (aks holda `strip_invalid_citations`)."""
    from app.services import search as S

    by_id = {r.message_id: r for r in rows}
    wanted: list[int] = []
    for items in ex.items.values():
        for it in items[:ROWS_PER_KIND]:
            if it.message_id in by_id and it.message_id not in wanted:
                wanted.append(it.message_id)
    hits = []
    for mid in wanted:
        r = by_id[mid]
        chat = ctx.chats_by_id.get(r.chat_id)
        hits.append(
            S.Hit(
                message_id=r.message_id,
                tg_msg_id=r.tg_msg_id,
                score=0.0,
                source="extract",
                chat_id=r.chat_id,
                chat_title=chat.title if chat is not None else "",
                date=r.date,
                text=U.neutralize(strip_phones(r.text))[:600],
                media_type=r.media_type,
            )
        )
    return hits


async def run_extract(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    from app.services.skills import SkillResult

    kinds = normalize_kinds(args.get("kind"), ctx.raw_question)
    notes = [EXTRACT_NOTE, OF.FORMAT_NOTE]
    source = "evidence"
    if ctx.ranked:
        rows = _rows_from_hits(ctx.ranked)
    else:
        source = "window"
        now = datetime.now(UTC)
        since = getattr(ctx.plan, "since", None)
        until = getattr(ctx.plan, "until", None) or now
        if not isinstance(since, datetime) or until <= since:
            since, until = now - timedelta(days=DEFAULT_DAYS), now
            notes.append(f"extract: no period in the question — used the last {DEFAULT_DAYS} days")
        chat_ids = _scope(ctx, args)
        rows = await _window_rows(ctx, chat_ids, since, until) if chat_ids else []
        if len(rows) >= EXTRACT_MAX_MSGS:
            notes.append(f"extract: only the newest {EXTRACT_MAX_MSGS} messages were scanned")
    ex = extract(rows, kinds)
    if _PHONE_ASK_RE.search(ctx.raw_question):
        notes.append("extract: phone numbers are never output (privacy) — give only ex_phone_n")
    if source == "window":
        hits = _cited_hits(ctx, ex, rows)
        taint = getattr(ctx, "taint", None)
        if taint is not None and hasattr(taint, "add_hits"):
            taint.add_hits(hits)
        ctx.add_evidence(hits)
    block = render_block(ex, kinds, getattr(ctx, "locale", "uz"))
    if block:
        ctx.claims.append(OF.wrap_block(block))
    summary: dict[str, Any] = {"source": source, "messages": ex.scanned, "phones": ex.phones}
    summary.update({k: ex.n(k) for k in kinds})
    return SkillResult(facts=facts_for(ex, kinds), notes=notes, summary=summary)


__all__ = [
    "EXTRACT_NOTE",
    "KINDS",
    "PHONE_RE",
    "Extraction",
    "Item",
    "Row",
    "count_phones",
    "domain_of",
    "extract",
    "facts_for",
    "find_addresses",
    "find_dates",
    "find_files",
    "find_links",
    "find_mentions",
    "find_prices",
    "normalize_kinds",
    "render_block",
    "run_extract",
    "strip_phones",
]
