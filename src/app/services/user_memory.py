"""Foydalanuvchi xotirasi — alias / afzallik / fakt (migratsiya 0026 `user_memories`).

Muammo: foydalanuvchi har suhbatda "Ishxona deganda «Ish chat» guruhini
nazarda tutaman", "javobni ruscha ber", "men Samarqanddaman" ni qayta aytardi.
Suhbat holati (`conv_state`) bitta suhbat ichida yashaydi; bu esa suhbatlar
**orasida**.

Rozilik modeli (qaror): xotira **sukut bo'yicha yoqiq**, lekin to'liq ko'rinadi:

* har saqlangan yozuv Sozlamalar → Xotira'da (tur, qiymat, manba suhbat, sana)
  — alohida o'chirish / o'chirib qo'yish, global kalit (`users.memory_enabled`);
* yordamchi saqlaganini **bitta qisqa qator** bilan aytadi ("Eslab qoldim: …");
* kalit o'chiq → hech narsa ishlatilmaydi **va** saqlanmaydi.

Nimadan olinadi: **faqat** foydalanuvchining o'z turn'idan (`question`) —
tarix, dalil, xabar matni, oldingi javob hech qachon. Sir (telefon, kod, parol,
karta, token) hech qachon saqlanmaydi: `is_sensitive` turn darajasida (bunday
turn'dan umuman hech narsa olinmaydi) va qiymat darajasida.

Ajratish: avval deterministik naqshlar (uz/ru/en), keyin **bitta** arzon
`Task.ROUTE` chaqiruvi — faqat turn'da eslatma so'zi bo'lsa ("esla", "eslab
qol", "unutma", "bundan keyin", "har doim", "запомни", "remember") va naqshlar
hech narsa topmagan bo'lsa. Har turn'da LLM yo'q. LLM qaytargan alias/fakt
qiymati foydalanuvchi matnida bo'lishi shart (`_grounded`).

Ishlatish (`orchestrator.run_pipeline`): planner va sintez ixcham
`user_memory` blokini oladi; savolda nomlangan alias chatga **kodda**
yechiladi (`alias_matches`) va `conv_state.resolved_entities` ga tushadi;
til afzalligi `ctx.locale` ni, uzunlik — `output_budget` ni belgilaydi.

Foydalanuvchiga ≤`MAX_ITEMS` (100) yozuv; dedup fold qilingan kalit bo'yicha
(`fold_key`), bazada `uq_user_memory_key` oxirgi to'siq.
"""

# Naqshlar uch tilda (kirill ham) — mahsulot matni, ASCII o'xshashligi emas.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update

from app.db.models import User, UserMemory
from app.llm import LLMError, Msg, Task
from app.logging import get_logger, scrub_text
from app.services import textnorm as TN
from app.services import untrusted as U

log = get_logger(__name__)

KINDS = ("fact", "preference", "alias")
MAX_ITEMS = 100
MAX_KEY_CHARS = 80
MAX_VALUE_CHARS = 200
#: Planner/sintez blokiga tushadigan yozuvlar qopqog'i (token uchun).
MAX_BLOCK_ITEMS = 30
#: LLM ajratish chaqiruvi — kichik JSON.
EXTRACT_MAX_TOKENS = 300
EXTRACT_MAX_ITEMS = 3
#: Alias kaliti "meni X deb chaqir" uchun.
SELF_KEY = "self"
#: Afzallik kalitlari va ruxsat etilgan qiymatlari.
PREF_VALUES: dict[str, frozenset[str]] = {
    "language": frozenset({"uz", "ru", "en"}),
    "length": frozenset({"short", "detailed"}),
    "format": frozenset({"table", "list"}),
}

ChatFn = Callable[..., Awaitable[Any]]


# ─── sirlar: hech qachon saqlanmaydi ─────────────────────────────────────────

#: Sir so'zlari — turn'da bo'lsa undan **hech narsa** olinmaydi.
SENSITIVE_WORD_RE = re.compile(
    r"\bparol\w*|\bpassword\w*|\bpasscode\w*|парол\w*|\bkod\w*|\bcode\w*|(?<!\w)код\w*"
    r"|\bpin\b|пин-?код\w*|\btoken\w*|токен\w*|\bsecret\w*|секрет\w*|\bmaxfiy\w*"
    r"|\bcvv\b|\bcvc\b|karta\s+raqam\w*|card\s+number|номер\s+карт\w*|\b2fa\b|\botp\b"
    r"|\bsms\b|смс|\bapi[\s_-]?key\w*|\bsession\w*|сесси\w*",
    re.I,
)
#: Telefon raqami (+998 90 123-45-67, 901234567).
PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{6,}\d")
#: 4+ ketma-ket raqam — kod, PIN, karta, hisob raqami.
LONG_DIGITS_RE = re.compile(r"\d{4,}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def is_sensitive(text: str | None) -> bool:
    """Matnda sir bo'lishi mumkinmi: telefon, kod/parol so'zi, 4+ raqam, e-mail, token shakli."""
    body = TN.strip_invisible(text or "")
    if not body:
        return False
    return bool(
        SENSITIVE_WORD_RE.search(body)
        or any(sum(ch.isdigit() for ch in m.group(0)) >= 7 for m in PHONE_RE.finditer(body))
        or LONG_DIGITS_RE.search(body)
        or _EMAIL_RE.search(body)
        or scrub_text(body) != body
    )


# ─── eslatma so'zi (LLM darvozasi) ───────────────────────────────────────────

CUE_RE = re.compile(
    r"\besla\w*|\beslab\s+qol\w*|\bunutma\w*|\bbundan\s+(?:buyon|keyin)\b|\bhar\s+doim\b"
    r"|\bdoimo\b|\bdoim\b|запомн\w*|не\s+забыва\w*|не\s+забудь\w*|всегда|впредь|отныне"
    r"|\bremember\b|\bdon'?t\s+forget\b|\bfrom\s+now\s+on\b|\balways\b|\bgoing\s+forward\b",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[‘’ʻʼ`]", "'", TN.strip_invisible(text or "")).strip()


def has_cue(text: str) -> bool:
    return bool(CUE_RE.search(_norm(text)))


def fold_key(key: str) -> str:
    return re.sub(r"\s+", " ", TN.uz_fold(key or "")).strip()


# ─── deterministik naqshlar ──────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Candidate:
    kind: str
    key: str
    value: str


_QUOTES = "\"'«»“”‘’`"
_LEAD_RE = re.compile(
    r"^(?:(?:eslab\s+qol\w*|esla\w*|unutma\w*|bundan\s+(?:buyon|keyin)|har\s+doim|doimo|men|biz"
    r"|запомни\w*|remember(?:\s+that)?|from\s+now\s+on|always|и|va|and|ok|xo'p|mayli)"
    r"[\s,:;-]*)+",
    re.I,
)
#: Qiymat oxiridagi tur so'zi — faqat **turlangan** shakli ("Ish chat guruhini",
#: "Oila chatni") yoki ingliz "group/channel"; yalang "chat" sarlavhaning qismi bo'lishi mumkin.
_KIND_TAIL_RE = re.compile(
    r"\s+(?:guruh(?:i|ni|ini|iga|ida)|gruppa\w+|chat(?:i|ni|ini|iga|ida)"
    r"|kanal(?:i|ni|ini|iga|ida)|group|channel)$",
    re.I,
)
#: Qiymat boshidagi tur so'zi / artikl: "группу Ish chat", "the Ish chat".
_KIND_HEAD_RE = re.compile(r"^(?:групп\w*|чат\w*|канал\w*|the|a|my|our)\s+", re.I)

_ALIAS_RES = (
    re.compile(
        r"(?P<key>[^,.;:!?\n]{1,60}?)\s+deganda(?:\s+(?:men|biz))?\s+"
        r"(?P<val>[^,.;:!?\n]{1,80}?)\s*(?:ni\s+)?nazarda\s+tut",
        re.I,
    ),
    re.compile(
        r"(?:когда\s+я\s+(?:говорю|пишу)|под)\s+(?P<key>[^,.;:!?\n]{1,60}?),?\s+"
        r"я\s+имею\s+в\s+виду\s+(?P<val>[^,.;:!?\n]{1,80})",
        re.I,
    ),
    re.compile(
        r"(?:when\s+i\s+(?:say|write)|by)\s+(?P<key>[^,.;:!?\n]{1,60}?),?\s+i\s+mean\s+"
        r"(?P<val>[^,.;:!?\n]{1,80})",
        re.I,
    ),
)
_SELF_RES = (
    re.compile(r"\bmeni\s+(?P<val>[^,.;:!?\n]{1,40}?)\s+deb\s+chaqir", re.I),
    re.compile(r"\bmening\s+ismim\s+(?P<val>[A-ZА-ЯЁЎҚҒҲ][\w'-]{1,30})"),
    re.compile(r"(?:зови|называй)\s+меня\s+(?P<val>[^,.;:!?\n]{1,40})", re.I),
    re.compile(r"\bcall\s+me\s+(?P<val>[^,.;:!?\n]{1,40})", re.I),
    re.compile(r"\bmy\s+name\s+is\s+(?P<val>[A-Z][\w'-]{1,30})"),
)

_LANG_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bruscha\b|\brus\s+tilida\b|по-русски|на\s+русском|\bin\s+russian\b", re.I),
        "ru",
    ),
    (
        re.compile(
            r"\binglizcha\b|\bingliz\s+tilida\b|по-английски|на\s+английском|\bin\s+english\b",
            re.I,
        ),
        "en",
    ),
    (
        re.compile(
            r"\bo'?zbekcha\b|\bo'?zbek\s+tilida\b|по-узбекски|на\s+узбекском|\bin\s+uzbek\b",
            re.I,
        ),
        "uz",
    ),
)
_REPLY_VERB_RE = re.compile(
    r"\bjavob\w*|\byoz\w*|\bgapir\w*|\bber\b|\bbering\b|отвеч\w*|\bпиш\w*|ответ\w*"
    r"|\banswer\w*|\breply\w*|\brespond\w*|\bwrite\b|\btalk\b",
    re.I,
)
_SHORT_RE = re.compile(
    r"\bqisqa\w*|\bixcham\w*|кратк\w*|коротк\w*|короче|\bbrief\w*|\bshort\w*|\bconcise\w*", re.I
)
_DETAIL_RE = re.compile(r"\bbatafsil\w*|подробн\w*|\bin\s+detail\b|\bdetailed\b", re.I)
_TABLE_RE = re.compile(r"\bjadval\w*|таблиц\w*|\btables?\b", re.I)
_LIST_RE = re.compile(r"\bro'?yxat\w*|списк\w*|список|\blists?\b|\bbullet\w*", re.I)
#: Bir martalik so'rov belgisi: "shuni qisqa yoz" — oldingi javob, afzallik emas.
_REFERENT_RE = re.compile(
    r"\b(?:shu\w*|buni|bunga|uni|o'sha\w*|osha)\b|(?<!\w)(?:это\w*|эт[иу])(?!\w)"
    r"|\b(?:this|that|it)\b",
    re.I,
)
#: Savol turn'i: "bugun nima bo'ldi? qisqa yoz" — shu javob uchun, doimiy emas.
_QUESTION_RE = re.compile(
    r"\?|\b(?:nima\w*|qanday|qancha|nechta|kim\w*|qachon|qaysi|qayer\w*)\b"
    r"|(?<!\w)(?:что|как|сколько|кто|когда|где|какой|какие)(?!\w)"
    r"|\b(?:what|how|who|when|where|which)\b",
    re.I,
)

_LOC_RES = (
    re.compile(r"\bmen\s+(?P<val>[A-ZА-ЯЁЎҚҒҲ][\w'-]{2,30}?)(?:da|ta)(?:man|yashayman)\b"),
    re.compile(r"\bmen\s+(?P<val>[A-ZА-ЯЁЎҚҒҲ][\w'-]{2,30}?)(?:da|ta)\s+yashayman\b"),
    re.compile(r"(?:я|Я)\s+(?:живу|нахожусь)\s+в\s+(?P<val>[А-ЯЁA-Z][\w-]{2,30})"),
    re.compile(r"\b[Ii]\s+(?:live|am\s+based|am\s+located)\s+in\s+(?P<val>[A-Z][\w-]{2,30})"),
)
_HOURS = r"\d{1,2}(?:[:.]\d{2})?"
_HOURS_RES = (
    re.compile(
        rf"\bish\s+vaqtim\s*(?:[:=\-–—]\s*)?(?:soat\s+)?(?P<val>{_HOURS}\s*(?:[-–—]|dan)\s*{_HOURS})",
        re.I,
    ),
    re.compile(
        rf"(?:рабочее\s+время|я\s+работаю)\s*(?:[:\-–—]\s*|с\s+)?(?P<val>{_HOURS}\s*(?:[-–—]|до)\s*{_HOURS})",
        re.I,
    ),
    re.compile(
        rf"\b(?:my\s+)?work(?:ing)?\s+hours\s+(?:are\s+)?(?P<val>{_HOURS}\s*(?:[-–—]|to)\s*{_HOURS})",
        re.I,
    ),
)


def _clean(text: str, *, max_chars: int) -> str:
    out = re.sub(r"\s+", " ", text or "").strip().strip(_QUOTES).strip()
    return out[:max_chars].strip()


def _alias_key(raw: str) -> str:
    key = _clean(raw, max_chars=120)
    prev = None
    while prev != key:
        prev = key
        key = _LEAD_RE.sub("", key).strip().strip(_QUOTES).strip()
    words = key.split()
    return _clean(" ".join(words[-4:]), max_chars=MAX_KEY_CHARS)


def _alias_value(raw: str) -> str:
    val = _clean(raw, max_chars=120)
    had_kind = bool(_KIND_TAIL_RE.search(val))
    val = _KIND_TAIL_RE.sub("", val).strip()
    val = _KIND_HEAD_RE.sub("", val).strip().strip(_QUOTES).strip()
    # "Otabekni" — tushum kelishigi (tur so'zi bo'lmasa, odam/chat nomining o'zida).
    if not had_kind:
        for suffix in ("ini", "ni"):
            if val.lower().endswith(suffix) and len(val) - len(suffix) >= 3:
                val = val[: -len(suffix)]
                break
    return _clean(val, max_chars=MAX_VALUE_CHARS)


def extract_rules(text: str) -> list[Candidate]:
    """Deterministik naqshlar — LLM'siz. Sir bo'lishi mumkin bo'lgan turn → bo'sh ro'yxat."""
    body = _norm(text)
    if not body or is_sensitive(body):
        return []
    out: list[Candidate] = []
    cue = has_cue(body)
    one_off = bool(_REFERENT_RE.search(body)) or bool(_QUESTION_RE.search(body))
    for rx in _ALIAS_RES:
        m = rx.search(body)
        if m:
            key, val = _alias_key(m["key"]), _alias_value(m["val"])
            if key and val and fold_key(key) != fold_key(val):
                out.append(Candidate("alias", key, val))
            break
    for rx in _SELF_RES:
        m = rx.search(body)
        if m:
            val = _clean(m["val"], max_chars=60)
            if val:
                out.append(Candidate("alias", SELF_KEY, val))
            break
    replyish = bool(_REPLY_VERB_RE.search(body))
    if replyish and (cue or not one_off):
        for rx, lang in _LANG_RES:
            if rx.search(body):
                out.append(Candidate("preference", "language", lang))
                break
        if _DETAIL_RE.search(body):
            out.append(Candidate("preference", "length", "detailed"))
        elif _SHORT_RE.search(body):
            out.append(Candidate("preference", "length", "short"))
    if cue:
        if _TABLE_RE.search(body):
            out.append(Candidate("preference", "format", "table"))
        elif _LIST_RE.search(body):
            out.append(Candidate("preference", "format", "list"))
    for rx in _LOC_RES:
        m = rx.search(body)
        if m:
            out.append(Candidate("fact", "location", _clean(m["val"], max_chars=60)))
            break
    for rx in _HOURS_RES:
        m = rx.search(body)
        if m:
            val = re.sub(r"\s*(?:dan|до|to)\s*", "–", m["val"], flags=re.I)
            out.append(Candidate("fact", "work_hours", _clean(val, max_chars=40)))
            break
    return [c for c in _dedupe(out) if valid(c)]


def _dedupe(items: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for c in items:
        k = (c.kind, fold_key(c.key))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _fact_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", TN.uz_fold(key)).strip("_")[:40]


def valid(c: Candidate) -> bool:
    """Tur, kalit, qiymat tekshiruvi + sir qo'riqchisi."""
    if c.kind not in KINDS or not c.key or not c.value:
        return False
    if len(c.key) > MAX_KEY_CHARS or len(c.value) > MAX_VALUE_CHARS:
        return False
    if c.kind == "preference":
        allowed = PREF_VALUES.get(c.key)
        if allowed is None or c.value not in allowed:
            return False
    if c.kind == "fact" and _fact_key(c.key) != c.key:
        return False
    return not is_sensitive(f"{c.key} {c.value}")


def _grounded(c: Candidate, text: str) -> bool:
    """LLM qiymati foydalanuvchi matnida bormi — "faqat user turn'idan" kafolati."""
    if c.kind == "preference":
        return True
    folded = fold_key(text)
    words = [w for w in re.findall(r"[a-z0-9]+", fold_key(c.value)) if len(w) >= 3]
    if not words:
        return bool(fold_key(c.value)) and fold_key(c.value) in folded
    if c.kind == "alias" and c.key != SELF_KEY:
        key_words = [w for w in re.findall(r"[a-z0-9]+", fold_key(c.key)) if len(w) >= 2]
        if not all(w in folded for w in key_words):
            return False
    return all(w in folded for w in words)


def parse_llm_items(raw: str, text: str) -> list[Candidate]:
    """`{"items": [...]}` → tekshirilgan nomzodlar (≤3). Buzuq JSON → bo'sh."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[Candidate] = []
    for item in items[: EXTRACT_MAX_ITEMS * 2]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        key = _clean(str(item.get("key") or ""), max_chars=MAX_KEY_CHARS)
        value = _clean(str(item.get("value") or ""), max_chars=MAX_VALUE_CHARS)
        if kind == "fact":
            key = _fact_key(key)
        elif kind == "preference":
            key, value = key.lower(), value.lower()
        c = Candidate(kind, key, value)
        if valid(c) and _grounded(c, text):
            out.append(c)
    return _dedupe(out)[:EXTRACT_MAX_ITEMS]


async def extract(text: str, chat: ChatFn | None) -> list[Candidate]:
    """Naqshlar, keyin (faqat eslatma so'zi bo'lsa va naqsh topmasa) **bitta** `Task.ROUTE`.

    `chat` — `meter.chat` (narx shu yugurish hisobiga). LLM xatosi — jim bo'sh.
    """
    body = _norm(text)
    if not body or is_sensitive(body):
        if body and has_cue(body):
            log.info("memory.sensitive_skipped")
        return []
    found = extract_rules(body)
    if found or chat is None or not has_cue(body):
        return found
    from app.services.prompts import MEMORY_EXTRACT_PROMPT

    try:
        res = await chat(
            Task.ROUTE,
            [
                Msg.system(MEMORY_EXTRACT_PROMPT),
                Msg.user(f"<user_turn>{U.neutralize(body)[:1000]}</user_turn>"),
            ],
            json_mode=True,
            max_tokens=EXTRACT_MAX_TOKENS,
        )
    except LLMError as exc:
        log.warning("memory.extract_failed", error=str(exc)[:200])
        return []
    return parse_llm_items(getattr(res, "text", "") or "", body)


# ─── yozuvlar ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MemoryItem:
    id: int
    kind: str
    key: str
    value: str
    source_conversation_id: int | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    enabled: bool = True

    @classmethod
    def of(cls, row: Any) -> MemoryItem:
        return cls(
            id=int(row.id),
            kind=str(row.kind),
            key=str(row.key),
            value=str(row.value),
            source_conversation_id=row.source_conversation_id,
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            enabled=bool(row.enabled),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
            "source_conversation_id": self.source_conversation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class Memory:
    """Yugurish uchun xotira: global kalit + **yoqiq** yozuvlar."""

    enabled: bool = False
    items: list[MemoryItem] = field(default_factory=list)

    def pref(self, key: str) -> str:
        for it in self.items:
            if it.kind == "preference" and it.key == key and it.enabled:
                return it.value
        return ""

    def self_name(self) -> str:
        for it in self.items:
            if it.kind == "alias" and it.key == SELF_KEY and it.enabled:
                return it.value
        return ""

    def of_kind(self, kind: str) -> list[MemoryItem]:
        return [it for it in self.items if it.kind == kind and it.enabled]


async def load(session: Any, user_id: int) -> Memory:
    """Global kalit o'chiq yoki foydalanuvchi yo'q → bo'sh (`enabled=False`)."""
    user = await session.get(User, user_id)
    if user is None or not bool(getattr(user, "memory_enabled", True)):
        return Memory(enabled=False)
    rows = (
        (
            await session.execute(
                select(UserMemory)
                .where(UserMemory.user_id == user_id, UserMemory.enabled.is_(True))
                .order_by(UserMemory.id)
                .limit(MAX_ITEMS)
            )
        )
        .scalars()
        .all()
    )
    return Memory(enabled=True, items=[MemoryItem.of(r) for r in rows])


async def list_all(session: Any, user_id: int) -> list[MemoryItem]:
    rows = (
        (
            await session.execute(
                select(UserMemory)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.created_at.desc(), UserMemory.id.desc())
                .limit(MAX_ITEMS)
            )
        )
        .scalars()
        .all()
    )
    return [MemoryItem.of(r) for r in rows]


async def save(
    session: Any,
    user_id: int,
    candidates: Sequence[Candidate],
    *,
    conversation_id: int | None,
) -> list[MemoryItem]:
    """Nomzodlarni yozadi: fold qilingan kalit bo'yicha dedup, ≤`MAX_ITEMS`.

    Qaytadi — **yangi yoki o'zgargan** yozuvlar ("Eslab qoldim" qatori shular
    haqida). Bir xil qiymat qayta aytilsa — hech narsa (qator yo'q).
    O'chirib qo'yilgan yozuvni foydalanuvchi yana aytsa — qayta yoqiladi.
    """
    cands = [c for c in _dedupe(candidates) if valid(c)]
    if not cands:
        return []
    rows = list(
        (await session.execute(select(UserMemory).where(UserMemory.user_id == user_id)))
        .scalars()
        .all()
    )
    by_key = {(r.kind, fold_key(r.key)): r for r in rows}
    count = len(rows)
    saved: list[Any] = []
    for c in cands:
        row = by_key.get((c.kind, fold_key(c.key)))
        if row is not None:
            if row.value == c.value and row.enabled:
                continue
            row.value = c.value
            row.enabled = True
            row.source_conversation_id = conversation_id
            saved.append(row)
            continue
        if count >= MAX_ITEMS:
            log.info("memory.cap_reached", user_id=user_id, cap=MAX_ITEMS)
            continue
        row = UserMemory(
            user_id=user_id,
            kind=c.kind,
            key=c.key,
            value=c.value,
            source_conversation_id=conversation_id,
            enabled=True,
        )
        session.add(row)
        by_key[(c.kind, fold_key(c.key))] = row
        count += 1
        saved.append(row)
    if saved:
        await session.flush()
    return [MemoryItem.of(r) for r in saved]


async def touch(session: Any, user_id: int, ids: Sequence[int]) -> None:
    if not ids:
        return
    await session.execute(
        update(UserMemory)
        .where(UserMemory.user_id == user_id, UserMemory.id.in_(list(ids)))
        .values(last_used_at=datetime.now(UTC))
    )


async def set_item_enabled(
    session: Any, user_id: int, item_id: int, enabled: bool
) -> MemoryItem | None:
    row = await session.get(UserMemory, item_id)
    if row is None or row.user_id != user_id:
        return None
    row.enabled = bool(enabled)
    await session.flush()
    return MemoryItem.of(row)


async def delete_item(session: Any, user_id: int, item_id: int) -> bool:
    res = await session.execute(
        delete(UserMemory).where(UserMemory.user_id == user_id, UserMemory.id == item_id)
    )
    return bool(getattr(res, "rowcount", 0))


async def delete_all(session: Any, user_id: int) -> int:
    res = await session.execute(delete(UserMemory).where(UserMemory.user_id == user_id))
    return int(getattr(res, "rowcount", 0) or 0)


async def get_enabled(session: Any, user_id: int) -> bool:
    user = await session.get(User, user_id)
    return bool(getattr(user, "memory_enabled", True)) if user is not None else False


async def set_enabled(session: Any, user_id: int, enabled: bool) -> bool:
    user = await session.get(User, user_id)
    if user is None:
        return False
    user.memory_enabled = bool(enabled)
    await session.flush()
    return True


# ─── yugurish ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RunMemory:
    """Bitta orkestrator yugurishi uchun: xotira, shu turn saqlagani, yechilgan aliaslar."""

    memory: Memory = field(default_factory=Memory)
    saved: list[MemoryItem] = field(default_factory=list)
    #: `(yozuv, chat maslahati)` — savolda nomlangan va chatga yechilgan aliaslar.
    aliases: list[tuple[MemoryItem, Any]] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.memory.enabled


async def prepare(
    session_factory: Any,
    chat: ChatFn | None,
    *,
    user_id: int,
    conversation_id: int | None,
    question: str,
) -> RunMemory:
    """Yuklash + shu turn'dan ajratish + saqlash. Xato — javobni buzmaydi (bo'sh xotira).

    Kalit o'chiq bo'lsa ajratish ham, LLM ham, yozish ham **yo'q**.
    """
    try:
        async with session_factory() as session:
            memory = await load(session, user_id)
    except Exception as exc:
        log.warning("memory.load_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return RunMemory()
    out = RunMemory(memory=memory)
    if not memory.enabled:
        return out
    candidates = await extract(question, chat)
    if not candidates:
        return out
    try:
        async with session_factory() as session:
            out.saved = await save(session, user_id, candidates, conversation_id=conversation_id)
    except Exception as exc:
        log.warning("memory.save_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return out
    if out.saved:
        ids = {it.id for it in out.saved}
        memory.items = [it for it in memory.items if it.id not in ids] + list(out.saved)
        log.info("memory.saved", user_id=user_id, kinds=[it.kind for it in out.saved])
    return out


async def touch_used(session_factory: Any, run: RunMemory, *, user_id: int) -> None:
    """`last_used_at` — bitta UPDATE; xato javobni buzmaydi."""
    ids = used_ids(run)
    if not ids:
        return
    try:
        async with session_factory() as session:
            await touch(session, user_id, ids)
    except Exception as exc:
        log.warning("memory.touch_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")


def alias_matches(
    question: str, items: Sequence[MemoryItem], chats: Sequence[Any]
) -> list[tuple[MemoryItem, Any]]:
    """Savolda nomlangan aliaslar → chat maslahati (`agent.ChatHint`) — **kodda**.

    Kalitning har so'zi savolda bo'lsin (qo'shimchali shakllar bilan,
    `agent.chat_hints` bilan bir mezon). Qiymat chat sarlavhasiga
    `chat_hints` orqali yechiladi: eng yaxshi moslik ≥0.5 va teng raqibsiz.
    Chatga yechilmagan alias (odam, mavzu) ro'yxatga kirmaydi — u blokda
    oddiy satr bo'lib qoladi.
    """
    from app.services import agent as AG

    forms: set[str] = set()
    for word in AG._hint_words(question):
        if len(word) >= 2:
            forms |= AG._word_forms(word)
    if not forms:
        return []
    out: list[tuple[MemoryItem, Any]] = []
    seen: set[int] = set()
    for item in items:
        if item.kind != "alias" or item.key == SELF_KEY or not item.enabled:
            continue
        words = [w for w in AG._hint_words(item.key) if len(w) >= 2]
        if not words or not all(
            max(AG._similar(f, w) for f in forms) >= AG.CHAT_HINT_MIN_RATIO for w in words
        ):
            continue
        hints = AG.chat_hints(item.value, list(chats), limit=2)
        if not hints or hints[0].score < 0.5:
            continue
        if len(hints) > 1 and hints[1].score == hints[0].score:
            continue
        top = hints[0]
        if int(top.chat_id) in seen:
            continue
        seen.add(int(top.chat_id))
        out.append((item, top))
    return out


def _ordered(memory: Memory, aliases: Sequence[tuple[MemoryItem, Any]]) -> list[MemoryItem]:
    matched = {it.id for it, _h in aliases}
    first = [it for it in memory.items if it.kind in ("preference", "fact") and it.enabled]
    first += [it for it in memory.items if it.kind == "alias" and it.key == SELF_KEY]
    first += [it for it, _h in aliases]
    rest = [
        it
        for it in memory.items
        if it.kind == "alias" and it.key != SELF_KEY and it.id not in matched and it.enabled
    ]
    return (first + rest)[:MAX_BLOCK_ITEMS]


def render_block(run: RunMemory) -> str:
    """Planner uchun ixcham `user_memory` bloki (bo'sh xotira → bo'sh satr).

    Qiymatlar foydalanuvchining o'z so'zlari, alias chat sarlavhalari esa
    Telegram'dan — blok konvert ichida, har satr `neutralize` dan o'tgan.
    """
    if not run.memory.enabled or not run.memory.items:
        return ""
    resolved = {it.id: hint for it, hint in run.aliases}
    lines: list[str] = []
    for it in _ordered(run.memory, run.aliases):
        key = U.attr(it.key, max_chars=MAX_KEY_CHARS)
        value = U.attr(it.value, max_chars=MAX_VALUE_CHARS)
        if it.kind == "alias" and it.key == SELF_KEY:
            lines.append(f'alias self = "{value}" (address the user so)')
        elif it.kind == "alias":
            hint = resolved.get(it.id)
            tail = (
                f' → chat "{U.attr(str(hint.title), max_chars=80)}" [id:{int(hint.chat_id)}] '
                "(named in this question — resolved by code)"
                if hint is not None
                else ""
            )
            lines.append(f'alias "{key}" = "{value}"{tail}')
        else:
            lines.append(f"{it.kind} {key} = {value}")
    body = "\n".join(U.neutralize(x) for x in lines)
    out = (
        f'<untrusted_data kind="user_memory" count="{len(lines)}">\n{body}\n</untrusted_data>\n'
        "user_memory: saved by the user (Settings → Memory). An alias named in the question is "
        "already resolved by code to the chat id shown — target it. Preferences set the "
        "answer's language/length/format unless this question asks otherwise; facts are plain "
        "facts about the user (rule 16)."
    )
    if run.saved:
        out += (
            f"\nmemory_saved: this turn saved {len(run.saved)} item(s) — if the user only asked "
            "to remember something, plan one `answer_direct` step (rule 16)."
        )
    return out


_LANG_NAME = {"uz": "Uzbek", "ru": "Russian", "en": "English"}


def synth_note(run: RunMemory) -> str:
    """Sintez uchun qisqa izoh — afzallik va faktlar (qiymatlar `neutralize` dan o'tgan)."""
    mem = run.memory
    if not mem.enabled or not mem.items:
        return ""
    parts: list[str] = []
    lang = mem.pref("language")
    if lang:
        parts.append(f"answer language: {_LANG_NAME.get(lang, lang)}")
    length = mem.pref("length")
    if length:
        parts.append(f"length: {length}")
    fmt = mem.pref("format")
    if fmt == "table":
        parts.append("format: a Markdown table whenever 3+ comparable items are listed")
    elif fmt == "list":
        parts.append("format: bullet lists")
    facts = [
        f"{U.attr(it.key, max_chars=40)}={U.attr(it.value, max_chars=80)}"
        for it in mem.of_kind("fact")[:8]
    ]
    name = mem.self_name()
    out: list[str] = []
    if parts:
        out.append(
            "user preferences (saved memory — follow unless this question asks otherwise): "
            + "; ".join(parts)
        )
    if facts:
        out.append("facts about the user (saved memory): " + ", ".join(facts))
    if name:
        out.append(f'address the user as "{U.attr(name, max_chars=60)}"')
    return U.neutralize("\n".join(out))


_SAVED_LABEL = {
    "uz": ("Eslab qoldim", "Sozlamalar → Xotira"),
    "ru": ("Запомнил", "Настройки → Память"),
    "en": ("Remembered", "Settings → Memory"),
}
_PREF_TEXT = {
    "uz": {
        "language": {"uz": "javob tili — o'zbekcha", "ru": "javob tili — ruscha",
                     "en": "javob tili — inglizcha"},
        "length": {"short": "javoblar qisqa", "detailed": "javoblar batafsil"},
        "format": {"table": "jadval ko'rinishi", "list": "ro'yxat ko'rinishi"},
    },
    "ru": {
        "language": {"uz": "язык ответа — узбекский", "ru": "язык ответа — русский",
                     "en": "язык ответа — английский"},
        "length": {"short": "короткие ответы", "detailed": "подробные ответы"},
        "format": {"table": "в виде таблицы", "list": "в виде списка"},
    },
    "en": {
        "language": {"uz": "answer language — Uzbek", "ru": "answer language — Russian",
                     "en": "answer language — English"},
        "length": {"short": "short answers", "detailed": "detailed answers"},
        "format": {"table": "as a table", "list": "as a list"},
    },
}  # fmt: skip
_SELF_TEXT = {"uz": "sizga «{v}» deb murojaat", "ru": "обращаться к вам «{v}»",
              "en": "call you “{v}”"}  # fmt: skip


_FACT_TEXT = {
    "uz": {"location": "shahar", "work_hours": "ish vaqti"},
    "ru": {"location": "город", "work_hours": "рабочее время"},
    "en": {"location": "city", "work_hours": "working hours"},
}


def describe(it: MemoryItem, locale: str) -> str:
    loc = locale if locale in _SAVED_LABEL else "uz"
    value = U.attr(it.value, max_chars=60)
    if it.kind == "preference":
        return _PREF_TEXT[loc].get(it.key, {}).get(it.value, f"{it.key}: {value}")
    if it.kind == "alias" and it.key == SELF_KEY:
        return _SELF_TEXT[loc].format(v=value)
    if it.kind == "alias":
        return f"«{U.attr(it.key, max_chars=40)}» = {value}"
    label = _FACT_TEXT[loc].get(it.key) or U.attr(it.key, max_chars=40).replace("_", " ")
    return f"{label}: {value}"


def saved_note(saved: Sequence[MemoryItem], locale: str) -> str:
    """Bitta qisqa qator: "Eslab qoldim: … (Sozlamalar → Xotira)". Bo'sh → bo'sh satr."""
    if not saved:
        return ""
    loc = locale if locale in _SAVED_LABEL else "uz"
    label, where = _SAVED_LABEL[loc]
    items = "; ".join(describe(it, loc) for it in saved[:4])
    return U.neutralize(f"_{label}: {items} ({where})_")


def used_ids(run: RunMemory) -> list[int]:
    """`last_used_at` yangilanadiganlari: afzallik/fakt/self + savolda nomlangan aliaslar."""
    if not run.memory.enabled:
        return []
    ids = [
        it.id
        for it in run.memory.items
        if it.kind in ("preference", "fact") or (it.kind == "alias" and it.key == SELF_KEY)
    ]
    ids += [it.id for it, _h in run.aliases]
    return sorted(set(ids))


__all__ = [
    "KINDS",
    "MAX_ITEMS",
    "Candidate",
    "Memory",
    "MemoryItem",
    "RunMemory",
    "alias_matches",
    "delete_all",
    "delete_item",
    "extract",
    "extract_rules",
    "has_cue",
    "is_sensitive",
    "list_all",
    "load",
    "parse_llm_items",
    "prepare",
    "render_block",
    "save",
    "saved_note",
    "set_enabled",
    "set_item_enabled",
    "synth_note",
    "touch",
    "used_ids",
]
