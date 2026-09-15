"""Entity chiqarish va indekslash — regex bilan, LLM'siz (arxitektura v2, 3.5-band).

Nega kerak: "nurbek nima degandi?" degan savolda leksik kanal `nurbek` so'zini
matn ichidan qidiradi, vektor kanali esa ma'noga qaraydi — ikkalasi ham
`@nurbek` mention'ini, `t.me/nurbek` havolasini yoki `#deploy` tegini
**aniq** signal sifatida ko'rmaydi. Entity indeksi shu bo'shliqni yopadi:
xabar ichidagi identifikatorlar alohida jadvalda (`node_entities`) yotadi va
aniq tenglik bo'yicha topiladi.

Ikki qat'iy chegara:

  * **LLM chaqirilmaydi.** Hammasi sof regex — deterministik, bepul,
    prompt injection yuzasi yo'q (3-tamoyil: deterministik ish — koddan).
  * **Telefon raqami va email hech qachon chiqarilmaydi.** `phone_like`
    turi ataylab **yo'q**: natija LLM promptiga va indeksga tushadi, raqam
    u yerda kerak emas (`list_members` bilan bir xil mezon). Email esa
    skanerdan **oldin** matndan olib tashlanadi — aks holda `a@b.com`
    dagi `@b` username, `b.com` esa domen bo'lib ko'rinardi.
    `tests/test_entities.py` ikkalasini ham qo'riqlaydi.

Chiqariladigan turlar:

  ``username``      ``@nurbek`` (5–32 belgi, harf bilan boshlanadi)
  ``hashtag``       ``#deploy`` / ``#деплой`` (unicode harflar, kirill ham)
  ``cashtag``       ``$BTC`` (Telegram cashtag'i — faqat katta harf)
  ``url_domain``    URL → ro'yxatga olinadigan domen (``www.`` olib tashlanadi)
  ``tme_link``      ``t.me/<x>``, ``t.me/c/<id>/<msg>`` → qiymat = yo'l qismi
  ``mention_name``  **faqat** Telegram `MessageEntityMentionName` dan; erkin
                    matndan ism **taxmin qilinmaydi**

Normalizatsiya (`value`):

  * ``username`` / ``cashtag`` / ``url_domain`` / ``tme_link`` — `lower()`;
  * ``hashtag`` / ``mention_name`` — `uz_fold` (``#Деплой`` va ``#deploy``
    bitta qiymatga tushadi, ``Oʻtkir`` va ``O'tkir`` ham).

`value_raw` — matnda qanday ko'rinsa shunday (UI va debug uchun).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import NodeEntity
from app.logging import get_logger
from app.services.textnorm import uz_fold

log = get_logger(__name__)

# ─── turlar ──────────────────────────────────────────────────────────────────

KIND_USERNAME = "username"
KIND_HASHTAG = "hashtag"
KIND_CASHTAG = "cashtag"
KIND_URL_DOMAIN = "url_domain"
KIND_TME_LINK = "tme_link"
KIND_MENTION_NAME = "mention_name"

#: DB `CHECK` bilan bir xil ro'yxat (migratsiya 0012). `phone_like` **yo'q**.
KINDS: frozenset[str] = frozenset(
    {
        KIND_USERNAME,
        KIND_HASHTAG,
        KIND_CASHTAG,
        KIND_URL_DOMAIN,
        KIND_TME_LINK,
        KIND_MENTION_NAME,
    }
)

#: Bitta xabardan olinadigan entity qopqog'i — spam post 300 ta hashtag bilan
#: keladi, ularning hammasi indeksga tushishi shart emas.
MAX_ENTITIES_PER_MESSAGE = 32
#: Qiymat uzunligi qopqog'i (indeks kaliti uzun bo'lmasin).
MAX_VALUE_CHARS = 128

# ─── regexlar ────────────────────────────────────────────────────────────────

#: Email — **chiqarilmaydi**, skanerdan oldin matndan olib tashlanadi.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
#: URL: sxema bilan, `www.` bilan yoki **sxemasiz `t.me/`** (chatda eng ko'p
#: uchraydigan shakl — `t.me/nurbek` deb yoziladi, `https://` yozilmaydi).
_URL_RE = re.compile(
    r"(?<![\w.@/-])(?:https?://|www\.|(?:t|telegram)\.me/|telegram\.dog/)[^\s<>\"'()\[\]{}]+",
    re.IGNORECASE,
)
#: Telegram username: 5–32 belgi, harf bilan boshlanadi, harf/raqam/pastki chiziq.
_USERNAME_RE = re.compile(r"(?<![\w@./+-])@([A-Za-z][A-Za-z0-9_]{4,31})(?![\w])")
#: Hashtag: birinchi belgi — harf (unicode; kirill ham), keyin \w.
_HASHTAG_RE = re.compile(r"(?<![\w#])#([^\W\d_]\w{0,63})", re.UNICODE)
#: Cashtag: Telegram faqat katta harfni cashtag deb hisoblaydi.
_CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Z]{1,8})(?![\w])")

#: `t.me` oilasi — bu hostlar `url_domain` emas, `tme_link` beradi.
_TME_HOSTS: frozenset[str] = frozenset({"t.me", "telegram.me", "telegram.dog", "telesco.pe"})

#: Ikki bo'g'inli public suffix'lar (PSL'siz minimal ro'yxat). Bularda
#: ro'yxatga olinadigan domen **uch** yorliqdan iborat: `shop.uz` emas,
#: `foo.com.uz`.
_MULTI_SUFFIXES: frozenset[str] = frozenset(
    {
        "com.uz",
        "org.uz",
        "net.uz",
        "gov.uz",
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "com.ru",
        "com.tr",
        "com.br",
        "co.jp",
        "com.au",
        "com.ua",
        "co.kr",
        "com.cn",
    }
)


@dataclass(frozen=True, slots=True)
class Entity:
    """Bitta topilgan identifikator (DB'ga yozilishidan oldingi sof qiymat)."""

    kind: str
    value: str  # normallashtirilgan (lower yoki uz_fold)
    value_raw: str  # matnda qanday ko'ringan bo'lsa


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def _norm(kind: str, raw: str) -> str:
    """Tur bo'yicha normalizatsiya: nomlar `uz_fold`, qolganlari `lower`."""
    value = uz_fold(raw) if kind in {KIND_HASHTAG, KIND_MENTION_NAME} else raw.lower()
    return value.strip()[:MAX_VALUE_CHARS]


def registrable_domain(host: str) -> str:
    """Host → ro'yxatga olinadigan domen (`www.` yo'q, port yo'q, lowercase).

    PSL ishlatilmaydi (kutubxona qo'shmaslik uchun): oxirgi **ikki** yorliq
    olinadi, `_MULTI_SUFFIXES` dagi hollarda esa uchta.

    Examples:
        >>> registrable_domain("WWW.Sub.Example.com:8443")
        'example.com'
        >>> registrable_domain("shop.example.com.uz")
        'example.com.uz'
    """
    h = host.strip().lower().rstrip(".")
    h = h.split("@")[-1].split(":")[0]
    if h.startswith("www."):
        h = h[4:]
    labels = [x for x in h.split(".") if x]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in _MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _split_url(url: str) -> tuple[str, str]:
    """URL → `(host, path)`; sxemasiz (`www.…`) variant ham qabul qilinadi."""
    body = url
    if "://" in body:
        body = body.split("://", 1)[1]
    body = body.split("#", 1)[0]
    host, _, rest = body.partition("/")
    path = rest.split("?", 1)[0]
    return host, path.strip("/")


def _url_entity(url: str) -> Entity | None:
    """URL → `tme_link` (t.me oilasi) yoki `url_domain`."""
    trimmed = url.rstrip(".,;:!?)»\"'")
    host, path = _split_url(trimmed)
    bare = host.lower().removeprefix("www.").split(":")[0]
    if bare in _TME_HOSTS:
        if not path:
            return None
        raw = f"{bare}/{path}"
        return Entity(KIND_TME_LINK, _norm(KIND_TME_LINK, path), raw)
    domain = registrable_domain(host)
    if "." not in domain:
        return None
    return Entity(KIND_URL_DOMAIN, _norm(KIND_URL_DOMAIN, domain), domain)


def _mention_names(text: str, tg_entities: Sequence[Any] | None) -> list[Entity]:
    """`MessageEntityMentionName` → ism (matndan **taxmin qilinmaydi**).

    Telegram offset'lari UTF-16 kod birliklarida o'lchanadi, shuning uchun
    kesish `utf-16-le` fazosida bajariladi — aks holda emoji bor xabarda
    ism noto'g'ri joydan kesilardi.
    """
    if not tg_entities or not text:
        return []
    units = text.encode("utf-16-le")
    out: list[Entity] = []
    for ent in tg_entities:
        name = type(ent).__name__
        if "MentionName" not in name:
            continue
        offset = getattr(ent, "offset", None)
        length = getattr(ent, "length", None)
        if offset is None or length is None or int(length) <= 0:
            continue
        lo, hi = int(offset) * 2, (int(offset) + int(length)) * 2
        if lo < 0 or hi > len(units):
            continue
        raw = units[lo:hi].decode("utf-16-le", errors="ignore").strip()
        value = _norm(KIND_MENTION_NAME, raw)
        if len(value) >= 2:
            out.append(Entity(KIND_MENTION_NAME, value, raw[:MAX_VALUE_CHARS]))
    return out


def extract_entities(text: str, tg_entities: Sequence[Any] | None = None) -> list[Entity]:
    """Matn (+ ixtiyoriy Telegram entity'lari) → identifikatorlar ro'yxati.

    Sof funksiya: I/O yo'q, LLM yo'q, natija bir xil kirishda har doim bir xil.
    Takrorlar `(kind, value)` bo'yicha yig'iladi, tartib — matndagi tartib.

    Args:
        text: xabar matni (`strip_invisible` dan o'tgan bo'lishi kutiladi).
        tg_entities: Telethon `message.entities` — faqat `mention_name` uchun.

    Returns:
        `MAX_ENTITIES_PER_MESSAGE` tagacha `Entity`. Telefon raqami va email
        **hech qachon** qaytarilmaydi.

    Examples:
        >>> [e.value for e in extract_entities("@nurbek #Deploy https://www.example.com/a")]
        ['@nurbek', '#deploy', 'example.com']
        >>> extract_entities("aloqa: ali@example.com, +998 90 123 45 67")
        []
    """
    if not text and not tg_entities:
        return []
    found: list[Entity] = []
    # 1) email — skanerdan oldin butunlay olib tashlanadi (domen ham chiqmasin).
    body = _EMAIL_RE.sub(" ", text or "")
    # 2) URL'lar: topilgach matndan olib tashlanadi, aks holda `?a=#x` fragmenti
    #    hashtag, `t.me/@user` esa username bo'lib ko'rinardi.
    for m in _URL_RE.finditer(body):
        ent = _url_entity(m.group(0))
        if ent is not None:
            found.append(ent)
    body = _URL_RE.sub(" ", body)
    for m in _USERNAME_RE.finditer(body):
        raw = "@" + m.group(1)
        found.append(Entity(KIND_USERNAME, _norm(KIND_USERNAME, raw), raw))
    for m in _HASHTAG_RE.finditer(body):
        raw = "#" + m.group(1)
        found.append(Entity(KIND_HASHTAG, "#" + _norm(KIND_HASHTAG, m.group(1)), raw))
    for m in _CASHTAG_RE.finditer(body):
        raw = "$" + m.group(1)
        found.append(Entity(KIND_CASHTAG, _norm(KIND_CASHTAG, raw), raw))
    found.extend(_mention_names(text or "", tg_entities))

    out: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for e in found:
        key = (e.kind, e.value)
        if not e.value or key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= MAX_ENTITIES_PER_MESSAGE:
            break
    return out


def parse_entity_query(raw: str) -> tuple[str, str] | None:
    """Planner bergan satr → `(kind, value)`; shakli tanilmasa `None`.

    Rejadagi `entities` erkin matn (`"@nurbek"`, `"#deploy"`, `"example.com"`,
    `"t.me/c/12/34"`). Bu yerda u indeksdagi kalit fazosiga tushiriladi —
    `entity_search` aynan shu juftlik bo'yicha tenglik qidiradi.

    Examples:
        >>> parse_entity_query("@Nurbek")
        ('username', '@nurbek')
        >>> parse_entity_query("https://t.me/c/12/34")
        ('tme_link', 'c/12/34')
        >>> parse_entity_query("salom dunyo") is None
        True
    """
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("@"):
        m = _USERNAME_RE.search(" " + s)
        return (KIND_USERNAME, _norm(KIND_USERNAME, "@" + m.group(1))) if m else None
    if s.startswith("#"):
        m = _HASHTAG_RE.search(" " + s)
        return (KIND_HASHTAG, "#" + _norm(KIND_HASHTAG, m.group(1))) if m else None
    if s.startswith("$"):
        m = _CASHTAG_RE.search(" " + s)
        return (KIND_CASHTAG, _norm(KIND_CASHTAG, "$" + m.group(1))) if m else None
    host = s.split("://", 1)[-1].split("/", 1)[0].lower().removeprefix("www.")
    if host in _TME_HOSTS:
        ent = _url_entity(s if "://" in s else "https://" + s)
        return (ent.kind, ent.value) if ent is not None else None
    if _EMAIL_RE.fullmatch(s):  # email hech qachon indeksda yo'q
        return None
    if "." in host and " " not in s:
        domain = registrable_domain(host)
        return (KIND_URL_DOMAIN, domain) if "." in domain else None
    return None


def resolve_entities(raw: Sequence[str]) -> list[tuple[str, str]]:
    """`parse_entity_query` ni ro'yxatga qo'llaydi, takrorlarni yig'adi."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        pair = parse_entity_query(item)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


# ─── DB qatlami ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EntityRef:
    """Indekslanadigan bitta xabar (batch uchun)."""

    message_id: int
    account_id: int
    chat_id: int
    text: str
    seen_at: datetime | None = None
    tg_entities: Sequence[Any] | None = None


async def index_messages_entities(session: AsyncSession, refs: Sequence[EntityRef]) -> int:
    """Bir nechta xabar uchun entity indeksini yangilaydi — idempotent.

    Uch qadam (batchga bitta so'rovdan):

      1. mavjud qatorlarni o'qish;
      2. endi mavjud bo'lmagan `(kind, value)` larni o'chirish (xabar
         tahrirlangan bo'lishi mumkin);
      3. yangilarini `ON CONFLICT DO NOTHING` bilan qo'shish — shu sababli
         qayta sync yoki takroriy chaqiruv hech narsani o'zgartirmaydi va
         `node_id` (epizod bog'lanishi) saqlanib qoladi.

    Returns:
        Qo'shilgan qatorlar soni (taxminiy — konfliktlar hisobga olinmaydi).
    """
    if not refs:
        return 0
    wanted: dict[int, list[Entity]] = {}
    for ref in refs:
        wanted[int(ref.message_id)] = extract_entities(ref.text, ref.tg_entities)
    ids = list(wanted)

    rows = (
        await session.execute(
            select(NodeEntity.id, NodeEntity.message_id, NodeEntity.kind, NodeEntity.value).where(
                NodeEntity.message_id.in_(ids)
            )
        )
    ).all()
    keys: dict[int, set[tuple[str, str]]] = {
        mid: {(e.kind, e.value) for e in ents} for mid, ents in wanted.items()
    }
    existing: dict[int, set[tuple[str, str]]] = {}
    stale: list[int] = []
    for row in rows:
        key = (str(row[2]), str(row[3]))
        mid = int(row[1])
        existing.setdefault(mid, set()).add(key)
        if key not in keys.get(mid, set()):
            stale.append(int(row[0]))
    if stale:
        await session.execute(delete(NodeEntity).where(NodeEntity.id.in_(stale)))

    values: list[dict[str, Any]] = []
    by_id = {int(r.message_id): r for r in refs}
    for mid, ents in wanted.items():
        have = existing.get(mid, set())
        ref = by_id[mid]
        for e in ents:
            if (e.kind, e.value) in have:
                continue
            values.append(
                {
                    "account_id": int(ref.account_id),
                    "chat_id": int(ref.chat_id),
                    "message_id": mid,
                    "kind": e.kind,
                    "value": e.value,
                    "value_raw": e.value_raw,
                    "seen_at": ref.seen_at,
                }
            )
    if not values:
        return 0
    stmt = pg_insert(NodeEntity).values(values)
    await session.execute(stmt.on_conflict_do_nothing(constraint="uq_node_entity_value"))
    return len(values)


async def index_message_entities(
    session: AsyncSession,
    account_id: int,
    chat_id: int,
    message_id: int,
    text: str,
    tg_entities: Sequence[Any] | None = None,
    *,
    seen_at: datetime | None = None,
) -> int:
    """Bitta xabar uchun `index_messages_entities` ustidagi qulaylik."""
    return await index_messages_entities(
        session,
        [
            EntityRef(
                message_id=message_id,
                account_id=account_id,
                chat_id=chat_id,
                text=text,
                seen_at=seen_at,
                tg_entities=tg_entities,
            )
        ],
    )


__all__ = [
    "KINDS",
    "KIND_CASHTAG",
    "KIND_HASHTAG",
    "KIND_MENTION_NAME",
    "KIND_TME_LINK",
    "KIND_URL_DOMAIN",
    "KIND_USERNAME",
    "MAX_ENTITIES_PER_MESSAGE",
    "Entity",
    "EntityRef",
    "extract_entities",
    "index_message_entities",
    "index_messages_entities",
    "parse_entity_query",
    "registrable_domain",
    "resolve_entities",
]
