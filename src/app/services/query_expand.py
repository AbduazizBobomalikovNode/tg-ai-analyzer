"""Ko'p tilli so'rov kengaytirish — bitta `Task.ROUTE` chaqiruvi + Redis kesh.

Muammo: arxiv uch tilda (o'zbek lotin/kirill, rus, ingliz) va odam savolni bir
tilda yozadi, xabar esa boshqa tilda yozilgan. "to'lov" deb qidirgan odam
"оплата" yozilgan xabarni topa olmaydi; leksik kanal (tsv/trigram) bu
bo'shliqni o'zi yopa olmaydi — u faqat yozilishni biladi, ma'noni emas.

Yechim: **bitta** arzon model chaqiruvi savolni variantlarga yoyadi, natija
Redis'da 24 soat yashaydi (kalit `qx:<md5(uz_fold(query))>` — imlo/registr/
apostrof farqi bir xil kalitga tushadi). Bir xil savol ikkinchi marta hech
narsa turmaydi.

Xavfsizlik: bu bosqich **begona matn ko'rmaydi** — kirish faqat foydalanuvchining
o'z savoli (dual-LLM naqshi, arxitektura v2, 7-bo'lim). Shuning uchun konvert
kerak emas; uzunlik esa 300 belgiga kesiladi (prompt bo'rtib ketmasin).

Har qanday nosozlik — LLM xatosi, Redis yo'qligi, buzuq JSON — **kengaytirishsiz**
davom etish demakdir: qidiruv asl so'rov bilan ishlaydi.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import get_settings
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services.prompts_retrieval import QUERY_EXPAND_PROMPT
from app.services.textnorm import uz_fold

log = get_logger(__name__)

MAX_QUERY_CHARS = 300
MAX_PER_LIST = 4
MAX_TOTAL_VARIANTS = 12  # narx qopqog'i: undan ko'p variant = ko'p SQL kanali
MAX_TERM_CHARS = 48
CACHE_PREFIX = "qx:"

_FIELDS = ("uz_lat", "uz_cyr", "ru", "en", "synonyms", "slang")
_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_JSON_RE = re.compile(r"\{.*\}", re.S)


@dataclass(slots=True)
class Expansion:
    """So'rov variantlari. Bo'sh `Expansion` — "kengaytirish yo'q" degani."""

    uz_lat: list[str] = field(default_factory=list)
    uz_cyr: list[str] = field(default_factory=list)
    ru: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    slang: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, f) for f in _FIELDS)

    def all_terms(self) -> list[str]:
        """Barcha variantlar, takrorsiz, umumiy qopqoq bilan."""
        out: list[str] = []
        for f in _FIELDS:
            for t in getattr(self, f):
                if t not in out:
                    out.append(t)
        return out[:MAX_TOTAL_VARIANTS]

    def latin_terms(self) -> list[str]:
        """`tsv_uz` / trigram kanallari uchun — kirill bo'lmagan hamma narsa."""
        return [t for t in self.all_terms() if not _CYR_RE.search(t)]

    def cyrillic_terms(self) -> list[str]:
        """`tsv_ru` kanali uchun — ru ro'yxati va kirill yozuvidagi variantlar."""
        out = [t for t in self.ru if t]
        for t in self.all_terms():
            if _CYR_RE.search(t) and t not in out:
                out.append(t)
        return out[:MAX_TOTAL_VARIANTS]

    def to_dict(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in asdict(self).items()}


def cache_key(query: str) -> str:
    """Kalit fold fazosida: apostrof/registr/yozuv/bo'shliq farqi bir kalitga tushadi."""
    folded = " ".join(uz_fold(query).split())[:MAX_QUERY_CHARS]
    digest = hashlib.md5(folded.encode(), usedforsecurity=False).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


def parse_expansion(raw: str) -> Expansion:
    """Model javobidan `Expansion`. Buzuq JSON — bo'sh kengaytirish, xato emas."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return Expansion()
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return Expansion()
    if not isinstance(data, dict):
        return Expansion()
    return from_dict(data)


def from_dict(data: dict[str, Any]) -> Expansion:
    kwargs: dict[str, list[str]] = {}
    for f in _FIELDS:
        kwargs[f] = _clean_list(data.get(f))
    return Expansion(**kwargs)


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        term = " ".join(item.split()).strip().lower()[:MAX_TERM_CHARS]
        if len(term) < 2 or term in out:
            continue
        out.append(term)
        if len(out) >= MAX_PER_LIST:
            break
    return out


# ─── Redis kesh (yo'q bo'lsa jim o'tiladi) ───────────────────────────────────


async def _cache_get(key: str) -> Expansion | None:
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]
        try:
            raw = await r.get(key)
        finally:
            await r.aclose()
    except Exception as exc:  # kesh hech qachon qidiruvni to'xtatmaydi
        log.debug("query_expand.cache_read_failed", error=str(exc)[:100])
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return from_dict(data) if isinstance(data, dict) else None


async def _cache_set(key: str, exp: Expansion) -> None:
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]
        try:
            await r.set(key, json.dumps(exp.to_dict()), ex=get_settings().search_expand_ttl_s)
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("query_expand.cache_write_failed", error=str(exc)[:100])


# ─── asosiy kirish nuqtasi ───────────────────────────────────────────────────


async def expand(query: str, *, llm: LLM | None = None, cache: bool = True) -> Expansion:
    """Savolni ko'p tilli variantlarga yoyadi. Nosozlikda — bo'sh `Expansion`."""
    q = " ".join((query or "").split())[:MAX_QUERY_CHARS]
    if len(q) < 3:
        return Expansion()

    key = cache_key(q)
    if cache:
        hit = await _cache_get(key)
        if hit is not None:
            log.debug("query_expand.cache_hit", key=key)
            return hit

    msgs = [Msg.system(QUERY_EXPAND_PROMPT), Msg.user(q)]
    try:
        res = await (llm or LLM()).chat(Task.ROUTE, msgs, json_mode=True, max_tokens=400)
    except LLMError as exc:
        log.warning("query_expand.failed", error=str(exc)[:200])
        return Expansion()
    exp = parse_expansion(res.text)
    if cache and not exp.is_empty:
        await _cache_set(key, exp)
    log.debug("query_expand.done", variants=len(exp.all_terms()))
    return exp


__all__ = [
    "CACHE_PREFIX",
    "MAX_QUERY_CHARS",
    "MAX_TOTAL_VARIANTS",
    "Expansion",
    "cache_key",
    "expand",
    "from_dict",
    "parse_expansion",
]
