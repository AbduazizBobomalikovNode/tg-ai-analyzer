"""Suhbat bo'yicha tool natijalari keshi (Redis, qisqa TTL).

Nega kerak (eval `econ-01`): foydalanuvchi bir xil savolni ikki marta bersa —
"yana ayt", "o'sha savol", sahifani yangilab qayta yuborish — agent aynan
o'sha `search_messages` ni qaytadan bajarardi. Natija o'zgarmaydi, DB/MTProto
ishi va tokenlar esa yana to'lanadi. `agent.seen_calls` bu yerda yordam
bermaydi: u **bitta yugurish** ichidagi takrorni to'sadi, bu esa turn'lar
orasida.

Chegaralar ongli:

* **Kalit suhbatga bog'langan** (`conversation_id`) — kesh akkauntlar yoki
  foydalanuvchilar orasida oqmaydi. Tool natijasi Telegram kontenti, ya'ni
  uni suhbatdan tashqariga chiqarish maxfiylik masalasi.
* **TTL qisqa** (`TTL_S`, 10 daqiqa) — "yangi xabar keldimi" degan savolga
  eskirgan javob bermasin. Uzoq kesh `get_recent_messages` ni yolg'onchi
  qilardi.
* **Faqat o'qish tool'lari** (chaqiruvchi hal qiladi) va faqat muvaffaqiyatli
  natija. Yozish tool'i hech qachon keshlanmaydi (u DB qatori yasaydi).
* **Redis yo'q bo'lsa kesh ham yo'q** — jim o'tiladi, xato ko'tarilmaydi.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

TTL_S = 600  # 10 daqiqa
KEY_PREFIX = "tgai:toolcache:"
MAX_VALUE_CHARS = 60_000  # bundan uzun natija keshlanmaydi (Redis xotirasi)
#: Kesh'dan kelgan natijaga qo'shiladigan belgi — model "bu yangi so'rov
#: emas" ekanini ko'rsin va kerak bo'lsa boshqa yo'l tanlasin.
CACHED_NOTE = "(cached)"


def _fold(value: Any) -> Any:
    """Argumentni taqqoslash shakliga keltiradi (registr/bo'shliq/tinish)."""
    from app.services.agent import fold_query

    if isinstance(value, str):
        return fold_query(value)
    if isinstance(value, dict):
        return {str(k): _fold(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_fold(v) for v in value]
    return value


def cache_key(conversation_id: int | None, tool: str, args: dict[str, Any]) -> str:
    """`(suhbat, tool, buklangan argumentlar)` → barqaror kalit.

    Argumentlar buklanadi, ya'ni `chat="Ish Chat"` va `chat="ish  chat"`
    bitta kalitga tushadi — bu aynan `agent.fold_query` bilan bir mantiq.
    """
    if not conversation_id or not tool:
        return ""
    blob = json.dumps(
        {"t": tool, "a": _fold(args or {})}, sort_keys=True, ensure_ascii=False, default=str
    )
    digest = hashlib.md5(blob.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{KEY_PREFIX}{int(conversation_id)}:{digest}"


async def _redis() -> Any:
    import redis.asyncio as aioredis

    return aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]


async def get(key: str) -> str | None:
    """Keshdagi natija matni (yoki `None`). Nosozlik — kesh promahi."""
    if not key:
        return None
    try:
        r = await _redis()
        try:
            raw = await r.get(key)
        finally:
            await r.aclose()
    except Exception as exc:  # kesh hech qachon javobni to'xtatmaydi
        log.debug("tool_cache.read_failed", error=str(exc)[:100])
        return None
    if not raw:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)


async def put(key: str, text: str, *, ttl_s: int = TTL_S) -> None:
    """Natijani keshga yozadi; bo'sh yoki juda uzun matn yozilmaydi."""
    if not key or not text or len(text) > MAX_VALUE_CHARS:
        return
    try:
        r = await _redis()
        try:
            await r.set(key, text, ex=max(1, ttl_s))
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("tool_cache.write_failed", error=str(exc)[:100])


__all__ = ["CACHED_NOTE", "KEY_PREFIX", "MAX_VALUE_CHARS", "TTL_S", "cache_key", "get", "put"]
