"""structlog sozlamasi.

MUHIM: session string, master key, 2FA parol va OTP kod hech qachon log'ga
tushmasligi kerak. `_redact` processor'i shubhali kalitlarni maskalaydi.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

MASK = "***REDACTED***"

# Kalit nomida shu bo'laklardan biri **uchrasa** (registrga qaramay) qiymat
# maskalanadi: `_SECRET_KEYS` aniq nomlari kam edi — `user_api_key`,
# `Authorization`, `refresh_token` kabi nomlar o'tib ketardi.
_SECRET_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "session",
    "api_key",
    "apikey",
    "authorization",
    "otp",
    "code",
    "master_key",
    "dek",
    "twofa",
    "two_fa",
    "api_hash",
)

# Erkin matn ichidagi tanish sir shakllari. Kalit nomiga tayanib bo'lmaydi:
# sir xato matniga yoki `event` satriga tushishi mumkin.
_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Telethon StringSession: '1' bilan boshlanadigan uzun base64
    re.compile(r"\b1[A-Za-z0-9+/_-]{199,}={0,2}"),
    # Bot API tokeni: 123456789:AA...
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    # OpenAI/Anthropic uslubidagi kalitlar va Google API kaliti
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),
)

# `code` bo'lagi juda keng tutadi (`status_code`, `error_code`…) — bular sir
# emas va ularni yo'qotish diagnostikani ko'r qiladi. Aniq nom "code" bo'lsa
# baribir maskalanadi: login kodi aynan shunday yoziladi.
_KEY_ALLOW = frozenset(
    {
        "status_code",
        "error_code",
        "err_code",
        "http_code",
        "country_code",
        "lang_code",
        "code_type",
        "code_length",
        "next_type",
        "exit_code",
        "reason_code",
    }
)

# Maskalash chuqurligi qopqog'i — log yozuvi tsiklik/juda chuqur bo'lsa ham
# processor cheksiz ketmasin.
_MAX_DEPTH = 6

_SECRET_KEYS = frozenset(
    {
        "session",
        "session_blob",
        "session_string",
        "master_key",
        "master_key_b64",
        "dek",
        "password",
        "twofa",
        "two_fa",
        "code",
        "phone_code",
        "otp",
        "api_hash",
        "bot_token",
        "gemini_api_key",
        "token",
        "authorization",
    }
)


def is_secret_key(key: Any) -> bool:
    k = str(key).lower()
    if k in _KEY_ALLOW:
        return False
    return k in _SECRET_KEYS or any(part in k for part in _SECRET_KEY_PARTS)


def scrub_text(text: str) -> str:
    """Matn ichidagi tanish sir shakllarini maskalaydi (kalit nomi bo'lmasa ham)."""
    for pat in _VALUE_PATTERNS:
        text = pat.sub(MASK, text)
    return text


def scrub(node: Any, _depth: int = 0) -> Any:
    """Dict/list/matn bo'ylab **rekursiv** maskalash.

    Ikki qatlam: (1) kalit nomi sirni bildirsa — qiymat butunlay almashtiriladi;
    (2) qolgan satrlarda tanish sir shakllari (`_VALUE_PATTERNS`) qidiriladi.
    Bitta implementatsiya: structlog processor'i ham, Sentry `before_send` ham
    (`observability._scrub`) shu funksiyani chaqiradi.
    """
    if _depth > _MAX_DEPTH:
        return node
    if isinstance(node, dict):
        return {k: (MASK if is_secret_key(k) else scrub(v, _depth + 1)) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v, _depth + 1) for v in node]
    if isinstance(node, tuple):
        return tuple(scrub(v, _depth + 1) for v in node)
    if isinstance(node, str):
        return scrub_text(node)
    if isinstance(node, bytes):
        return node  # baytlar log'ga baribir repr bo'lib chiqadi — tegmaymiz
    return node


def _redact(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return scrub(event_dict)  # type: ignore[no-any-return]


def setup_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO)
    )
    # Telethon juda gapdon — faqat warning
    logging.getLogger("telethon").setLevel(logging.WARNING)

    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,  # type: ignore[list-item]
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
