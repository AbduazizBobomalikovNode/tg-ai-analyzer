"""structlog sozlamasi.

MUHIM: session string, master key, 2FA parol va OTP kod hech qachon log'ga
tushmasligi kerak. `_redact` processor'i shubhali kalitlarni maskalaydi.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

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


def _redact(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "***REDACTED***"
    return event_dict


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
