"""Oddiy JSON-asosli i18n. uz / ru / en.

Kalit topilmasa — default locale'ga, u ham bo'lmasa kalitning o'zi qaytadi
(sukut bilan yiqilmaydi, lekin log'da ko'rinadi).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

LOCALES_DIR = Path(__file__).parent / "locales"
SUPPORTED: tuple[str, ...] = ("uz", "ru", "en")
FALLBACK = "uz"


@lru_cache(maxsize=8)
def _load(locale: str) -> dict[str, str]:
    path = LOCALES_DIR / f"{locale}.json"
    if not path.exists():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def normalize(locale: str | None) -> str:
    if not locale:
        return FALLBACK
    short = locale.split("-")[0].lower()
    return short if short in SUPPORTED else FALLBACK


def t(key: str, locale: str = FALLBACK, /, **kwargs: Any) -> str:
    loc = normalize(locale)
    value = _load(loc).get(key) or _load(FALLBACK).get(key) or key
    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError):
            return value
    return value


class Translator:
    """Bir locale'ga bog'langan qulay wrapper: `_ = Translator("ru"); _("btn.help")`."""

    __slots__ = ("locale",)

    def __init__(self, locale: str | None) -> None:
        self.locale = normalize(locale)

    def __call__(self, key: str, **kwargs: Any) -> str:
        return t(key, self.locale, **kwargs)


__all__ = ["FALLBACK", "SUPPORTED", "Translator", "normalize", "t"]
