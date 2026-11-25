"""Foydalanuvchi UI sozlamalari (`users.prefs`, migratsiya 0018).

Bu modul — `users.prefs` JSONB ustuniga **yagona yozish nuqtasi**. Ustun
ixtiyoriy JSON omborxonasi emas: `merge()` faqat `ALLOWED` dagi kalitlarni
va faqat validatordan o'tgan qiymatni yozadi, qolgani jimgina tashlanadi.
Shu tufayli `PUT /api/settings/prefs` orqali begona struktura ham, cheksiz
katta ro'yxat ham kira olmaydi.

Sof funksiyalar (DB yo'q) — testlanadi.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.i18n import SUPPORTED

THEMES = ("system", "light", "dark")
DENSITIES = ("comfortable", "compact")
MODES = ("auto", "agent", "direct", "deep")
STRATEGIES = ("auto", "recent", "search", "window")

#: `recent_chats` uzunligi (eng yangi birinchi). UI "oxirgi ochilganlar"
#: ro'yxati — undan uzun bo'lsa foyda yo'q, cookie/JSON esa o'sadi.
RECENT_MAX = 10
#: Pin'langan chatlar qopqog'i — ro'yxat cheksiz o'smasin.
PINNED_MAX = 50

DEFAULTS: dict[str, Any] = {
    "theme": "system",
    "locale": "",  # bo'sh = cookie/Accept-Language qaroriga qo'yiladi
    "default_mode": "auto",
    "default_strategy": "auto",
    "deep_default": False,
    "density": "comfortable",
    "onboarding_done": False,
    "pinned_chats": [],
    "recent_chats": [],
}


def _enum(choices: tuple[str, ...]) -> Callable[[Any], Any]:
    def check(v: Any) -> Any:
        return v if isinstance(v, str) and v in choices else None

    return check


def _bool(v: Any) -> Any:
    return v if isinstance(v, bool) else None


def _locale(v: Any) -> Any:
    if not isinstance(v, str):
        return None
    raw = v.strip().lower()
    if raw == "":
        return ""  # "avtomatik" — tozalash uchun ataylab ruxsat etilgan
    return raw if raw in SUPPORTED else None


def _ids(cap: int) -> Callable[[Any], Any]:
    """`list[int]` — takrorlarsiz, tartib saqlanadi, `cap` tagacha kesiladi."""

    def check(v: Any) -> Any:
        if not isinstance(v, list):
            return None
        out: list[int] = []
        for item in v:
            # `bool` — `int` ning avlodi; `True` chat id bo'lib qolmasin.
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if item <= 0 or item in out:
                continue
            out.append(item)
            if len(out) >= cap:
                break
        return out

    return check


#: kalit → validator. Validator `None` qaytarsa qiymat **yozilmaydi**
#: (mavjud qiymat o'z joyida qoladi) — 422 emas, chunki bu prezentatsiya
#: sozlamasi: bitta buzuq kalit butun `PUT` ni yiqitmasligi kerak.
ALLOWED: dict[str, Callable[[Any], Any]] = {
    "theme": _enum(THEMES),
    "locale": _locale,
    "default_mode": _enum(MODES),
    "default_strategy": _enum(STRATEGIES),
    "deep_default": _bool,
    "density": _enum(DENSITIES),
    "onboarding_done": _bool,
    "pinned_chats": _ids(PINNED_MAX),
    "recent_chats": _ids(RECENT_MAX),
}


def validate(patch: Any) -> dict[str, Any]:
    """Kirish JSON'idan faqat tanilgan va to'g'ri kalitlarni ajratadi."""
    if not isinstance(patch, dict):
        return {}
    out: dict[str, Any] = {}
    for key, check in ALLOWED.items():
        if key not in patch:
            continue
        value = check(patch[key])
        if value is not None:
            out[key] = value
    return out


def normalize(raw: Any) -> dict[str, Any]:
    """DB'dagi qiymat → to'liq, defaultlar bilan to'ldirilgan lug'at.

    Eski/buzuq qiymatlar (qo'lda tahrir, eski versiya) default bilan
    almashtiriladi — UI hech qachon kutilmagan tur ko'rmaydi.
    """
    stored = raw if isinstance(raw, dict) else {}
    out = {k: (v[:] if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    out.update(validate(stored))
    return out


def merge(current: Any, patch: Any) -> dict[str, Any]:
    """Joriy qiymat + validatsiyadan o'tgan patch → yangi to'liq lug'at."""
    out = normalize(current)
    out.update(validate(patch))
    return out
