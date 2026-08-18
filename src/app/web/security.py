"""Web sessiya (cookie) va so'rov himoyasi.

* Cookie — `itsdangerous` bilan imzolangan JSON: {"uid", "iat"}. Kalit master
  key'dan HMAC bilan hosil qilinadi (alohida sir kerak emas, rotatsiya master
  key bilan birga). HttpOnly + SameSite=Lax + (HTTPS'da) Secure.
* CSRF — barcha o'zgartiruvchi API'lar `X-Requested-With: fetch` header'ini
  talab qiladi (oddiy forma/`<img>` orqali yuborib bo'lmaydi), cookie SameSite=Lax.
* Locale — `?lang=` → `lang` cookie → `Accept-Language` → uz.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeSerializer

from app.config import get_settings
from app.i18n import SUPPORTED, Translator, normalize

SESSION_COOKIE = "tgai_session"
LANG_COOKIE = "lang"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "fetch"


@dataclass(frozen=True, slots=True)
class WebIdentity:
    user_id: int
    issued_at: int


def _serializer() -> URLSafeSerializer:
    key = hmac.new(get_settings().master_key, b"web-session-v1", hashlib.sha256).digest()
    return URLSafeSerializer(key, salt="tgai.web.session")


def issue_session(user_id: int) -> str:
    return _serializer().dumps({"uid": int(user_id), "iat": int(time.time())})


def read_session(token: str | None) -> WebIdentity | None:
    if not token:
        return None
    try:
        data: Any = _serializer().loads(token)
    except BadSignature:
        return None
    if not isinstance(data, dict) or "uid" not in data or "iat" not in data:
        return None
    ttl = get_settings().web_session_ttl_hours * 3600
    if time.time() - int(data["iat"]) > ttl:
        return None
    return WebIdentity(user_id=int(data["uid"]), issued_at=int(data["iat"]))


def is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto == "https" or request.url.scheme == "https"


def set_session_cookie(request: Request, response: Response, user_id: int) -> None:
    """`Secure` — so'rov HTTPS bo'lsa (yoki config HTTPS'ni talab qilsa).

    Lokal http://localhost'da Secure qo'yilmaydi, aks holda brauzer cookie'ni
    saqlamaydi va login "ishlamaydi" bo'lib ko'rinadi.
    """
    s = get_settings()
    response.set_cookie(
        SESSION_COOKIE,
        issue_session(user_id),
        max_age=s.web_session_ttl_hours * 3600,
        httponly=True,
        secure=is_https(request) or (s.web_secure_cookies and s.is_prod),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def identity_of(request: Request) -> WebIdentity | None:
    return read_session(request.cookies.get(SESSION_COOKIE))


def require_identity(request: Request) -> WebIdentity:
    """API dependency — 401 JSON."""
    ident = identity_of(request)
    if ident is None:
        raise HTTPException(status_code=401, detail={"code": "unauthorized"})
    return ident


def require_csrf(request: Request) -> None:
    """O'zgartiruvchi API'lar uchun dependency."""
    if request.headers.get(CSRF_HEADER, "").lower() != CSRF_VALUE:
        raise HTTPException(status_code=403, detail={"code": "csrf"})


def locale_of(request: Request) -> str:
    q = request.query_params.get("lang")
    if q and q.lower() in SUPPORTED:
        return q.lower()
    c = request.cookies.get(LANG_COOKIE)
    if c and c in SUPPORTED:
        return c
    accept = request.headers.get("accept-language", "")
    for part in accept.split(","):
        code = part.split(";")[0].strip()
        if code and code.split("-")[0].lower() in SUPPORTED:
            return normalize(code)
    return "uz"


def translator_of(request: Request) -> Translator:
    return Translator(locale_of(request))


def client_ip(request: Request) -> str:
    # Reverse-proxy ortida — X-Forwarded-For'ning birinchisi
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


class RateLimiter:
    """Oddiy sliding-window (jarayon xotirasi). Auth endpoint'lari uchun yetarli."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        now = time.monotonic()
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(hits) >= self.limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
