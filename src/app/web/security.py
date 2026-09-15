"""Web sessiya (cookie) va so'rov himoyasi.

* Cookie — `itsdangerous` bilan imzolangan JSON: {"uid", "iat", "v"}. Kalit
  master key'dan HMAC bilan hosil qilinadi (alohida sir kerak emas, rotatsiya
  master key bilan birga). HttpOnly + SameSite=Lax + (HTTPS'da) Secure.
  `v` — `SESSION_VERSION`: kelajakda hamma sessiyani bir zarbada bekor qilish
  uchun (raqamni oshirish kifoya, migratsiya shart emas).
* Logout — **server tomonda ham** bekor qilinadi: Redis'da
  `web:sess:revoked:<uid>` = shu daqiqa; undan oldin berilgan (`iat <= chegara`)
  cookie'lar rad etiladi. Redis yo'q bo'lsa **fail-open** (ogohlantirish
  log'i bilan): web umuman ishlamay qolgandan ko'ra eski cookie bir muddat
  yashagani afzal, imzo va TTL baribir kuchda.
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
from app.logging import get_logger

log = get_logger(__name__)

SESSION_COOKIE = "tgai_session"
LANG_COOKIE = "lang"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "fetch"

# Cookie formati versiyasi. Oshirilsa — eski cookie'lar darhol rad etiladi
# (global logout), migratsiya ham, DB ustuni ham kerak emas.
SESSION_VERSION = 1
REVOKE_PREFIX = "web:sess:revoked:"
# Bekor qilish yozuvi cookie TTL'idan uzoqroq yashashi kerak, aks holda
# eskirgan chegara o'chib, allaqachon TTL ichidagi cookie qayta tirilardi.
REVOKE_TTL_EXTRA = 3600


@dataclass(frozen=True, slots=True)
class WebIdentity:
    user_id: int
    issued_at: int


def _serializer() -> URLSafeSerializer:
    key = hmac.new(get_settings().master_key, b"web-session-v1", hashlib.sha256).digest()
    return URLSafeSerializer(key, salt="tgai.web.session")


def issue_session(user_id: int) -> str:
    return _serializer().dumps({"uid": int(user_id), "iat": int(time.time()), "v": SESSION_VERSION})


def read_session(token: str | None) -> WebIdentity | None:
    """Sof token tekshiruvi: imzo, versiya, TTL. **Bekor qilish bu yerda emas**
    — u Redis'ga boradi, ya'ni async (`identity_of`)."""
    if not token:
        return None
    try:
        data: Any = _serializer().loads(token)
    except BadSignature:
        return None
    if not isinstance(data, dict) or "uid" not in data or "iat" not in data:
        return None
    # `v` yo'q = versiyalashdan oldingi cookie → 0 deb qaraladi va rad etiladi.
    if int(data.get("v", 0)) != SESSION_VERSION:
        return None
    ttl = get_settings().web_session_ttl_hours * 3600
    if time.time() - int(data["iat"]) > ttl:
        return None
    return WebIdentity(user_id=int(data["uid"]), issued_at=int(data["iat"]))


# ─── server tomonda bekor qilish (logout) ────────────────────────────────────


async def _redis() -> Any:
    import redis.asyncio as aioredis

    return aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]


async def revoke_sessions(user_id: int, *, at: int | None = None) -> bool:
    """Shu foydalanuvchining barcha cookie'larini bekor qiladi. `False` — Redis yo'q."""
    threshold = int(at if at is not None else time.time())
    try:
        r = await _redis()
        try:
            await r.set(
                REVOKE_PREFIX + str(int(user_id)),
                str(threshold),
                ex=get_settings().web_session_ttl_hours * 3600 + REVOKE_TTL_EXTRA,
            )
        finally:
            await r.aclose()
    except Exception as exc:
        log.warning("web.session_revoke_failed", user_id=user_id, error=str(exc)[:200])
        return False
    return True


async def is_revoked(ident: WebIdentity) -> bool:
    """`iat <= chegara` bo'lsa — bekor qilingan. Redis yo'q → **fail-open**."""
    try:
        r = await _redis()
        try:
            raw = await r.get(REVOKE_PREFIX + str(ident.user_id))
        finally:
            await r.aclose()
    except Exception as exc:
        log.warning("web.session_revoke_check_failed", error=str(exc)[:200])
        return False
    if raw is None:
        return False
    try:
        threshold = int(raw.decode() if isinstance(raw, bytes) else raw)
    except (ValueError, AttributeError):
        return False
    return ident.issued_at <= threshold


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


async def identity_of(request: Request) -> WebIdentity | None:
    ident = read_session(request.cookies.get(SESSION_COOKIE))
    if ident is None:
        return None
    if await is_revoked(ident):
        return None
    return ident


async def require_identity(request: Request) -> WebIdentity:
    """API dependency — 401 JSON."""
    ident = await identity_of(request)
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


def _is_trusted(ip: str, nets: tuple[Any, ...]) -> bool:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in n for n in nets)


def client_ip(request: Request) -> str:
    """Mijoz IP'si — shift kaliti.

    `X-Forwarded-For` **shartsiz ishonchsiz**: uni istalgan mijoz yozib
    yuborishi mumkin va ishonsak, har so'rovda boshqa "IP" ko'rsatib shiftni
    aylanib o'tish mumkin. Shuning uchun header faqat TCP peer `TRUSTED_PROXIES`
    (CIDR ro'yxati, default bo'sh) ichida bo'lsa o'qiladi va zanjir **o'ngdan
    chapga** yuriladi: birinchi ishonchsiz hop — haqiqiy mijoz.
    """
    peer = request.client.host if request.client else "?"
    nets = get_settings().trusted_proxy_networks
    if not nets or not _is_trusted(peer, nets):
        return peer
    xff = request.headers.get("x-forwarded-for") or ""
    for part in reversed([p.strip() for p in xff.split(",") if p.strip()]):
        if not _is_trusted(part, nets):
            return part
    # Zanjirdagi hamma hop ishonchli (yoki header yo'q) — peer'ning o'zi.
    return peer


class RateLimiter:
    """Oddiy sliding-window (jarayon xotirasi). Auth endpoint'lari uchun yetarli.

    `_hits` cheksiz o'smaydi: har chaqiruvda eskirgan/bo'sh kalitlar
    tozalanadi (amortizatsiya uchun har `_SWEEP_EVERY` chaqiruvda bir marta
    to'liq supurish). Aks holda har yangi IP abadiy qator qoldirardi.
    """

    _SWEEP_EVERY = 256
    _MAX_KEYS = 1024

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._calls = 0

    def _sweep(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
            del self._hits[key]

    def check(self, key: str) -> bool:
        now = time.monotonic()
        self._calls += 1
        if self._calls % self._SWEEP_EVERY == 0 or len(self._hits) > self._MAX_KEYS:
            self._sweep(now)
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(hits) >= self.limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
