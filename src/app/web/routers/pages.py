"""HTML sahifalar (Jinja2). Ma'lumot — /api orqali, sahifa faqat qobiq."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.i18n import SUPPORTED, _load
from app.logging import get_logger
from app.web.security import LANG_COOKIE, WebIdentity, identity_of, locale_of

log = get_logger(__name__)

router = APIRouter(include_in_schema=False)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


# ─── static fingerprinting (ui/R1b) ──────────────────────────────────────────
#
# `asset("app.css")` → `/static/app.css?v=<sha1[:10]>`. Versiya fayl
# **mazmunidan** olinadi (mtime emas: konteyner qayta qurilganda mtime
# o'zgaradi, mazmun esa yo'q — foydalanuvchi keshini bekorga kuydirmaymiz).
#
# Shu bilan `web/main.py` dagi static mount `v=` bo'lsa bir yillik
# `immutable` kesh beradi: fayl o'zgarsa URL o'zgaradi, ya'ni eski javob
# hech qachon noto'g'ri bo'lmaydi.

_ASSET_CACHE: dict[str, str] = {}
ASSET_HASH_CHARS = 10


def _fingerprint(rel: str) -> str:
    """Fayl mazmunining sha1 prefiksi; fayl yo'q bo'lsa bo'sh satr."""
    path = (STATIC_DIR / rel).resolve()
    # Yo'l chegarasi: `asset("../secrets")` static papkadan chiqib ketmasin.
    if not path.is_relative_to(STATIC_DIR.resolve()) or not path.is_file():
        return ""
    digest = hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()
    return digest[:ASSET_HASH_CHARS]


def asset(path: str) -> str:
    """Jinja global: fingerprint'li static URL.

    Dev rejimida (`ENV=dev`) har chaqiruvda qayta hisoblanadi — CSS'ni
    tahrirlab sahifani yangilash yetarli bo'lsin. Prod'da bir marta
    (`prime_assets()` startupda, keyin kesh).
    """
    rel = path.lstrip("/")
    if get_settings().env == "dev":
        version = _fingerprint(rel)
    else:
        cached = _ASSET_CACHE.get(rel)
        version = cached if cached is not None else _fingerprint(rel)
        _ASSET_CACHE[rel] = version
    return f"/static/{rel}?v={version}" if version else f"/static/{rel}"


def prime_assets() -> int:
    """Startupda barcha static fayllarni keshlaydi. → keshlangan fayllar soni."""
    _ASSET_CACHE.clear()
    if not STATIC_DIR.is_dir():
        return 0
    for item in STATIC_DIR.rglob("*"):
        if item.is_file():
            rel = item.relative_to(STATIC_DIR).as_posix()
            _ASSET_CACHE[rel] = _fingerprint(rel)
    return len(_ASSET_CACHE)


templates.env.globals["asset"] = asset


def _strings(locale: str) -> dict[str, str]:
    """JS'ga beriladigan tarjimalar (faqat web.*, *.err.*, error.*)."""
    merged = {**_load("uz"), **_load(locale)}
    return {
        k: v
        for k, v in merged.items()
        if k.startswith(
            (
                "web.",
                "auth.err.",
                "pool.err.",
                "chat.err.",
                "action.err.",
                "sync.err.",
                "alert.err.",  # phase3/C
                "briefing.err.",  # phase3/B
                "error.",
            )
        )
    }


async def _locale_for(request: Request, ident: WebIdentity | None) -> str:
    """Til: `prefs.locale` (o'rnatilgan bo'lsa) → aks holda `?lang=`/cookie/header.

    Cookie mantiqi **o'z joyida qoladi** (kirmagan foydalanuvchi va til
    almashtirgich shunga tayanadi); sozlamada aniq til tanlangan bo'lsa u
    ustun turadi — foydalanuvchi buni bir marta aytgan.

    DB'ga borib bo'lmasa (test, baza yo'q) — jimgina cookie qaroriga
    tushamiz: sahifa tili tufayli butun UI yiqilmasin.
    """
    fallback = locale_of(request)
    if ident is None or request.query_params.get("lang") in SUPPORTED:
        return fallback  # aniq `?lang=` — foydalanuvchining shu safargi tanlovi
    try:
        from app.db.base import session_scope
        from app.db.models import User
        from app.services import prefs as P

        async with session_scope() as db:
            user = await db.get(User, ident.user_id)
            chosen = P.normalize(user.prefs).get("locale") if user is not None else ""
    except Exception as exc:
        log.debug("pages.prefs_locale_failed", error=str(exc)[:200])
        return fallback
    return chosen if chosen in SUPPORTED else fallback


def _ctx(request: Request, locale: str, **extra: Any) -> dict[str, Any]:
    strings = _strings(locale)
    return {
        "request": request,
        "lang": locale,
        "langs": SUPPORTED,
        "t": lambda key, **kw: (
            (strings.get(key) or key).format(**kw) if kw else strings.get(key, key)
        ),
        "strings_json": json.dumps(strings, ensure_ascii=False),
        "env": get_settings().env,
        **extra,
    }


def _finish_html(request: Request, resp: Response, locale: str) -> Response:
    """HTML javobi: `no-cache` + ETag, va `?lang=` bo'lsa cookie.

    `no-cache` "keshlamang" degani emas — "har safar serverdan so'rang"
    degani: ETag mos kelsa javob 304, ya'ni tarmoqda faqat sarlavhalar
    ketadi, lekin foydalanuvchi hech qachon eski qobiqni ko'rmaydi
    (static fayllar esa aksincha — `immutable`, chunki URL'da versiya bor).
    """
    resp.headers["Cache-Control"] = "no-cache"
    body = getattr(resp, "body", b"")
    if body:
        digest = hashlib.sha1(body, usedforsecurity=False).hexdigest()[:16]
        etag = f'"{digest}"'
        resp.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            resp = Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
    if request.query_params.get("lang") in SUPPORTED:
        resp.set_cookie(LANG_COOKIE, locale, max_age=365 * 86400, samesite="lax")
    return resp


def _render(request: Request, name: str, locale: str, **extra: Any) -> Response:
    resp = templates.TemplateResponse(request, name, _ctx(request, locale, **extra))
    return _finish_html(request, resp, locale)


async def _prefs_for(ident: WebIdentity | None) -> dict[str, Any]:
    """`users.prefs` (normallashtirilgan). DB'ga borib bo'lmasa — defaultlar.

    Sahifa marshruti prefs tufayli yiqilmasligi kerak: baza yo'q bo'lsa
    (test, migratsiya oldidan) bo'sh lug'at qaytadi va chaqiruvchi
    `normalize()` defaultlariga tushadi.
    """
    if ident is None:
        return {}
    try:
        from app.db.base import session_scope
        from app.db.models import User
        from app.services import prefs as P

        async with session_scope() as db:
            user = await db.get(User, ident.user_id)
            return P.normalize(user.prefs if user is not None else None)
    except Exception as exc:
        log.debug("pages.prefs_failed", error=str(exc)[:200])
        return {}


async def _has_synced_chats(ident: WebIdentity) -> bool:
    """Foydalanuvchida hech bo'lmasa bir sinxronlangan chatli akkaunt bormi.

    "Sinxronlangan" — `auth.py::_chat_counts` bilan bir xil mezon:
    `Chat.synced_total > 0` (tarix boshigacha yetmagan bo'lsa ham qidiruv
    uchun foydali). DB'ga borib bo'lmasa **True** qaytadi — `index()` shu
    holatda onboarding'ga emas, `/today`ga tushsin (H6: noaniqlikda eski
    ulangan foydalanuvchini qayta onboarding'ga qamab qo'ymaymiz).
    """
    try:
        from sqlalchemy import func, select

        from app.db.base import session_scope
        from app.db.models import Account, Chat

        async with session_scope() as db:
            row = await db.execute(
                select(func.count(Chat.id))
                .join(Account, Account.id == Chat.account_id)
                .where(Account.user_id == ident.user_id, Chat.synced_total > 0)
            )
            return int(row.scalar() or 0) > 0
    except Exception as exc:
        log.debug("pages.synced_chats_check_failed", error=str(exc)[:200])
        return True


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    """Kirish nuqtasi: akkaunt yo'q → `/login`, yangi foydalanuvchi →
    `/onboarding`, aks holda `/today` (ui/R2c, `docs/ui-research.md` § 2).

    "Yangi" mezoni — `prefs.onboarding_done` bayrog'i **va** hech qanday
    akkauntda sinxronlangan chat yo'qligi (H6): faqat bayroqqa qaralsa
    allaqachon ulangan, chatlari sinxronlangan foydalanuvchi ham (masalan
    bayroq eski sessiyada saqlanmagan bo'lsa) har safar `/onboarding`ga
    qaytarilardi. Bayroq `/onboarding` ning oxirgi qadamida qo'yiladi —
    oqimni tugatmagan (va hali chati yo'q) foydalanuvchi davom etadi.
    """
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    prefs = await _prefs_for(ident)
    target = "/today"
    if not prefs.get("onboarding_done") and not await _has_synced_chats(ident):
        target = "/onboarding"
    return RedirectResponse(target, status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Any:
    ident = await identity_of(request)
    return _render(
        request, "login.html", await _locale_for(request, ident), logged_in=ident is not None
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Any:
    """Eski marshrut — `/ask` ga ko'chdi (ui/R2b). Havolalar buzilmasin.

    Yo'naltirish **autentifikatsiyadan keyin**: bekor qilingan cookie bilan
    kelgan so'rov `/ask` ga emas, `/login` ga tushishi kerak. Aks holda
    `/chat` sessiya tekshiruvini aylanib o'tadigan yo'lga aylanardi
    (`tests/test_hardening.py::test_logout_refuses_the_old_cookie`).
    """
    if await identity_of(request) is None:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/ask", status_code=302)


@router.get("/ask", response_class=HTMLResponse)
@router.get("/ask/{conversation_id}", response_class=HTMLResponse)
async def ask_page(request: Request, conversation_id: int = 0) -> Any:
    """AI suhbat yuzasi (ui/R2b).

    Sahifa faqat qobiq: suhbatlar, xabarlar va oqim `/api` orqali keladi.
    `conversation_id` — chuqur havola; egalik **API tomonda** tekshiriladi
    (bu yerda faqat qobiqqa uzatiladi, DB'ga borilmaydi).
    """
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    s = get_settings()
    return _render(
        request,
        "ask.html",
        await _locale_for(request, ident),
        context_default=s.web_context_default_messages,
        context_max=s.web_context_max_messages,
        conversation_id=conversation_id,
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> Any:
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "dashboard.html", await _locale_for(request, ident))


# ── phase3/B: shaxsiy briefing (Mini App chuqur havolasi shu sahifaga tushadi) ──


@router.get("/briefings", response_class=HTMLResponse)
@router.get("/briefings/{day}", response_class=HTMLResponse)
async def briefings_page(request: Request, day: str = "") -> Any:
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    # `day` faqat sahifa qobig'iga beriladi; ma'lumot /api/briefings dan keladi
    # va u yerda egalik tekshiriladi.
    return _render(request, "briefings.html", await _locale_for(request, ident), day=day[:10])


# ── ui/R2a: chat ro'yxati va tafsiloti ──────────────────────────────────────


@router.get("/chats", response_class=HTMLResponse)
async def chats_page(request: Request) -> Any:
    """Chat ro'yxati. Mobilda bitta panel (`pane="list"`), desktopda ikkitasi."""
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(
        request,
        "chats.html",
        await _locale_for(request, ident),
        nav_current="chats",
        pane="list",
        selected_id=0,
    )


@router.get("/chats/{chat_id}", response_class=HTMLResponse)
async def chat_detail_page(request: Request, chat_id: int) -> Any:
    """Bitta chat tafsiloti.

    `chat_id` faqat qobiqqa beriladi — egalik `GET /api/accounts/{id}/chats/{id}`
    da tekshiriladi (begona chat 404). Mobilda `pane="content"`: ro'yxat
    ustiga **push** bo'ladi va orqaga qaytish brauzer tarixi orqali ishlaydi.
    """
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(
        request,
        "chat_detail.html",
        await _locale_for(request, ident),
        nav_current="chats",
        pane="content",
        selected_id=chat_id,
    )


# ── ui/R2c: /today · /alerts · /actions · /settings · /onboarding ────────────


@router.get("/today", response_class=HTMLResponse)
async def today_page(request: Request) -> Any:
    """Uy ekrani: briefing + alert lentasi + "so'rang" + sync holati.

    Sahifa faqat qobiq — barcha ma'lumot `/api` dan keladi va egalik o'sha
    yerda tekshiriladi.
    """
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "today.html", await _locale_for(request, ident))


@router.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request) -> Any:
    """Alert qoidalari va hodisalar (ilgari dashboard ichida edi)."""
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "alerts.html", await _locale_for(request, ident))


@router.get("/actions", response_class=HTMLResponse)
async def actions_page(request: Request) -> Any:
    """Tasdiq kutayotgan yozish amallari (invariant 4 ning yuzasi)."""
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "actions.html", await _locale_for(request, ident))


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> Any:
    """Sozlamalar: akkaunt, ko'rinish, AI standartlari, yozish rejimi,
    auto-reply, briefing/jim soatlar, maxfiylik.

    Kontekst chegaralari **server sozlamasi** (pref emas) — shuning uchun
    faqat o'qish uchun qobiqqa beriladi.
    """
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    s = get_settings()
    return _render(
        request,
        "settings.html",
        await _locale_for(request, ident),
        context_default=s.web_context_default_messages,
        context_max=s.web_context_max_messages,
    )


@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request) -> Any:
    """To'rt qadamli birinchi tajriba — **kirish talab qilmaydi**.

    1-qadam aynan kirmagan foydalanuvchi uchun (QR login), shuning uchun
    bu yerda `/login` ga yo'naltirish yo'q. Kirgan foydalanuvchida klient
    birinchi qadamni o'tkazib yuboradi.
    """
    ident = await identity_of(request)
    return _render(
        request, "onboarding.html", await _locale_for(request, ident), logged_in=ident is not None
    )
