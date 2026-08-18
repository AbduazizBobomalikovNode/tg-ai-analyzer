"""HTML sahifalar (Jinja2). Ma'lumot — /api orqali, sahifa faqat qobiq."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.i18n import SUPPORTED, _load
from app.web.security import LANG_COOKIE, identity_of, locale_of

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _strings(locale: str) -> dict[str, str]:
    """JS'ga beriladigan tarjimalar (faqat web.*, *.err.*, error.*)."""
    merged = {**_load("uz"), **_load(locale)}
    return {
        k: v
        for k, v in merged.items()
        if k.startswith(
            ("web.", "auth.err.", "pool.err.", "chat.err.", "action.err.", "sync.err.", "error.")
        )
    }


def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    locale = locale_of(request)
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


def _render(request: Request, name: str, **extra: Any) -> HTMLResponse:
    resp = templates.TemplateResponse(request, name, _ctx(request, **extra))
    if request.query_params.get("lang") in SUPPORTED:
        resp.set_cookie(LANG_COOKIE, locale_of(request), max_age=365 * 86400, samesite="lax")
    return resp


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return RedirectResponse("/chat" if identity_of(request) else "/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Any:
    return _render(request, "login.html", logged_in=identity_of(request) is not None)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Any:
    if identity_of(request) is None:
        return RedirectResponse("/login", status_code=302)
    s = get_settings()
    return _render(
        request,
        "chat.html",
        context_default=s.web_context_default_messages,
        context_max=s.web_context_max_messages,
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> Any:
    if identity_of(request) is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "dashboard.html")
