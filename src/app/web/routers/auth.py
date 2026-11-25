"""Telefon orqali Telegram auth — HTTPS API (rejaning 4.1-bandi).

    POST /api/auth/phone     {phone}              → {flow_id, status, code_type, ...}
    POST /api/auth/code      {flow_id, code}      → {status: needs_2fa | done}
    POST /api/auth/password  {flow_id, password}  → {status: done}
    POST /api/auth/cancel    {flow_id}
    POST /api/auth/logout                          — web cookie o'chadi (session saqlanadi)
    POST /api/auth/telegram  {init_data}           — Mini App: initData HMAC → cookie
    GET  /api/me                                   — user (+prefs) + akkauntlar

`done` bo'lganda javob cookie o'rnatadi. Xatolar: 400 {"code": "auth.err.<x>"}.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Chat, User
from app.logging import get_logger
from app.services import prefs as P
from app.services.accounts import link_account, list_accounts
from app.services.auth_flow import AuthError, AuthFlowStore, FlowStatus
from app.web.security import (
    SESSION_COOKIE,
    RateLimiter,
    WebIdentity,
    clear_session_cookie,
    client_ip,
    identity_of,
    read_session,
    require_csrf,
    require_identity,
    revoke_sessions,
    set_session_cookie,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["auth"])


def flows(request: Request) -> AuthFlowStore:
    return request.app.state.auth_flows  # type: ignore[no-any-return]


def limiter(request: Request) -> RateLimiter:
    return request.app.state.auth_limiter  # type: ignore[no-any-return]


class PhoneIn(BaseModel):
    phone: str = Field(min_length=6, max_length=32)


class CodeIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)
    code: str = Field(min_length=3, max_length=12)


class PasswordIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)
    password: str = Field(min_length=1, max_length=512)


class FlowIn(BaseModel):
    flow_id: str = Field(min_length=8, max_length=64)


def _flow_out(flow: Any) -> dict[str, Any]:
    return {
        "flow_id": flow.id,
        "status": str(flow.status),
        "code_type": flow.code_type,
        "next_type": flow.next_type,  # bo'sh = "qayta yuborish" tugmasi ko'rsatilmaydi
        "code_length": flow.code_length,
        "timeout": flow.timeout,
    }


def _auth_http(exc: AuthError) -> HTTPException:
    return HTTPException(
        status_code=429 if exc.code == "flood" else 400,
        detail={"code": f"auth.err.{exc.code}", "retry_after": exc.retry_after},
    )


@router.post("/auth/phone", dependencies=[Depends(require_csrf)])
async def auth_phone(body: PhoneIn, request: Request) -> dict[str, Any]:
    if not limiter(request).check(client_ip(request)):
        raise HTTPException(status_code=429, detail={"code": "auth.err.rate_limited"})
    ident = await identity_of(request)
    try:
        flow = await flows(request).start(
            body.phone, owner_user_id=ident.user_id if ident else None
        )
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return _flow_out(flow)


async def _finish(request: Request, response: Response, flow_id: str) -> dict[str, Any]:
    """DONE oqimni DB'ga tushiradi va cookie beradi."""
    store = flows(request)
    flow = store.get(flow_id)
    if flow.status is not FlowStatus.DONE:
        return _flow_out(flow)
    result = store.take_result(flow_id)
    try:
        async with session_scope() as db:
            linked = await link_account(db, result, owner_user_id=flow.owner_user_id)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    finally:
        del result  # session string RAM'da qolmasin
    await set_session_cookie(request, response, linked.user_id)
    await _kick_sync(linked.account_id)
    return {"flow_id": flow_id, "status": "done", "account_id": linked.account_id}


async def _kick_sync(account_id: int) -> None:
    """Login'dan keyin birinchi ingestion (Redis yo'q bo'lsa login baribir o'tadi)."""
    try:
        from app.worker.queue import enqueue
        from app.worker.tasks import job_id_account

        await enqueue("sync_account", account_id, True, job_id=job_id_account(account_id))
    except Exception as exc:  # navbat mavjud emas — cron keyin baribir oladi
        log.warning("auth.sync_enqueue_failed", account_id=account_id, error=str(exc)[:200])


def _qr_data_uri(url: str) -> str:
    """`tg://login?token=…` → SVG data URI.

    Server tomonda chiziladi: brauzerga tashqi kutubxona qo'shmaymiz (CDN yo'q)
    va natija `<img src>` bo'lib beriladi — `innerHTML` ishlatilmaydi, ya'ni
    XSS yuzasi yo'q.
    """
    import base64
    import io

    import qrcode
    from qrcode.image.svg import SvgPathImage

    img = qrcode.make(url, image_factory=SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# Oqim bo'yicha oxirgi chizilgan QR: `{flow_id: (qr_url, data_uri)}`.
# Frontend har 2 soniyada `/auth/qr/poll` so'raydi, token esa ~1 daqiqada bir
# marta yangilanadi — keshsiz har poll'da SVG qaytadan chizilardi (CPU va
# ~10 KB base64 behuda). Kesh jarayon xotirasida, `AuthFlowStore` kabi
# (`api` bitta uvicorn worker).
_QR_CACHE: dict[str, tuple[str, str]] = {}
_QR_CACHE_MAX = 64


def qr_data_uri_cached(flow_id: str, url: str) -> str:
    cached = _QR_CACHE.get(flow_id)
    if cached is not None and cached[0] == url:
        return cached[1]
    if len(_QR_CACHE) >= _QR_CACHE_MAX:  # oqim bekor qilinmay qolsa ham o'smasin
        _QR_CACHE.clear()
    uri = _qr_data_uri(url)
    _QR_CACHE[flow_id] = (url, uri)
    return uri


def _qr_forget(flow_id: str) -> None:
    _QR_CACHE.pop(flow_id, None)


def _qr_out(flow: Any) -> dict[str, Any]:
    out = _flow_out(flow)
    if flow.status is FlowStatus.QR_PENDING and flow.qr_url:
        out["qr"] = qr_data_uri_cached(flow.id, flow.qr_url)
    else:
        _qr_forget(flow.id)
    if flow.error:
        out["error"] = flow.error
    return out


@router.post("/auth/qr", dependencies=[Depends(require_csrf)])
async def auth_qr(request: Request) -> dict[str, Any]:
    """QR login boshlash — telefon ham, kod ham kerak emas."""
    if not limiter(request).check(client_ip(request)):
        raise HTTPException(status_code=429, detail={"code": "auth.err.rate_limited"})
    ident = await identity_of(request)
    try:
        flow = await flows(request).start_qr(owner_user_id=ident.user_id if ident else None)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return _qr_out(flow)


@router.post("/auth/qr/poll", dependencies=[Depends(require_csrf)])
async def auth_qr_poll(body: FlowIn, request: Request, response: Response) -> dict[str, Any]:
    """Skanerlandimi? Frontend shu yerni so'rab turadi (rate limit yo'q — poll)."""
    try:
        flow = flows(request).get(body.flow_id)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    if flow.status is FlowStatus.DONE:
        return await _finish(request, response, body.flow_id)
    return _qr_out(flow)


@router.post("/auth/resend", dependencies=[Depends(require_csrf)])
async def auth_resend(body: FlowIn, request: Request) -> dict[str, Any]:
    """Kodni qayta yuborish — Telegram odatda SMS'ga o'tkazadi.

    IP shifti `/auth/phone` bilan bir xil: aks holda bu yangi oqim ochmasdan
    cheksiz kod yuborish yo'liga aylanardi.
    """
    if not limiter(request).check(client_ip(request)):
        raise HTTPException(status_code=429, detail={"code": "auth.err.rate_limited"})
    try:
        flow = await flows(request).resend(body.flow_id)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return _flow_out(flow)


@router.post("/auth/code", dependencies=[Depends(require_csrf)])
async def auth_code(body: CodeIn, request: Request, response: Response) -> dict[str, Any]:
    try:
        flow = await flows(request).submit_code(body.flow_id, body.code)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    if flow.status is FlowStatus.NEEDS_2FA:
        return _flow_out(flow)
    return await _finish(request, response, body.flow_id)


@router.post("/auth/password", dependencies=[Depends(require_csrf)])
async def auth_password(body: PasswordIn, request: Request, response: Response) -> dict[str, Any]:
    try:
        await flows(request).submit_password(body.flow_id, body.password)
    except AuthError as exc:
        raise _auth_http(exc) from exc
    return await _finish(request, response, body.flow_id)


@router.post("/auth/cancel", dependencies=[Depends(require_csrf)])
async def auth_cancel(body: FlowIn, request: Request) -> dict[str, str]:
    await flows(request).cancel(body.flow_id)
    _qr_forget(body.flow_id)
    return {"status": "cancelled"}


@router.post("/auth/logout", dependencies=[Depends(require_csrf)])
async def auth_logout(request: Request, response: Response) -> dict[str, str]:
    """Cookie'ni o'chiradi **va** server tomonda bekor qiladi.

    Faqat cookie'ni o'chirish yetarli emas: nusxa olingan (o'g'irlangan) cookie
    14 kun davomida amal qilardi. Redis'dagi `web:sess:revoked:<uid>` chegarasi
    shu daqiqagacha berilgan hamma cookie'ni rad etadi. Redis yo'q bo'lsa
    `revoked=false` qaytadi — fail-open, `security.is_revoked` hujjatiga qarang.
    """
    ident = read_session(request.cookies.get(SESSION_COOKIE))
    clear_session_cookie(response)
    revoked = await revoke_sessions(ident.user_id) if ident else False
    return {"status": "ok", "revoked": "true" if revoked else "false"}


async def _chat_counts(db: Any, account_ids: list[int]) -> dict[int, tuple[int, int]]:
    """`{account_id: (chats_total, chats_synced)}` — bitta agregat so'rov.

    "synced" = `synced_total > 0`, ya'ni chatdan hech bo'lmasa bir xabar
    olingan. `sync_state = done` emas: tarix boshiga yetmagan chat ham
    qidiruv uchun foydali va UI'da "bo'sh" ko'rinmasligi kerak.
    """
    if not account_ids:
        return {}
    from sqlalchemy import case, func, select

    rows = await db.execute(
        select(
            Chat.account_id,
            func.count(Chat.id),
            func.sum(case((Chat.synced_total > 0, 1), else_=0)),
        )
        .where(Chat.account_id.in_(account_ids))
        .group_by(Chat.account_id)
    )
    return {int(aid): (int(total or 0), int(synced or 0)) for aid, total, synced in rows.all()}


@router.get("/me")
async def me(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        user = await db.get(User, ident.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "unauthorized"})
        accounts = await list_accounts(db, user.id)
        counts = await _chat_counts(db, [a.id for a in accounts])
        # ui/R1b: `display_name` ustuni bo'lsa (keyingi migratsiya) — u,
        # aks holda username. Avatar hozircha yo'q: `null` qaytadi va UI
        # bosh harflarga tushadi (endpoint qo'shilganda faqat shu qiymat
        # to'ladi, shartnoma o'zgarmaydi).
        display = (getattr(user, "display_name", "") or user.username or "").strip()
        return {
            "user": {
                "id": user.id,
                "tg_user_id": user.tg_user_id,
                "username": user.username,
                "display_name": display or f"id{user.tg_user_id}",
                "avatar_url": None,
                "prefs": P.normalize(user.prefs),
            },
            "accounts": [
                {
                    "id": a.id,
                    "label": a.label,
                    "status": a.status,
                    "tg_account_id": a.tg_account_id,
                    "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
                    "chats_total": counts.get(a.id, (0, 0))[0],
                    "chats_synced": counts.get(a.id, (0, 0))[1],
                }
                for a in accounts
            ],
            "max_accounts": get_settings().max_accounts,
        }


# ─── Telegram Mini App (initData) ────────────────────────────────────────────


class InitDataIn(BaseModel):
    # Telegram `initData` — URL-kodlangan query satr. 4096 — Telegram'ning
    # o'z chegarasidan (`tgWebAppData`) ancha keng zaxira bilan.
    init_data: str = Field(min_length=1, max_length=4096)


def verify_init_data(init_data: str, bot_token: str, *, max_age: int = 86_400) -> dict[str, Any]:
    """Mini App `initData` imzosini tekshiradi.

    https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app

    Kalit = `HMAC_SHA256("WebAppData", <bot token>)`, imzo = shu kalit bilan
    `data_check_string` (kalitlar alifbo tartibida, `k=v`, `\\n` bilan) ustidan
    HMAC-SHA256. Taqqoslash **doimiy vaqtli**: oddiy `!=` imzoni belgima-belgi
    tiklash uchun vaqt kanalini ochadi.

    Bot token **hech qachon** log'ga yoki xato matniga tushmaydi — bu yerda u
    faqat HMAC kaliti sifatida ishlatiladi.

    Returns:
        Tekshirilgan maydonlar (`user` JSON bo'lib ochilgan).

    Raises:
        AuthError("init_data"): imzo/format noto'g'ri.
        AuthError("init_data_expired"): `auth_date` juda eski.
    """
    import hashlib
    import hmac as _hmac
    from urllib.parse import parse_qsl

    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise AuthError("init_data") from exc
    fields = dict(pairs)
    got = fields.pop("hash", "")
    if not got or not fields:
        raise AuthError("init_data")

    check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = _hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    want = _hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not _hmac.compare_digest(want, got.lower()):
        raise AuthError("init_data")

    try:
        auth_date = int(fields.get("auth_date", "0"))
    except ValueError as exc:
        raise AuthError("init_data") from exc
    # Kelajakdagi sana ham rad etiladi (soat siljishi uchun kichik zaxira).
    age = time.time() - auth_date
    if auth_date <= 0 or age > max_age or age < -300:
        raise AuthError("init_data_expired")

    out: dict[str, Any] = dict(fields)
    if fields.get("user"):
        try:
            out["user"] = json.loads(fields["user"])
        except ValueError as exc:
            raise AuthError("init_data") from exc
    return out


@router.post("/auth/telegram", dependencies=[Depends(require_csrf)])
async def auth_telegram(body: InitDataIn, request: Request, response: Response) -> dict[str, Any]:
    """Mini App ichida avtomatik kirish: `initData` → o'sha web cookie.

    **Akkaunt yaratmaydi.** `initData` faqat "bu brauzerni shu Telegram
    foydalanuvchisi ochdi" deyishga yetadi; MTProto sessiyasi esa QR login'da
    tug'iladi. Shuning uchun `tg_user_id` bazadagi userga mos kelmasa — 403,
    UI QR login'ga yuboradi.
    """
    if not limiter(request).check(client_ip(request)):
        raise HTTPException(status_code=429, detail={"code": "auth.err.rate_limited"})
    s = get_settings()
    try:
        data = verify_init_data(body.init_data, s.bot_token, max_age=s.miniapp_init_data_max_age)
    except AuthError as exc:
        raise _auth_http(exc) from exc

    tg_user = data.get("user")
    tg_user_id = int(tg_user["id"]) if isinstance(tg_user, dict) and tg_user.get("id") else 0
    if not tg_user_id:
        raise HTTPException(status_code=400, detail={"code": "auth.err.init_data"})

    async with session_scope() as db:
        user = (
            await db.execute(select(User).where(User.tg_user_id == tg_user_id))
        ).scalar_one_or_none()
        if user is None:
            log.info("auth.miniapp_unlinked", tg_user_id=tg_user_id)
            raise HTTPException(status_code=403, detail={"code": "auth.err.not_linked"})
        user_id = int(user.id)
    await set_session_cookie(request, response, user_id)
    log.info("auth.miniapp_ok", user_id=user_id)
    return {"status": "done", "user_id": user_id}
