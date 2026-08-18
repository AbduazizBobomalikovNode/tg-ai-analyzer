"""Telefon raqami orqali MTProto login oqimi (telefon → kod → 2FA).

Kod va parol **faqat HTTPS web/Mini App forma** orqali keladi (rejaning
4.1-bandi). Chat orqali hech qachon — Telegram chat'da yuborilgan kodni bekor
qiladi. Bu modul transportni bilmaydi: web router uni chaqiradi.

Holat mashinasi (bitta `PendingLogin`):

    start(phone) ──► CODE_SENT ──submit_code──► DONE
                                    │
                                    └─(2FA)──► NEEDS_2FA ──submit_password──► DONE

Har oqim o'z `GuardedTelegramClient` ini ushlab turadi (send_code va sign_in
bir xil klient/sessiyada bo'lishi shart). Shuning uchun store **jarayon
xotirasida** — `api` servisi bitta jarayon bo'lib ishlaydi (uvicorn workers=1).
TTL o'tsa oqim va klient tozalanadi.

Barcha auth TL metodlari faqat `auth_window()` ichida — allowlist shunday
talab qiladi; oqim tashqarisida agent hech qachon logout/login qila olmaydi.

Sirlar: telefon (faqat hash), kod, parol, session string — log'ga tushmaydi.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from telethon import errors as tg_errors

from app.config import get_settings
from app.logging import get_logger
from app.mtproto import auth_window, build_client
from app.services.session_store import phone_hash

log = get_logger(__name__)


class FlowStatus(StrEnum):
    CODE_SENT = "code_sent"
    NEEDS_2FA = "needs_2fa"
    DONE = "done"


class AuthError(Exception):
    """Foydalanuvchiga ko'rsatiladigan, i18n kalitli xato."""

    def __init__(self, code: str, *, retry_after: int | None = None, detail: str = "") -> None:
        self.code = code  # i18n: auth.err.<code>
        self.retry_after = retry_after
        self.detail = detail
        super().__init__(code)


@dataclass(slots=True)
class LoggedIn:
    """Muvaffaqiyatli login natijasi — DB'ga yozish uchun kerak bo'lgan hamma narsa."""

    tg_user_id: int
    first_name: str
    username: str | None
    phone_hash: str
    session_string: str  # ⚠️ sir — faqat session_store.store_session() ga beriladi

    def __repr__(self) -> str:
        return f"LoggedIn(tg_user_id={self.tg_user_id}, username={self.username!r})"


@dataclass(slots=True)
class PendingLogin:
    id: str
    client: Any
    phone_hash: str
    created_at: float
    owner_user_id: int | None = None  # web'da allaqachon login bo'lgan user (multi-account)
    status: FlowStatus = FlowStatus.CODE_SENT
    phone: str = field(default="", repr=False)  # sign_in uchun kerak, saqlanmaydi
    phone_code_hash: str = field(default="", repr=False)
    code_type: str = ""  # app | sms | call | flash_call | ...
    code_length: int | None = None
    timeout: int | None = None
    attempts: int = 0
    result: LoggedIn | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


MAX_CODE_ATTEMPTS = 5


def _code_type_name(sent_type: Any) -> str:
    name = type(sent_type).__name__  # SentCodeTypeApp / SentCodeTypeSms / ...
    return name.removeprefix("SentCodeType").lower() or "unknown"


class AuthFlowStore:
    """Jarayon ichidagi oqimlar. Bitta instansiya — `web.state.auth_flows`."""

    def __init__(self, *, ttl_seconds: float | None = None, client_factory: Any = None) -> None:
        self._flows: dict[str, PendingLogin] = {}
        self._ttl = ttl_seconds or get_settings().web_auth_flow_ttl_min * 60
        # testlarda soxta klient ulash uchun
        self._client_factory = client_factory or (lambda: build_client(None))

    # ── lifecycle ────────────────────────────────────────────────────────────

    def get(self, flow_id: str) -> PendingLogin:
        self._gc()
        flow = self._flows.get(flow_id)
        if flow is None:
            raise AuthError("flow_expired")
        return flow

    def _gc(self) -> None:
        now = time.monotonic()
        dead = [fid for fid, f in self._flows.items() if now - f.created_at > self._ttl]
        for fid in dead:
            flow = self._flows.pop(fid)
            _fire_and_forget(_disconnect(flow.client))
            log.info("auth.flow_expired", flow_id=fid[:8])

    async def cancel(self, flow_id: str) -> None:
        flow = self._flows.pop(flow_id, None)
        if flow is not None:
            await _disconnect(flow.client)

    async def close_all(self) -> None:
        flows = list(self._flows.values())
        self._flows.clear()
        for f in flows:
            await _disconnect(f.client)

    @property
    def active_count(self) -> int:
        self._gc()
        return len(self._flows)

    # ── qadamlar ─────────────────────────────────────────────────────────────

    async def start(self, phone: str, *, owner_user_id: int | None = None) -> PendingLogin:
        phone = _normalize_phone(phone)
        if len(phone) < 8:
            raise AuthError("invalid_phone")

        client = self._client_factory()
        try:
            await client.connect()
            with auth_window():
                sent = await client.send_code_request(phone)
        except tg_errors.PhoneNumberInvalidError as exc:
            await _disconnect(client)
            raise AuthError("invalid_phone") from exc
        except tg_errors.PhoneNumberBannedError as exc:
            await _disconnect(client)
            raise AuthError("phone_banned") from exc
        except tg_errors.FloodWaitError as exc:
            await _disconnect(client)
            raise AuthError("flood", retry_after=int(exc.seconds)) from exc
        except tg_errors.RPCError as exc:
            await _disconnect(client)
            raise AuthError("telegram", detail=type(exc).__name__) from exc
        except (OSError, ConnectionError, TimeoutError) as exc:
            await _disconnect(client)
            raise AuthError("network", detail=str(exc)[:100]) from exc

        flow = PendingLogin(
            id=secrets.token_urlsafe(24),
            client=client,
            phone_hash=phone_hash(phone),
            created_at=time.monotonic(),
            owner_user_id=owner_user_id,
            phone=phone,
            phone_code_hash=getattr(sent, "phone_code_hash", "") or "",
            code_type=_code_type_name(getattr(sent, "type", None)),
            code_length=getattr(getattr(sent, "type", None), "length", None),
            timeout=getattr(sent, "timeout", None),
        )
        self._flows[flow.id] = flow
        log.info(
            "auth.code_sent",
            flow_id=flow.id[:8],
            phone_hash=flow.phone_hash[:12],
            code_type=flow.code_type,
        )
        return flow

    async def submit_code(self, flow_id: str, code: str) -> PendingLogin:
        flow = self.get(flow_id)
        code = "".join(ch for ch in code if ch.isdigit())
        if not code:
            raise AuthError("invalid_code")

        async with flow.lock:
            if flow.status is not FlowStatus.CODE_SENT:
                raise AuthError("wrong_step")
            flow.attempts += 1
            if flow.attempts > MAX_CODE_ATTEMPTS:
                await self.cancel(flow_id)
                raise AuthError("too_many_attempts")
            try:
                with auth_window():
                    await flow.client.sign_in(
                        flow.phone, code, phone_code_hash=flow.phone_code_hash
                    )
            except tg_errors.SessionPasswordNeededError:
                flow.status = FlowStatus.NEEDS_2FA
                log.info("auth.needs_2fa", flow_id=flow.id[:8])
                return flow
            except tg_errors.PhoneCodeInvalidError as exc:
                raise AuthError("invalid_code") from exc
            except tg_errors.PhoneCodeExpiredError as exc:
                await self.cancel(flow_id)
                raise AuthError("code_expired") from exc
            except tg_errors.PhoneNumberUnoccupiedError as exc:
                await self.cancel(flow_id)
                raise AuthError("signup_required") from exc
            except tg_errors.FloodWaitError as exc:
                raise AuthError("flood", retry_after=int(exc.seconds)) from exc
            except tg_errors.RPCError as exc:
                raise AuthError("telegram", detail=type(exc).__name__) from exc

            await self._complete(flow)
            return flow

    async def submit_password(self, flow_id: str, password: str) -> PendingLogin:
        flow = self.get(flow_id)
        if not password:
            raise AuthError("invalid_password")

        async with flow.lock:
            if flow.status is not FlowStatus.NEEDS_2FA:
                raise AuthError("wrong_step")
            flow.attempts += 1
            if flow.attempts > MAX_CODE_ATTEMPTS * 2:
                await self.cancel(flow_id)
                raise AuthError("too_many_attempts")
            try:
                with auth_window():
                    await flow.client.sign_in(password=password)
            except tg_errors.PasswordHashInvalidError as exc:
                raise AuthError("invalid_password") from exc
            except tg_errors.FloodWaitError as exc:
                raise AuthError("flood", retry_after=int(exc.seconds)) from exc
            except tg_errors.RPCError as exc:
                raise AuthError("telegram", detail=type(exc).__name__) from exc

            await self._complete(flow)
            return flow

    async def _complete(self, flow: PendingLogin) -> None:
        """Login bo'ldi: `me` ni olib, session'ni chiqarib, klientni yopamiz."""
        me = await flow.client.get_me()
        session_string = flow.client.session.save()
        flow.result = LoggedIn(
            tg_user_id=int(me.id),
            first_name=getattr(me, "first_name", "") or "",
            username=getattr(me, "username", None),
            phone_hash=flow.phone_hash,
            session_string=session_string,
        )
        flow.status = FlowStatus.DONE
        flow.phone = ""
        flow.phone_code_hash = ""
        log.info("auth.done", flow_id=flow.id[:8], tg_user_id=me.id)

    def take_result(self, flow_id: str) -> LoggedIn:
        """Natijani oladi va oqimni yopadi (bir marta)."""
        flow = self._flows.pop(flow_id, None)
        if flow is None or flow.result is None:
            raise AuthError("flow_expired")
        _fire_and_forget(_disconnect(flow.client))
        return flow.result


# ─── yordamchilar ────────────────────────────────────────────────────────────


def _normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return f"+{digits}" if digits else ""


async def _disconnect(client: Any) -> None:
    try:
        res = client.disconnect()
        if asyncio.iscoroutine(res):
            await res
    except Exception as exc:  # yopishda xato — sessiya baribir tashlab yuboriladi
        log.debug("auth.disconnect_failed", error=str(exc))


def _fire_and_forget(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        return
    task = loop.create_task(coro)
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
