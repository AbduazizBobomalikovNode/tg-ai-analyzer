"""Telefon → kod → 2FA holat mashinasi — soxta Telethon klient bilan, tarmoqsiz.

Qo'riqlaydi:
  * auth TL metodlari faqat `auth_window()` ichida chaqiriladi
  * 2FA yo'li, xato mapping'i, urinish limiti, TTL
  * natija bir marta olinadi va klient yopiladi; sir repr'ga chiqmaydi
"""

from __future__ import annotations

from typing import Any

import pytest
from telethon import errors as tg_errors

from app.mtproto.allowlist import _auth_open
from app.services import auth_flow as AF


class _Me:
    id = 777
    first_name = "Ali"
    username = "ali"


class _Session:
    def save(self) -> str:
        return "1AbCsessionSTRING"


class SentCodeTypeApp:  # Telethon'dagi nom bilan bir xil — code_type shundan olinadi
    length = 5


class _Sent:
    phone_code_hash = "pch"
    timeout = 60
    type = SentCodeTypeApp()


class FakeClient:
    """Sozlanuvchi ssenariy: `send_code_error`, `sign_in_errors` (navbat), `needs_2fa`."""

    def __init__(
        self,
        *,
        send_code_error: Exception | None = None,
        sign_in_errors: list[Exception] | None = None,
        needs_2fa: bool = False,
        password_error: Exception | None = None,
    ) -> None:
        self.send_code_error = send_code_error
        self.sign_in_errors = list(sign_in_errors or [])
        self.needs_2fa = needs_2fa
        self.password_error = password_error
        self.connected = False
        self.disconnected = False
        self.calls: list[tuple[str, bool]] = []  # (metod, auth_window ochiqmi)
        self.session = _Session()

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def send_code_request(self, phone: str) -> Any:
        self.calls.append(("send_code", _auth_open.get()))
        if self.send_code_error:
            raise self.send_code_error
        return _Sent()

    async def sign_in(self, phone: str | None = None, code: str | None = None, **kw: Any) -> Any:
        if "password" in kw:
            self.calls.append(("check_password", _auth_open.get()))
            if self.password_error:
                raise self.password_error
            return _Me()
        self.calls.append(("sign_in", _auth_open.get()))
        if self.sign_in_errors:
            raise self.sign_in_errors.pop(0)
        if self.needs_2fa:
            raise tg_errors.SessionPasswordNeededError(request=None)
        return _Me()

    async def get_me(self) -> Any:
        return _Me()


def _store(client: FakeClient, ttl: float = 600) -> AF.AuthFlowStore:
    return AF.AuthFlowStore(ttl_seconds=ttl, client_factory=lambda: client)


async def test_happy_path_without_2fa() -> None:
    client = FakeClient()
    store = _store(client)
    flow = await store.start("+998 90 123-45-67")
    assert flow.status is AF.FlowStatus.CODE_SENT
    assert flow.code_type == "app" and flow.code_length == 5
    assert client.connected

    flow = await store.submit_code(flow.id, "12345")
    assert flow.status is AF.FlowStatus.DONE
    result = store.take_result(flow.id)
    assert result.tg_user_id == 777 and result.username == "ali"
    assert result.session_string == "1AbCsessionSTRING"
    assert "session" not in repr(result).lower() or "STRING" not in repr(result)
    # oqim yopildi, qayta olib bo'lmaydi
    with pytest.raises(AF.AuthError, match="flow_expired"):
        store.take_result(flow.id)
    # auth metodlari faqat auth_window ichida chaqirilgan
    assert client.calls == [("send_code", True), ("sign_in", True)]
    assert _auth_open.get() is False  # oyna yopilgan


async def test_2fa_path() -> None:
    client = FakeClient(needs_2fa=True)
    store = _store(client)
    flow = await store.start("+998901234567")
    flow = await store.submit_code(flow.id, "11111")
    assert flow.status is AF.FlowStatus.NEEDS_2FA
    # kodni qayta yuborish — noto'g'ri qadam
    with pytest.raises(AF.AuthError, match="wrong_step"):
        await store.submit_code(flow.id, "11111")
    flow = await store.submit_password(flow.id, "hunter2")
    assert flow.status is AF.FlowStatus.DONE
    assert ("check_password", True) in client.calls
    assert flow.phone == "" and flow.phone_code_hash == ""  # tozalangan


async def test_phone_hash_not_raw() -> None:
    client = FakeClient()
    flow = await _store(client).start("+998901234567")
    assert "998901234567" not in flow.phone_hash and len(flow.phone_hash) == 64


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (tg_errors.PhoneNumberInvalidError(request=None), "invalid_phone"),
        (tg_errors.PhoneNumberBannedError(request=None), "phone_banned"),
        (tg_errors.FloodWaitError(request=None, capture=42), "flood"),
    ],
)
async def test_send_code_errors_are_mapped(exc: Exception, code: str) -> None:
    client = FakeClient(send_code_error=exc)
    store = _store(client)
    with pytest.raises(AF.AuthError) as ei:
        await store.start("+998901234567")
    assert ei.value.code == code
    if code == "flood":
        assert ei.value.retry_after == 42
    assert client.disconnected  # klient tashlab yuborildi
    assert store.active_count == 0


async def test_invalid_phone_rejected_before_network() -> None:
    client = FakeClient()
    with pytest.raises(AF.AuthError, match="invalid_phone"):
        await _store(client).start("123")
    assert not client.connected


async def test_invalid_code_then_success_counts_attempts() -> None:
    client = FakeClient(sign_in_errors=[tg_errors.PhoneCodeInvalidError(request=None)])
    store = _store(client)
    flow = await store.start("+998901234567")
    with pytest.raises(AF.AuthError, match="invalid_code"):
        await store.submit_code(flow.id, "00000")
    assert flow.attempts == 1
    flow = await store.submit_code(flow.id, "12345")
    assert flow.status is AF.FlowStatus.DONE


async def test_too_many_attempts_cancels_flow() -> None:
    errs = [tg_errors.PhoneCodeInvalidError(request=None) for _ in range(10)]
    client = FakeClient(sign_in_errors=errs)
    store = _store(client)
    flow = await store.start("+998901234567")
    for _ in range(AF.MAX_CODE_ATTEMPTS):
        with pytest.raises(AF.AuthError, match="invalid_code"):
            await store.submit_code(flow.id, "00000")
    with pytest.raises(AF.AuthError, match="too_many_attempts"):
        await store.submit_code(flow.id, "00000")
    assert client.disconnected and store.active_count == 0


async def test_code_expired_and_signup_required_cancel() -> None:
    for exc, code in (
        (tg_errors.PhoneCodeExpiredError(request=None), "code_expired"),
        (tg_errors.PhoneNumberUnoccupiedError(request=None), "signup_required"),
    ):
        client = FakeClient(sign_in_errors=[exc])
        store = _store(client)
        flow = await store.start("+998901234567")
        with pytest.raises(AF.AuthError, match=code):
            await store.submit_code(flow.id, "12345")
        assert store.active_count == 0 and client.disconnected


async def test_wrong_password() -> None:
    client = FakeClient(
        needs_2fa=True, password_error=tg_errors.PasswordHashInvalidError(request=None)
    )
    store = _store(client)
    flow = await store.start("+998901234567")
    await store.submit_code(flow.id, "12345")
    with pytest.raises(AF.AuthError, match="invalid_password"):
        await store.submit_password(flow.id, "bad")
    assert flow.status is AF.FlowStatus.NEEDS_2FA  # qayta urinsa bo'ladi


async def test_flow_expires_by_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    store = _store(client, ttl=1)
    flow = await store.start("+998901234567")
    flow.created_at -= 5  # vaqtni "oldinga surish"
    with pytest.raises(AF.AuthError, match="flow_expired"):
        store.get(flow.id)
    assert store.active_count == 0


async def test_unknown_flow() -> None:
    store = _store(FakeClient())
    with pytest.raises(AF.AuthError, match="flow_expired"):
        await store.submit_code("nope-nope-nope", "12345")


async def test_cancel_disconnects() -> None:
    client = FakeClient()
    store = _store(client)
    flow = await store.start("+998901234567")
    await store.cancel(flow.id)
    assert client.disconnected and store.active_count == 0
