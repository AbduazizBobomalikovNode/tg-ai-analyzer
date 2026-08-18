"""Guardrail testlari — loyihaning eng muhim invarianti.

Bu testlar sinsa, "agent o'chira olmaydi" kafolati buzilgan bo'ladi.
"""

from __future__ import annotations

import pytest

from app.mtproto.allowlist import (
    ALLOWED,
    AUTH,
    DENIED,
    RpcBlocked,
    auth_window,
    check_request,
    is_write,
    request_key,
)


def fake_request(module: str, name: str) -> object:
    """`telethon.tl.functions.<module>.<name>` ni taqlid qiluvchi obyekt."""
    cls = type(name, (), {})
    cls.__module__ = f"telethon.tl.functions.{module}"
    return cls()


# ─── invariantlar ────────────────────────────────────────────────────────────


def test_allowlist_and_denylist_never_intersect() -> None:
    assert not (ALLOWED | AUTH) & DENIED


def test_no_delete_method_is_allowed() -> None:
    leaked = {k for k in ALLOWED | AUTH if "delete" in k.lower()}
    assert not leaked, f"o'chirish metodi oq ro'yxatga sizib kirdi: {leaked}"


def test_no_logout_or_leave_is_allowed() -> None:
    banned = ("logout", "leavechannel", "resetauthorization")
    leaked = {k for k in ALLOWED | AUTH if any(b in k.lower().replace(".", "") for b in banned)}
    assert not leaked, f"xavfli metod oq ro'yxatda: {leaked}"


# ─── request_key ─────────────────────────────────────────────────────────────


def test_request_key_namespaced() -> None:
    assert (
        request_key(fake_request("messages", "GetHistoryRequest")) == "messages.GetHistoryRequest"
    )


def test_request_key_distinguishes_namespaces() -> None:
    """messages.DeleteMessages va channels.DeleteMessages — boshqa-boshqa metodlar."""
    a = request_key(fake_request("messages", "DeleteMessagesRequest"))
    b = request_key(fake_request("channels", "DeleteMessagesRequest"))
    assert a != b
    assert a in DENIED and b in DENIED


# ─── check_request ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("messages", "DeleteMessagesRequest"),
        ("channels", "DeleteMessagesRequest"),
        ("messages", "DeleteHistoryRequest"),
        ("channels", "DeleteChannelRequest"),
        ("channels", "LeaveChannelRequest"),
        ("account", "DeleteAccountRequest"),
        ("auth", "LogOutRequest"),
        ("channels", "EditBannedRequest"),
    ],
)
def test_destructive_requests_blocked(module: str, name: str) -> None:
    with pytest.raises(RpcBlocked, match="qora ro'yxatda"):
        check_request(fake_request(module, name))


def test_unknown_request_blocked_by_default() -> None:
    with pytest.raises(RpcBlocked, match="deny-by-default"):
        check_request(fake_request("messages", "SomeBrandNewRequest"))


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("messages", "GetHistoryRequest"),
        ("messages", "SearchRequest"),
        ("messages", "SendMessageRequest"),
        ("messages", "EditMessageRequest"),
        ("messages", "ForwardMessagesRequest"),
        ("messages", "UpdatePinnedMessageRequest"),
        ("stats", "GetBroadcastStatsRequest"),
    ],
)
def test_business_requests_allowed(module: str, name: str) -> None:
    assert check_request(fake_request(module, name))


# ─── auth oynasi ─────────────────────────────────────────────────────────────


def test_auth_blocked_outside_window() -> None:
    with pytest.raises(RpcBlocked, match="login oynasidan tashqarida"):
        check_request(fake_request("auth", "SendCodeRequest"))


def test_auth_allowed_inside_window() -> None:
    with auth_window():
        assert check_request(fake_request("auth", "ExportLoginTokenRequest"))


def test_auth_window_closes() -> None:
    with auth_window():
        pass
    with pytest.raises(RpcBlocked):
        check_request(fake_request("auth", "SignInRequest"))


def test_logout_blocked_even_inside_auth_window() -> None:
    """DENIED har doim ustun — login oynasi ham ochib bera olmaydi."""
    with auth_window(), pytest.raises(RpcBlocked, match="qora ro'yxatda"):
        check_request(fake_request("auth", "LogOutRequest"))


# ─── write belgisi ───────────────────────────────────────────────────────────


def test_is_write_flags() -> None:
    assert is_write("messages.SendMessageRequest")
    assert not is_write("messages.GetHistoryRequest")


# ─── haqiqiy Telethon klasslari bilan ─────────────────────────────────────────


def test_with_real_telethon_classes() -> None:
    tl = pytest.importorskip("telethon.tl.functions")
    from telethon.tl.functions import messages as m

    assert request_key(m.GetHistoryRequest.__new__(m.GetHistoryRequest)) == (
        "messages.GetHistoryRequest"
    )
    with pytest.raises(RpcBlocked):
        check_request(m.DeleteMessagesRequest.__new__(m.DeleteMessagesRequest))
    assert tl is not None
