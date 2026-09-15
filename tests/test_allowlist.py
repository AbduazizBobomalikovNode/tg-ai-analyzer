"""Guardrail testlari — loyihaning eng muhim invarianti.

Bu testlar sinsa, "agent o'chira olmaydi" kafolati buzilgan bo'ladi.
"""

from __future__ import annotations

import pytest

from app.mtproto.allowlist import (
    ALLOWED,
    AUTH,
    DENIED,
    READ,
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


def test_messages_delete_topic_history_denied() -> None:
    """messages.DeleteTopicHistoryRequest — channels.* bilan bir xil, DENIED'ga qo'shildi."""
    assert "messages.DeleteTopicHistoryRequest" in DENIED
    with pytest.raises(RpcBlocked, match="qora ro'yxatda"):
        check_request(fake_request("messages", "DeleteTopicHistoryRequest"))


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


# ─── qamrov kengaytmasi (1-bosqich, arxitektura v2 §8) ───────────────────────


NEW_READ_METHODS = [
    ("messages", "GetSearchResultsCalendarRequest"),
    ("messages", "GetSearchResultsPositionsRequest"),
    ("messages", "GetSavedDialogsRequest"),
    ("messages", "GetSavedHistoryRequest"),
    ("messages", "GetUnreadMentionsRequest"),
    ("messages", "GetUnreadReactionsRequest"),
    ("messages", "GetForumTopicsRequest"),
    ("messages", "GetForumTopicsByIDRequest"),
    ("channels", "GetForumTopicsRequest"),
    ("channels", "GetForumTopicsByIDRequest"),
    ("channels", "GetChannelRecommendationsRequest"),
    ("channels", "CheckSearchPostsFloodRequest"),
    ("channels", "SearchPostsRequest"),
    ("stats", "GetMessagePublicForwardsRequest"),
]


@pytest.mark.parametrize(("module", "name"), NEW_READ_METHODS)
def test_new_read_methods_are_allowed_and_not_writes(module: str, name: str) -> None:
    key = f"{module}.{name}"
    assert check_request(fake_request(module, name)) == key
    assert not is_write(key), f"{key} o'qish bo'lishi kerak"


def test_search_posts_sibling_payment_method_stays_denied() -> None:
    """Stars sarflash — moliyaviy amal: `channels.SearchPosts` qo'shildi, pul yo'li YO'Q."""
    assert "payments.SendStarsFormRequest" in DENIED
    assert "payments.SendPaymentFormRequest" in DENIED
    with pytest.raises(RpcBlocked, match="qora ro'yxatda"):
        check_request(fake_request("payments", "SendStarsFormRequest"))


def test_read_mentions_is_still_a_write() -> None:
    """`GetUnreadMentions` (o'qish) bilan `ReadMentions` (belgilash) chalkashmasin."""
    assert is_write("messages.ReadMentionsRequest")
    assert not is_write("messages.GetUnreadMentionsRequest")
    assert "messages.ReadMentionsRequest" not in READ


def test_new_methods_exist_in_the_installed_telethon() -> None:
    """Allowlist'dagi nom haqiqiy TL klassiga mos kelishi kerak (telethon>=1.44)."""
    pytest.importorskip("telethon.tl.functions")
    from telethon.tl import functions as tlf

    required = [
        ("messages", "GetSearchResultsCalendarRequest"),
        ("messages", "GetSearchResultsPositionsRequest"),
        ("messages", "GetSavedDialogsRequest"),
        ("messages", "GetSavedHistoryRequest"),
        ("messages", "GetUnreadMentionsRequest"),
        ("messages", "GetUnreadReactionsRequest"),
        ("channels", "GetChannelRecommendationsRequest"),
        ("channels", "CheckSearchPostsFloodRequest"),
        ("channels", "SearchPostsRequest"),
        ("stats", "GetMessagePublicForwardsRequest"),
    ]
    for module, name in required:
        cls = getattr(getattr(tlf, module), name, None)
        assert cls is not None, f"{module}.{name} — telethon'da yo'q (pin: telethon>=1.44)"
        assert request_key(cls.__new__(cls)) in ALLOWED

    # Forum topiclari 1.44+ da `messages.*` ga ko'chdi — ikkala nom ham ruxsat etilgan
    forum = getattr(tlf.messages, "GetForumTopicsRequest", None) or getattr(
        tlf.channels, "GetForumTopicsRequest", None
    )
    assert forum is not None
    assert request_key(forum.__new__(forum)) in ALLOWED


def test_search_posts_has_no_paid_parameter_forced() -> None:
    """`allow_paid_stars` ixtiyoriy bo'lib qolsin — tool uni hech qachon bermaydi."""
    pytest.importorskip("telethon.tl.functions")
    import inspect

    from telethon.tl.functions.channels import SearchPostsRequest

    params = inspect.signature(SearchPostsRequest.__init__).parameters
    assert params["allow_paid_stars"].default is None


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
