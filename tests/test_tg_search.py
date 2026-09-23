"""Telegram server tool'lari: qidiruv, kalendar, forum, saqlangan, kashfiyot.

Qo'riqlaydi:
  * hammasi `READ_TOOLS` da va yozish registry'siga tushmagan;
  * ishlatiladigan TL metodlar allowlist'da ruxsat etilgan va destruktiv emas;
  * bucket shiftlari va `TG_SEARCH_RATE_PER_MINUTE=0` (o'chirish) ishlaydi;
  * FloodWait retry qilinmaydi — modelga matn sifatida qaytadi;
  * `channels.SearchPosts` HECH QACHON `allow_paid_stars` bilan yuborilmaydi
    va bepul kvota tugaganda umuman chaqirilmaydi (Stars — moliyaviy amal);
  * telefon raqami hech qayerda chiqmaydi.

Real Telegram chaqiruvi yo'q — klientlar soxta (`_Client`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import errors as tg_errors
from telethon.tl import types as tlt

from app.config import get_settings
from app.mtproto import allowlist as A
from app.mtproto.ratelimit import BucketSpec, MethodBuckets
from app.services import tools as T


def _ctx(account_id: int = 1) -> T.ToolContext:
    return T.ToolContext(session=None, account_id=account_id)  # type: ignore[arg-type]


# ─── soxta TL obyektlari ─────────────────────────────────────────────────────


def _msg(
    mid: int,
    text: str = "salom",
    *,
    user_id: int | None = 1,
    channel_id: int | None = 100,
) -> Any:
    return SimpleNamespace(
        id=mid,
        date=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        message=text,
        media=None,
        from_id=SimpleNamespace(user_id=user_id) if user_id else None,
        peer_id=SimpleNamespace(channel_id=channel_id) if channel_id else None,
    )


def _entity(
    ident: int,
    *,
    title: str | None = None,
    first: str | None = None,
    username: str | None = None,
    **extra: Any,
) -> Any:
    return SimpleNamespace(
        id=ident, title=title, first_name=first, last_name=None, username=username, **extra
    )


def _messages(
    msgs: list[Any], *, users: list[Any] | None = None, chats: list[Any] | None = None
) -> Any:
    return SimpleNamespace(
        messages=list(msgs), users=list(users or []), chats=list(chats or []), count=len(msgs)
    )


class _Client:
    """`client(request)` ni yozib oladi; `get_messages`/`get_participants` ham bor."""

    def __init__(
        self,
        replies: dict[str, Any] | None = None,
        *,
        raise_: Exception | None = None,
        msgs: list[Any] | None = None,
    ) -> None:
        self.replies = dict(replies or {})
        self.raise_ = raise_
        self.requests: list[Any] = []
        self.calls: list[dict[str, Any]] = []  # get_messages/get_participants
        self.msgs = msgs or []

    @property
    def seen(self) -> list[str]:
        return [type(r).__name__ for r in self.requests]

    async def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        if self.raise_ is not None:
            raise self.raise_
        reply = self.replies.get(type(request).__name__)
        if isinstance(reply, Exception):
            raise reply
        return reply if reply is not None else _messages([])

    async def get_messages(self, peer: Any, **kw: Any) -> list[Any]:
        self.calls.append({"peer": peer, **kw})
        if self.raise_ is not None:
            raise self.raise_
        return self.msgs


def _req(client: _Client, name: str) -> Any:
    return next(r for r in client.requests if type(r).__name__ == name)


@pytest.fixture(autouse=True)
def _reset_rate(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Har test toza, saxiy bucket'lar bilan boshlansin (jitter o'chiq)."""
    monkeypatch.setenv("TG_SEARCH_RATE_PER_MINUTE", "10")
    get_settings.cache_clear()
    monkeypatch.setattr(T, "_buckets", _buckets(50))
    yield
    get_settings.cache_clear()


def _buckets(rate: int, burst: int | None = None) -> MethodBuckets:
    specs = {
        name: BucketSpec(name, rate, burst if burst is not None else rate)
        for name in ("history", "search", "resolve", "discover")
    }
    return MethodBuckets(specs, jitter=0.0)


def _patch_pool(monkeypatch: pytest.MonkeyPatch, client: _Client, chat: Any = None) -> None:
    import app.mtproto.pool as P

    async def get(account_id: int) -> Any:
        return client

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return f"peer:{peer_id}"

    monkeypatch.setattr(P.pool, "get", get)
    monkeypatch.setattr(P.pool, "input_peer", input_peer)
    if chat is not None:

        async def resolve(ctx: Any, args: Any) -> Any:
            return chat

        monkeypatch.setattr(T, "_resolve_chat", resolve)


def _chat(**kw: Any) -> Any:
    base = {"id": 1, "tg_peer_id": 555, "title": "IELTS", "type": "channel", "username": "ielts"}
    return SimpleNamespace(**{**base, **kw})


# ─── registry / invariantlar ─────────────────────────────────────────────────


SERVER_TOOLS = (
    "search_telegram",
    "search_global",
    "search_calendar",
    "list_forum_topics",
    "saved_messages_search",
    "discover_channels",
)


def test_tools_are_registered_and_read_only() -> None:
    from app.services import write_tools as W

    for name in SERVER_TOOLS:
        assert name in T.READ_TOOLS
        assert name not in W.WRITE_TOOL_NAMES


def test_used_tl_methods_are_allowed_reads() -> None:
    """Qidiruv/kashfiyot metodlari ruxsat etilgan va yozish deb belgilanmagan."""
    for method in (
        "messages.SearchRequest",
        "messages.SearchGlobalRequest",
        "messages.GetSearchResultsCalendarRequest",
        "messages.GetForumTopicsRequest",
        "channels.GetChannelRecommendationsRequest",
        "channels.CheckSearchPostsFloodRequest",
        "channels.SearchPostsRequest",
        "contacts.ResolveUsernameRequest",
    ):
        assert method in A.ALLOWED
        assert method not in A.DENIED
        assert not A.is_write(method)


def test_every_new_tool_is_described_in_the_agent_prompt() -> None:
    from app.services import prompts as P

    for name in SERVER_TOOLS:
        assert name in P.AGENT_SYSTEM_PROMPT


# ─── xatti-harakat: qidiruv ──────────────────────────────────────────────────


async def test_search_global_lists_chat_per_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "SearchGlobalRequest": _messages(
                [_msg(7, "promo post", channel_id=100), _msg(9, "narx", channel_id=200)],
                users=[_entity(1, first="Ali")],
                chats=[_entity(100, title="IELTS"), _entity(200, title="Zuhal")],
            )
        }
    )
    _patch_pool(monkeypatch, client)

    res = await T.search_global(_ctx(), {"query": "promo", "limit": 5})

    assert res.ok and res.meta["source"] == "telegram"
    assert res.meta["hits"] == 2
    assert 'in "IELTS"' in res.text and 'in "Zuhal"' in res.text
    assert "#7" in res.text and "Ali" in res.text
    req = _req(client, "SearchGlobalRequest")
    assert req.q == "promo" and req.limit == 5
    assert req.broadcasts_only is None and req.folder_id is None


async def test_search_telegram_scopes_to_one_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"SearchRequest": _messages([_msg(3)], users=[_entity(1, first="Ali")])})
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_telegram(_ctx(), {"query": "salom"})

    assert res.ok and 'chat="IELTS"' in res.text
    assert _req(client, "SearchRequest").peer == "peer:555"


async def test_limit_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)
    await T.search_global(_ctx(), {"query": "x", "limit": 9_999})
    assert _req(client, "SearchGlobalRequest").limit == T.TG_SEARCH_LIMIT_MAX


async def test_empty_query_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)
    res = await T.search_global(_ctx(), {"query": "   "})
    assert not res.ok and "query" in res.text
    assert not client.requests  # so'rov umuman yuborilmagan


async def test_flood_wait_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    err = tg_errors.FloodWaitError(request=None)
    err.seconds = 42
    client = _Client(raise_=err)
    _patch_pool(monkeypatch, client)

    res = await T.search_global(_ctx(), {"query": "x"})

    assert not res.ok and "flood_wait 42s" in res.text
    assert len(client.requests) == 1  # bitta urinish, retry yo'q


async def test_rate_limit_blocks_before_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_buckets", _buckets(2, burst=2))
    client = _Client()
    _patch_pool(monkeypatch, client)

    for _ in range(2):
        assert (await T.search_global(_ctx(), {"query": "x"})).ok
    blocked = await T.search_global(_ctx(), {"query": "x"})

    assert not blocked.ok and "rate_limited:search" in blocked.text
    assert "search_messages" in blocked.text  # modelga lokal yo'l ko'rsatiladi
    assert len(client.requests) == 2  # uchinchisi MTProto'ga umuman bormadi


async def test_buckets_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qidiruv kvotasi tugasa ham tarix o'qish ishlashi kerak."""
    monkeypatch.setattr(T, "_buckets", _buckets(1, burst=1))
    client = _Client(msgs=[])
    _patch_pool(monkeypatch, client, chat=_chat())

    assert (await T.search_global(_ctx(), {"query": "x"})).ok
    assert "rate_limited:search" in (await T.search_global(_ctx(), {"query": "x"})).text
    # `history` bucket'i alohida — bu chaqiruv serverga boradi
    assert await T.list_scheduled_messages(_ctx(), {"chat": "IELTS"}) is not None
    assert client.calls and client.calls[0]["scheduled"] is True


async def test_scheduled_messages_respect_the_history_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(T, "_buckets", _buckets(0, burst=0))
    client = _Client(msgs=[])
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.list_scheduled_messages(_ctx(), {"chat": "IELTS"})

    assert not res.ok and "rate_limited:history" in res.text
    assert not client.calls  # shift MTProto'dan oldin ishladi


async def test_disabled_by_zero_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_SEARCH_RATE_PER_MINUTE", "0")
    get_settings.cache_clear()
    client = _Client()
    _patch_pool(monkeypatch, client)

    res = await T.search_global(_ctx(), {"query": "x"})

    assert not res.ok and "disabled" in res.text
    assert not client.requests


async def test_missing_entity_cache_falls_back_to_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Javobda entity kelmasa `user:<id>` / `chat:<id>` — "—" emas, RPC ham yo'q."""
    client = _Client(
        {"SearchGlobalRequest": _messages([_msg(5, "salom", user_id=777, channel_id=123)])}
    )
    _patch_pool(monkeypatch, client)

    res = await T.search_global(_ctx(), {"query": "salom"})

    assert "user:777" in res.text and 'in "chat:123"' in res.text


# ─── yangi qidiruv parametrlari ──────────────────────────────────────────────


async def test_search_telegram_narrowing_params(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"SearchRequest": _messages([_msg(3)])})
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_telegram(
        _ctx(),
        {
            "chat": "IELTS",
            "query": "narx",
            "topic_id": 42,
            "filter": "url",
            "since": "2026-01-01",
            "until": "2026-02-01",
            "from_user": "777",
        },
    )

    assert res.ok
    req = _req(client, "SearchRequest")
    assert req.top_msg_id == 42
    assert isinstance(req.filter, tlt.InputMessagesFilterUrl)
    assert req.min_date == datetime(2026, 1, 1, tzinfo=UTC)
    assert req.max_date == datetime(2026, 2, 1, 23, 59, 59, tzinfo=UTC)
    assert req.from_id == "peer:555"  # lokal chat orqali yechildi (resolve kerak emas)
    assert "since=2026-01-01" in res.text and "filter=url" in res.text


async def test_search_global_scope_and_archived(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)

    await T.search_global(_ctx(), {"query": "x", "scope": "channels", "archived": True})

    req = _req(client, "SearchGlobalRequest")
    assert req.broadcasts_only is True and req.groups_only is None
    assert req.folder_id == 1


async def test_bad_filter_scope_and_date_are_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)

    for args, needle in (
        ({"query": "x", "filter": "stickers"}, "unknown filter"),
        ({"query": "x", "scope": "planets"}, "unknown scope"),
        ({"query": "x", "since": "yesterday"}, "ISO date"),
    ):
        res = await T.search_global(_ctx(), args)
        assert not res.ok and needle in res.text
    assert not client.requests


async def test_from_user_username_uses_the_resolve_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lokal DB'da yo'q @username → ResolveUsername, alohida bucket ostida."""
    client = _Client(
        {
            "ResolveUsernameRequest": SimpleNamespace(
                users=[tlt.User(id=9, first_name="Nurbek", access_hash=7)], chats=[]
            ),
            "SearchRequest": _messages([_msg(3)]),
        }
    )
    _patch_pool(monkeypatch, client)

    async def resolve(ctx: Any, args: Any) -> Any:
        return _chat() if args.get("chat") == "IELTS" else None

    monkeypatch.setattr(T, "_resolve_chat", resolve)

    res = await T.search_telegram(
        _ctx(), {"chat": "IELTS", "query": "narx", "from_user": "@nurbek"}
    )

    assert res.ok and "ResolveUsernameRequest" in client.seen
    assert isinstance(_req(client, "SearchRequest").from_id, tlt.InputPeerUser)

    # resolve bucket'i bo'sh bo'lsa — tushunarli matn, RPC yo'q
    monkeypatch.setattr(T, "_buckets", _buckets(0, burst=0))
    blocked = await T.search_telegram(
        _ctx(), {"chat": "IELTS", "query": "narx", "from_user": "@nurbek"}
    )
    assert not blocked.ok and "rate_limited" in blocked.text


# ─── search_calendar ─────────────────────────────────────────────────────────


def _period(day: int, count: int) -> Any:
    return SimpleNamespace(
        date=datetime(2026, 3, day, tzinfo=UTC), count=count, min_msg_id=1, max_msg_id=9
    )


async def test_search_calendar_returns_per_day_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "GetSearchResultsCalendarRequest": SimpleNamespace(
                count=15, periods=[_period(1, 3), _period(9, 12)], messages=[], chats=[], users=[]
            )
        }
    )
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_calendar(_ctx(), {"chat": "IELTS", "filter": "photo"})

    assert res.ok and res.meta["days"] == 2
    lines = [ln for ln in res.text.splitlines() if ln.startswith("2026-")]
    assert lines == ["2026-03-09  12", "2026-03-01  3"]  # yangisidan eskisiga
    assert isinstance(
        _req(client, "GetSearchResultsCalendarRequest").filter, tlt.InputMessagesFilterPhotos
    )


async def test_search_calendar_without_query_or_filter_is_explained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_calendar(_ctx(), {"chat": "IELTS"})

    assert not res.ok and "filter" in res.text
    assert not client.requests  # Telegram rad etadigan so'rov yuborilmadi


# ─── list_forum_topics ───────────────────────────────────────────────────────


def _topic(tid: int, title: str, *, unread: int = 0) -> Any:
    return SimpleNamespace(
        id=tid, title=title, top_message=tid * 10, unread_count=unread, closed=False, pinned=False
    )


async def test_list_forum_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "GetForumTopicsRequest": SimpleNamespace(
                count=2,
                topics=[_topic(1, "General"), _topic(7, "Narxlar", unread=3)],
                messages=[],
                chats=[],
                users=[],
                pts=1,
            )
        }
    )
    _patch_pool(monkeypatch, client, chat=_chat(type="supergroup"))

    res = await T.list_forum_topics(_ctx(), {"chat": "IELTS"})

    assert res.ok and res.meta["n"] == 2
    assert 'topic_id=7 "Narxlar" top_msg=#70 unread=3' in res.text
    assert "topic_id" in res.text and "search_telegram" in res.text


async def test_list_forum_topics_non_forum_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"GetForumTopicsRequest": tg_errors.ChannelForumMissingError(request=None)})
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.list_forum_topics(_ctx(), {"chat": "IELTS"})

    assert not res.ok and "not a forum" in res.text


# ─── saved_messages_search ───────────────────────────────────────────────────


async def test_saved_messages_search_targets_self(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"SearchRequest": _messages([_msg(4, "eslatma")])})
    _patch_pool(monkeypatch, client)

    res = await T.saved_messages_search(_ctx(), {"query": "eslatma"})

    assert res.ok and "Saved Messages" in res.text
    req = _req(client, "SearchRequest")
    assert isinstance(req.peer, tlt.InputPeerSelf)
    assert req.saved_peer_id is None  # oddiy Saved Messages qidiruvi


# ─── discover_channels — Stars hech qachon sarflanmaydi ──────────────────────


def _flood(*, remains: int, free: bool = False, wait_till: int | None = None) -> Any:
    return SimpleNamespace(
        total_daily=5, remains=remains, stars_amount=100, query_is_free=free, wait_till=wait_till
    )


async def test_discover_channels_never_offers_stars(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "CheckSearchPostsFloodRequest": _flood(remains=3),
            "SearchPostsRequest": _messages(
                [_msg(11, "post matni", channel_id=100)],
                chats=[_entity(100, title="IELTS", username="ielts")],
            ),
        }
    )
    _patch_pool(monkeypatch, client)

    res = await T.discover_channels(_ctx(), {"query": "ielts"})

    assert res.ok and "#11" in res.text
    assert "https://t.me/ielts/11" in res.text
    posts = _req(client, "SearchPostsRequest")
    assert posts.query == "ielts"
    assert getattr(posts, "allow_paid_stars", None) is None  # PULLIK yo'l yopiq
    assert client.seen.index("CheckSearchPostsFloodRequest") < client.seen.index(
        "SearchPostsRequest"
    )  # avval kvota, keyin qidiruv


async def test_discover_channels_stops_when_free_quota_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client({"CheckSearchPostsFloodRequest": _flood(remains=0, wait_till=1_800_000_000)})
    _patch_pool(monkeypatch, client)

    res = await T.discover_channels(_ctx(), {"query": "ielts"})

    assert res.ok
    assert "not spending Stars" in res.text and "remains=0" in res.text
    assert "SearchPostsRequest" not in client.seen  # umuman chaqirilmadi


async def test_discover_channels_hashtag_still_works_without_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(
        {
            "CheckSearchPostsFloodRequest": _flood(remains=0),
            "SearchPostsRequest": _messages([_msg(12, "hashtag post")]),
        }
    )
    _patch_pool(monkeypatch, client)

    res = await T.discover_channels(_ctx(), {"query": "#ielts"})

    assert res.ok and "hashtag" in res.text
    posts = _req(client, "SearchPostsRequest")
    assert posts.hashtag == "ielts" and posts.query is None
    assert getattr(posts, "allow_paid_stars", None) is None


async def test_discover_channels_similar_to(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "GetChannelRecommendationsRequest": SimpleNamespace(
                chats=[
                    _entity(
                        200,
                        title="IELTS Pro",
                        username="ieltspro",
                        participants_count=1234,
                        phone="+998901234567",
                    )
                ],
                users=[],
            )
        }
    )
    _patch_pool(monkeypatch, client)

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return tlt.InputPeerChannel(channel_id=peer_id, access_hash=5)

    import app.mtproto.pool as P

    monkeypatch.setattr(P.pool, "input_peer", input_peer)

    async def resolve(ctx: Any, args: Any) -> Any:
        return _chat()

    monkeypatch.setattr(T, "_resolve_chat", resolve)

    res = await T.discover_channels(_ctx(), {"similar_to": "@ielts"})

    assert res.ok and "@ieltspro — IELTS Pro members=1234" in res.text
    assert "998901234567" not in res.text  # telefon hech qachon chiqmaydi
    assert isinstance(_req(client, "GetChannelRecommendationsRequest").channel, tlt.InputChannel)
    assert "SearchPostsRequest" not in client.seen


# ─── a'zolar ro'yxati ────────────────────────────────────────────────────────


def _user(uid: int, uname: str | None, first: str, **flags: Any) -> Any:
    return SimpleNamespace(
        id=uid,
        username=uname,
        first_name=first,
        last_name=None,
        bot=flags.get("bot", False),
        deleted=flags.get("deleted", False),
        is_self=flags.get("is_self", False),
        phone="+998901234567",  # ATAYLAB: natijaga tushmasligi tekshiriladi
    )


class _MembersClient(_Client):
    def __init__(self, users: list[Any] | None = None, raise_: Exception | None = None) -> None:
        super().__init__(raise_=raise_)
        self.users = users or []

    async def get_participants(self, peer: Any, **kw: Any) -> Any:
        self.calls.append({"peer": peer, **kw})
        if self.raise_ is not None:
            raise self.raise_
        out = list(self.users)
        return _TotalList(out, len(out))  # Telethon ro'yxat + `.total` qaytaradi


class _TotalList(list):  # type: ignore[type-arg]
    def __init__(self, items: list[Any], total: int) -> None:
        super().__init__(items)
        self.total = total


def _patch_members_pool(monkeypatch: pytest.MonkeyPatch, client: Any, chat: Any = None) -> None:
    import app.mtproto.pool as P

    async def get(account_id: int) -> Any:
        return client

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return f"peer:{peer_id}"

    async def resolve(ctx: Any, args: Any) -> Any:
        return chat or SimpleNamespace(id=1, tg_peer_id=42, title="Blog", type="channel")

    monkeypatch.setattr(P.pool, "get", get)
    monkeypatch.setattr(P.pool, "input_peer", input_peer)
    monkeypatch.setattr(T, "_resolve_chat", resolve)


def test_list_members_is_registered_and_read_only() -> None:
    assert "list_members" in T.READ_TOOLS
    from app.services import write_tools as W

    assert "list_members" not in W.WRITE_TOOL_NAMES
    assert "channels.GetParticipantsRequest" in A.ALLOWED
    assert not A.is_write("channels.GetParticipantsRequest")


async def test_list_members_never_leaks_phone(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _MembersClient([_user(1, "ali", "Ali"), _user(2, None, "Vali", bot=True)])
    _patch_members_pool(monkeypatch, client)

    res = await T.list_members(_ctx(), {"chat": "Blog"})

    assert res.ok and res.meta["n"] == 2
    assert "998901234567" not in res.text  # telefon hech qachon chiqmaydi
    assert "@ali — Ali" in res.text
    assert "id:2 — Vali [bot]" in res.text
    assert "members=2 of 2" in res.text


async def test_list_members_private_chat_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = SimpleNamespace(id=9, tg_peer_id=7, title="Firuz", type="private")
    client = _MembersClient([])
    _patch_members_pool(monkeypatch, client, chat=chat)

    res = await T.list_members(_ctx(), {"chat": "Firuz"})

    assert not res.ok and "private" in res.text
    assert not client.calls  # MTProto'ga umuman borilmadi


async def test_list_members_admin_required(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _MembersClient(raise_=tg_errors.ChatAdminRequiredError(request=None))
    _patch_members_pool(monkeypatch, client)

    res = await T.list_members(_ctx(), {"chat": "Blog"})

    assert not res.ok and "admin rights required" in res.text


async def test_list_members_limit_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _MembersClient([])
    _patch_members_pool(monkeypatch, client)
    await T.list_members(_ctx(), {"chat": "Blog", "limit": 10_000})
    assert client.calls[0]["limit"] == T.MEMBERS_LIMIT_MAX


# ─── profil / kontekst tool'lari ─────────────────────────────────────────────


class _TLClient(_Client):
    """`client(request)` ni so'rov turi nomi bo'yicha javobga bog'laydi."""

    def __init__(self, replies: dict[str, Any]) -> None:
        super().__init__(replies)


def _patch_peer(monkeypatch: pytest.MonkeyPatch, client: Any, peer: Any, chat: Any) -> None:
    import app.mtproto.pool as P

    async def get(account_id: int) -> Any:
        return client

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return peer

    async def resolve(ctx: Any, args: Any) -> Any:
        return chat

    monkeypatch.setattr(P.pool, "get", get)
    monkeypatch.setattr(P.pool, "input_peer", input_peer)
    monkeypatch.setattr(T, "_resolve_chat", resolve)


def test_profile_tools_registered_and_read_only() -> None:
    from app.services import write_tools as W

    for name in ("get_chat_info", "get_user_info", "get_reactions", "get_post_comments"):
        assert name in T.READ_TOOLS
        assert name not in W.WRITE_TOOL_NAMES
    for method in (
        "channels.GetFullChannelRequest",
        "messages.GetFullChatRequest",
        "users.GetFullUserRequest",
        "messages.GetMessageReactionsListRequest",
        "messages.GetRepliesRequest",
        "messages.GetSearchCountersRequest",
    ):
        assert method in A.ALLOWED and not A.is_write(method)


async def test_get_chat_info_rejects_private(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _TLClient({})
    chat = SimpleNamespace(id=1, tg_peer_id=5, title="Firuz", type="private", username=None)
    _patch_peer(monkeypatch, client, tlt.InputPeerUser(user_id=5, access_hash=0), chat)

    res = await T.get_chat_info(_ctx(), {"chat": "Firuz"})

    assert not res.ok and "get_user_info" in res.text
    assert not client.seen  # noto'g'ri TL metod umuman yuborilmadi


async def test_get_user_info_rejects_group(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _TLClient({})
    chat = SimpleNamespace(id=1, tg_peer_id=5, title="Blog", type="channel", username=None)
    _patch_peer(monkeypatch, client, tlt.InputPeerChannel(channel_id=5, access_hash=0), chat)

    res = await T.get_user_info(_ctx(), {"chat": "Blog"})

    assert not res.ok and "get_chat_info" in res.text
    assert not client.seen


async def test_get_user_info_never_leaks_phone(monkeypatch: pytest.MonkeyPatch) -> None:
    user = _user(5, "firuz", "Firuz")
    full = SimpleNamespace(id=5, about="1986", common_chats_count=3, blocked=False)
    client = _TLClient({"GetFullUserRequest": SimpleNamespace(full_user=full, users=[user])})
    chat = SimpleNamespace(id=1, tg_peer_id=5, title="Firuz", type="private", username=None)
    _patch_peer(monkeypatch, client, tlt.InputPeerUser(user_id=5, access_hash=0), chat)

    res = await T.get_user_info(_ctx(), {"chat": "Firuz"})

    assert res.ok and "@firuz — Firuz" in res.text
    assert "common_chats=3" in res.text and "bio: 1986" in res.text
    assert "998901234567" not in res.text


async def test_get_reactions_broadcast_message_is_explained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telegram kanalda kim bosganini bermaydi — model tushunadigan xabar qaytadi."""
    err = tg_errors.BroadcastForbiddenError(request=None)
    client = _TLClient({"GetMessageReactionsListRequest": err})
    chat = SimpleNamespace(id=1, tg_peer_id=5, title="Blog", type="channel", username=None)
    _patch_peer(monkeypatch, client, tlt.InputPeerChannel(channel_id=5, access_hash=0), chat)

    res = await T.get_reactions(_ctx(), {"chat": "Blog", "message_id": 17})

    assert not res.ok and "get_chat_stats" in res.text


async def test_message_id_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _TLClient({})
    chat = SimpleNamespace(id=1, tg_peer_id=5, title="Blog", type="channel", username=None)
    _patch_peer(monkeypatch, client, tlt.InputPeerChannel(channel_id=5, access_hash=0), chat)

    for tool in (T.get_reactions, T.get_post_comments):
        res = await tool(_ctx(), {"chat": "Blog"})
        assert not res.ok and "message_id" in res.text
    assert not client.seen


# ─── havolalar va bo'sh so'rov (conv-20260918-111653) ────────────────────────
#
# Jonli yugurishda server topilmalari **havolasiz** qator bo'lib berilardi
# (`_hit_line(..., with_link=False)`), model esa havolani o'zi yasardi:
# `https://t.me/c/<lokal DB id>/<msg>` — bunday URL mavjud emas. `ServerHit`
# da haqiqiy havola allaqachon bor, shuning uchun u har doim chiqadi.


def _hit(**kw: Any) -> Any:
    from app.services import tg_search as TS

    base = {
        "tg_msg_id": 7,
        "tg_peer_id": 100,
        "date": datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        "sender": "Ali",
        "text": "promo post",
        "chat_title": "IELTS",
        "chat_id": 3830,  # lokal DB id — javobga ham, havolaga ham tushmaydi
        "link": "https://t.me/ielts/7",
    }
    return TS.ServerHit(**{**base, **kw})


def test_hit_line_carries_the_real_link_and_never_the_db_id() -> None:
    line = T._hit_line(_hit(), with_chat=True)
    assert "https://t.me/ielts/7" in line
    assert "3830" not in line, "lokal DB id modelga havola sifatida ko'rinmaydi"


def test_rendered_server_hits_are_linked_by_default() -> None:
    from app.services import tg_search as TS

    opts = {"q": "promo", "filter_name": "", "limit": 5, "min_date": None, "max_date": None}
    res = T._render_hits(TS.ServerHits(hits=[_hit()], calls=1), opts, 'chat="IELTS"')

    assert res.ok and "https://t.me/ielts/7" in res.text
    assert "3830" not in res.text


async def test_search_global_hits_come_back_with_links(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "SearchGlobalRequest": _messages(
                [_msg(7, "promo post", channel_id=100)],
                users=[_entity(1, first="Ali")],
                chats=[_entity(100, title="IELTS", username="ielts")],
            )
        }
    )
    _patch_pool(monkeypatch, client)

    res = await T.search_global(_ctx(), {"query": "promo"})

    assert "https://t.me/ielts/7" in res.text


async def test_search_telegram_accepts_an_empty_query_with_from_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Shu odam nima yozdi" — so'zsiz so'rov `from_id` bilan ma'noli."""
    client = _Client({"SearchRequest": _messages([_msg(3)], users=[_entity(1, first="Ali")])})
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_telegram(_ctx(), {"query": "", "from_user": "@ali"})

    assert res.ok, res.text
    req = _req(client, "SearchRequest")
    assert req.q == "" and req.from_id is not None
    assert 'query=""' not in res.text and "from=@ali" in res.text


async def test_search_telegram_accepts_an_empty_query_with_a_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client({"SearchRequest": _messages([_msg(3)], users=[_entity(1, first="Ali")])})
    _patch_pool(monkeypatch, client, chat=_chat())

    res = await T.search_telegram(_ctx(), {"filter": "url"})

    assert res.ok and _req(client, "SearchRequest").q == ""


async def test_empty_query_without_any_narrowing_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client, chat=_chat())
    res = await T.search_telegram(_ctx(), {"query": "  "})
    assert not res.ok and "query" in res.text
    assert not client.requests
