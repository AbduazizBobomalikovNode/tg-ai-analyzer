"""`services/tg_search.py` — server qidiruvining ichki (konvertsiz) API'si.

Qo'riqlanadigan narsalar:

  * TL so'rovlari **to'g'ri** quriladi (peer, filter, sana, from_id, topic);
  * har chaqiruv `MethodBuckets` shiftidan o'tadi — bo'sh bucket'da MTProto'ga
    umuman borilmaydi va `rate_limited` bucket nomi bilan qaytadi;
  * `FloodWaitError` **retry qilinmaydi** — `error_text` bo'lib qaytadi
    (CLAUDE.md 4.2: retry-storm yo'q);
  * `channels.SearchPosts` hech qachon `allow_paid_stars` bilan yuborilmaydi
    va bepul kvota tugaganda umuman chaqirilmaydi;
  * saqlash **faqat** lokal `chats` da bor chat uchun — begona chatdan kelgan
    hit DB'ga tushmaydi va `chat_id` `None` bo'lib qoladi;
  * telefon raqami natijaga hech qachon tushmaydi.

Real Telegram/DB yo'q: klient ham, sessiya ham soxta.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import errors as tg_errors
from telethon.tl import types as tlt

from app.config import get_settings
from app.mtproto.ratelimit import BucketSpec, MethodBuckets
from app.services import tg_search as TS
from app.services import tools as T

PHONE = "+998901234567"  # ATAYLAB: natijaga tushmasligi tekshiriladi


# ─── soxta TL obyektlari ─────────────────────────────────────────────────────


def _msg(mid: int, text: str = "salom", *, user_id: int = 1, channel_id: int = 100) -> Any:
    return SimpleNamespace(
        id=mid,
        date=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        message=text,
        media=None,
        from_id=SimpleNamespace(user_id=user_id),
        peer_id=SimpleNamespace(channel_id=channel_id),
    )


def _entity(ident: int, *, title: str | None = None, username: str | None = None) -> Any:
    return SimpleNamespace(
        id=ident,
        title=title,
        first_name=None,
        last_name=None,
        username=username,
        phone=PHONE,
    )


def _messages(msgs: list[Any], **kw: Any) -> Any:
    return SimpleNamespace(
        messages=list(msgs),
        users=list(kw.get("users") or []),
        chats=list(kw.get("chats") or []),
        count=len(msgs),
    )


class _Client:
    def __init__(self, replies: dict[str, Any] | None = None, raise_: Exception | None = None):
        self.replies = dict(replies or {})
        self.raise_ = raise_
        self.requests: list[Any] = []

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


def _req(client: _Client, name: str) -> Any:
    return next(r for r in client.requests if type(r).__name__ == name)


def _buckets(rate: int, burst: int | None = None) -> MethodBuckets:
    specs = {
        name: BucketSpec(name, rate, burst if burst is not None else rate)
        for name in ("history", "search", "resolve", "discover")
    }
    return MethodBuckets(specs, jitter=0.0)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TG_SEARCH_RATE_PER_MINUTE", "10")
    get_settings.cache_clear()
    monkeypatch.setattr(T, "_buckets", _buckets(50))
    yield
    get_settings.cache_clear()


def _patch_pool(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    import app.mtproto.pool as P

    async def get(account_id: int) -> Any:
        return client

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return f"peer:{peer_id}"

    monkeypatch.setattr(P.pool, "get", get)
    monkeypatch.setattr(P.pool, "input_peer", input_peer)


# ─── server_search_chat ──────────────────────────────────────────────────────


async def test_server_search_chat_builds_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"SearchRequest": _messages([_msg(3)], chats=[_entity(100, title="IELTS")])})
    _patch_pool(monkeypatch, client)

    res = await TS.server_search_chat(
        1,
        7,
        "narx",
        from_peer="peer:9",
        topic_id=42,
        filter="url",
        since=datetime(2026, 1, 1, tzinfo=UTC),
        until=datetime(2026, 2, 1, tzinfo=UTC),
        limit=5,
        peer="peer:555",
        chat_title="IELTS",
    )

    assert res.error_text is None and res.rate_limited is None and res.calls == 1
    req = _req(client, "SearchRequest")
    assert req.peer == "peer:555" and req.q == "narx" and req.limit == 5
    assert req.top_msg_id == 42 and req.from_id == "peer:9"
    assert isinstance(req.filter, tlt.InputMessagesFilterUrl)
    assert req.min_date == datetime(2026, 1, 1, tzinfo=UTC)

    hit = res.hits[0]
    assert hit.tg_msg_id == 3 and hit.chat_id == 7 and hit.tg_peer_id == 100
    assert hit.chat_title == "IELTS" and hit.sender == "user:1" and hit.text == "salom"


async def test_server_search_chat_caps_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)
    await TS.server_search_chat(1, 7, "x", limit=9_999, peer="p")
    assert _req(client, "SearchRequest").limit == TS.SERVER_LIMIT_MAX


async def test_empty_query_never_reaches_mtproto(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)
    res = await TS.server_search_chat(1, 7, "  ", peer="p")
    assert res.error_text and "query" in res.error_text
    assert not client.requests


# ─── shift ───────────────────────────────────────────────────────────────────


async def test_bucket_guard_blocks_before_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_buckets", _buckets(1, burst=1))
    client = _Client()
    _patch_pool(monkeypatch, client)

    assert (await TS.server_search_global(1, "x")).rate_limited is None
    blocked = await TS.server_search_global(1, "x")

    assert blocked.rate_limited == TS.BUCKET_SEARCH
    assert blocked.error_text and "rate_limited:search" in blocked.error_text
    assert blocked.calls == 0
    assert len(client.requests) == 1  # ikkinchisi serverga umuman bormadi


async def test_discover_uses_its_own_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qidiruv kvotasi tugasa ham kashfiyot alohida hisoblanadi (va aksincha)."""
    buckets = MethodBuckets(
        {
            "history": BucketSpec("history", 50, 50),
            "search": BucketSpec("search", 0, 0),
            "resolve": BucketSpec("resolve", 50, 50),
            "discover": BucketSpec("discover", 50, 50),
        },
        jitter=0.0,
    )
    monkeypatch.setattr(T, "_buckets", buckets)
    client = _Client({"CheckSearchPostsFloodRequest": SimpleNamespace(remains=0, wait_till=None)})
    _patch_pool(monkeypatch, client)

    assert (await TS.server_search_global(1, "x")).rate_limited == TS.BUCKET_SEARCH
    found = await TS.server_discover(1, "x")
    assert found.rate_limited is None and "CheckSearchPostsFloodRequest" in client.seen


async def test_disabled_by_zero_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_SEARCH_RATE_PER_MINUTE", "0")
    get_settings.cache_clear()
    client = _Client()
    _patch_pool(monkeypatch, client)

    res = await TS.server_search_global(1, "x")

    assert res.rate_limited == TS.BUCKET_SEARCH and res.error_text
    assert "disabled" in res.error_text
    assert not client.requests


# ─── FloodWait — retry yo'q ──────────────────────────────────────────────────


async def test_flood_wait_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    err = tg_errors.FloodWaitError(request=None)
    err.seconds = 42
    client = _Client(raise_=err)
    _patch_pool(monkeypatch, client)

    res = await TS.server_search_global(1, "x")

    assert res.rate_limited is None  # shift emas — Telegram tomondagi holat
    assert res.error_text and "flood_wait 42s" in res.error_text
    assert res.hits == []
    assert len(client.requests) == 1  # bitta urinish, retry-storm yo'q


# ─── server_search_global ────────────────────────────────────────────────────


async def test_global_scope_flags_and_archived(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    _patch_pool(monkeypatch, client)

    await TS.server_search_global(1, "x", scope="channels", archived=True)

    req = _req(client, "SearchGlobalRequest")
    assert req.broadcasts_only is True and req.groups_only is None and req.folder_id == 1
    bad = await TS.server_search_global(1, "x", scope="planets")
    assert bad.error_text and "unknown scope" in bad.error_text


async def test_global_falls_back_to_ids_without_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client({"SearchGlobalRequest": _messages([_msg(5, user_id=777, channel_id=123)])})
    _patch_pool(monkeypatch, client)

    hit = (await TS.server_search_global(1, "salom")).hits[0]

    assert hit.sender == "user:777" and hit.chat_title == "chat:123"
    assert hit.chat_id is None  # lokal chat bilan hali bog'lanmagan


# ─── kalendar ────────────────────────────────────────────────────────────────


async def test_server_calendar_returns_sorted_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = [
        SimpleNamespace(date=datetime(2026, 3, 1, tzinfo=UTC), count=3),
        SimpleNamespace(date=datetime(2026, 3, 9, tzinfo=UTC), count=12),
    ]
    client = _Client(
        {"GetSearchResultsCalendarRequest": SimpleNamespace(count=15, periods=periods)}
    )
    _patch_pool(monkeypatch, client)

    rows, total, err = await TS.calendar_rows(1, 7, filter="photo", peer="peer:1")

    assert err is None and total == 15
    assert [c for _, c in rows] == [12, 3]  # yangisidan eskisiga


async def test_server_calendar_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kalendar — yordamchi signal: yo'qligi quvurni to'xtatmaydi."""
    monkeypatch.setattr(T, "_buckets", _buckets(0, burst=0))
    client = _Client()
    _patch_pool(monkeypatch, client)

    assert await TS.server_calendar(1, 7, "photo") == []
    assert not client.requests


# ─── discover — Stars hech qachon ────────────────────────────────────────────


def _flood(*, remains: int, free: bool = False, wait_till: int | None = None) -> Any:
    return SimpleNamespace(
        total_daily=5, remains=remains, stars_amount=100, query_is_free=free, wait_till=wait_till
    )


async def test_discover_never_offers_stars(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "CheckSearchPostsFloodRequest": _flood(remains=3),
            "SearchPostsRequest": _messages(
                [_msg(11, "post")], chats=[_entity(100, title="IELTS", username="ielts")]
            ),
        }
    )
    _patch_pool(monkeypatch, client)

    res = await TS.server_discover(1, "ielts")

    assert res.calls == 2 and res.mode == "free text"
    posts = _req(client, "SearchPostsRequest")
    assert getattr(posts, "allow_paid_stars", None) is None  # PULLIK yo'l yopiq
    assert client.seen.index("CheckSearchPostsFloodRequest") < client.seen.index(
        "SearchPostsRequest"
    )
    assert res.hits[0].link == "https://t.me/ielts/11"


async def test_discover_stops_when_free_quota_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client({"CheckSearchPostsFloodRequest": _flood(remains=0, wait_till=1_800_000_000)})
    _patch_pool(monkeypatch, client)

    res = await TS.server_discover(1, "ielts")

    assert res.hits == [] and res.error_text is None
    assert res.note and "not spending Stars" in res.note
    assert "SearchPostsRequest" not in client.seen


async def test_discover_hashtag_mode_without_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(
        {
            "CheckSearchPostsFloodRequest": _flood(remains=0),
            "SearchPostsRequest": _messages([_msg(12, "hashtag post")]),
        }
    )
    _patch_pool(monkeypatch, client)

    res = await TS.server_discover(1, "#ielts")

    assert res.mode == "hashtag"
    posts = _req(client, "SearchPostsRequest")
    assert posts.hashtag == "ielts" and posts.query is None
    assert getattr(posts, "allow_paid_stars", None) is None


# ─── saqlash — faqat ma'lum chatlar ──────────────────────────────────────────


class _Rows:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """Ikkita `execute`: (1) chatlar, (2) yozilgan xabar id'lari."""

    def __init__(self, chats: list[Any], msgs: list[Any]) -> None:
        self.chats, self.msgs = chats, msgs
        self.n = 0

    async def execute(self, stmt: Any) -> Any:
        self.n += 1
        return _Rows(self.chats if self.n == 1 else self.msgs)


async def test_persist_only_touches_known_chats(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[list[dict[str, Any]]] = []

    async def upsert(rows: list[dict[str, Any]]) -> int:
        written.append(rows)
        return len(rows)

    from app.services import ingestion as ING

    monkeypatch.setattr(ING, "upsert_messages", upsert)

    known = TS.hit_of(_msg(3, "ma'lum chat", channel_id=100), {})
    external = TS.hit_of(_msg(9, "begona kanal", channel_id=999), {})
    session = _FakeSession(chats=[(5, 100, "Ish chat", "https://t.me/c/5/")], msgs=[(77, 5, 3)])

    n = await TS.persist_server_hits(session, 1, [known, external])  # type: ignore[arg-type]

    assert n == 1
    assert [r["tg_msg_id"] for r in written[0]] == [3]  # begona xabar DB'ga tushmadi
    assert known.chat_id == 5 and known.message_id == 77
    assert known.chat_title == "Ish chat" and known.link == "https://t.me/c/5/3"
    # Begona chat: saqlanmaydi, lokal id olmaydi — faqat `t.me` havolasi qoladi
    assert external.chat_id is None and external.message_id is None


async def test_persist_is_a_no_op_without_known_chats(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    from app.services import ingestion as ING

    async def upsert(rows: list[dict[str, Any]]) -> int:
        calls.append(rows)
        return len(rows)

    monkeypatch.setattr(ING, "upsert_messages", upsert)
    hit = TS.hit_of(_msg(9, "begona", channel_id=999), {})
    session = _FakeSession(chats=[], msgs=[])

    assert await TS.persist_server_hits(session, 1, [hit]) == 0  # type: ignore[arg-type]
    assert not calls and hit.chat_id is None


# ─── maxfiylik ───────────────────────────────────────────────────────────────


async def test_no_phone_anywhere_in_a_server_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity'da telefon bo'lsa ham `ServerHit` ga tushmaydi (LLM promptiga boradi)."""
    client = _Client(
        {
            "SearchGlobalRequest": _messages(
                [_msg(5, f"raqamim {PHONE} emas", user_id=7, channel_id=100)],
                users=[_entity(7, title=None, username="ali")],
                chats=[_entity(100, title="IELTS", username="ielts")],
            )
        }
    )
    _patch_pool(monkeypatch, client)

    hits = (await TS.server_search_global(1, "x")).hits

    assert not any(f.name == "phone" for f in hits[0].__dataclass_fields__.values())
    rendered = f"{hits[0].sender}|{hits[0].chat_title}|{hits[0].link}"
    assert PHONE not in rendered  # faqat `text` da (bu xabarning o'z mazmuni)
