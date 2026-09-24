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
from app.mtproto.ratelimit import BUCKETS, BucketSpec, MethodBuckets
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
        for name in BUCKETS  # yangi bucket qo'shilsa test helperi ortda qolmasin
    }
    return MethodBuckets(specs, jitter=0.0)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TG_SEARCH_RATE_PER_MINUTE", "10")
    get_settings.cache_clear()
    monkeypatch.setattr(T, "_buckets", _buckets(50))

    # `TS._redis` — standart holda xato ko'taradi (fail-open): bu fayldagi
    # testlar Premium xatosini haqiqiy Redis'ga (7 kunlik TTL bilan, butun
    # test suit bilan baham ko'rilgan) yozib qo'ymasin. `tests/test_tg_search.py`
    # dagi izohga qarang — sabab bir xil.
    async def _no_redis() -> Any:
        raise RuntimeError("redis disabled in this test by default")

    monkeypatch.setattr(TS, "_redis", _no_redis)
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
            "contacts": BucketSpec("contacts", 50, 50),
        },
        jitter=0.0,
    )
    monkeypatch.setattr(T, "_buckets", buckets)
    client = _Client({"CheckSearchPostsFloodRequest": SimpleNamespace(remains=0, wait_till=None)})
    _patch_pool(monkeypatch, client)

    assert (await TS.server_search_global(1, "x")).rate_limited == TS.BUCKET_SEARCH
    found = await TS.server_discover(1, "x")
    assert found.rate_limited is None and "CheckSearchPostsFloodRequest" in client.seen


# ─── Premium'siz kashfiyot: probe zaxira yo'lni ochlikda qoldirmasin ─────────


class PremiumAccountRequiredError(Exception):
    """Nomi muhim: `tg_search.is_premium_error` xato **klassi nomiga** qaraydi."""


class _TwoSearchClient:
    """`contacts.Search` va `messages.Search` klass nomi bir xil — modulga qaraymiz."""

    def __init__(self, found: Any, posts: Any) -> None:
        self.found, self.posts = found, posts
        self.requests: list[Any] = []

    @property
    def seen(self) -> list[str]:
        return [
            f"{type(r).__module__.rsplit('.', 1)[-1]}.{type(r).__name__}" for r in self.requests
        ]

    async def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        if type(request).__name__ == "CheckSearchPostsFloodRequest":
            raise PremiumAccountRequiredError("PREMIUM_ACCOUNT_REQUIRED")
        if "contacts" in type(request).__module__:
            return self.found
        return self.posts


def _public_channel() -> Any:
    return SimpleNamespace(
        id=555,
        access_hash=7,
        title="AI kanal",
        username="ai_uz",
        broadcast=True,
        megagroup=False,
        participants_count=1200,
        phone=PHONE,
    )


async def test_premium_fallback_reaches_contacts_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jonli regressiya: probe `discover` burst'ini yeb, zaxira yo'lni o'ldirardi.

    **Soxta bucket yo'q** — `.env` dagi haqiqiy tezliklar bilan, bitta soniya
    ichida: `CheckSearchPostsFlood` (probe) → Premium xatosi →
    `contacts.Search` → topilgan kanalda `messages.Search`.
    """
    TS.PREMIUM_MISSING.discard(1)
    monkeypatch.setattr(T, "_buckets", MethodBuckets.from_settings(get_settings(), jitter=0.0))
    client = _TwoSearchClient(
        found=SimpleNamespace(chats=[_public_channel()], users=[]),
        posts=_messages([_msg(3, channel_id=555)], chats=[_entity(555, title="AI kanal")]),
    )
    _patch_pool(monkeypatch, client)

    probe = await TS.server_discover(1, "sun'iy intellekt")
    assert probe.note_code == "premium_required" and not probe.hits

    res = await TS.server_discover_by_name(1, "sun'iy intellekt", known_peer_ids=set())

    assert "contacts.SearchRequest" in client.seen  # ← aynan shu bajarilmasdi
    assert res.rate_limited is None and res.error_text is None
    assert res.searched == ["ai_uz"] and res.hits
    assert res.hits[0].chat_id is None  # tashqi: lokal `chats` da yo'q
    assert PHONE not in str(res.channels)
    TS.PREMIUM_MISSING.discard(1)


# ─── discover_name_queries: mavzu → "nomga o'xshagan" qisqa variantlar ───────


@pytest.mark.parametrize(
    ("subqueries", "entities", "expected"),
    [
        # Jonli misol: to'liq ibora `contacts.Search`da 0 ta natija berardi
        # ("IT ta'lim O'zbekistonda"), qisqa variant esa beradi.
        (["IT ta'lim O'zbekistonda"], [], ["IT ta'lim"]),
        # Reja shu ikkitasini alohida entity qilib bergan bo'lsa — ikkalasi
        # ham (joy kelishigi qo'shimchasi "-da" olib tashlanadi).
        ([], ["IT ta'lim", "O'zbekistonda"], ["IT ta'lim", "O'zbekiston"]),
        # Faqat joy nomi — bitta so'zli entity ham 1 tokenda qoladi.
        ([], ["Toshkentda"], ["Toshkent"]),
        # Entity to'liq stop-so'zlardan iborat — tashlab ketiladi, subquery
        # boshiga tushiladi.
        (["muhim mavzu haqida"], ["haqida bo'yicha"], ["muhim mavzu", "muhim"]),
        # Ingliz: 3 so'zgacha entity o'zgarishsiz qoladi.
        ([], ["Cloud Computing Conference"], ["Cloud Computing Conference"]),
        # Rus: birinchi 2 so'z + birinchi so'z (≥4 belgi) — ikkalasi ham.
        (["новости про Пятигорск фестиваль"], [], ["новости про", "новости"]),
        # Bir xil variant ikki marta tushmasin (fold bo'yicha).
        (["IT ta'lim narxi qancha"], ["IT ta'lim"], ["IT ta'lim"]),
        # Juda qisqa nomzodlar (< 3 belgi) — hammasi tashlab ketiladi.
        (["a", "b"], [], []),
        (["", "   "], [], []),
    ],
)
def test_discover_name_queries_table(
    subqueries: list[str], entities: list[str], expected: list[str]
) -> None:
    assert TS.discover_name_queries(subqueries, entities) == expected


def test_discover_name_queries_respects_max_queries() -> None:
    out = TS.discover_name_queries(
        [], ["IT ta'lim", "O'zbekistonda", "Toshkent shahri"], max_queries=1
    )
    assert out == ["IT ta'lim"]


@pytest.mark.parametrize(
    ("subqueries", "entities", "max_queries", "expected"),
    [
        # Jonli `global-04` rejasi: reja joy nomini **birinchi** bergan edi va
        # `contacts.Search` shu bilan gigant/umumiy kanal topardi. Endi mavzu
        # birinchi.
        (
            ["IT ta'lim O'zbekistonda"],
            ["O'zbekiston", "IT ta'lim"],
            2,
            ["IT ta'lim", "O'zbekiston"],
        ),
        # Joy nomi qopqoqqa sig'masa — **butunlay** tashlanadi.
        (["IT ta'lim O'zbekistonda"], ["O'zbekiston", "IT ta'lim"], 1, ["IT ta'lim"]),
        # Rus/ingliz variantlari ham (fold bitta fazoga tushiradi).
        ([], ["Новости", "ИИ стартапы"], 2, ["ИИ стартапы", "Новости"]),
        ([], ["world", "quantum computing"], 2, ["quantum computing", "world"]),
        # Joy nomi + mavzu bitta iborada — bu **mavzu** nomzodi, surilmaydi.
        ([], ["IT Uzbekistan", "Toshkent"], 2, ["IT Uzbekistan", "Toshkent"]),
        # Faqat joy nomlari bo'lsa — baribir qaytadi (boshqa nomzod yo'q).
        ([], ["Toshkent", "Samarqand"], 2, ["Toshkent", "Samarqand"]),
    ],
)
def test_discover_name_queries_puts_location_terms_last(
    subqueries: list[str], entities: list[str], max_queries: int, expected: list[str]
) -> None:
    assert TS.discover_name_queries(subqueries, entities, max_queries=max_queries) == expected


def test_discover_name_queries_is_deterministic_and_pure() -> None:
    args = (["sun'iy intellekt haqida nima deyishyapti"], ["sun'iy intellekt"])
    assert TS.discover_name_queries(*args) == TS.discover_name_queries(*args)


async def test_contacts_search_never_draws_from_the_discover_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`discover` butunlay o'chiq bo'lsa ham nom qidiruvi ishlaydi (va aksincha)."""
    buckets = MethodBuckets(
        {
            "history": BucketSpec("history", 50, 50),
            "search": BucketSpec("search", 50, 50),
            "resolve": BucketSpec("resolve", 50, 50),
            "discover": BucketSpec("discover", 0, 0),
            "contacts": BucketSpec("contacts", 50, 50),
        },
        jitter=0.0,
    )
    monkeypatch.setattr(T, "_buckets", buckets)
    TS.PREMIUM_MISSING.discard(1)
    client = _TwoSearchClient(
        found=SimpleNamespace(chats=[_public_channel()], users=[]), posts=None
    )
    _patch_pool(monkeypatch, client)

    found = await TS.server_channel_search(1, "ai")
    assert found.rate_limited is None and found.channels

    assert (await TS.server_discover(1, "ai")).rate_limited == TS.BUCKET_DISCOVER

    buckets.specs["contacts"] = BucketSpec("contacts", 0, 0)
    blocked = await TS.server_channel_search(1, "ai")
    assert blocked.rate_limited == TS.BUCKET_CONTACTS


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
    session = _FakeSession(
        chats=[(5, 100, "Ish chat", "https://t.me/c/5/", "channel")], msgs=[(77, 5, 3)]
    )

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


# ─── F1: peer turi va Saved Messages ─────────────────────────────────────────


def test_peer_kind_is_carried_on_every_hit() -> None:
    """M6: raqamli id tur ichida unikal — tur hit bilan birga yuradi."""
    assert TS.hit_of(_msg(1, channel_id=100), {}).peer_kind == "channel"
    user_msg = SimpleNamespace(
        id=2,
        date=None,
        message="salom",
        media=None,
        from_id=SimpleNamespace(user_id=5),
        peer_id=SimpleNamespace(user_id=5),
    )
    assert TS.hit_of(user_msg, {}).peer_kind == "user"
    group_msg = SimpleNamespace(
        id=3,
        date=None,
        message="salom",
        media=None,
        from_id=SimpleNamespace(user_id=5),
        peer_id=SimpleNamespace(chat_id=77),
    )
    assert TS.hit_of(group_msg, {}).peer_kind == "chat"


def test_peer_kind_matches_maps_channel_to_supergroup_too() -> None:
    assert TS.peer_kind_matches("channel", "channel")
    assert TS.peer_kind_matches("channel", "supergroup")  # supergruh ham PeerChannel
    assert TS.peer_kind_matches("chat", "group")
    assert TS.peer_kind_matches("user", "private")
    assert not TS.peer_kind_matches("user", "channel")
    assert not TS.peer_kind_matches(None, "channel")


async def test_persist_refuses_a_peer_of_the_wrong_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: `PeerChannel(100)` lokal `private` chat 100 ga **yozilmaydi**."""
    from app.services import ingestion as ING

    written: list[list[dict[str, Any]]] = []

    async def upsert(rows: list[dict[str, Any]]) -> int:
        written.append(rows)
        return len(rows)

    monkeypatch.setattr(ING, "upsert_messages", upsert)

    hit = TS.hit_of(_msg(3, "kanal posti", channel_id=100), {})
    # Lokal chatda **o'sha raqam**, lekin turi `private` (PeerUser(100)).
    session = _FakeSession(chats=[(5, 100, "Ali", "https://t.me/c/5/", "private")], msgs=[])

    assert await TS.persist_server_hits(session, 1, [hit]) == 0  # type: ignore[arg-type]
    assert not written
    assert hit.chat_id is None and hit.message_id is None


async def test_saved_messages_hits_have_no_chat_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """M7: "Saved Messages" `chat_id=0` emas, `None` — 0 haqiqiy chat emas."""
    from telethon.tl import types as tlt

    client = _Client({"SearchRequest": _messages([_msg(4, "eslatma", channel_id=0)])})
    _patch_pool(monkeypatch, client)

    res = await TS.server_search_saved(1, "eslatma")

    assert res.hits and all(h.chat_id is None for h in res.hits)
    assert all(h.saved for h in res.hits)
    assert res.hits[0].chat_title == "Saved Messages"
    req = _req(client, "SearchRequest")
    assert isinstance(req.peer, tlt.InputPeerSelf)
