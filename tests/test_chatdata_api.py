"""Chat ma'lumot endpoint'lari (ui/R1a) — `web/routers/chatdata.py` + avatar.

Qo'riqlaydi:
  * har yo'lda autentifikatsiya va **egalik** (boshqa user'ning akkaunti → 404);
  * chat qidiruvi fold fazosida ishlaydi va reyting tartibi to'g'ri
    (prefiks > so'z boshi > ichida);
  * `stats` javobining shakli (`by_hour` 24, `by_weekday` 7, `series`);
  * digest javobi 62 kun bilan chegaralangan;
  * xabar qidiruvi **LLM ham, rerank ham** chaqirmaydi va `expand=False`.
"""

# Fold korpusi uchun kirill/apostrofli matn ataylab 'ma'lumot' sifatida turadi
# (`test_textnorm.py`, `test_alerts.py` bilan bir xil sabab).
# ruff: noqa: RUF001, RUF002
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.web import security as sec
from app.web.routers import chatdata as CD

USER_ID = 7
ACC_ID = 5
CHAT_ID = 1


def _chat(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "id": CHAT_ID,
        "account_id": ACC_ID,
        "tg_peer_id": 777,
        "title": "Toshkent yangiliklari",
        "type": "channel",
        "username": "tashnews",
        "is_admin": False,
        "is_writable": False,
        "write_mode": "read_only",
        "participants_count": 10,
        "sync_state": "done",
        "synced_total": 100,
        "total_estimate": 100,
        "sync_error": None,
        "last_sync_at": None,
        "last_message_at": None,
        "about": "Kanal haqida",
        "avatar_photo_id": None,
        "avatar_mime": None,
        "is_verified": True,
        "is_scam": False,
        "is_fake": False,
        "unread_count": 3,
        "last_message_text": "salom",
        "link_prefix": "https://t.me/tashnews/",
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    def __init__(self, account_user_id: int | None, rows: list[Any]) -> None:
        self.account_user_id = account_user_id
        self.rows = rows

    async def get(self, model: Any, ident: int) -> Any:
        name = getattr(model, "__name__", "")
        if name == "Account":
            if self.account_user_id is None:
                return None
            return SimpleNamespace(id=ident, user_id=self.account_user_id)
        if name == "Chat":
            return self.rows[0] if self.rows else None
        return None

    async def execute(self, stmt: Any) -> _Result:
        return _Result(self.rows)


class _Scope:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *exc: Any) -> None: ...


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    from app.web import main as web_main

    state: dict[str, Any] = {"owner": USER_ID, "rows": [_chat()], "chat": _chat()}

    def scope() -> _Scope:
        return _Scope(_FakeSession(state["owner"], state["rows"]))

    async def owned(session: Any, user_id: int, chat_id: int) -> Any:
        chat = state["chat"]
        if chat is None or state["owner"] != user_id:
            return None
        return chat

    monkeypatch.setattr(CD, "session_scope", scope)
    monkeypatch.setattr(CD.AV, "get_owned_chat", owned)
    from app.web.routers import tg as tg_router

    monkeypatch.setattr(tg_router, "session_scope", scope)
    with TestClient(web_main.app) as c:
        c.state = state  # type: ignore[attr-defined]
        yield c


PATHS = [
    f"/api/accounts/{ACC_ID}/chats/search?q=test",
    f"/api/accounts/{ACC_ID}/messages/search?q=test",
    f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}",
    f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/stats",
    f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/digests",
    f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/avatar",
]


# ─── auth / egalik ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", PATHS)
def test_requires_login(api: TestClient, path: str) -> None:
    api.cookies.clear()
    assert api.get(path).status_code == 401


@pytest.mark.parametrize("path", PATHS)
def test_foreign_account_is_not_found(api: TestClient, path: str) -> None:
    api.state["owner"] = 99  # akkaunt boshqa foydalanuvchiniki
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    assert api.get(path).status_code == 404


def test_foreign_chat_is_not_found(api: TestClient) -> None:
    api.state["chat"] = None
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}")
    assert r.status_code == 404


# ─── chat qidiruvi ───────────────────────────────────────────────────────────


def test_chat_out_shape(api: TestClient) -> None:
    out = CD.chat_out(_chat())
    for key in (
        "id",
        "peer_id",
        "title",
        "type",
        "username",
        "about",
        "avatar_url",
        "is_verified",
        "is_scam",
        "is_fake",
        "unread_count",
        "last_message_text",
        "link_prefix",
        "progress",
    ):
        assert key in out, key
    assert out["avatar_url"] is None  # fayl yo'q → havola ham yo'q
    assert out["unread_count"] == 3


async def test_search_ranking_is_fold_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Oʻzbekiston` / `O'zbekiston` / `Ўзбекистон` — bitta so'rov."""
    from app.services.textnorm import uz_fold

    rows = [
        _chat(id=1, title="Kecha Oʻzbekiston haqida", username=None),  # so'z boshi → 2
        _chat(id=2, title="Ўзбекистон yangiliklari", username=None),  # prefiks → 3
        _chat(id=3, title="Sport", username=None, about="ozbekiston futboli"),  # about → 1
        _chat(id=4, title="Boshqa", username=None, about=None),  # moslik yo'q
    ]
    session = _FakeSession(USER_ID, rows)
    folded = uz_fold("O'zbekiston")
    out = await CD._search_python(session, [], folded, 10)
    assert [c.id for c in out] == [2, 1, 3]


def test_search_endpoint_falls_back_to_python(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("function uz_fold(text) does not exist")

    monkeypatch.setattr(CD, "_search_sql", boom)
    api.state["rows"] = [_chat(id=2, title="Ўзбекистон yangiliklari")]
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/search?q=o'zbekiston")
    assert r.status_code == 200
    body = r.json()
    assert body["folded"] == "ozbekiston"
    assert [i["id"] for i in body["items"]] == [2]


def test_search_rejects_unknown_type(api: TestClient) -> None:
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/search?q=a&type=nope")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "chat.err.bad_type"


def test_search_sql_uses_uz_fold() -> None:
    """SQL yo'li fold qilingan ustunlar ustidan boradi (paritet uchun muhim)."""
    from app.db.models import Chat

    assert "uz_fold" in str(CD._fold_col(Chat.title))


# ─── xabar qidiruvi ──────────────────────────────────────────────────────────


def test_message_search_never_uses_llm_or_rerank(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import rerank as RR
    from app.services import search as S

    calls: dict[str, Any] = {}

    async def fake_single(session: Any, chat_id: int, q: str, **kw: Any) -> list[Any]:
        calls["single"] = kw
        return [SimpleNamespace(message_id=11)]

    async def fake_multi(session: Any, account_id: int, q: str, **kw: Any) -> list[Any]:
        calls["multi"] = kw
        return [SimpleNamespace(message_id=22)]

    async def no_rerank(*a: Any, **kw: Any) -> Any:
        raise AssertionError("rerank chaqirildi — bu yo'lda model ishlatilmaydi")

    async def fake_hydrate(db: Any, hits: list[Any], chat: Any) -> list[dict[str, Any]]:
        return [
            {
                "chat_id": CHAT_ID,
                "chat_title": "T",
                "tg_msg_id": int(h.message_id),
                "date": None,
                "sender": "—",
                "text": "x",
                "link": None,
            }
            for h in hits
        ]

    monkeypatch.setattr(S, "hybrid_search", fake_single)
    monkeypatch.setattr(S, "hybrid_search_multi", fake_multi)
    monkeypatch.setattr(RR, "rerank", no_rerank)
    monkeypatch.setattr(CD, "_hydrate", fake_hydrate)
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))

    r = api.get(f"/api/accounts/{ACC_ID}/messages/search?q=salom&chat_id={CHAT_ID}")
    assert r.status_code == 200
    assert calls["single"]["expand"] is False
    assert r.json()["items"][0]["tg_msg_id"] == 11

    r = api.get(f"/api/accounts/{ACC_ID}/messages/search?q=salom")
    assert r.status_code == 200
    assert calls["multi"]["expand"] is False
    assert r.json()["items"][0]["tg_msg_id"] == 22


def test_message_search_empty_query_is_cheap(api: TestClient) -> None:
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/messages/search?q=%20")
    assert r.status_code == 200 and r.json()["items"] == []


def test_message_search_is_rate_limited(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import search as S

    async def fake_multi(*a: Any, **kw: Any) -> list[Any]:
        return []

    monkeypatch.setattr(S, "hybrid_search_multi", fake_multi)
    monkeypatch.setattr(CD, "_msg_search_limiter", sec.RateLimiter(2, 60.0))
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    codes = [api.get(f"/api/accounts/{ACC_ID}/messages/search?q=a").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ─── statistika va digest'lar ────────────────────────────────────────────────


def test_stats_shape(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import analytics as AN

    async def fake_stats(session: Any, chat: Any, **kw: Any) -> Any:
        return SimpleNamespace(
            by_hour={"09": 4},
            by_weekday={"Mon": 2},
            to_dict=lambda: {"chat": chat.title, "posts": 6, "period_days": kw.get("days")},
        )

    async def fake_series(db: Any, chat_id: int, *, days: int) -> list[dict[str, Any]]:
        return [{"day": "2026-09-01", "posts": 1, "views": 10}]

    monkeypatch.setattr(AN, "chat_stats", fake_stats)
    monkeypatch.setattr(CD, "_series", fake_series)
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    body = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/stats?days=7").json()
    assert body["period_days"] == 7
    assert len(body["by_hour"]) == 24 and body["by_hour"][9] == 4
    assert len(body["by_weekday"]) == 7 and body["by_weekday"][0] == 2
    assert body["series"][0]["views"] == 10


def test_digests_are_capped(api: TestClient) -> None:
    day = datetime(2026, 9, 1, tzinfo=UTC)
    api.state["rows"] = [SimpleNamespace(day=day, digest=f"d{i}", msg_count=i) for i in range(200)]
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/digests?days=14")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == CD.DIGEST_MAX == 62
    assert items[0]["day"] == "2026-09-01"

    # 62 kundan katta oyna umuman qabul qilinmaydi
    assert api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/digests?days=90").status_code == 422


class _SparkSession:
    """`spark_series` uchun minimal soxta sessiya — bitta so'rov, tuple qatorlar."""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.stmt: Any = None

    async def execute(self, stmt: Any) -> Any:
        self.stmt = stmt
        rows = self.rows

        class _R:
            def all(self) -> list[Any]:
                return rows

        return _R()


async def test_spark_series_is_one_grouped_query() -> None:
    """M3: bitta so'rov, `deleted_at IS NULL` + guruhlangan chat_id/kun."""
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC)
    rows = [(1, today - timedelta(days=1), 3), (2, today, 5)]
    session = _SparkSession(rows)

    out = await CD.spark_series(session, [1, 2, 3], days=14)

    assert set(out) == {1, 2, 3}
    assert all(len(v) == 14 for v in out.values())
    assert sum(out[1]) == 3 and sum(out[2]) == 5 and sum(out[3]) == 0

    sql = str(session.stmt).lower()
    assert "group by" in sql
    assert "deleted_at is null" in sql
    assert "chat_id" in sql


async def test_spark_series_empty_chat_ids_skips_query() -> None:
    session = _SparkSession([])
    out = await CD.spark_series(session, [], days=14)
    assert out == {}
    assert session.stmt is None


def test_chat_detail_has_sync_block(api: TestClient) -> None:
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    body = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}").json()
    assert body["id"] == CHAT_ID
    assert body["sync"] == {
        "state": "done",
        "progress": 100,
        "synced_total": 100,
        "total_estimate": 100,
        "last_sync_at": None,
        "error": None,
    }


# ─── avatar ──────────────────────────────────────────────────────────────────


def test_avatar_404_without_file(api: TestClient) -> None:
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/avatar")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "chat.err.no_avatar"


def test_avatar_served_with_etag_and_private_cache(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from app.web.routers import tg as tg_router

    path = tmp_path / "1.jpg"
    path.write_bytes(b"\xff\xd8\xffdata")
    monkeypatch.setattr(tg_router.AV, "path_for", lambda chat_id, mime=None: path)
    api.state["rows"] = [_chat(avatar_photo_id=999, avatar_mime="image/jpeg")]
    api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = api.get(f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/avatar")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, max-age=86400"
    assert r.headers["etag"] == '"999"'
    assert r.content == b"\xff\xd8\xffdata"


# ─── bitta xabar (sitata hover kartasi, ui/R2b) ──────────────────────────────


def _message(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "id": 99,
        "chat_id": CHAT_ID,
        "tg_msg_id": 4321,
        "sender_id": 12345,
        "published_at": datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        "text": "Salom\nikkinchi qator",
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _OneResult:
    """`scalar_one_or_none()` beradigan minimal natija (bitta xabar so'rovi)."""

    def __init__(self, row: Any) -> None:
        self._row = row

    def scalar_one_or_none(self) -> Any:
        return self._row


@pytest.fixture
def msg_api(monkeypatch: pytest.MonkeyPatch):
    from app.web import main as web_main

    state: dict[str, Any] = {"owner": USER_ID, "chat": _chat(), "msg": _message()}

    class _Session:
        async def get(self, model: Any, ident: int) -> Any:
            if getattr(model, "__name__", "") == "Account":
                return SimpleNamespace(id=ident, user_id=state["owner"])
            return None

        async def execute(self, stmt: Any) -> Any:
            return _OneResult(state["msg"])

    class _MsgScope:
        async def __aenter__(self) -> Any:
            return _Session()

        async def __aexit__(self, *exc: Any) -> None: ...

    async def owned(session: Any, user_id: int, chat_id: int) -> Any:
        return state["chat"] if state["owner"] == user_id else None

    monkeypatch.setattr(CD, "session_scope", lambda: _MsgScope())
    monkeypatch.setattr(CD.AV, "get_owned_chat", owned)
    with TestClient(web_main.app) as c:
        c.state = state  # type: ignore[attr-defined]
        yield c


MSG_PATH = f"/api/accounts/{ACC_ID}/chats/{CHAT_ID}/messages/4321"


def test_message_detail_requires_login(msg_api: TestClient) -> None:
    msg_api.cookies.clear()
    assert msg_api.get(MSG_PATH).status_code == 401


def test_message_detail_checks_ownership(msg_api: TestClient) -> None:
    msg_api.state["owner"] = 99
    msg_api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    assert msg_api.get(MSG_PATH).status_code == 404


def test_message_detail_shape_and_link(msg_api: TestClient) -> None:
    """Hover kartasi uchun: matn bir qatorga yig'iladi va `link` beriladi."""
    msg_api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = msg_api.get(MSG_PATH)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["chat_id"] == CHAT_ID and data["tg_msg_id"] == 4321
    assert data["chat_title"] == "Toshkent yangiliklari"
    assert "\n" not in data["text"], "matn bir qatorga yig'ilishi kerak"
    assert data["link"] == "https://t.me/tashnews/4321"
    assert data["date"].startswith("2026-09-01")
    # telefon raqami/`sender_id` xom holda chiqmaydi — faqat `user:<id>` yorlig'i
    assert data["sender"] == "user:12345"


def test_message_detail_caps_the_text(msg_api: TestClient) -> None:
    """Javob hajmi chegaralangan — hover kartasi butun postni tortmaydi."""
    msg_api.state["msg"] = _message(text="x" * 5000)
    msg_api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = msg_api.get(MSG_PATH)
    assert r.status_code == 200
    assert len(r.json()["text"]) == CD.CITATION_TEXT_CHARS


def test_message_detail_missing_message_is_404(msg_api: TestClient) -> None:
    msg_api.state["msg"] = None
    msg_api.cookies.set(sec.SESSION_COOKIE, sec.issue_session(USER_ID))
    r = msg_api.get(MSG_PATH)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "chat.err.not_found"
