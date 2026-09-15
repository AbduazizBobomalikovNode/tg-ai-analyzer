"""ui/R1b: oqimli chat, suhbat boshqaruvi, prefs, sync holati, Mini App auth.

Qo'riqlaydi:
  * SSE ramkasi va hodisalar **tartibi** (`status` → `token` → `done`),
    saqlash oddiy yo'l bilan bir xil, byudjet/shift xatosi LLM'gacha `error`;
  * suhbat: sarlavha (zararsizlantirilgan), kursorli sahifalash, regenerate;
  * prefs: faqat tanilgan kalitlar yoziladi, noma'lum/buzuq jimgina tashlanadi;
  * `DELETE /api/accounts/{id}` — session envelope o'chadi, `auth.LogOut` **yo'q**;
  * static fingerprint va kesh sarlavhalari, HTML ETag;
  * Mini App `initData` HMAC'i (sintetik bot token, qo'lda hisoblangan imzo).

Tarmoq ham, DB ham yo'q — servislar soxtalanadi (`test_web.py` uslubi).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from app.services import chat_service as cs
from app.services import prefs as P
from app.web import main as web_main
from app.web import security as sec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

H = {"X-Requested-With": "fetch"}


class _Scope:
    """`session_scope()` o'rnini bosuvchi — ichida soxta sessiya."""

    def __init__(self, db: Any = None) -> None:
        self._db = db if db is not None else SimpleNamespace()

    async def __aenter__(self) -> Any:
        return self._db

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(web_main.app) as c:
        c.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        yield c


# ─── SSE parser (frontend bilan bir xil shartnoma) ───────────────────────────


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        kind = ""
        data: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                kind = line[7:].strip()
            elif line.startswith("data: "):
                data.append(line[6:])
        if kind:
            out.append((kind, json.loads("\n".join(data)) if data else {}))
    return out


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def test_chunk_answer_splits_and_rejoins() -> None:
    text = "x" * 1000 + "ЎЗ"
    parts = cs.chunk_answer(text, size=400)
    assert len(parts) == 3 and "".join(parts) == text
    assert cs.chunk_answer("") == []


def test_prefs_validate_drops_unknown_and_bad() -> None:
    got = P.validate(
        {
            "theme": "dark",
            "density": "cosmic",  # noma'lum qiymat
            "locale": "en",
            "deep_default": "yes",  # bool emas
            "pinned_chats": [1, 1, 2, True, -5, "x"],
            "recent_chats": list(range(1, 40)),
            "evil": {"drop": "me"},
        }
    )
    assert got == {
        "theme": "dark",
        "locale": "en",
        "pinned_chats": [1, 2],
        "recent_chats": list(range(1, P.RECENT_MAX + 1)),
    }
    assert "evil" not in got and "density" not in got


def test_prefs_merge_keeps_current_and_fills_defaults() -> None:
    merged = P.merge({"theme": "dark", "junk": 1}, {"density": "compact"})
    assert merged["theme"] == "dark" and merged["density"] == "compact"
    assert merged["onboarding_done"] is False and merged["pinned_chats"] == []
    assert "junk" not in merged
    # bo'sh satr — "avtomatik" (tozalash), noto'g'ri til esa yozilmaydi
    assert P.merge({"locale": "ru"}, {"locale": ""})["locale"] == ""
    assert P.merge({"locale": "ru"}, {"locale": "kz"})["locale"] == "ru"


# ─── Mini App initData ───────────────────────────────────────────────────────

BOT_TOKEN = "123456:TEST-SYNTHETIC"  # noqa: S105 — sintetik, haqiqiy token emas


def make_init_data(token: str, *, user_id: int = 555, auth_date: int | None = None) -> str:
    """Telegram hujjatidagi algoritm bo'yicha imzolangan `initData`."""
    fields = {
        "auth_date": str(int(auth_date if auth_date is not None else time.time())),
        "query_id": "AAE",
        "user": json.dumps({"id": user_id, "first_name": "A"}, separators=(",", ":")),
    }
    check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def test_verify_init_data_accepts_valid_and_rejects_tampered() -> None:
    from app.services.auth_flow import AuthError
    from app.web.routers.auth import verify_init_data

    raw = make_init_data(BOT_TOKEN)
    data = verify_init_data(raw, BOT_TOKEN)
    assert data["user"]["id"] == 555

    # boshqa token — imzo mos kelmaydi
    with pytest.raises(AuthError):
        verify_init_data(raw, BOT_TOKEN + "x")
    # maydon o'zgartirilgan (imzo eski)
    with pytest.raises(AuthError):
        verify_init_data(raw.replace("AAE", "AAF"), BOT_TOKEN)
    # hash yo'q
    with pytest.raises(AuthError):
        verify_init_data("auth_date=1&user=%7B%7D", BOT_TOKEN)


def test_verify_init_data_rejects_old_auth_date() -> None:
    from app.services.auth_flow import AuthError
    from app.web.routers.auth import verify_init_data

    old = make_init_data(BOT_TOKEN, auth_date=int(time.time()) - 90_000)
    with pytest.raises(AuthError) as exc:
        verify_init_data(old, BOT_TOKEN)
    assert exc.value.code == "init_data_expired"


def test_auth_telegram_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings
    from app.web.routers import auth as auth_router

    monkeypatch.setattr(get_settings(), "bot_token", BOT_TOKEN)
    client.cookies.clear()  # kirmagan holat — initData kirishga yetadi

    known: dict[str, Any] = {"user": SimpleNamespace(id=7)}

    class _DB:
        async def execute(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(scalar_one_or_none=lambda: known["user"])

    monkeypatch.setattr(auth_router, "session_scope", lambda: _Scope(_DB()))

    raw = make_init_data(BOT_TOKEN)
    assert client.post("/api/auth/telegram", json={"init_data": raw}).status_code == 403  # CSRF

    r = client.post("/api/auth/telegram", json={"init_data": raw}, headers=H)
    assert r.status_code == 200 and r.json() == {"status": "done", "user_id": 7}
    ident = sec.read_session(r.cookies[sec.SESSION_COOKIE])
    assert ident is not None and ident.user_id == 7

    # akkaunt ulanmagan → 403, akkaunt **yaratilmaydi**
    known["user"] = None
    r = client.post("/api/auth/telegram", json={"init_data": raw}, headers=H)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "auth.err.not_linked"

    # buzuq imzo
    r = client.post("/api/auth/telegram", json={"init_data": raw + "x"}, headers=H)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "auth.err.init_data"


def test_auth_telegram_rate_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings
    from app.web.routers import auth as auth_router

    class _DB:
        async def execute(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    monkeypatch.setattr(auth_router, "session_scope", lambda: _Scope(_DB()))
    monkeypatch.setattr(get_settings(), "bot_token", BOT_TOKEN)
    client.app.state.auth_limiter = sec.RateLimiter(1, 600)  # type: ignore[attr-defined]
    raw = make_init_data(BOT_TOKEN, user_id=999)
    client.post("/api/auth/telegram", json={"init_data": raw}, headers=H)
    r = client.post("/api/auth/telegram", json={"init_data": raw}, headers=H)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "auth.err.rate_limited"


# ─── static fingerprint / kesh ───────────────────────────────────────────────


def test_asset_returns_versioned_url() -> None:
    from app.web.routers import pages

    url = pages.asset("app.css")
    assert url.startswith("/static/app.css?v=") and len(url.split("v=")[1]) == 10
    # mavjud bo'lmagan fayl — versiyasiz (sahifa baribir yiqilmasin)
    assert pages.asset("nope.css") == "/static/nope.css"
    # yo'l chegarasi: static papkadan chiqib bo'lmaydi
    assert "?v=" not in pages.asset("../config.py")
    assert pages.prime_assets() > 0


def test_static_cache_headers(client: TestClient) -> None:
    r = client.get("/static/app.css?v=abc123")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    r = client.get("/static/app.css")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_html_no_cache_and_etag(client: TestClient) -> None:
    client.cookies.clear()
    r = client.get("/login")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    etag = r.headers["etag"]
    r2 = client.get("/login", headers={"If-None-Match": etag})
    assert r2.status_code == 304 and r2.headers["etag"] == etag


def test_vendor_telegram_sdk_is_local() -> None:
    from app.web.routers import pages

    path = pages.STATIC_DIR / "vendor" / "telegram-web-app.js"
    assert path.is_file() and path.stat().st_size > 10_000
    assert (pages.STATIC_DIR / "vendor" / "VENDOR.md").is_file()


# ─── oqimli chat ─────────────────────────────────────────────────────────────


@pytest.fixture
def chat_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.services import actions as ACT
    from app.services import evaluation as ev
    from app.services import limits
    from app.web.routers import chat as chat_router

    conv = SimpleNamespace(id=1, title="", account_id=3, user_id=7)
    state: dict[str, Any] = {"persisted": [], "conv": conv}

    async def fake_conv(db: Any, user_id: int, cid: int) -> Any:
        if cid != conv.id:
            raise cs.ChatError("not_found")
        return conv

    async def fake_send(db: Any, **kw: Any) -> cs.Reply:
        events = kw.get("events")
        # Agent sikli bir tool chaqirgandek — hodisalar tartibi shu yerda tug'iladi
        await cs.emit(events, "status", {"stage": "tool", "tool": "search_messages"})
        state["persisted"].append(kw["text"])
        answer = "J" * 900
        await cs._stream_answer(events, answer)
        return cs.Reply(
            conversation_id=conv.id,
            user_message_id=10,
            assistant_message_id=11,
            text=answer,
            model="m",
            provider="p",
            tokens_in=5,
            tokens_out=6,
            context_used=None,
            latency_ms=12,
            cost_usd=0.001,
        )

    async def no_budget(db: Any, user_id: int) -> Any:
        return None

    async def fake_eval(*a: Any, **kw: Any) -> None:
        return None

    async def no_actions(db: Any, ids: Any) -> dict[int, Any]:
        return {}

    monkeypatch.setattr(chat_router, "session_scope", lambda: _Scope())
    monkeypatch.setattr(chat_router.cs, "get_conversation", fake_conv)
    monkeypatch.setattr(chat_router.cs, "send_message", fake_send)
    monkeypatch.setattr(limits, "assert_within_budget", no_budget)
    monkeypatch.setattr(limits, "check_chat_rate", lambda uid: True)
    monkeypatch.setattr(ACT, "actions_for_runs", no_actions)
    monkeypatch.setattr(ev, "auto_evaluate", fake_eval)
    client.state = state  # type: ignore[attr-defined]
    return client


def test_stream_requires_auth_and_csrf(chat_client: TestClient) -> None:
    r = chat_client.post("/api/conversations/1/messages/stream", json={"text": "salom"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf"
    chat_client.cookies.clear()
    r = chat_client.post("/api/conversations/1/messages/stream", json={"text": "salom"}, headers=H)
    assert r.status_code == 401


def test_stream_event_order_and_persistence(chat_client: TestClient) -> None:
    r = chat_client.post("/api/conversations/1/messages/stream", json={"text": "savol"}, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "status" and kinds[-1] == "done"
    # thinking → tool → writing → token* → done
    stages = [d["stage"] for k, d in events if k == "status"]
    assert stages == ["thinking", "tool", "writing"]
    assert [d for k, d in events if k == "status"][1]["tool"] == "search_messages"
    tokens = [d["text"] for k, d in events if k == "token"]
    assert len(tokens) == 3 and "".join(tokens) == "J" * 900

    done = events[-1][1]
    for key in (
        "assistant_message_id",
        "model",
        "provider",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "latency_ms",
        "actions",
    ):
        assert key in done
    assert done["assistant_message_id"] == 11 and done["actions"] == []
    # saqlash oddiy yo'l bilan bir xil — `send_message` bitta yozuvchi
    assert chat_client.state["persisted"] == ["savol"]  # type: ignore[attr-defined]


def test_stream_rate_limit_error_before_llm(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import limits

    monkeypatch.setattr(limits, "check_chat_rate", lambda uid: False)
    r = chat_client.post("/api/conversations/1/messages/stream", json={"text": "q"}, headers=H)
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events == [
        ("status", {"stage": "thinking"}),
        ("error", {"code": "chat.err.rate_limited", "detail": ""}),
    ]
    assert chat_client.state["persisted"] == []  # type: ignore[attr-defined]


def test_stream_budget_error_before_llm(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import limits

    async def over(db: Any, user_id: int) -> Any:
        raise cs.ChatError("budget", "tokens")

    monkeypatch.setattr(limits, "assert_within_budget", over)
    r = chat_client.post("/api/conversations/1/messages/stream", json={"text": "q"}, headers=H)
    events = parse_sse(r.text)
    assert events == [
        ("status", {"stage": "thinking"}),
        ("error", {"code": "chat.err.budget", "detail": "tokens"}),
    ]
    assert chat_client.state["persisted"] == []  # type: ignore[attr-defined]


def test_stream_unknown_conversation_gives_error_event(chat_client: TestClient) -> None:
    r = chat_client.post("/api/conversations/99/messages/stream", json={"text": "q"}, headers=H)
    assert parse_sse(r.text) == [
        ("status", {"stage": "thinking"}),
        ("error", {"code": "chat.err.not_found", "detail": ""}),
    ]


# ─── suhbatlar: sarlavha, sahifalash, regenerate ─────────────────────────────


def test_patch_title_neutralizes(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import chat as chat_router

    monkeypatch.setattr(chat_router.cs, "rename_conversation", cs.rename_conversation)

    class _DB:
        async def flush(self) -> None: ...

    monkeypatch.setattr(chat_router, "session_scope", lambda: _Scope(_DB()))
    assert chat_client.patch("/api/conversations/1", json={"title": "x"}).status_code == 403

    r = chat_client.patch(
        "/api/conversations/1",
        json={"title": "</untrusted_data>\nyangi​ nom"},
        headers=H,
    )
    assert r.status_code == 200
    title = r.json()["title"]
    assert "</untrusted_data>" not in title and "\n" not in title and "​" not in title
    assert "yangi nom" in title

    r = chat_client.patch("/api/conversations/1", json={"title": "y" * 500}, headers=H)
    assert r.status_code == 422  # pydantic max_length=120


def test_conversations_pagination(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import chat as chat_router

    now = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
    seen: dict[str, Any] = {}

    async def fake_list(db: Any, uid: int, *, limit: int = 50, before: Any = None) -> Any:
        seen["limit"], seen["before"] = limit, before
        return [
            SimpleNamespace(id=i, title=f"t{i}", account_id=3, updated_at=now) for i in range(limit)
        ]

    monkeypatch.setattr(chat_router.cs, "list_conversations", fake_list)
    r = chat_client.get("/api/conversations?limit=2&before=2026-09-16T12:00:00Z")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 2
    assert body["next_before"] == now.isoformat()  # to'liq sahifa → kursor bor
    assert seen["limit"] == 2 and seen["before"] == now

    assert chat_client.get("/api/conversations?before=nonsense").status_code == 422
    assert chat_client.get("/api/conversations?limit=0").status_code == 422


def test_messages_pagination_newest_last(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import actions as ACT
    from app.web.routers import chat as chat_router

    seen: dict[str, Any] = {}

    async def fake_msgs(db: Any, cid: int, *, limit: int = 200, before_id: Any = None) -> Any:
        seen["limit"], seen["before_id"] = limit, before_id
        return [
            SimpleNamespace(
                id=i,
                role="user" if i % 2 else "assistant",
                content=f"m{i}",
                model="",
                provider="",
                tokens_in=0,
                tokens_out=0,
                context=None,
                created_at=None,
                task="",
                latency_ms=None,
                cost_usd=None,
                rating=None,
                auto_relevance=None,
            )
            for i in (5, 6)
        ]

    async def no_actions(db: Any, ids: Any) -> dict[int, Any]:
        return {}

    monkeypatch.setattr(chat_router.cs, "list_messages", fake_msgs)
    monkeypatch.setattr(ACT, "actions_for_runs", no_actions)
    r = chat_client.get("/api/conversations/1/messages?limit=2&before=7")
    assert r.status_code == 200
    body = r.json()
    assert [m["id"] for m in body["items"]] == [5, 6]  # eng yangisi oxirida
    assert body["next_before"] == 5
    assert seen == {"limit": 2, "before_id": 7}


def test_regenerate_replaces_last_turn(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.web.routers import chat as chat_router

    calls: list[int] = []

    async def fake_take(db: Any, uid: int, cid: int, mid: int) -> Any:
        calls.append(mid)
        if mid == 999:
            raise cs.ChatError("not_last")
        return chat_client.state["conv"], "eski savol", None  # type: ignore[attr-defined]

    monkeypatch.setattr(chat_router.cs, "take_for_regenerate", fake_take)
    no_csrf = chat_client.post("/api/conversations/1/messages/11/regenerate", json={})
    assert no_csrf.status_code == 403

    r = chat_client.post("/api/conversations/1/messages/11/regenerate", json={}, headers=H)
    assert r.status_code == 200 and r.json()["assistant_message_id"] == 11
    # matn saqlangan savoldan olindi, so'rov tanasidan emas
    assert chat_client.state["persisted"] == ["eski savol"]  # type: ignore[attr-defined]
    assert calls == [11]

    r = chat_client.post("/api/conversations/1/messages/999/regenerate", json={}, headers=H)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "chat.err.not_last"


def test_regenerate_streaming(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import chat as chat_router

    async def fake_take(db: Any, uid: int, cid: int, mid: int) -> Any:
        return chat_client.state["conv"], "eski savol", None  # type: ignore[attr-defined]

    monkeypatch.setattr(chat_router.cs, "take_for_regenerate", fake_take)
    r = chat_client.post(
        "/api/conversations/1/messages/11/regenerate", json={"stream": True}, headers=H
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    kinds = [k for k, _ in parse_sse(r.text)]
    assert kinds[0] == "status" and kinds[-1] == "done"


# ─── prefs / akkaunt / sync API ──────────────────────────────────────────────


@pytest.fixture
def settings_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.web.routers import settings as settings_router

    user = SimpleNamespace(id=7, prefs={"theme": "dark"})
    account = SimpleNamespace(
        id=3, user_id=7, status="active", session_ciphertext=b"x", session_wrapped_dek=b"y"
    )
    store = {"user": user, "account": account, "aborted": []}

    class _DB:
        async def get(self, model: Any, pk: int) -> Any:
            name = getattr(model, "__name__", "")
            if name == "User":
                return store["user"] if pk == 7 else None
            if name == "Account":
                return store["account"] if pk == 3 else None
            return None

        async def flush(self) -> None: ...

        async def execute(self, *a: Any, **kw: Any) -> Any:
            rows = store.get("rows", [])
            return SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: rows), all=lambda: rows
            )

    async def fake_abort(account_id: int) -> None:
        store["aborted"].append(account_id)

    monkeypatch.setattr(settings_router, "session_scope", lambda: _Scope(_DB()))
    monkeypatch.setattr(settings_router, "_abort_jobs", fake_abort)
    client.store = store  # type: ignore[attr-defined]
    return client


def test_prefs_get_put(settings_client: TestClient) -> None:
    r = settings_client.get("/api/settings/prefs")
    assert r.status_code == 200 and r.json()["prefs"]["theme"] == "dark"
    assert r.json()["prefs"]["density"] == "comfortable"  # default to'ldirildi

    assert settings_client.put("/api/settings/prefs", json={"theme": "light"}).status_code == 403
    r = settings_client.put(
        "/api/settings/prefs",
        json={"theme": "light", "pinned_chats": [4, 4, 9], "evil": "x", "density": "nope"},
        headers=H,
    )
    assert r.status_code == 200
    got = r.json()["prefs"]
    assert got["theme"] == "light" and got["pinned_chats"] == [4, 9]
    assert got["density"] == "comfortable" and "evil" not in got
    assert settings_client.store["user"].prefs["theme"] == "light"  # type: ignore[attr-defined]


def test_prefs_requires_login(settings_client: TestClient) -> None:
    settings_client.cookies.clear()
    assert settings_client.get("/api/settings/prefs").status_code == 401
    assert settings_client.put("/api/settings/prefs", json={}, headers=H).status_code == 401


def test_unlink_account_clears_session_without_logout(settings_client: TestClient) -> None:
    assert settings_client.delete("/api/accounts/3").status_code == 403  # CSRF
    r = settings_client.delete("/api/accounts/3", headers=H)
    assert r.status_code == 200 and r.json() == {"status": "revoked", "id": 3}
    acc = settings_client.store["account"]  # type: ignore[attr-defined]
    assert acc.session_ciphertext is None and acc.session_wrapped_dek is None
    assert acc.status == "revoked"
    assert settings_client.store["aborted"] == [3]  # type: ignore[attr-defined]
    assert settings_client.delete("/api/accounts/4", headers=H).status_code == 404


def test_unlink_never_calls_logout() -> None:
    """`auth.LogOut` DENIED — uzish yo'lida u chaqirilmasligi kerak (invariant 1)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "app" / "web" / "routers" / "settings.py"
    text = src.read_text(encoding="utf-8")
    assert "LogOut" not in text.replace("`auth.LogOut`", "")


def test_sync_status(settings_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import settings as settings_router

    async def fake_status(db: Any, account_id: int) -> dict[str, Any]:
        return {
            "running": True,
            "chats": [
                {
                    "chat_id": 1,
                    "state": "running",
                    "progress": 42,
                    "synced_total": 42,
                    "total_estimate": 100,
                    "error": None,
                }
            ],
            "counts": {"running": 1},
            "poll_seconds": 2.0,
        }

    monkeypatch.setattr(settings_router, "_sync_status", fake_status)
    r = settings_client.get("/api/accounts/3/sync/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["chats"][0] == {
        "chat_id": 1,
        "state": "running",
        "progress": 42,
        "synced_total": 42,
        "total_estimate": 100,
        "error": None,
    }
    assert settings_client.get("/api/accounts/4/sync/status").status_code == 404


def test_sync_stream_ends_with_done(
    settings_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.web.routers import settings as settings_router

    async def fake_status(db: Any, account_id: int) -> dict[str, Any]:
        return {"running": False, "chats": [], "counts": {}, "poll_seconds": 2.0}

    monkeypatch.setattr(settings_router, "_sync_status", fake_status)
    r = settings_client.get("/api/accounts/3/sync/stream")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    kinds = [k for k, _ in parse_sse(r.text)]
    assert kinds == ["status", "done"]


def test_me_includes_prefs_and_counts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import auth as auth_router

    user = SimpleNamespace(id=7, tg_user_id=555, username="ali", prefs={"theme": "dark"})

    class _DB:
        async def get(self, model: Any, pk: int) -> Any:
            return user

        async def execute(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(all=lambda: [(3, 12, 5)])

    async def fake_accounts(db: Any, uid: int) -> Any:
        return [
            SimpleNamespace(
                id=3, label="@ali", status="active", tg_account_id=555, last_seen_at=None
            )
        ]

    monkeypatch.setattr(auth_router, "session_scope", lambda: _Scope(_DB()))
    monkeypatch.setattr(auth_router, "list_accounts", fake_accounts)
    r = client.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["display_name"] == "ali" and body["user"]["avatar_url"] is None
    assert body["user"]["prefs"]["theme"] == "dark"
    assert body["accounts"][0]["chats_total"] == 12
    assert body["accounts"][0]["chats_synced"] == 5
