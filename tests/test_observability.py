"""8-bosqich: metrikalar, limitlar, /metrics va /api/stats/system — tarmoqsiz."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import observability as O
from app.config import get_settings
from app.services import limits as L
from app.web import security as sec


def _metric_value(name: str, **labels: str) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_record_llm_counts_tokens_and_cost() -> None:
    before = _metric_value(
        "tgai_llm_tokens_total", provider="claude", model="claude-opus-5", dir="in"
    )
    O.record_llm(
        provider="claude",
        model="claude-opus-5",
        task="search",
        ok=True,
        tokens_in=1000,
        tokens_out=100,
        seconds=1.2,
    )
    after = _metric_value(
        "tgai_llm_tokens_total", provider="claude", model="claude-opus-5", dir="in"
    )
    assert after - before == 1000
    assert (
        _metric_value(
            "tgai_llm_requests_total",
            provider="claude",
            model="claude-opus-5",
            task="search",
            ok="true",
        )
        >= 1
    )
    assert _metric_value("tgai_llm_cost_usd_total", provider="claude", model="claude-opus-5") > 0


def test_metrics_payload_has_our_metrics() -> None:
    body, ctype = O.metrics_payload()
    assert b"tgai_llm_requests_total" in body and b"tgai_write_actions_total" in body
    assert "text/plain" in ctype


def test_scrub_masks_secrets() -> None:
    event = {
        "extra": {"session_string": "abc", "nested": [{"password": "x", "ok": 1}]},
        "message": "m",
    }
    out = O._scrub(event, None)
    assert out is not None
    masked = "***"
    assert out["extra"]["session_string"] == masked
    assert out["extra"]["nested"][0]["password"] == masked
    assert out["extra"]["nested"][0]["ok"] == 1


def test_init_sentry_disabled_without_dsn() -> None:
    assert O.init_sentry("test") is False


# ─── limitlar ────────────────────────────────────────────────────────────────


def test_usage_exceeded_flags() -> None:
    assert L.Usage(10, 0.1, 1, 100, 1.0).exceeded is None
    assert L.Usage(100, 0.1, 1, 100, 1.0).exceeded == "tokens"
    assert L.Usage(1, 5.0, 1, 100, 1.0).exceeded == "cost"
    assert L.Usage(10**9, 10**9, 1, 0, 0.0).exceeded is None  # 0 = cheksiz
    d = L.Usage(50, 0.5, 3, 100, 2.0).to_dict()
    assert d["tokens_pct"] == 50.0 and d["cost_pct"] == 25.0


async def test_assert_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.chat_service import ChatError

    async def over(session: Any, user_id: int) -> L.Usage:
        return L.Usage(200, 0.0, 1, 100, 0.0)

    monkeypatch.setattr(L, "daily_usage", over)
    with pytest.raises(ChatError, match="budget"):
        await L.assert_within_budget(object(), 1)  # type: ignore[arg-type]


def test_chat_rate_limiter() -> None:
    rl = sec.RateLimiter(2, 60)
    assert rl.check("u1") and rl.check("u1") and not rl.check("u1")


# ─── endpoint'lar ────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.web import main as web_main

    with TestClient(web_main.app) as c:
        yield c


def test_metrics_endpoint_and_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200 and "tgai_http_requests_total" in r.text
    monkeypatch.setattr(get_settings(), "metrics_token", "sekret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics?token=sekret").status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer sekret"}).status_code == 200
    monkeypatch.setattr(get_settings(), "metrics_token", "")


def test_http_middleware_counts_requests(client: TestClient) -> None:
    before = _metric_value("tgai_http_requests_total", method="GET", route="/login", status="200")
    client.get("/login")
    after = _metric_value("tgai_http_requests_total", method="GET", route="/login", status="200")
    assert after == before + 1


def test_system_status_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import stats as stats_router

    class _DB:
        async def execute(self, *a: Any) -> Any:
            return None

    class _Scope:
        async def __aenter__(self) -> Any:
            return _DB()

        async def __aexit__(self, *a: Any) -> None: ...

    async def hb() -> dict[str, Any]:
        return {"snapshot_metrics": {"at": "2026-08-18T12:05:00+00:00"}}

    async def usage(db: Any, user_id: int) -> L.Usage:
        return L.Usage(10, 0.01, 1, 100, 1.0)

    monkeypatch.setattr(stats_router, "session_scope", lambda: _Scope())
    monkeypatch.setattr(O, "heartbeats", hb)
    monkeypatch.setattr(L, "daily_usage", usage)
    client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
    r = client.get("/api/stats/system")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["db"]["ok"] and d["redis"]["ok"] and d["heartbeats"]["snapshot_metrics"]["at"]
    assert d["budget"]["tokens"] == 10 and d["version"]


def test_health_has_version(client: TestClient) -> None:
    assert "version" in client.get("/health").json()


def test_settings_defaults() -> None:
    s = get_settings()
    assert s.llm_daily_token_budget > 0 and s.chat_rate_per_minute >= 1
    assert isinstance(SimpleNamespace(v=s.app_version).v, str)
