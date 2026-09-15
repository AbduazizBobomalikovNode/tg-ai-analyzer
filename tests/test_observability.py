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
    # Sentry va structlog bitta implementatsiyani bo'lishadi (`app.logging.scrub`)
    from app.logging import MASK as masked

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


# ─── M3: fon ishlari uchun byudjet qopqog'i ──────────────────────────────────


class _BudgetSession:
    """Ikki `execute`: (1) egalar bo'yicha jamlama, (2) akkaunt id'lari."""

    def __init__(self, totals: list[Any], accounts: list[Any]) -> None:
        self.totals, self.accounts = totals, accounts
        self.n = 0

    async def execute(self, stmt: Any) -> Any:
        self.n += 1
        rows = self.totals if self.n == 1 else self.accounts
        return SimpleNamespace(all=lambda: rows, scalars=lambda: rows)


async def test_blocked_user_ids_groups_by_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "llm_daily_token_budget", 100)
    monkeypatch.setattr(s, "llm_daily_cost_budget_usd", 0.0)
    session = _BudgetSession([(1, 150, 0.0), (2, 10, 0.0), (3, 100, 0.0)], [])
    assert await L.blocked_user_ids(session) == {1, 3}  # type: ignore[arg-type]


async def test_blocked_user_ids_is_empty_without_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "llm_daily_token_budget", 0)
    monkeypatch.setattr(s, "llm_daily_cost_budget_usd", 0.0)
    session = _BudgetSession([(1, 10**9, 10**9)], [])
    assert await L.blocked_user_ids(session) == set()  # type: ignore[arg-type]


async def test_blocked_account_ids_maps_owners_to_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "llm_daily_token_budget", 100)
    monkeypatch.setattr(s, "llm_daily_cost_budget_usd", 0.0)
    session = _BudgetSession([(1, 150, 0.0)], [11, 12])
    assert await L.blocked_account_ids(session) == {11, 12}  # type: ignore[arg-type]


async def test_worker_embed_skips_blocked_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """M3: `embed_messages` byudjeti tugagan egalarning chatlarini chetlab o'tadi."""
    from app.worker import tasks as W

    async def blocked() -> set[int]:
        return {42}

    captured: list[str] = []

    class _Sess:
        async def execute(self, stmt: Any) -> Any:
            captured.append(str(stmt))
            return SimpleNamespace(all=lambda: [])

    class _Scope:
        async def __aenter__(self) -> Any:
            return _Sess()

        async def __aexit__(self, *a: Any) -> None: ...

    monkeypatch.setattr(W, "_budget_blocked_accounts", blocked)
    monkeypatch.setattr("app.db.base.session_scope", lambda: _Scope())
    s = get_settings()
    monkeypatch.setattr(s, "embed_enabled", True)
    monkeypatch.setattr(s, "embed_mode", "sync")

    out = await W.embed_messages({})

    assert out == {"embedded": 0}
    sql = captured[0]
    assert "chats.account_id NOT IN" in sql
    # phase4: embedding **joriy model** bo'yicha yetishmayotganini qidiradi
    assert "message_embeddings.model" in sql


async def test_worker_reembed_skips_a_blocked_account(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.worker import tasks as W

    async def blocked() -> set[int]:
        return {42}

    monkeypatch.setattr(W, "_budget_blocked_accounts", blocked)
    monkeypatch.setattr(get_settings(), "embed_enabled", True)
    assert await W.reembed_all({}, 42) == {"skipped": "budget", "account_id": 42}


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
    # Endpoint modul-global `web_main.s` ni o'qiydi; `get_settings()` keshi
    # boshqa testda tozalangan bo'lsa u boshqa obyektga aylanadi.
    from app.web import main as web_main

    monkeypatch.setattr(web_main.s, "metrics_token", "sekret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics?token=sekret").status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer sekret"}).status_code == 200
    monkeypatch.setattr(web_main.s, "metrics_token", "")


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


# ─── phase4/B: alert va briefing metrikalari ─────────────────────────────────


def test_alert_and_briefing_metrics_exist() -> None:
    """Nomlar barqaror — dashboard/alert qoidalari shularga tayanadi."""
    O.ALERT_SCANS.inc()
    O.ALERT_EVENTS.labels("keyword").inc()
    O.ALERT_PUSHES.labels("ok").inc()
    O.BRIEFING_BUILDS.labels("ok").inc()
    O.BRIEFING_PUSHES.labels("sent").inc()
    body, _ = O.metrics_payload()
    for name in (
        b"tgai_alert_scans_total",
        b"tgai_alert_events_total",
        b"tgai_alert_pushes_total",
        b"tgai_briefing_builds_total",
        b"tgai_briefing_pushes_total",
    ):
        assert name in body
    assert _metric_value("tgai_alert_events_total", kind="keyword") >= 1
    assert _metric_value("tgai_briefing_pushes_total", status="sent") >= 1


def test_briefing_delivery_result_is_counted() -> None:
    from datetime import date

    from app.services.briefing import DeliveryResult, _counted

    before = _metric_value("tgai_briefing_pushes_total", status="skipped")
    res = _counted(DeliveryResult(1, date(2026, 9, 15), "skipped", "quiet"))
    assert res.status == "skipped"
    assert _metric_value("tgai_briefing_pushes_total", status="skipped") - before == 1
