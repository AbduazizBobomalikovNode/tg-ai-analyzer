"""Foydalanuvchi bo'yicha kunlik byudjet (migratsiya 0023): override + bugungi boost.

Qo'riqlaydi:
  * `compute_budget` — global / override / faol boost / muddati o'tgan boost /
    shift (ceiling) qisish / cheksiz baza;
  * `effective_budget` — kesh → DB → global (Redis ham, DB ham yiqilsa fail-open);
  * **har** tekshiruv samarali limitdan o'tadi: global limitdan oshgan, lekin
    faol boost ichidagi foydalanuvchi **to'silmaydi** (`check_daily_budget`,
    `assert_within_budget`, `blocked_user_ids`);
  * API: auth + CSRF, validatsiya, PUT/POST/DELETE, GET shakli;
  * haqiqiy Postgres: set/boost/clear aylanishi (TEST_DATABASE_URL bo'lsa).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services import limits as L
from app.web import main as web_main
from app.web import security as sec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

H = {"X-Requested-With": "fetch"}
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def glob(monkeypatch: pytest.MonkeyPatch) -> Any:
    s = get_settings()
    monkeypatch.setattr(s, "llm_daily_token_budget", 4_000_000)
    monkeypatch.setattr(s, "llm_daily_cost_budget_usd", 10.0)
    monkeypatch.setattr(s, "llm_daily_token_budget_max", 0)
    monkeypatch.setattr(s, "llm_daily_cost_budget_usd_max", 0.0)
    return s


# ─── compute_budget (sof) ────────────────────────────────────────────────────


def test_global_default(glob: Any) -> None:
    b = L.compute_budget(None, now=NOW)
    assert (b.tokens, b.usd, b.source, b.boost_active) == (4_000_000, 10.0, "global", False)
    assert b.ceiling_tokens is None and b.ceiling_usd is None


def test_override_replaces_global(glob: Any) -> None:
    b = L.compute_budget(L.BudgetRow(token_budget=6_000_000), now=NOW)
    assert (b.tokens, b.usd, b.source) == (6_000_000, 10.0, "custom")
    b = L.compute_budget(L.BudgetRow(cost_budget_usd=2.5), now=NOW)
    assert (b.tokens, b.usd, b.source) == (4_000_000, 2.5, "custom")


def test_active_boost_adds(glob: Any) -> None:
    row = L.BudgetRow(boost_tokens=1_000_000, boost_usd=3.0, boost_until=NOW + timedelta(hours=5))
    b = L.compute_budget(row, now=NOW)
    assert (b.tokens, b.usd) == (5_000_000, 13.0)
    assert b.boost_active and b.boost_until == row.boost_until and b.source == "global"


def test_expired_boost_is_ignored(glob: Any) -> None:
    row = L.BudgetRow(boost_tokens=1_000_000, boost_usd=3.0, boost_until=NOW - timedelta(seconds=1))
    b = L.compute_budget(row, now=NOW)
    assert (b.tokens, b.usd, b.boost_active, b.boost_until) == (4_000_000, 10.0, False, None)


def test_ceiling_clamps_override_and_boost(glob: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glob, "llm_daily_token_budget_max", 5_000_000)
    monkeypatch.setattr(glob, "llm_daily_cost_budget_usd_max", 12.0)
    row = L.BudgetRow(
        token_budget=9_000_000,
        boost_tokens=1_000_000,
        boost_usd=5.0,
        boost_until=NOW + timedelta(hours=1),
    )
    b = L.compute_budget(row, now=NOW)
    assert (b.tokens, b.usd) == (5_000_000, 12.0)
    assert (b.ceiling_tokens, b.ceiling_usd) == (5_000_000, 12.0)
    # Foydalanuvchi "0 = cheksiz" desa ham shift ustun.
    assert L.compute_budget(L.BudgetRow(token_budget=0), now=NOW).tokens == 5_000_000
    # Global standart esa (o'zgartirilmagan) admin qarori — qisilmaydi.
    monkeypatch.setattr(glob, "llm_daily_token_budget_max", 1_000_000)
    assert L.compute_budget(None, now=NOW).tokens == 4_000_000


def test_unlimited_base_stays_unlimited_with_boost(
    glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glob, "llm_daily_token_budget", 0)
    row = L.BudgetRow(boost_tokens=1_000_000, boost_until=NOW + timedelta(hours=1))
    assert L.compute_budget(row, now=NOW).tokens == 0


def test_budget_row_json_roundtrip() -> None:
    row = L.BudgetRow(
        token_budget=5, cost_budget_usd=1.25, boost_tokens=7, boost_usd=0.5, boost_until=NOW
    )
    assert L.BudgetRow.from_json(row.to_json()) == row
    assert L.BudgetRow.from_json("{}") is None


def test_next_local_midnight_uses_account_tz() -> None:
    now = datetime(2026, 9, 25, 20, 30, tzinfo=UTC)  # Toshkentda 26-sent 01:30
    assert L.next_local_midnight("Asia/Tashkent", now=now) == datetime(
        2026, 9, 26, 19, 0, tzinfo=UTC
    )
    assert L.next_local_midnight("UTC", now=now) == datetime(2026, 9, 26, 0, 0, tzinfo=UTC)
    # Buzuq zona — yiqilmaydi, standartga tushadi.
    assert L.next_local_midnight("Nope/Zone", now=now) == datetime(2026, 9, 26, 19, 0, tzinfo=UTC)


# ─── effective_budget: kesh → DB → global ────────────────────────────────────


class _Rows:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def scalar_one_or_none(self) -> Any:
        return self.obj


class _DBSession:
    """`user_budgets` o'qishi uchun soxta sessiya."""

    def __init__(self, ub: Any = None, *, fail: bool = False) -> None:
        self.ub, self.fail, self.calls = ub, fail, 0

    async def execute(self, stmt: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("db down")
        return _Rows(self.ub)


def _ub(**kw: Any) -> Any:
    base = {
        "token_budget": None,
        "cost_budget_usd": None,
        "boost_tokens": 0,
        "boost_usd": 0.0,
        "boost_until": None,
    }
    return SimpleNamespace(**{**base, **kw})


async def test_effective_budget_redis_error_reads_db(
    glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken() -> Any:
        raise ConnectionError("redis down")

    monkeypatch.setattr(L, "_redis", broken)
    session = _DBSession(_ub(token_budget=7_000_000))
    b = await L.effective_budget(session, 1)  # type: ignore[arg-type]
    assert b.tokens == 7_000_000 and session.calls == 1


async def test_effective_budget_fails_open_to_global(
    glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken() -> Any:
        raise ConnectionError("redis down")

    monkeypatch.setattr(L, "_redis", broken)
    b = await L.effective_budget(_DBSession(fail=True), 1)  # type: ignore[arg-type]
    assert (b.tokens, b.usd, b.source) == (4_000_000, 10.0, "global")


class _FakeRedis:
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value.encode()
        self.store[f"ttl:{key}"] = ex

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def aclose(self) -> None: ...


async def test_effective_budget_is_cached_60s_and_invalidated(
    glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: dict[str, Any] = {}

    async def fake() -> Any:
        return _FakeRedis(store)

    monkeypatch.setattr(L, "_redis", fake)
    session = _DBSession(_ub(cost_budget_usd=3.0))
    assert (await L.effective_budget(session, 5)).usd == 3.0  # type: ignore[arg-type]
    assert (await L.effective_budget(session, 5)).usd == 3.0  # type: ignore[arg-type]
    assert session.calls == 1, "ikkinchi o'qish keshdan bo'lishi kerak"
    assert store[f"ttl:{L.CACHE_PREFIX}5"] == L.CACHE_TTL_S == 60
    await L.invalidate(5)
    session.ub = _ub(cost_budget_usd=4.0)
    assert (await L.effective_budget(session, 5)).usd == 4.0  # type: ignore[arg-type]
    assert session.calls == 2
    # Qator yo'q — ham keshlanadi ("{}"), DB'ga qayta borilmaydi.
    empty = _DBSession(None)
    assert (await L.effective_budget(empty, 6)).source == "global"  # type: ignore[arg-type]
    assert (await L.effective_budget(empty, 6)).source == "global"  # type: ignore[arg-type]
    assert empty.calls == 1


# ─── tekshiruv yo'llari samarali limitdan o'tadi ─────────────────────────────


class _UsageSession:
    """`daily_usage` jamlama so'rovi: `(tokens, cost, count)`."""

    def __init__(self, tokens: int, cost: float) -> None:
        self.row = (tokens, cost, 3)

    async def execute(self, stmt: Any) -> Any:
        return SimpleNamespace(one=lambda: self.row)


@pytest.fixture
def boosted(glob: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Foydalanuvchi 1: faol boost +1M token / +$3 (5 soat)."""

    async def row(session: Any, user_id: int) -> L.BudgetRow | None:
        if user_id != 1:
            return None
        return L.BudgetRow(
            boost_tokens=1_000_000,
            boost_usd=3.0,
            boost_until=datetime.now(UTC) + timedelta(hours=5),
        )

    async def no_hit(user_id: int, kind: str) -> None: ...

    monkeypatch.setattr(L, "budget_row", row)
    monkeypatch.setattr(L, "_note_hit", no_hit)


async def test_over_global_but_under_boost_is_not_blocked(boosted: None) -> None:
    session = _UsageSession(4_050_000, 4.0)  # global 4M dan oshgan, 5M dan kam
    usage = await L.daily_usage(session, 1)  # type: ignore[arg-type]
    assert (usage.token_limit, usage.cost_limit) == (5_000_000, 13.0)
    assert await L.check_daily_budget(session, 1) is None  # type: ignore[arg-type]
    assert (await L.assert_within_budget(session, 1)).exceeded is None  # type: ignore[arg-type]
    # Boost'siz foydalanuvchi o'sha sarf bilan to'siladi.
    assert await L.check_daily_budget(session, 2) == "tokens"  # type: ignore[arg-type]
    from app.services.chat_service import ChatError

    with pytest.raises(ChatError, match="budget"):
        await L.assert_within_budget(session, 2)  # type: ignore[arg-type]


async def test_blocked_user_ids_respects_overrides_and_boost(
    glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = {
        1: L.BudgetRow(boost_tokens=1_000_000, boost_until=datetime.now(UTC) + timedelta(hours=1)),
        2: L.BudgetRow(token_budget=100),
    }

    async def load(session: Any) -> dict[int, L.BudgetRow]:
        return rows

    class _S:
        async def execute(self, stmt: Any) -> Any:
            data = [(1, 4_500_000, 1.0), (2, 150, 0.0), (3, 4_500_000, 1.0)]
            return SimpleNamespace(all=lambda: data)

    monkeypatch.setattr(L, "_load_rows", load)
    assert await L.blocked_user_ids(_S()) == {2, 3}  # type: ignore[arg-type]

    # Global cheksiz, lekin override bor — override egasi baribir tekshiriladi.
    monkeypatch.setattr(glob, "llm_daily_token_budget", 0)
    monkeypatch.setattr(glob, "llm_daily_cost_budget_usd", 0.0)
    assert await L.blocked_user_ids(_S()) == {2}  # type: ignore[arg-type]


def test_no_direct_global_budget_comparisons_outside_limits() -> None:
    """Global sozlama faqat `limits.compute_budget` da o'qiladi."""
    src = ROOT / "src/app"
    hits = []
    for p in src.rglob("*.py"):
        if p.name in {"limits.py", "config.py"}:
            continue
        text = p.read_text(encoding="utf-8")
        for name in (".llm_daily_token_budget", ".llm_daily_cost_budget_usd"):
            if name in text:
                hits.append(f"{p.relative_to(src)}: {name}")
    assert not hits, hits


# ─── API ─────────────────────────────────────────────────────────────────────


class _Scope:
    def __init__(self, db: Any) -> None:
        self.db = db

    async def __aenter__(self) -> Any:
        return self.db

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, dict[str, Any]]]:
    from app.web.routers import settings as R

    calls: dict[str, Any] = {"set": [], "boost": [], "clear": 0, "invalidate": 0}
    account = SimpleNamespace(id=3, user_id=7, tz="Asia/Tashkent")

    class _DB:
        async def get(self, model: Any, pk: int) -> Any:
            return account if getattr(model, "__name__", "") == "Account" and pk == 3 else None

        async def execute(self, stmt: Any) -> Any:
            return SimpleNamespace(scalar_one_or_none=lambda: "Europe/Moscow")

    async def view(db: Any, user_id: int, *, tz: str | None) -> dict[str, Any]:
        return {"user_id": user_id, "tz": tz}

    async def set_budget(db: Any, user_id: int, **kw: Any) -> None:
        calls["set"].append(kw)

    async def boost(db: Any, user_id: int, **kw: Any) -> None:
        calls["boost"].append(kw)

    async def clear(db: Any, user_id: int) -> None:
        calls["clear"] += 1

    async def inval(user_id: int) -> None:
        calls["invalidate"] += 1

    monkeypatch.setattr(R, "session_scope", lambda: _Scope(_DB()))
    monkeypatch.setattr(R.LIM, "budget_view", view)
    monkeypatch.setattr(R.LIM, "set_user_budget", set_budget)
    monkeypatch.setattr(R.LIM, "add_boost", boost)
    monkeypatch.setattr(R.LIM, "clear_boost", clear)
    monkeypatch.setattr(R.LIM, "invalidate", inval)
    with TestClient(web_main.app) as c:
        c.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        yield c, calls


def test_budget_api_requires_login_and_csrf(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, calls = api
    assert c.put("/api/settings/budget", json={"token_budget": 5}).status_code == 403
    assert c.post("/api/settings/budget/boost", json={"tokens": 5}).status_code == 403
    assert c.delete("/api/settings/budget/boost").status_code == 403
    c.cookies.clear()
    assert c.get("/api/settings/budget").status_code == 401
    assert c.put("/api/settings/budget", json={"token_budget": 5}, headers=H).status_code == 401
    assert c.post("/api/settings/budget/boost", json={"tokens": 5}, headers=H).status_code == 401
    assert c.delete("/api/settings/budget/boost", headers=H).status_code == 401
    assert not calls["set"] and not calls["boost"] and not calls["clear"]


def test_budget_get_uses_account_tz(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, _ = api
    r = c.get("/api/settings/budget?account_id=3")
    assert r.status_code == 200 and r.json() == {"user_id": 7, "tz": "Asia/Tashkent"}
    # account_id berilmasa — birinchi akkauntning zonasi.
    assert c.get("/api/settings/budget").json()["tz"] == "Europe/Moscow"
    # Begona akkaunt — 404.
    assert c.get("/api/settings/budget?account_id=99").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"token_budget": -1},
        {"token_budget": 1_000_000_001},
        {"cost_budget_usd": -0.01},
        {"cost_budget_usd": 1000.01},
    ],
)
def test_budget_put_validation(api: tuple[TestClient, dict[str, Any]], body: Any) -> None:
    c, calls = api
    r = c.put("/api/settings/budget", json=body, headers=H)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "web.budget.err_invalid"
    assert not calls["set"]


def test_budget_put_passes_only_given_fields(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, calls = api
    for body in (
        {"token_budget": 6_000_000},
        {"cost_budget_usd": None},
        {"token_budget": None, "cost_budget_usd": 12.5},
    ):
        assert c.put("/api/settings/budget", json=body, headers=H).status_code == 200
    assert calls["set"] == [
        {"token_budget": 6_000_000},
        {"cost_budget_usd": None},
        {"token_budget": None, "cost_budget_usd": 12.5},
    ]
    assert calls["invalidate"] == 3  # commit'dan keyin kesh qayta o'chiriladi


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({}, "web.budget.err_boost_empty"),
        ({"tokens": 0, "usd": 0}, "web.budget.err_boost_empty"),
        ({"tokens": -5}, "web.budget.err_invalid"),
        ({"usd": 1001}, "web.budget.err_invalid"),
        ({"tokens": 5, "hours": 0}, "web.budget.err_invalid"),
        ({"tokens": 5, "hours": 49}, "web.budget.err_invalid"),
    ],
)
def test_boost_validation(api: tuple[TestClient, dict[str, Any]], body: Any, code: str) -> None:
    c, calls = api
    r = c.post("/api/settings/budget/boost", json=body, headers=H)
    assert r.status_code == 400 and r.json()["detail"]["code"] == code
    assert not calls["boost"]


def test_boost_defaults_to_local_midnight(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, calls = api
    r = c.post(
        "/api/settings/budget/boost",
        json={"tokens": 1_000_000, "usd": 3, "account_id": 3},
        headers=H,
    )
    assert r.status_code == 200
    kw = calls["boost"][0]
    assert kw["tokens"] == 1_000_000 and kw["usd"] == 3.0
    assert kw["until"] == L.next_local_midnight("Asia/Tashkent")

    before = datetime.now(UTC)
    c.post("/api/settings/budget/boost", json={"usd": 1.5, "hours": 2}, headers=H)
    until = calls["boost"][1]["until"]
    assert timedelta(hours=1.99) < until - before < timedelta(hours=2.01)


def test_boost_delete(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, calls = api
    r = c.delete("/api/settings/budget/boost?account_id=3", headers=H)
    assert r.status_code == 200 and calls["clear"] == 1


async def test_budget_view_shape(glob: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    until = datetime.now(UTC) + timedelta(hours=3)

    async def row(session: Any, user_id: int) -> L.BudgetRow:
        return L.BudgetRow(token_budget=3_000_000, boost_tokens=1_000_000, boost_until=until)

    async def hit(user_id: int) -> dict[str, Any]:
        return {"at": "2026-09-25T10:00:00+00:00", "kind": "tokens"}

    async def resets(session: Any, user_id: int, usage: Any) -> datetime | None:
        return None

    monkeypatch.setattr(L, "budget_row", row)
    monkeypatch.setattr(L, "last_hit", hit)
    monkeypatch.setattr(L, "_resets_at", resets)
    out = await L.budget_view(_UsageSession(2_000_000, 1.5), 1, tz="Asia/Tashkent")  # type: ignore[arg-type]
    assert out["usage_24h"] == {"tokens": 2_000_000, "usd": 1.5, "requests": 3}
    lim = out["limits"]
    assert (lim["tokens"], lim["usd"], lim["source"]) == (4_000_000, 10.0, "custom")
    assert lim["global"] == {"tokens": 4_000_000, "usd": 10.0}
    assert lim["custom"] == {"tokens": 3_000_000, "usd": None}
    assert lim["ceiling"] is None
    assert out["boost"] == {"tokens": 1_000_000, "usd": 0.0, "until": until.isoformat()}
    assert out["blocked"] is None and out["last_hit"]["kind"] == "tokens"
    assert out["resets_hint"] == {"window_hours": 24, "at": None}
    assert out["tz"] == "Asia/Tashkent" and out["boost_default_until"]


# ─── haqiqiy Postgres ────────────────────────────────────────────────────────


async def test_budget_roundtrip_on_postgres(
    pg_engine: Any, glob: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from test_db_integration import _fixture_chat

    async def no_cache() -> Any:
        raise ConnectionError("no redis in this test")

    monkeypatch.setattr(L, "_redis", no_cache)
    account_id, _ = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        uid = int(
            (
                await conn.execute(
                    text("SELECT user_id FROM accounts WHERE id = :a"), {"a": account_id}
                )
            ).scalar_one()
        )
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as s:
        assert (await L.effective_budget(s, uid)).source == "global"
        await L.set_user_budget(s, uid, token_budget=6_000_000)
        await s.commit()
    async with maker() as s:
        b = await L.effective_budget(s, uid)
        assert (b.tokens, b.usd, b.source) == (6_000_000, 10.0, "custom")
        until = datetime.now(UTC) + timedelta(hours=2)
        await L.add_boost(s, uid, tokens=1_000_000, usd=3.0, until=until)
        await L.add_boost(s, uid, tokens=500_000, usd=0.0, until=until - timedelta(hours=1))
        await s.commit()
    async with maker() as s:
        b = await L.effective_budget(s, uid)
        # Faol boost ustiga qo'shildi, muddat — kechrog'i.
        assert (b.tokens, b.usd, b.boost_until) == (7_500_000, 13.0, until)
        await L.set_user_budget(s, uid, token_budget=None)
        await L.clear_boost(s, uid)
        await s.commit()
    async with maker() as s:
        b = await L.effective_budget(s, uid)
        assert (b.tokens, b.usd, b.source, b.boost_active) == (4_000_000, 10.0, "global", False)
        view = await L.budget_view(s, uid, tz="Asia/Tashkent")
        assert view["limits"]["tokens"] == 4_000_000 and view["boost"] is None
