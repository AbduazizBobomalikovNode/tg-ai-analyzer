"""Hardening pass regressiyalari.

Har test bitta aniq zaiflik/isrof uchun: logout server tomonda bekor qiladimi,
`X-Forwarded-For` ishonchsizmi, log maskalash rekursivmi, `/metrics` doimiy
vaqtda taqqoslaydimi, shift lug'ati o'sadimi, QR keshlanadimi, sync `COUNT(*)`
ni takrorlaydimi, backfill "tugadi" ni byudjetdan xulosa qiladimi, ARQ eski
natija tufayli "already_running" deb yolg'on gapiradimi, judge/rasm tokenlari
byudjetga tushadimi, yozish tool'i qopqoq ostidami, worker metrikalari
ko'tariladimi, snapshot qopqoqlimi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.web import security as sec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ─── 1. logout → server tomonda bekor qilish ─────────────────────────────────


class FakeRedis:
    """Minimal `redis.asyncio` o'rnini bosuvchi (set/get/aclose)."""

    def __init__(self, *, fail: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.fail = fail
        self.closed = 0

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise ConnectionError("redis yo'q")
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise ConnectionError("redis yo'q")
        return self.data.get(key)

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    r = FakeRedis()

    async def _get() -> FakeRedis:
        return r

    monkeypatch.setattr(sec, "_redis", _get)
    return r


def test_session_cookie_carries_version() -> None:
    """Cookie'da `v` da'vosi bor — kelajakda global rotatsiya migratsiyasiz."""
    from itsdangerous import URLSafeSerializer

    token = sec.issue_session(42)
    raw = sec._serializer().loads(token)
    assert raw["v"] == sec.SESSION_VERSION
    # boshqa versiyali (eski yoki kelajakdagi) cookie qabul qilinmaydi
    ser: URLSafeSerializer = sec._serializer()
    stale = ser.dumps({"uid": 42, "iat": int(time.time()), "v": sec.SESSION_VERSION + 1})
    assert sec.read_session(stale) is None
    legacy = ser.dumps({"uid": 42, "iat": int(time.time())})  # `v` dan oldingi format
    assert sec.read_session(legacy) is None


async def test_revoked_session_is_refused(fake_redis: FakeRedis) -> None:
    token = sec.issue_session(7)
    ident = sec.read_session(token)
    assert ident is not None
    assert await sec.is_revoked(ident) is False

    # logout: chegara "hozir" — undan oldin berilgan hamma cookie rad etiladi
    assert await sec.revoke_sessions(7) is True
    assert fake_redis.data[sec.REVOKE_PREFIX + "7"]
    assert await sec.is_revoked(ident) is True
    # boshqa foydalanuvchiga ta'sir qilmaydi
    other = sec.read_session(sec.issue_session(8))
    assert other is not None and await sec.is_revoked(other) is False
    # logout'dan **keyin** berilgan cookie yashaydi
    fresh = sec.WebIdentity(user_id=7, issued_at=int(time.time()) + 5)
    assert await sec.is_revoked(fresh) is False


async def test_revocation_fails_open_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis yo'q — web ishlashda davom etadi (hujjatlashtirilgan fail-open)."""

    async def _boom() -> Any:
        raise ConnectionError("no redis")

    monkeypatch.setattr(sec, "_redis", _boom)
    ident = sec.read_session(sec.issue_session(3))
    assert ident is not None
    assert await sec.is_revoked(ident) is False
    assert await sec.revoke_sessions(3) is False


@pytest.fixture
def app_client(fake_redis: FakeRedis) -> Any:
    from fastapi.testclient import TestClient

    from app.web import main as web_main

    with TestClient(web_main.app) as c:
        yield c


async def test_logout_refuses_the_old_cookie(app_client: Any, fake_redis: FakeRedis) -> None:
    """Asosiy regressiya: ilgari logout faqat cookie'ni o'chirardi — nusxasi
    qolgan bo'lsa 14 kun ishlaydigan kalit edi."""
    token = sec.issue_session(7)
    app_client.cookies.set(sec.SESSION_COOKIE, token)
    r = app_client.post("/api/auth/logout", headers={"X-Requested-With": "fetch"})
    assert r.status_code == 200 and r.json()["revoked"] == "true"

    # "o'g'irlangan" nusxani qo'lda qaytaramiz — server endi qabul qilmaydi
    app_client.cookies.set(sec.SESSION_COOKIE, token)
    assert app_client.get("/api/me").status_code == 401
    # sahifa ham login'ga qaytaradi
    r = app_client.get("/chat", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"


# ─── 4. /metrics tokenini taqqoslash ─────────────────────────────────────────


def test_metrics_token_is_compared_in_constant_time(
    app_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hmac as hmac_mod

    from app.web import main as web_main

    seen: list[tuple[bytes, bytes]] = []
    real = hmac_mod.compare_digest

    def spy(a: Any, b: Any) -> bool:
        seen.append((a, b))
        return bool(real(a, b))

    monkeypatch.setattr(web_main.hmac, "compare_digest", spy)
    monkeypatch.setattr(web_main.s, "metrics_token", "s3cret-token")

    assert app_client.get("/metrics?token=wrong").status_code == 401
    r = app_client.get("/metrics", headers={"Authorization": "Bearer s3cret-token"})
    assert r.status_code == 200 and b"tgai_" in r.content
    assert len(seen) == 2  # `!=` emas, aynan compare_digest


def test_healthz_is_open_and_trivial(app_client: Any) -> None:
    r = app_client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


# ─── 2. client_ip / TRUSTED_PROXIES ──────────────────────────────────────────


def _req(peer: str, xff: str | None = None) -> Any:
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


@pytest.fixture
def with_proxies(monkeypatch: pytest.MonkeyPatch) -> Any:
    """`TRUSTED_PROXIES` ni **keshni tozalamasdan** o'zgartiradi: `get_settings`
    lru_cache'ini tozalash boshqa modullarda ushlab turilgan `Settings`
    havolasini eskirtiradi (`web.main.s`)."""
    from app.config import get_settings

    def _set(value: str) -> None:
        monkeypatch.setattr(get_settings(), "trusted_proxies", value)

    return _set


def test_client_ip_ignores_xff_by_default(with_proxies: Any) -> None:
    """Default (bo'sh ro'yxat) — header umuman o'qilmaydi, aks holda shift soxta."""
    with_proxies("")
    assert sec.client_ip(_req("203.0.113.9", "1.2.3.4")) == "203.0.113.9"


def test_client_ip_uses_rightmost_untrusted_hop(with_proxies: Any) -> None:
    with_proxies("10.0.0.0/8")
    # Mijoz o'zi "1.1.1.1" yozib yuborgan, undan keyin ikkita ishonchli proxy
    req = _req("10.0.0.5", "1.1.1.1, 198.51.100.7, 10.0.0.9")
    assert sec.client_ip(req) == "198.51.100.7"
    # Ishonchsiz peer — header butunlay e'tiborsiz
    assert sec.client_ip(_req("203.0.113.1", "1.1.1.1")) == "203.0.113.1"
    # Hamma hop ishonchli — peer'ning o'zi
    assert sec.client_ip(_req("10.0.0.5", "10.0.0.9")) == "10.0.0.5"


# ─── 3. log maskalash ────────────────────────────────────────────────────────


def test_redact_is_recursive_and_matches_key_substrings() -> None:
    from app.logging import MASK, _redact

    out = _redact(
        None,
        "info",
        {
            "event": "auth.done",
            "user": {"api_key": "AIza" + "b" * 30, "nested": [{"Refresh-Token": "abc"}]},
            "status_code": 200,
            "code": "12345",
        },
    )
    assert out["user"]["api_key"] == MASK
    assert out["user"]["nested"][0]["Refresh-Token"] == MASK
    assert out["code"] == MASK
    assert out["status_code"] == 200  # diagnostikani yo'qotmaymiz
    assert out["event"] == "auth.done"


def test_redact_scrubs_secret_shapes_in_free_text() -> None:
    from app.logging import MASK, _redact

    session_string = "1" + "A" * 300
    text = (
        f"failed to connect with {session_string} and bot 123456789:"
        f"{'A' * 35} plus sk-{'x' * 32} and AIza{'y' * 30}"
    )
    out = _redact(None, "warning", {"event": "x", "error": text})
    scrubbed = out["error"]
    assert session_string not in scrubbed
    assert "123456789:" not in scrubbed
    assert "sk-" not in scrubbed
    assert "AIza" not in scrubbed
    assert scrubbed.count(MASK) == 4


def test_sentry_scrubber_shares_one_implementation() -> None:
    from app.logging import MASK
    from app.observability import _scrub

    ev = _scrub({"extra": {"session_string": "s3cret", "ok": 1}}, None)
    assert ev is not None and ev["extra"]["session_string"] == MASK and ev["extra"]["ok"] == 1


# ─── 5. RateLimiter xotirasi ─────────────────────────────────────────────────


def test_rate_limiter_evicts_stale_keys() -> None:
    rl = sec.RateLimiter(5, 60)
    base = time.monotonic()
    # 400 xil IP, hammasi oynadan eski
    for i in range(400):
        rl.check(f"10.0.0.{i}")
    for v in rl._hits.values():  # ularni "eski" qilamiz
        v[:] = [base - 1000]
    before = len(rl._hits)
    for _ in range(sec.RateLimiter._SWEEP_EVERY):
        rl.check("fresh")
    assert len(rl._hits) < before
    assert len(rl._hits) == 1 and "fresh" in rl._hits


# ─── 6. QR keshi ─────────────────────────────────────────────────────────────


def test_qr_data_uri_cached_per_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.web.routers import auth as auth_router

    calls: list[str] = []

    def fake_render(url: str) -> str:
        calls.append(url)
        return f"data:{url}"

    monkeypatch.setattr(auth_router, "_qr_data_uri", fake_render)
    auth_router._QR_CACHE.clear()
    flow = SimpleNamespace(id="f1")
    a = auth_router.qr_data_uri_cached(flow.id, "tg://login?token=A")
    b = auth_router.qr_data_uri_cached(flow.id, "tg://login?token=A")
    assert a == b and calls == ["tg://login?token=A"]  # ikkinchi poll chizmaydi
    auth_router.qr_data_uri_cached(flow.id, "tg://login?token=B")  # token yangilandi
    assert len(calls) == 2
    auth_router._QR_CACHE.clear()


# ─── 7/8. ingestion: COUNT va backfill tugashi ───────────────────────────────


class _FakeIter:
    """`client.iter_messages` o'rnini bosadi (`total` atributi bilan)."""

    def __init__(self, messages: list[Any], total: int | None = None) -> None:
        self._messages = messages
        self.total = total

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for m in self._messages:
                yield m

        return gen()


def _msg(i: int) -> Any:
    return SimpleNamespace(
        id=i,
        message=f"m{i}",
        date=datetime(2026, 1, 1, tzinfo=UTC),
        media=None,
        entities=None,
        out=False,
    )


@dataclass(slots=True)
class _PullEnv:
    counts: list[int]
    progress: list[int | None]


@pytest.fixture
def pull_env(monkeypatch: pytest.MonkeyPatch) -> _PullEnv:
    from app.services import ingestion as ing

    env = _PullEnv(counts=[], progress=[])

    async def fake_count(chat_id: int) -> int:
        env.counts.append(chat_id)
        return 1000

    async def fake_upsert(rows: list[Any], **kw: Any) -> int:
        return len(rows)

    async def fake_progress(
        chat_id: int, report: Any, total: int | None, *, synced_total: int | None = None
    ) -> None:
        env.progress.append(synced_total)

    monkeypatch.setattr(ing, "_message_count", fake_count)
    monkeypatch.setattr(ing, "upsert_messages", fake_upsert)
    monkeypatch.setattr(ing, "_progress", fake_progress)
    return env


async def test_pull_counts_once_and_tracks_running_total(pull_env: _PullEnv) -> None:
    """`_progress` har batchda chaqiriladi, `COUNT(*)` esa bitta."""
    from app.services import ingestion as ing

    msgs = [_msg(i) for i in range(1, 251)]
    client = SimpleNamespace(iter_messages=lambda *a, **k: _FakeIter(msgs, total=250))
    report = ing.SyncReport(chat_id=1)
    fetched = await ing._pull(client, object(), 1, report, 250)

    assert fetched == 250
    assert pull_env.counts == [1]  # bitta COUNT(*) — batch boshiga emas
    # 2 ta to'la batch + yakuniy yozuv; hisoblagich yuguradi
    assert pull_env.progress == [1100, 1200, 1250]


@pytest.mark.parametrize(
    ("n_messages", "budget", "expect_done"),
    [
        (250, 250, False),  # byudjet tugadi — tarix tugagani noma'lum
        (100, 100, False),  # aynan bitta to'la batch = byudjet → hali bilmaymiz
        (230, 500, True),  # kalta batch keldi — rostdan tarix boshi
        (0, 500, True),  # umuman xabar yo'q
    ],
)
async def test_backfill_done_only_on_short_batch(
    pull_env: _PullEnv, n_messages: int, budget: int, expect_done: bool
) -> None:
    """`done` → `_finish` da `synced_min_id = 1` sentineli. Byudjet tugashi
    hech qachon "tarix tugadi" degani emas — aks holda qolgani abadiy yo'qoladi."""
    from app.services import ingestion as ing

    msgs = [_msg(i) for i in range(1, n_messages + 1)]
    client = SimpleNamespace(iter_messages=lambda *a, **k: _FakeIter(msgs))
    report = ing.SyncReport(chat_id=1)
    await ing._pull(client, object(), 1, report, budget, backfill=True)
    assert report.done is expect_done


# ─── 9. ARQ eski natija ──────────────────────────────────────────────────────


class _FakePool:
    """`ArqRedis` o'rnini bosadi: `job_id` band bo'lsa `enqueue_job` None qaytaradi."""

    def __init__(self, taken: set[str]) -> None:
        self.taken = set(taken)
        self.deleted: list[str] = []
        self.enqueues = 0

    async def enqueue_job(self, function: str, *args: Any, _job_id: str, **kw: Any) -> Any:
        self.enqueues += 1
        if _job_id in self.taken:
            return None
        return SimpleNamespace(job_id=_job_id)

    async def delete(self, key: str) -> int:
        self.deleted.append(key)
        self.taken.discard(key.removeprefix("arq:result:"))
        return 1


def _patch_status(monkeypatch: pytest.MonkeyPatch, status: Any) -> None:
    import arq.jobs as arq_jobs

    class _Job:
        def __init__(self, job_id: str, redis: Any = None, **kw: Any) -> None:
            self.job_id = job_id

        async def status(self) -> Any:
            return status

    monkeypatch.setattr(arq_jobs, "Job", _Job)


async def test_enqueue_clears_stale_result_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tugagan ishning natijasi `job_id` ni band qilib turardi → UI yolg'on
    "already_running" ko'rsatardi. Endi iz tozalanadi va ish navbatga tushadi."""
    from arq.jobs import JobStatus

    from app.worker import queue as Q

    pool = _FakePool({"sync:acc:5"})
    monkeypatch.setattr(Q, "get_queue", lambda: _coro(pool))
    _patch_status(monkeypatch, JobStatus.complete)

    got = await Q.enqueue("sync_account", 5, True, job_id="sync:acc:5")
    assert got == "sync:acc:5"
    assert pool.deleted == ["arq:result:sync:acc:5"]
    assert pool.enqueues == 2  # birinchi urinish + tozalashdan keyingisi


async def test_enqueue_keeps_dedup_for_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rostdan yugurayotgan ish — tegilmaydi (bir akkauntga parallel sync yo'q)."""
    from arq.jobs import JobStatus

    from app.worker import queue as Q

    pool = _FakePool({"sync:acc:5"})
    monkeypatch.setattr(Q, "get_queue", lambda: _coro(pool))
    _patch_status(monkeypatch, JobStatus.in_progress)

    assert await Q.enqueue("sync_account", 5, True, job_id="sync:acc:5") is None
    assert pool.deleted == [] and pool.enqueues == 1


def _coro(value: Any) -> Any:
    async def _inner() -> Any:
        return value

    return _inner()


# ─── 10. judge va rasm tokenlari byudjetga tushadi ───────────────────────────


class _UsageSession:
    """`session_scope()` ichidagi AsyncSession o'rnini bosadi."""

    def __init__(self, existing_conv: Any = None, src: Any = None) -> None:
        self.added: list[Any] = []
        self.existing = existing_conv
        self.src = src

    async def get(self, model: Any, pk: Any) -> Any:
        return self.src

    async def execute(self, *a: Any, **k: Any) -> Any:
        return SimpleNamespace(scalar_one_or_none=lambda: self.existing)

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)

    async def flush(self) -> None: ...


async def test_judge_tokens_are_persisted_for_daily_budget() -> None:
    """Judge har javobdan keyin yuguradi — tokeni saqlanmasa kunlik byudjet
    uni ko'rmaydi (`limits.daily_usage` faqat `conversation_messages` ni o'qiydi)."""
    from app.db.models import Conversation
    from app.services import evaluation as E

    src = Conversation(user_id=7, account_id=3, title="Suhbat")
    src.id = 11
    db = _UsageSession(src=src)
    row = SimpleNamespace(conversation_id=11)
    j = E.Judgement(4, 5, True, "yaxshi", tokens_in=900, tokens_out=60, model="claude-haiku-4-5")
    j.provider = "claude"
    await E._record_judge_usage(db, row, j)  # type: ignore[arg-type]

    convs = [o for o in db.added if type(o).__name__ == "Conversation"]
    msgs = [o for o in db.added if type(o).__name__ == "ConversationMessage"]
    assert convs and convs[0].title == E.JUDGE_CONVERSATION and convs[0].user_id == 7
    assert len(msgs) == 1
    m = msgs[0]
    assert m.role == "assistant" and m.task == "judge"
    assert (m.tokens_in, m.tokens_out) == (900, 60)
    assert m.cost_usd and m.cost_usd > 0
    # baholanayotgan suhbatga emas — transkriptga chiqmaydi
    assert m.conversation_id != 11


async def test_judge_usage_skipped_without_tokens() -> None:
    from app.services import evaluation as E

    db = _UsageSession()
    j = E.Judgement(4, 5, True, "x")
    await E._record_judge_usage(db, SimpleNamespace(conversation_id=1), j)  # type: ignore[arg-type]
    assert db.added == []


def test_image_model_has_a_price() -> None:
    """Narx jadvalida bo'lmasa `estimate_cost` None qaytaradi va kunlik
    xarajat limiti jimgina ishlamay qoladi."""
    from app.llm.pricing import PRICES, estimate_cost

    assert "gemini-2.5-flash-image" in PRICES
    cost = estimate_cost("gemini", "gemini-2.5-flash-image", 50, 1290)
    assert cost is not None and 0.03 < cost < 0.05  # ≈ $0.039 / rasm


async def test_image_generation_records_usage() -> None:
    from app.services import images as IMG

    db = _UsageSession()
    await IMG._record_image_usage(
        db,  # type: ignore[arg-type]
        user_id=7,
        account_id=3,
        prompt="a cat on a roof",
        provider="gemini",
        model="gemini-2.5-flash-image",
    )
    msgs = [o for o in db.added if type(o).__name__ == "ConversationMessage"]
    assert len(msgs) == 1 and msgs[0].task == "image"
    assert msgs[0].tokens_out == IMG.IMAGE_OUTPUT_TOKENS
    assert msgs[0].cost_usd and msgs[0].cost_usd > 0


# ─── 11. yozish tool'i ham MAX_CALLS_PER_TOOL ostida ─────────────────────────


async def test_write_tools_share_the_per_tool_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ilgari hisob faqat o'qish shoxida oshardi: model bitta raundda
    cheksiz taklif yasay olardi (har biri DB qatori + tasdiq so'rovi)."""
    from app.llm import ChatResult, Msg, Task, ToolCall, Usage
    from app.services import agent as A

    proposed: list[dict[str, Any]] = []

    async def fake_propose(session: Any, ctx: Any, *, run_id: int, tool: str, args: Any) -> Any:
        proposed.append({"tool": tool, "args": args})
        return A.T.ToolResult("proposed", meta={"action_id": len(proposed), "proposed": True})

    async def writable(session: Any, account_id: int) -> bool:
        return True

    monkeypatch.setattr(A.W, "propose_or_execute", fake_propose)
    monkeypatch.setattr(A.W, "writable_chats_exist", writable)

    name = next(iter(A.W.WRITE_TOOL_NAMES))
    calls = [ToolCall(f"c{i}", name, {"text": f"msg {i}"}) for i in range(4)]

    class _Session:
        def __init__(self) -> None:
            self.added: list[Any] = []

        def add(self, obj: Any) -> None:
            self.added.append(obj)
            if getattr(obj, "id", None) is None:
                obj.id = len(self.added)

        async def flush(self) -> None: ...

        async def execute(self, *a: Any, **k: Any) -> Any:
            return SimpleNamespace(
                first=lambda: (1,), scalars=lambda: SimpleNamespace(all=lambda: [])
            )

    class _LLM:
        def __init__(self, script: list[Any]) -> None:
            self.script = list(script)
            self.calls: list[Any] = []

        async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> Any:
            self.calls.append(list(messages))
            return self.script.pop(0)

    def _res(text: str = "", tcalls: list[ToolCall] | None = None) -> ChatResult:
        return ChatResult(
            text=text,
            tool_calls=tcalls or [],
            model="m",
            provider="p",
            usage=Usage(10, 2),
            finish_reason="tool_use" if tcalls else "end_turn",
        )

    llm = _LLM([_res(tcalls=calls), _res("ok")])
    await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="en",
        llm=llm,  # type: ignore[arg-type]
    )
    assert len(proposed) == A.MAX_CALLS_PER_TOOL  # 4-si to'silgan
    tool_msgs = [m for m in llm.calls[1] if m.role.value == "tool"]
    assert any("already called" in m.content for m in tool_msgs)


# ─── 12. list_scheduled_messages shift ostida ────────────────────────────────


async def test_list_scheduled_messages_is_rate_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Har chaqiruv real MTProto so'rovi — `search_*` bilan bitta kvotada."""
    from app.services import tools as T

    seen: list[tuple[int, str]] = []

    def fake_guard(account_id: int, bucket: str) -> str | None:
        seen.append((account_id, bucket))
        return "error: rate limited"

    monkeypatch.setattr(T, "_tg_guard", fake_guard)
    ctx = SimpleNamespace(account_id=5, user_id=1, locale="uz", chat=None, llm=None)
    res = await T.list_scheduled_messages(ctx, {"chat": "C"})  # type: ignore[arg-type]
    assert not res.ok and "rate limited" in res.text
    assert seen == [(5, T.BUCKET_HISTORY)]


# ─── 13. worker metrikalari ──────────────────────────────────────────────────


def test_worker_metrics_server_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import prometheus_client

    from app import observability as O

    started: list[int] = []
    monkeypatch.setattr(prometheus_client, "start_http_server", lambda p: started.append(p))
    assert O.start_metrics_server(0) == 0
    assert started == []
    assert O.start_metrics_server(9101) == 9101
    assert started == [9101]


# ─── 15. snapshot qopqog'i ───────────────────────────────────────────────────


def test_snapshot_candidates_are_capped_and_oldest_first() -> None:
    from app.services import ingestion as ing

    stmt = ing.snapshot_candidates_stmt(
        1, datetime(2026, 1, 1, tzinfo=UTC), timedelta(0), timedelta(days=1), 5000
    )
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
    assert "limit 5000" in sql
    assert "order by" in sql and "last_at" in sql  # eng eski snapshot birinchi
