"""Sync markazi — Redis ko'rsatkichi, `/sync/status` shakli, SSE va frontend.

Qo'riqlaydi:
  * `services/sync_progress` — kalitlar, TTL, EMA tezlik, ETA, pauza;
    **Redis yiqilsa ingestion to'xtamaydi** (eng muhim xossa).
  * `settings._sync_status` — sanoqlar, `pending` mezoni (`synced_min_id == 1`
    sentineli), umumiy foiz, navbat/tayyor/xato ro'yxatlari, egalik.
  * SSE hodisalar tartibi (`status` → `done`).
  * `SYNC_MAX_JOBS` ↔ `WorkerSettings.max_jobs` pariteti — ikki nusxa
    jimgina ajralib ketmasin.
  * Frontend: `web.sync.*` kalitlari uchchala tilda, `innerHTML` yo'qligi,
    halqa/ETA matematikasi (node).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services import sync_progress as SP
from app.web import main as web_main
from app.web import security as sec

H = {"X-Requested-With": "fetch"}
ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/app/web/static"
SYNC_JS = STATIC / "lib/sync-center.js"
NODE = shutil.which("node")


# ═══ soxta Redis ═════════════════════════════════════════════════════════════


class FakeRedis:
    """`set`/`get`/`delete` — TTL yozib olinadi, tarmoq yo'q."""

    def __init__(self, *, fail: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}
        self.fail = fail
        self.sets = 0

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise ConnectionError("redis yo'q")
        self.sets += 1
        self.data[key] = value
        self.ttl[key] = ex

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise ConnectionError("redis yo'q")
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        if self.fail:
            raise ConnectionError("redis yo'q")
        self.data.pop(key, None)

    async def aclose(self) -> None: ...


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    r = FakeRedis()

    async def _redis() -> Any:
        return r

    monkeypatch.setattr(SP, "_redis", _redis)
    monkeypatch.setattr(SP, "_client", None)
    return r


# ═══ sync_progress ═══════════════════════════════════════════════════════════


def test_keys_are_stable() -> None:
    """Kalit nomlari shartnoma: worker yozadi, api o'qiydi."""
    assert SP.progress_key(7) == "sync:progress:7"
    assert SP.job_key(7) == "sync:job:7"


@pytest.mark.asyncio
async def test_progress_publish_roundtrip_with_ttl(fake_redis: FakeRedis) -> None:
    tracker = SP.ChatTracker(2, 31, "Ish chat", budget=50_000, started_at=1000.0)
    await tracker.publish(
        fetched=100, synced_total=100, total_estimate=1000, phase="new", now=1001.0
    )

    raw = fake_redis.data[SP.progress_key(2)]
    got = json.loads(raw)
    assert got["chat_id"] == 31 and got["title"] == "Ish chat"
    assert got["phase"] == "new" and got["fetched"] == 100
    assert got["synced_total"] == 100 and got["total_estimate"] == 1000
    assert got["job_budget_left"] == 49_900
    assert got["paused_until"] is None
    assert fake_redis.ttl[SP.progress_key(2)] == SP.PROGRESS_TTL == 600

    stored = await SP.read_progress(2)
    assert stored is not None and stored["chat_id"] == 31


@pytest.mark.asyncio
async def test_rate_is_an_ema_and_eta_follows_it(fake_redis: FakeRedis) -> None:
    """Birinchi batch — o'lchangan tezlik; keyingilari EMA bilan silliqlanadi."""
    tr = SP.ChatTracker(1, 5, "c", budget=10_000, started_at=0.0)
    first = tr.payload(fetched=100, synced_total=100, total_estimate=1000, now=1.0)
    assert first["rate_per_s"] == 100.0
    # 100 xabar 2 soniyada = 50/s; EMA alpha=1/3 → 100 + (50-100)/3 ≈ 83.33
    second = tr.payload(fetched=200, synced_total=200, total_estimate=1000, now=3.0)
    assert second["rate_per_s"] == pytest.approx(83.33, abs=0.01)
    # ETA: taxmindan qolgani 800 xabar / 83.33 ≈ 9.6 s (byudjet qolgani kattaroq)
    assert second["eta_s"] == pytest.approx(9.6, abs=0.1)


@pytest.mark.asyncio
async def test_eta_is_none_without_a_rate(fake_redis: FakeRedis) -> None:
    tr = SP.ChatTracker(1, 5, "c", budget=100, started_at=0.0)
    p = tr.payload(fetched=0, synced_total=0, total_estimate=None, now=0.0)
    assert p["rate_per_s"] is None and p["eta_s"] is None


@pytest.mark.asyncio
async def test_paused_keeps_the_last_numbers(fake_redis: FakeRedis) -> None:
    tr = SP.ChatTracker(3, 9, "kanal", budget=1000, started_at=0.0)
    tr.payload(fetched=100, synced_total=100, total_estimate=500, now=1.0)
    await tr.publish_paused(120)
    got = json.loads(fake_redis.data[SP.progress_key(3)])
    assert got["synced_total"] == 100  # raqamlar saqlanadi
    assert got["rate_per_s"] is None and got["eta_s"] is None  # pauzada tezlik yo'q
    until = datetime.fromisoformat(got["paused_until"])
    assert until > datetime.now(UTC)


@pytest.mark.asyncio
async def test_job_publish_merges_and_keeps_started_at(fake_redis: FakeRedis) -> None:
    await SP.publish_job(
        4, state=SP.RUNNING, started_at="2026-09-17T10:00:00+00:00", chats_total=12
    )
    merged = await SP.publish_job(4, chats_done=3)
    assert merged["state"] == SP.RUNNING  # oldingi holat saqlandi
    assert merged["started_at"] == "2026-09-17T10:00:00+00:00"
    assert merged["chats_total"] == 12 and merged["chats_done"] == 3


@pytest.mark.asyncio
async def test_redis_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eng muhim xossa: ko'rsatkich yiqilsa ham ingestion davom etadi."""
    broken = FakeRedis(fail=True)

    async def _redis() -> Any:
        return broken

    monkeypatch.setattr(SP, "_redis", _redis)
    tr = SP.ChatTracker(1, 1, "x", budget=10)
    await tr.publish(fetched=1, synced_total=1, total_estimate=10)  # xato ko'tarilmaydi
    await tr.publish_paused(5)
    assert await SP.read_progress(1) is None
    assert await SP.read_job(1) is None
    await SP.clear_progress(1)


@pytest.mark.asyncio
async def test_corrupt_payload_reads_as_none(fake_redis: FakeRedis) -> None:
    fake_redis.data[SP.progress_key(1)] = "{buzuq"
    assert await SP.read_progress(1) is None
    fake_redis.data[SP.progress_key(1)] = "[1, 2]"  # lug'at emas
    assert await SP.read_progress(1) is None


@pytest.mark.asyncio
async def test_read_jobs_skips_missing(fake_redis: FakeRedis) -> None:
    await SP.publish_job(1, state=SP.RUNNING)
    jobs = await SP.read_jobs([1, 2, 3])
    assert set(jobs) == {1}


@pytest.mark.asyncio
async def test_ingestion_progress_writes_to_redis(
    fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ingestion._progress` DB bilan bir vaqtda ko'rsatkichni ham chiqaradi."""
    from app.services import ingestion as ing

    chat = SimpleNamespace(
        synced_max_id=None, synced_min_id=None, total_estimate=None, synced_total=0
    )

    class _DB:
        async def get(self, model: Any, pk: int) -> Any:
            return chat

        async def execute(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(scalar_one=lambda: 0)

    monkeypatch.setattr(ing, "session_scope", lambda: _Scope(_DB()))

    report = ing.SyncReport(chat_id=31)
    report.tracker = SP.ChatTracker(2, 31, "Ish chat", budget=50_000, started_at=0.0)
    report.newest_id, report.oldest_id, report.fetched = 900, 800, 100

    await ing._progress(31, report, 1000, synced_total=100, phase="backfill", fetched=100)

    assert chat.synced_total == 100 and chat.total_estimate == 1000  # DB yozildi
    got = json.loads(fake_redis.data[SP.progress_key(2)])  # Redis ham
    assert got["phase"] == "backfill" and got["synced_total"] == 100
    assert got["title"] == "Ish chat" and got["job_budget_left"] == 49_900


@pytest.mark.asyncio
async def test_ingestion_progress_without_a_tracker_is_silent(
    fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tracker=None` (eski chaqiruv, test) — Redis'ga hech narsa yozilmaydi."""
    from app.services import ingestion as ing

    chat = SimpleNamespace(synced_min_id=None, total_estimate=None, synced_total=0)

    class _DB:
        async def get(self, model: Any, pk: int) -> Any:
            return chat

        async def execute(self, *a: Any, **kw: Any) -> Any:
            return SimpleNamespace(scalar_one=lambda: 0)

    monkeypatch.setattr(ing, "session_scope", lambda: _Scope(_DB()))
    report = ing.SyncReport(chat_id=31)
    await ing._progress(31, report, None, synced_total=5)
    assert fake_redis.sets == 0


# ═══ /api/accounts/{id}/sync/status ══════════════════════════════════════════


def _chat(
    cid: int,
    *,
    state: str = "idle",
    synced: int = 0,
    est: int | None = None,
    min_id: int | None = None,
    error: str | None = None,
    title: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        title=title or f"chat {cid}",
        sync_state=state,
        synced_total=synced,
        total_estimate=est,
        synced_min_id=min_id,
        sync_error=error,
        last_sync_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        last_message_at=datetime(2026, 9, 17, 9, 0, tzinfo=UTC),
        avatar_photo_id=None,
        avatar_mime=None,
        account_id=3,
    )


class _Scope:
    def __init__(self, db: Any) -> None:
        self.db = db

    async def __aenter__(self) -> Any:
        return self.db

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StatusDB:
    """`_sync_status` ichidagi so'rovlarni **tartib bo'yicha** qaytaradi.

    Soxta sessiya SQL'ni talqin qilmaydi — u faqat shakl testidir; mantiq
    (pending mezoni, foiz) Python tomonda hisoblanadi va shu yerda tekshiriladi.
    """

    def __init__(self, plan: list[Any], account: Any) -> None:
        self.plan = list(plan)
        self.account = account
        self.calls = 0

    async def get(self, model: Any, pk: int) -> Any:
        name = getattr(model, "__name__", "")
        if name == "User":
            return SimpleNamespace(id=7, prefs={})
        if name == "Account":
            return self.account if pk == 3 else None
        return None

    async def execute(self, *a: Any, **kw: Any) -> Any:
        rows = self.plan[self.calls] if self.calls < len(self.plan) else []
        self.calls += 1
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(
                all=lambda: rows, first=lambda: rows[0] if rows else None
            ),
            all=lambda: rows,
        )


@pytest.fixture
def status_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """`/sync/status` uchun klient: egalik rost, DB soxta, Redis bo'sh."""
    from app.web.routers import settings as settings_router

    account = SimpleNamespace(
        id=3, user_id=7, status="active", created_at=datetime(2020, 1, 1, tzinfo=UTC)
    )
    running = _chat(10, state="running", synced=500, est=1000, title="Ish chat")
    failed = _chat(11, state="failed", error="ChannelPrivateError", title="Yopiq kanal")
    plan = [
        [running, failed],  # running + failed qatorlar (eski `chats` kaliti)
        [
            ("done", 5, 4000, 5000),
            ("idle", 2, 0, 0),
            ("running", 1, 500, 1000),
            ("failed", 1, 0, 0),
        ],
        [3],  # pending soni
        [_chat(20, est=900, title="Navbat A"), _chat(21, title="Navbat B")],  # navbat
        [_chat(30, state="done", synced=1200, title="Tayyor")],  # yaqinda tayyor
    ]
    db = _StatusDB(plan, account)
    monkeypatch.setattr(settings_router, "session_scope", lambda: _Scope(db))
    with TestClient(web_main.app) as client:
        client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        client.db = db  # type: ignore[attr-defined]
        yield client


def test_sync_status_shape_and_counts(status_client: TestClient, fake_redis: FakeRedis) -> None:
    r = status_client.get("/api/accounts/3/sync/status")
    assert r.status_code == 200
    body = r.json()

    # ── eski shartnoma buzilmagan ──
    assert body["running"] is True
    assert body["chats"][0]["chat_id"] == 10 and body["chats"][0]["progress"] == 50
    assert body["poll_seconds"] > 0

    # ── sanoqlar ──
    c = body["counts"]
    assert c["done"] == 5 and c["running"] == 1 and c["failed"] == 1
    assert c["pending"] == 3 and c["paused"] == 0
    assert c["total"] == 9 and c["idle"] == 2

    # ── umumiy foiz: taxmini bor chatlar bo'yicha 4500/6000 ──
    assert body["percent_overall"] == 75

    # ── ro'yxatlar ──
    assert [q["chat_id"] for q in body["queue"]] == [20, 21]
    assert body["queue"][0]["total_estimate"] == 900 and "avatar_url" in body["queue"][0]
    assert body["recent_done"] == [
        {
            "chat_id": 30,
            "title": "Tayyor",
            "synced_total": 1200,
            "finished_at": "2026-09-17T10:00:00+00:00",
        }
    ]
    assert body["failed"] == [
        {"chat_id": 11, "title": "Yopiq kanal", "error": "ChannelPrivateError"}
    ]

    # ── parallel va ramp ──
    assert body["parallel"]["max_jobs"] == 4
    assert body["ramp"]["limited"] is False  # akkaunt eski


def test_sync_status_reports_ramp_up_for_a_fresh_account(
    status_client: TestClient, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_client.db.account.created_at = datetime.now(UTC) - timedelta(hours=1)  # type: ignore[attr-defined]
    body = status_client.get("/api/accounts/3/sync/status").json()
    assert body["ramp"]["limited"] is True
    assert body["ramp"]["note"] == "web.sync.ramp_note"
    assert body["ramp"]["max_messages"] == 1000 and body["ramp"]["max_chats"] == 20


def test_sync_status_surfaces_the_current_chat_and_pause(
    status_client: TestClient, fake_redis: FakeRedis
) -> None:
    tr = SP.ChatTracker(3, 10, "Ish chat", budget=50_000, started_at=0.0)
    payload = tr.payload(
        fetched=100, synced_total=500, total_estimate=1000, phase="backfill", now=1.0
    )
    payload["paused_until"] = (datetime.now(UTC) + timedelta(seconds=300)).isoformat()
    fake_redis.data[SP.progress_key(3)] = json.dumps(payload)
    fake_redis.data[SP.job_key(3)] = json.dumps(
        {"state": SP.PAUSED, "chats_done": 5, "chats_total": 9}
    )

    body = status_client.get("/api/accounts/3/sync/status").json()
    assert body["current"]["chat_id"] == 10 and body["current"]["phase"] == "backfill"
    assert body["current"]["paused_until"] is not None
    assert body["counts"]["paused"] == 1
    assert body["job"]["state"] == "paused" and body["job"]["chats_done"] == 5
    assert body["running"] is True


def test_expired_pause_is_not_reported(
    status_client: TestClient, fake_redis: FakeRedis
) -> None:
    """TTL 10 daqiqa, FloodWait 30 soniya — eski kalit "pauzada" demasin."""
    fake_redis.data[SP.progress_key(3)] = json.dumps(
        {"chat_id": 10, "title": "x", "paused_until": SP.now_iso(0.0)}  # 1970
    )
    body = status_client.get("/api/accounts/3/sync/status").json()
    assert body["current"]["paused_until"] is None
    assert body["counts"]["paused"] == 0


def test_sync_status_ownership(status_client: TestClient, fake_redis: FakeRedis) -> None:
    assert status_client.get("/api/accounts/4/sync/status").status_code == 404
    status_client.cookies.clear()
    assert status_client.get("/api/accounts/3/sync/status").status_code == 401


def test_sync_stream_emits_status_then_done(
    status_client: TestClient, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.web.routers import settings as settings_router

    async def idle(db: Any, account_id: int, *a: Any, **kw: Any) -> dict[str, Any]:
        return {"running": False, "chats": [], "counts": {}, "poll_seconds": 5.0}

    monkeypatch.setattr(settings_router, "_sync_status", idle)
    r = status_client.get("/api/accounts/3/sync/stream")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    kinds = [line[7:].strip() for line in r.text.splitlines() if line.startswith("event: ")]
    assert kinds == ["status", "done"]


def test_sync_stream_ownership(status_client: TestClient, fake_redis: FakeRedis) -> None:
    assert status_client.get("/api/accounts/4/sync/stream").status_code == 404


def test_sync_overview_lists_accounts(
    status_client: TestClient, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.web.routers import settings as settings_router

    accounts = [
        SimpleNamespace(id=3, user_id=7, label="@ali", status="active"),
        SimpleNamespace(id=4, user_id=7, label="@vali", status="active"),
    ]
    db = _StatusDB([accounts], accounts[0])
    monkeypatch.setattr(settings_router, "session_scope", lambda: _Scope(db))

    body = status_client.get("/api/sync/overview").json()
    assert [a["account_id"] for a in body["accounts"]] == [3, 4]
    assert body["parallel"]["max_jobs"] == 4
    assert body["parallel"]["accounts_running"] == 0


def test_sync_overview_requires_login(status_client: TestClient) -> None:
    status_client.cookies.clear()
    assert status_client.get("/api/sync/overview").status_code == 401


def test_max_jobs_constant_matches_the_worker() -> None:
    """Ikki nusxa (`api` arq sozlamasini import qilmaydi) ajralib ketmasin."""
    from app.web.routers.settings import SYNC_MAX_JOBS
    from app.worker.settings import WorkerSettings

    assert SYNC_MAX_JOBS == WorkerSettings.max_jobs == 4


# ═══ frontend ════════════════════════════════════════════════════════════════


def test_sync_center_has_no_inner_html() -> None:
    """XAVFSIZLIK: chat sarlavhasi va xato matni — ishonchsiz ma'lumot."""
    import re

    src = SYNC_JS.read_text(encoding="utf-8")
    code = re.sub(r"/\*[\s\S]*?\*/", "", src)  # izohlar hisobga olinmaydi
    code = re.sub(r"(?m)^\s*//.*$", "", code)
    assert "innerHTML" not in code
    assert "outerHTML" not in code
    assert "insertAdjacentHTML" not in code


def test_sync_i18n_keys_exist_in_all_locales() -> None:
    src = SYNC_JS.read_text(encoding="utf-8")
    import re

    keys = set(re.findall(r't\("(web\.sync\.[a-z_]+)"', src))
    assert len(keys) >= 20, keys
    for lang in ("uz", "ru", "en"):
        data = json.loads((ROOT / f"src/app/i18n/locales/{lang}.json").read_text(encoding="utf-8"))
        missing = [k for k in keys if k.startswith("web.sync.") and k not in data]
        assert not missing, f"{lang}: {missing}"
        for unit in ("sec", "min", "hour"):
            assert f"web.sync.eta_{unit}" in data


def test_today_page_no_longer_shows_the_anonymous_strip() -> None:
    """Eski "Sinxronizatsiya: N chat" ikki anonim chiziq bilan — olib tashlangan."""
    src = (STATIC / "pages/today.js").read_text(encoding="utf-8")
    assert "web.today.sync_running" not in src
    assert "mountStrip" in src


@pytest.mark.skipif(NODE is None, reason="node yo'q")
def test_ring_and_eta_math_in_node() -> None:
    """Halqa geometriyasi, ETA birligi va sanoq — brauzersiz tekshiriladi."""
    script = f"""
import * as S from {json.dumps(SYNC_JS.as_uri())};
const iso = (ms) => new Date(ms).toISOString();
const out = {{}};
out.circumference = 2 * Math.PI * S.RING_R;
out.full = S.ringGeometry(100);
out.half = S.ringGeometry(50);
out.zero = S.ringGeometry(0);
out.clamped = S.ringGeometry(140).offset;
out.eta = [0, 45, 89, 90, 240, 7200, null].map(S.etaParts);
out.overall = [
  S.overallPercent(null),
  S.overallPercent({{ percent_overall: 42 }}),
  S.overallPercent({{ counts: {{ done: 3, total: 4 }} }}),
  S.overallPercent({{ percent_overall: 250 }}),
];
out.chat = [
  S.chatPercent(null),
  S.chatPercent({{ synced_total: 250, total_estimate: 1000 }}),
  S.chatPercent({{ synced_total: 5 }}),
];
out.countdown = [
  S.countdownText(iso(1000 + 201000), 1000),
  S.countdownText(iso(0), 1000),
];
out.elapsed = [S.elapsedText(iso(0), 125000), S.elapsedText(iso(0), 3725000)];
const cur = {{
  chat_id: 7, synced_total: 50, total_estimate: 200,
  phase: "backfill", paused_until: null,
}};
const status = {{ current: cur }};
out.live = [
  S.liveChat(status, 7),
  S.liveChat(status, 8),
  S.currentChatId(status),
  S.currentChatId({{}}),
];
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(  # noqa: S603
        [str(NODE), "--input-type=module", "--eval", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    # Halqa: to'liqda `offset == 0`, yarmida yarim aylana, noldakidan butun.
    assert got["full"]["offset"] == pytest.approx(0, abs=1e-9)
    assert got["half"]["dash"] == pytest.approx(got["circumference"] / 2, rel=1e-9)
    assert got["zero"]["offset"] == pytest.approx(got["circumference"], rel=1e-9)
    assert got["clamped"] == pytest.approx(0, abs=1e-9)  # 100 dan oshmaydi

    # ETA: 90 s chegarasi soniya→daqiqa, 90 daqiqa chegarasi daqiqa→soat.
    assert got["eta"] == [
        None,
        {"unit": "sec", "n": 45},
        {"unit": "sec", "n": 89},
        {"unit": "min", "n": 2},
        {"unit": "min", "n": 4},
        {"unit": "hour", "n": 2},
        None,
    ]

    assert got["overall"] == [0, 42, 75, 100]
    assert got["chat"] == [None, 25, None]
    assert got["countdown"] == ["03:21", None]
    assert got["elapsed"] == ["2:05", "1:02:05"]

    assert got["live"][0] == {"active": True, "percent": 25, "phase": "backfill", "paused": False}
    assert got["live"][1]["active"] is False
    assert got["live"][2] == 7 and got["live"][3] is None
