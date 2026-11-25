"""Eslatmalar: vaqt tahlili (akkaunt tz), taklif → tasdiq → `scheduled_jobs`, cron.

Qo'riqlaydi:
  * `parse_due` — uz/ru/en, nisbiy va aniq vaqt, **akkaunt tz** sida, o'tmish yo'q;
  * `remind` ko'nikmasi faqat taklif qiladi; vaqt/matn bo'lmasa so'raydi;
  * tasdiqdan keyin `scheduled_jobs(type="reminder", status="pending")`;
  * bekor qilish — `cancelled` (UPDATE), bir nechta bo'lsa aniqlashtiriladi;
  * yetkazish: push'dan **oldin** band qilish (ikkinchi tik hech narsa
    yubormaydi), xato → qayta urinish, `REMINDER_MAX_DELIVERY_ATTEMPTS` dan
    keyin `failed`; push **faqat** egasining `tg_user_id` siga (real Postgres).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.db.models import AgentAction, ScheduledJob
from app.services import actions as ACT
from app.services import scheduled as SCH
from app.services import skills_ext as SX
from app.worker import tasks as WT

TZ = "Asia/Tashkent"  # UTC+5
NOW = datetime(2026, 9, 26, 10, 0, tzinfo=UTC)  # Toshkentda 15:00


def _utc(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ertaga 10 da eslat: hisobotni yubor", _utc(2026, 9, 27, 5)),
        ("2 soatdan keyin eslat", NOW + timedelta(hours=2)),
        ("30 daqiqadan so'ng eslatib qo'y", NOW + timedelta(minutes=30)),
        ("через 30 минут напомни позвонить", NOW + timedelta(minutes=30)),
        ("через час напомни", NOW + timedelta(hours=1)),
        ("remind me in an hour to stretch", NOW + timedelta(hours=1)),
        ("bugun 18:30 da eslat", _utc(2026, 9, 26, 13, 30)),
        ("soat 9 da eslat", _utc(2026, 9, 27, 4)),  # bugungi 09:00 o'tgan → ertaga
        ("kechqurun 7 da eslat", _utc(2026, 9, 26, 14)),
        ("завтра в 9 напомни", _utc(2026, 9, 27, 4)),
        ("tomorrow at 8pm remind me", _utc(2026, 9, 27, 15)),
        ("30 sentyabr soat 14:00 da eslat", _utc(2026, 9, 30, 9)),
        ("12.10.2026 da eslat", _utc(2026, 10, 12, 4)),  # 12:10 emas — sana
        ("indinga eslat", _utc(2026, 9, 28, 4)),
    ],
)
def test_parse_due_in_account_tz(text: str, expected: datetime) -> None:
    assert SCH.parse_due(text, tz=TZ, now=NOW) == expected


def test_parse_due_tz_matters_and_rejects_past_or_missing() -> None:
    assert SCH.parse_due("ertaga 10 da eslat", tz="UTC", now=NOW) == _utc(2026, 9, 27, 10)
    assert SCH.parse_due("eslat", tz=TZ, now=NOW) is None
    assert SCH.parse_due("2020-01-01 da eslat", tz=TZ, now=NOW) is None
    assert SCH.parse_due("2030-01-01 da eslat", tz=TZ, now=NOW) is None  # > 1 yil
    assert SCH.parse_iso_due("2026-09-27T10:00", tz=TZ, now=NOW) == _utc(2026, 9, 27, 5)
    assert SCH.parse_iso_due("2026-09-27T10:00+00:00", tz=TZ, now=NOW) == _utc(2026, 9, 27, 10)
    assert SCH.parse_iso_due("yesterday", tz=TZ, now=NOW) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ertaga 10 da eslat: hisobotni yubor", "hisobotni yubor"),
        ("eslat 10:30 da: Aziz bilan uchrashuv", "Aziz bilan uchrashuv"),
        ("2 soatdan keyin Azizga qo'ng'iroq qilishni eslat", "Azizga qo'ng'iroq qilishni"),
        ("remind me tomorrow at 9 to pay rent", "pay rent"),
    ],
)
def test_extract_text(text: str, expected: str) -> None:
    assert SCH.extract_text(text) == expected


def test_render_html_escapes_owner_text() -> None:
    out = SCH.render_html("<b>x</b> & <untrusted_data>", locale="ru")
    assert out.startswith("⏰ <b>Напоминание</b>\n")
    assert "&lt;b&gt;x&lt;/b&gt; &amp;" in out and "<untrusted_data" not in out


# ─── ko'nikma: taklif ────────────────────────────────────────────────────────


class _Session:
    def __init__(self, pending: list[Any] | None = None, objects: dict[Any, Any] | None = None):
        self.added: list[Any] = []
        self.pending = pending or []
        self.objects = objects or {}
        self._seq = 200
        self.executed: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._seq += 1
            obj.id = self._seq

    async def flush(self) -> None: ...

    async def get(self, model: Any, key: Any) -> Any:
        return self.objects.get((model.__name__, int(key)))

    async def execute(self, stmt: Any, *a: Any, **k: Any) -> Any:
        self.executed.append(stmt)
        rows = self.pending
        return SimpleNamespace(
            rowcount=len(rows),
            scalars=lambda: SimpleNamespace(all=lambda: rows),
        )


class _Factory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Factory:
        return self

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *a: Any) -> None:
        return None


def _ctx(question: str, session: _Session, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "raw_question": question,
        "locale": "uz",
        "account_id": 5,
        "user_id": 1,
        "run_id": None,
        "action_ids": [],
        "tz": TZ,
        "session_factory": _Factory(session),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _proposals(s: _Session) -> list[AgentAction]:
    return [o for o in s.added if isinstance(o, AgentAction)]


async def test_remind_skill_proposes_with_utc_due() -> None:
    s = _Session()
    res = await SX._remind(_ctx("ertaga 10 da eslat: hisobotni yubor", s), {})
    [a] = _proposals(s)
    assert a.tool == "remind" and a.status == "proposed"
    due = datetime.fromisoformat(a.args["due_at"])
    assert due.tzinfo is not None and due.utcoffset() == timedelta(0)
    # lokal 10:00 (Toshkent) — qaysi kun bo'lishidan qat'i nazar UTC 05:00
    assert (due.hour, due.minute) == (5, 0) and a.args["due_local"].endswith("10:00")
    assert a.args["text"] == "hisobotni yubor" and a.args["tz"] == TZ
    assert not any(isinstance(o, ScheduledJob) for o in s.added)  # hali yaratilmagan
    assert res.stop and f"#{a.id}" in res.answer


async def test_remind_asks_when_time_or_text_missing() -> None:
    s = _Session()
    res = await SX._remind(_ctx("hisobotni eslat", s), {})
    assert res.summary["reason"] == "no_time" and _proposals(s) == []
    res = await SX._remind(_ctx("salom", s), {})
    assert res.summary["reason"] == "no_intent"
    # planner `when` (ISO) — faqat zaxira
    res = await SX._remind(
        _ctx("hisobotni eslat", s), {"when": "2099-01-01T09:00", "text": "hisobot"}
    )
    assert res.summary["reason"] == "no_time"  # > 1 yil — rad


async def test_cancel_reminder_single_ambiguous_and_all() -> None:
    def job(i: int, h: int) -> Any:
        return SimpleNamespace(id=i, run_at=_utc(2026, 9, 27, h), payload={"text": f"t{i}"})

    s = _Session(pending=[job(1, 5)])
    res = await SX._remind(_ctx("eslatmani bekor qil", s), {})
    [a] = _proposals(s)
    assert a.tool == "cancel_reminder" and a.args["job_ids"] == [1] and res.stop
    s2 = _Session(pending=[job(1, 5), job(2, 6)])
    res = await SX._remind(_ctx("eslatmani bekor qil", s2), {})
    assert res.summary["reason"] == "ambiguous" and _proposals(s2) == []
    assert "#1" in res.answer and "#2" in res.answer
    s3 = _Session(pending=[job(1, 5), job(2, 6)])
    await SX._remind(_ctx("hamma eslatmalarni bekor qil", s3), {})
    [c] = _proposals(s3)
    assert c.args["job_ids"] == [1, 2]
    s4 = _Session(pending=[])
    res = await SX._remind(_ctx("eslatmani bekor qil", s4), {})
    assert res.summary["reason"] == "none_pending"


# ─── tasdiq → bajarish ───────────────────────────────────────────────────────


def _action(tool: str, args: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=9,
        run_id=7,
        tool=tool,
        status="confirmed",
        args=args,
        target_peer_id=None,
        result_msg_id=None,
        error=None,
        confirmed_at=None,
    )


def _objs() -> dict[Any, Any]:
    return {
        ("AgentRun", 7): SimpleNamespace(id=7, account_id=5, user_id=1),
        ("Account", 5): SimpleNamespace(id=5, user_id=1, tz=TZ),
    }


async def test_confirmed_remind_creates_pending_job() -> None:
    s = _Session(objects=_objs())
    due = datetime.now(UTC) + timedelta(hours=3)
    a = _action("remind", {"account_id": 5, "due_at": due.isoformat(), "text": "hisobot", "tz": TZ})
    rid = await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    [job] = [o for o in s.added if isinstance(o, ScheduledJob)]
    assert job.type == "reminder" and job.status == "pending" and job.run_at == due
    assert job.payload == {"text": "hisobot", "tz": TZ, "action_id": 9}
    assert job.account_id == 5 and job.chat_id is None
    assert rid == job.id and a.status == "executed"


async def test_confirmed_remind_in_the_past_fails() -> None:
    s = _Session(objects=_objs())
    past = datetime.now(UTC) - timedelta(minutes=1)
    a = _action("remind", {"account_id": 5, "due_at": past.isoformat(), "text": "x", "tz": TZ})
    with pytest.raises(ACT.ActionError, match="due_passed"):
        await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    assert a.status == "failed" and not any(isinstance(o, ScheduledJob) for o in s.added)


async def test_confirmed_cancel_updates_status_only() -> None:
    s = _Session(pending=[object()], objects=_objs())
    a = _action("cancel_reminder", {"account_id": 5, "job_ids": [3]})
    await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    [stmt] = s.executed
    sql = str(stmt)
    assert sql.startswith("UPDATE scheduled_jobs") and "DELETE" not in sql
    assert a.status == "executed"


def test_worker_cron_registered() -> None:
    from app.worker.settings import WorkerSettings

    assert WT.deliver_reminders in WorkerSettings.functions
    names = {getattr(c, "name", "") for c in WorkerSettings.cron_jobs}
    assert any("deliver_reminders" in n for n in names)
    [cron] = WT.REMINDER_CRONS
    assert cron.minute == set(range(60))


# ─── yetkazish — haqiqiy Postgres (TEST_DATABASE_URL bo'lmasa skip) ──────────


async def _fixture_job(engine: AsyncEngine, *, run_at: datetime, status: str = "pending") -> tuple:
    async with engine.begin() as conn:
        tg_uid = int(
            (
                await conn.execute(text("SELECT floor(random() * 1e9)::bigint + 1000000"))
            ).scalar_one()
        )
        uid = (
            await conn.execute(
                text("INSERT INTO users (tg_user_id, locale) VALUES (:t, 'uz') RETURNING id"),
                {"t": tg_uid},
            )
        ).scalar_one()
        aid = (
            await conn.execute(
                text(
                    "INSERT INTO accounts (user_id, label, status) VALUES (:u, 't', 'active') "
                    "RETURNING id"
                ),
                {"u": uid},
            )
        ).scalar_one()
        jid = (
            await conn.execute(
                text(
                    "INSERT INTO scheduled_jobs (account_id, type, payload, run_at, status) "
                    "VALUES (:a, 'reminder', CAST(:p AS jsonb), :r, :s) RETURNING id"
                ),
                {"a": aid, "p": '{"text": "hisobot <b>"}', "r": run_at, "s": status},
            )
        ).scalar_one()
    return int(jid), int(aid), tg_uid


async def _row(engine: AsyncEngine, jid: int) -> Any:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT status, attempts, delivered_at, error FROM scheduled_jobs WHERE id = :j"
                ),
                {"j": jid},
            )
        ).one()


@pytest.mark.usefixtures("pg_engine")
async def test_deliver_claims_before_send_and_sends_once(pg_engine: AsyncEngine) -> None:
    now = datetime.now(UTC)
    jid, _aid, tg_uid = await _fixture_job(pg_engine, run_at=now - timedelta(seconds=5))
    future, _, _ = await _fixture_job(pg_engine, run_at=now + timedelta(hours=1))
    cancelled, _, _ = await _fixture_job(
        pg_engine, run_at=now - timedelta(seconds=5), status="cancelled"
    )
    sent: list[tuple[int, str]] = []

    async def send(uid: int, body: str) -> int:
        sent.append((uid, body))
        return 555

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as s:
        ids = await SCH.due_ids(s, now)
    assert jid in ids and future not in ids and cancelled not in ids
    async with maker() as s, s.begin():
        out = await SCH.deliver_one(s, jid, now=now, send=send)
    assert out.status == "sent"
    assert sent == [(tg_uid, "⏰ <b>Eslatma</b>\nhisobot &lt;b&gt;")]  # faqat egasiga
    async with maker() as s, s.begin():
        again = await SCH.deliver_one(s, jid, now=now, send=send)
    assert again.status == "skipped" and len(sent) == 1
    status, attempts, delivered_at, error = await _row(pg_engine, jid)
    assert (status, attempts, error) == ("sent", 1, None) and delivered_at is not None


@pytest.mark.usefixtures("pg_engine")
async def test_deliver_parallel_claim_is_exclusive(pg_engine: AsyncEngine) -> None:
    now = datetime.now(UTC)
    jid, _aid, _ = await _fixture_job(pg_engine, run_at=now - timedelta(seconds=1))
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as a, maker() as b:
        async with a.begin():
            first = await SCH.claim(a, jid, now)
        async with b.begin():
            second = await SCH.claim(b, jid, now)
    assert first == 1 and second is None


@pytest.mark.usefixtures("pg_engine")
async def test_deliver_retries_then_gives_up(pg_engine: AsyncEngine) -> None:
    now = datetime.now(UTC)
    jid, _aid, _ = await _fixture_job(pg_engine, run_at=now - timedelta(seconds=1))

    async def broken(uid: int, body: str) -> int | None:
        return None  # `send_html` xatoni yutadi va `None` qaytaradi

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    for i in range(1, SCH.REMINDER_MAX_DELIVERY_ATTEMPTS + 1):
        async with maker() as s, s.begin():
            out = await SCH.deliver_one(s, jid, now=now, send=broken)
        status, attempts, delivered_at, error = await _row(pg_engine, jid)
        assert attempts == i
        if i < SCH.REMINDER_MAX_DELIVERY_ATTEMPTS:
            assert out.status == "retry" and status == "pending" and delivered_at is None
        else:
            assert out.status == "failed" and status == "failed"
            assert error.startswith(f"gave_up_after_{i}")
    async with maker() as s, s.begin():
        after = await SCH.deliver_one(s, jid, now=now, send=broken)
    assert after.status == "skipped"


@pytest.mark.usefixtures("pg_engine")
async def test_create_and_cancel_on_real_db(pg_engine: AsyncEngine) -> None:
    now = datetime.now(UTC)
    _jid, aid, _ = await _fixture_job(pg_engine, run_at=now + timedelta(days=1))
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as s, s.begin():
        job = await SCH.create_reminder(
            s, account_id=aid, due_at=now + timedelta(hours=2), text="x", tz=TZ
        )
        pend = await SCH.pending_reminders(s, aid)
        assert {p.id for p in pend} == {job.id, _jid}
        n = await SCH.cancel_reminders(s, account_id=aid, job_ids=[job.id, 10**12])
        other = await SCH.cancel_reminders(s, account_id=aid + 10**6, job_ids=[_jid])
    assert n == 1 and other == 0
    status, *_ = await _row(pg_engine, job.id)
    assert status == "cancelled"
    assert (await _row(pg_engine, _jid))[0] == "pending"
