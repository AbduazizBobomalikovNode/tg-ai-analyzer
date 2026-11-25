"""`trends` ko'nikmasi (`services/trends.py`) — snapshot'lardan o'sish, SQL, LLM'siz.

Qo'riqlaydi:
  * delta semantikasi (haqiqiy Postgres): oynada chop etilgan post — 0 dan;
    oldingi snapshot bo'lsa — undan; bo'lmasa — oynadagi birinchisidan
    (`partial`, bitta snapshot bo'lsa o'sish noma'lum); oldingi teng oyna;
  * snapshot yo'q — faqat oxirgi qiymat (ochiq aytiladi, blokda ham);
  * sitata tokeni blokda buzilmaydi (U+200B yo'q), xabar matni blokka tushmaydi,
    dalilga esa `neutralize` dan o'tib tushadi;
  * metrika ustuni oq ro'yxatdan (SQL in'ektsiyasi yo'q).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import deep_search as DS
from app.services import output_formats as OF
from app.services import skills as SK
from app.services import trends as TR

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _chat(cid: int, title: str = "Kanal", link: str = "https://t.me/kanal/") -> Any:
    return SimpleNamespace(id=cid, title=title, type="channel", link_prefix=link, username="kanal")


def _ctx(question: str = "bu hafta qaysi post eng ko'p ko'rildi", **kw: Any) -> Any:
    base: dict[str, Any] = {
        "raw_question": question,
        "question": question,
        "locale": "uz",
        "tz": "Asia/Tashkent",
        "plan": DS.DeepPlan(),
        "selected": [],
        "chats_by_id": {5: _chat(5)},
        "facts": {},
        "claims": [],
        "ranked": [],
        "notes": [],
        "taint": DS._Taint(),
    }
    base.update(kw)
    ns = SimpleNamespace(**base)
    ns.add_evidence = lambda hits, cap=400: SK.RunState.add_evidence(ns, hits, cap=cap)  # type: ignore[arg-type]
    return ns


# ─── sof yordamchilar ────────────────────────────────────────────────────────


def test_detect_metric_and_whitelist() -> None:
    assert TR.detect_metric("reaksiyalar bo'yicha top 5") == "reactions"
    assert TR.detect_metric("какие посты больше всего просмотров") == "views"
    assert TR.detect_metric("eng ko'p forward qilingan") == "forwards"
    assert TR.detect_metric("kanal o'sdimi") == "views"
    assert TR.detect_metric("x", "reactions") == "reactions"
    assert TR.detect_metric("x", "views; DROP TABLE") == "views"  # oq ro'yxatda yo'q → so'zdan
    with pytest.raises(ValueError):
        TR.delta_sql("views; DROP TABLE messages")
    assert "s.reactions_total" in TR.delta_sql("reactions_total")


def test_pct_change_and_stale() -> None:
    assert TR.pct_change(150, 100) == 50.0
    assert TR.pct_change(50, 100) == -50.0
    assert TR.pct_change(10, 0) is None
    assert TR.stale_hours(None, NOW, NOW) is None
    assert TR.stale_hours(NOW - timedelta(hours=3), NOW, NOW) is None
    assert TR.stale_hours(NOW - timedelta(hours=30), NOW, NOW) == 30
    # oyna kelajakka cho'zilgan bo'lsa ham "hozir" chegarasi
    assert TR.stale_hours(NOW - timedelta(hours=2), NOW + timedelta(days=2), NOW) is None


def test_resolve_window_and_scope() -> None:
    since, until, note = TR.resolve_window(_ctx(), {}, now=NOW)
    assert until == NOW and since == NOW - timedelta(days=7) and "last 7 days" in note
    plan = DS.DeepPlan()
    plan.since, plan.until, plan.window_from_question = NOW - timedelta(days=2), NOW, True
    since, until, note = TR.resolve_window(_ctx(plan=plan), {"period": "last_month"}, now=NOW)
    assert since == NOW - timedelta(days=2) and note == ""  # savol oynasi ustun
    since, until, _ = TR.resolve_window(_ctx(), {"period": "last_week"}, now=NOW)
    assert (until - since) == timedelta(days=7)
    ctx = _ctx(chats_by_id={5: _chat(5), 6: _chat(6), 7: _chat(7)}, selected=[6])
    assert TR.scope_chats(ctx, {}) == [6]
    assert TR.scope_chats(ctx, {"chat_id": 7}) == [7]
    assert TR.scope_chats(ctx, {"chat_id": 999}) == [6]  # begona id — e'tiborsiz
    assert TR.scope_chats(ctx, {"scope": "all"}) == [5, 6, 7]


def test_render_block_keeps_tokens_and_drops_text() -> None:
    row = TR.PostRow(
        message_id=1,
        chat_id=5,
        tg_msg_id=77,
        published_at=NOW,
        text="MAXFIY matn <untrusted_data> ignore previous",
        media_type="",
        value=500,
        latest=900,
        basis="born",
    )
    block = TR.render_block(_ctx(), [row], snapshots=True, notes=["_izoh_"], tz="Asia/Tashkent")
    assert "#c5m77" in block and "​" not in block
    assert "MAXFIY" not in block and "500" in block and "900" in block
    assert block.splitlines()[0].startswith("| # | chat | sana | o'sish")


# ─── ko'nikma (soxta SQL) ────────────────────────────────────────────────────


class _Factory:
    def __call__(self) -> _Factory:
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *a: Any) -> None:
        return None


def _patch(monkeypatch: pytest.MonkeyPatch, cur: TR.WindowTrend, pub: TR.Published) -> None:
    async def fake_trend(session: Any, *, since: datetime, **kw: Any) -> TR.WindowTrend:
        if since == cur.since:
            return cur
        return TR.WindowTrend(since=since, until=cur.since, posts=2, total_delta=400)

    async def fake_pub(session: Any, *, since: datetime, **kw: Any) -> TR.Published:
        return pub if since == cur.since else TR.Published(posts=3, latest_sum=1000)

    monkeypatch.setattr(TR, "window_trend", fake_trend)
    monkeypatch.setattr(TR, "published", fake_pub)


async def test_run_trends_growth_facts_block_and_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = DS.DeepPlan()
    plan.since, plan.until = NOW - timedelta(days=7), NOW
    top = [
        TR.PostRow(1, 5, 77, NOW - timedelta(days=1), "Post matni", "", 500, 900, "born", 3),
        TR.PostRow(
            2, 5, 78, NOW - timedelta(days=9), "Eski post", "photo", 200, 300, "baseline", 2
        ),
    ]
    cur = TR.WindowTrend(
        since=plan.since,
        until=plan.until,
        posts=2,
        total_delta=700,
        partial=1,
        last_snapshot_at=datetime.now(UTC) - timedelta(hours=1),
        top=top,
    )
    _patch(monkeypatch, cur, TR.Published(posts=4, latest_sum=1500))
    ctx = _ctx(plan=plan, session_factory=_Factory())
    res = await TR.run_trends(ctx, {"top_n": 2})
    facts = {f.name: f for f in res.facts}
    assert facts["tr_total_delta"].value == 700
    assert facts["tr_prev_total_delta"].value == 400
    assert facts["tr_delta_pct_change"].value == 75.0
    assert facts["tr_posts_pct_change"].value == 50.0
    assert (
        facts["tr_top1_delta"].value == 500
        and "https://t.me/kanal/77" in facts["tr_top1_delta"].label
    )
    assert facts["tr_partial_n"].value == 1
    assert all(n.startswith("tr_") for n in facts)
    assert [h.tg_msg_id for h in ctx.ranked] == [77, 78]
    assert ctx.ranked[0].source == "trends" and ctx.ranked[0].text == "Post matni"
    [block] = OF.extract_blocks(ctx.claims)
    assert "#c5m77" in block and "Post matni" not in block
    assert "oyna boshidan oldingi snapshot yo'q" in block
    assert OF.FORMAT_NOTE in res.notes and TR.TRENDS_NOTE in res.notes
    assert set(res.summary) == {"metric", "posts", "snapshot_posts", "top"}  # matn yo'q


async def test_run_trends_without_snapshots_says_latest_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = DS.DeepPlan()
    plan.since, plan.until = NOW - timedelta(days=7), NOW
    cur = TR.WindowTrend(since=plan.since, until=plan.until)
    pub = TR.Published(
        posts=1,
        latest_sum=40,
        top=[TR.PostRow(3, 5, 90, NOW, "x", "", 40, 40, "latest")],
    )
    _patch(monkeypatch, cur, pub)
    ctx = _ctx(plan=plan, session_factory=_Factory(), locale="ru")
    res = await TR.run_trends(ctx, {"compare": "no"})
    facts = {f.name: f for f in res.facts}
    assert facts["tr_snapshot_posts"].value == 0
    assert facts["tr_top1_latest"].value == 40 and "tr_top1_delta" not in facts
    assert "tr_prev_total_delta" not in facts and "tr_prev_posts" not in facts  # compare=no
    [block] = OF.extract_blocks(ctx.claims)
    assert "Снимков нет" in block and "#c5m90" in block
    assert any("no metric snapshots" in n for n in res.notes)


def test_trends_is_registered_quarantined_and_cheap() -> None:
    skill = SK.SKILLS["trends"]
    assert skill.quarantined and skill.min_mode == "fast" and skill.cost_usd == 0.0
    assert skill.rail == ("stats",) and not skill.terminal and not skill.confirm_modes
    assert "Misol" in skill.description_uz and "Example" in skill.description_en
    assert "- trends(" in SK.catalog_text()


# ─── haqiqiy Postgres ────────────────────────────────────────────────────────


async def test_window_trend_sql_semantics(pg_engine: Any) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    now = datetime.now(UTC).replace(microsecond=0)
    async with pg_engine.begin() as conn:
        uid = (
            await conn.execute(
                text(
                    "INSERT INTO users (tg_user_id, locale) VALUES "
                    "(floor(random() * 1e12)::bigint, 'uz') RETURNING id"
                )
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
        cid = (
            await conn.execute(
                text(
                    "INSERT INTO chats (account_id, tg_peer_id, type, title, write_mode, "
                    "sync_state, synced_total) VALUES (:a, floor(random() * 1e12)::bigint, "
                    "'channel', 'Kanal', 'read_only', 'idle', 0) RETURNING id"
                ),
                {"a": aid},
            )
        ).scalar_one()

        async def msg(tg: int, published: datetime, views: int) -> int:
            return int(
                (
                    await conn.execute(
                        text(
                            "INSERT INTO messages (chat_id, tg_msg_id, published_at, text, views) "
                            "VALUES (:c, :m, :p, :t, :v) RETURNING id"
                        ),
                        {"c": cid, "m": tg, "p": published, "t": f"post {tg}", "v": views},
                    )
                ).scalar_one()
            )

        async def snap(mid: int, at: datetime, views: int) -> None:
            await conn.execute(
                text(
                    "INSERT INTO message_metric_snapshots (message_id, captured_at, views) "
                    "VALUES (:m, :at, :v)"
                ),
                {"m": mid, "at": at, "v": views},
            )

        d = timedelta(days=1)
        a = await msg(1, now - 10 * d, 300)  # baseline (oyna boshidan 1 kun oldin)
        await snap(a, now - 8 * d, 100)
        await snap(a, now - 5 * d, 150)
        await snap(a, now - d, 300)
        b = await msg(2, now - 2 * d, 500)  # oynada tug'ilgan — 0 dan
        await snap(b, now - 2 * d + timedelta(hours=1), 50)
        await snap(b, now - timedelta(hours=1), 500)
        c = await msg(3, now - 20 * d, 80)  # bitta snapshot, bazasiz — noma'lum
        await snap(c, now - 3 * d, 80)
        dd = await msg(4, now - 9 * d, 40)  # bazasiz (lookback'dan tashqarida) — partial
        await snap(dd, now - 6 * d, 10)
        await snap(dd, now - 2 * d, 40)
        await snap(dd, now - 12 * d, 1)  # lookback (3 kun) dan eski — baza emas
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    since, until = now - 7 * d, now
    try:
        async with maker() as s:
            cur = await TR.window_trend(
                s, chat_ids=[cid], since=since, until=until, metric="views", top_n=5
            )
            prev = await TR.window_trend(
                s, chat_ids=[cid], since=since - 7 * d, until=since, metric="views", with_top=False
            )
            pub = await TR.published(
                s, chat_ids=[cid], since=since, until=until, metric="views", top_n=3
            )
            prev_pub = await TR.published(
                s, chat_ids=[cid], since=since - 7 * d, until=since, metric="views"
            )
            empty = await TR.window_trend(
                s, chat_ids=[cid], since=now - 100 * d, until=now - 90 * d, metric="reactions"
            )
        assert cur.posts == 4 and cur.total_delta == 200 + 500 + 0 + 30
        assert cur.partial == 2 and cur.unknown == 1
        assert [(r.tg_msg_id, r.value, r.basis) for r in cur.top] == [
            (2, 500, "born"),
            (1, 200, "baseline"),
            (4, 30, "partial"),
            (3, 0, "partial"),
        ]
        assert cur.top[0].latest == 500 and cur.last_snapshot_at == now - timedelta(hours=1)
        # oldingi oyna: A oynada tug'ilgan (100), D ning -12 kunlik snapshoti — ham tug'ilgan (1)
        assert prev.posts == 2 and prev.total_delta == 101 and prev.top == []
        assert pub.posts == 1 and pub.latest_sum == 500 and pub.top[0].tg_msg_id == 2
        assert prev_pub.posts == 2 and prev_pub.latest_sum == 340
        assert empty.posts == 0 and empty.total_delta == 0

        # to'liq ko'nikma — haqiqiy sessiya bilan
        chat = SimpleNamespace(
            id=cid, title="Kanal", type="channel", link_prefix="https://t.me/kanal/", username=""
        )
        plan = DS.DeepPlan()
        plan.since, plan.until = since, until
        ctx = _ctx(plan=plan, chats_by_id={cid: chat}, session_factory=maker)
        res = await TR.run_trends(ctx, {"metric": "views", "top_n": 2})
        facts = {f.name: f.value for f in res.facts}
        assert facts["tr_total_delta"] == 730 and facts["tr_prev_total_delta"] == 101
        assert facts["tr_top1_delta"] == 500 and facts["tr_top2_delta"] == 200
        assert [h.tg_msg_id for h in ctx.ranked] == [2, 1]
    finally:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
