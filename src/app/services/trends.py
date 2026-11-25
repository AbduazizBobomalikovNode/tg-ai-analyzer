"""`trends` ko'nikmasi — ko'rish / reaksiya / forward o'sishi `message_metric_snapshots` dan.

Savollar: "bu hafta qaysi post eng ko'p ko'rildi", "kanal ko'rishlari o'sdimi",
"reaksiyalar bo'yicha top 5", "o'tgan haftaga nisbatan ko'rishlar". Hammasi
**SQL**, LLM yo'q (xarajat 0); raqam faqat kodda.

Semantika (oyna `[since, until)`, UTC, akkaunt `tz` da ko'rsatiladi):

* **Post o'sishi (delta)** — oyna ichida metrikasi bor kamida bitta snapshoti
  bo'lgan har post uchun:
  ``end`` = oynadagi **oxirgi** snapshot qiymati;
  ``start`` = (a) post oynada chop etilgan bo'lsa **0** (``basis="born"`` —
  hamma ko'rishlar shu oynada yig'ilgan); (b) aks holda oyna boshidan oldingi
  ``BASELINE_LOOKBACK`` ichidagi **oxirgi** snapshot (``basis="baseline"``);
  (c) u ham bo'lmasa oynadagi **birinchi** snapshot (``basis="partial"`` —
  o'sish kam hisoblanadi; bitta snapshot bo'lsa delta 0 va noma'lum).
  ``delta = end - start`` (kesilmaydi: reaksiya kamayishi mumkin).
* **Jami** — oynadagi deltalar yig'indisi; **oldingi teng oyna**
  ``[since - L, since)`` xuddi shu qoida bilan; ``%`` o'zgarish faqat oldingisi
  > 0 bo'lsa.
* **Chop etilganlar** (zaxira va ikkinchi o'lchov) — oynada chop etilgan
  postlar va ularning ``messages.<metric>`` **oxirgi** qiymati yig'indisi
  (`analytics.chat_stats` bilan bir ma'no). Snapshot umuman bo'lmasa reyting
  shundan tuziladi va bu ochiq aytiladi ("snapshot yo'q — faqat oxirgi qiymat").
* **Bo'shliq** — oynadagi eng oxirgi snapshot ``min(until, now)`` dan
  ``STALE_SNAPSHOT_H`` soatdan eski bo'lsa: snapshot cron'ida bo'shliq bor,
  delta shu nuqtagacha.

Faktlar ``tr_`` prefiksli (`compute` / `format_output` shular ustida ishlaydi),
top postlar dalil satri bo'lib (`S.Hit`, matn `neutralize` dan o'tgan)
sintezga boradi; jadval bloki — `output_formats.wrap_block` (xabar matni
blokka tushmaydi, faqat sarlavha, sana, raqam va sitata tokeni).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY

from app.services import output_formats as OF
from app.services import untrusted as U
from app.services.compute import Fact

if TYPE_CHECKING:
    from app.services.skills import RunState, SkillResult

#: Planner nomi → `messages` / `message_metric_snapshots` ustuni (oq ro'yxat —
#: SQL'ga faqat shu qiymatlar tushadi).
METRICS: dict[str, str] = {
    "views": "views",
    "reactions": "reactions_total",
    "forwards": "forwards",
    "replies": "replies_count",
}
DEFAULT_DAYS = 7
TOP_N_DEFAULT = 5
TOP_N_MAX = 20
PER_CHAT_MAX = 5
#: Oyna boshidan oldingi bazaviy snapshot shu oraliqda qidiriladi. 7-90 kunlik
#: postlar kuniga bir marta olinadi (`ingestion.snapshot_tiers`) — 3 kun zaxira.
BASELINE_LOOKBACK = timedelta(days=3)
STALE_SNAPSHOT_H = 24
EVIDENCE_TEXT_CHARS = 400

_METRIC_WORDS = (
    ("reactions", ("reaksiya", "reaktsiya", "реакци", "reaction", "like", "layk", "лайк")),
    ("forwards", ("forward", "ulash", "репост", "пересыл", "share")),
    ("replies", ("izoh", "komment", "коммент", "comment", "repl")),
    ("views", ("ko'ril", "ko'rish", "prosmotr", "просмотр", "view")),
)

TRENDS_NOTE = (
    "trends: growth numbers come from metric snapshots (tr_* facts). A post's growth = its "
    "last snapshot in the window minus its value at the window start (0 if published inside "
    "the window). Quote tr_* numbers exactly; say 'snapshot yo'q — faqat oxirgi qiymat' when "
    "tr_snapshot_posts is 0; never invent a view count."
)

_TEXT: dict[str, dict[str, Any]] = {
    "uz": {
        "headers": ("#", "chat", "sana", "o'sish", "oxirgi", "manba"),
        "latest_headers": ("#", "chat", "sana", "oxirgi qiymat", "manba"),
        "no_snap": "_Snapshot yo'q — faqat oxirgi qiymat (oynadagi o'sish noma'lum)._",
        "stale": "_Oxirgi snapshot {h} soat oldin olingan — snapshot'larda bo'shliq bor._",
        "partial": "_{n} ta post uchun oyna boshidan oldingi snapshot yo'q — o'sish kam "
        "hisoblangan bo'lishi mumkin._",
        "empty": "_Bu oynada post ham, snapshot ham yo'q._",
    },
    "ru": {
        "headers": ("#", "чат", "дата", "прирост", "последнее", "источник"),
        "latest_headers": ("#", "чат", "дата", "последнее значение", "источник"),
        "no_snap": "_Снимков нет — только последнее значение (прирост за период неизвестен)._",
        "stale": "_Последний снимок сделан {h} ч назад — в снимках есть пробел._",
        "partial": "_Для {n} постов нет снимка до начала периода — прирост может быть занижен._",
        "empty": "_За этот период нет ни постов, ни снимков._",
    },
    "en": {
        "headers": ("#", "chat", "date", "growth", "latest", "source"),
        "latest_headers": ("#", "chat", "date", "latest value", "source"),
        "no_snap": "_No snapshots — latest stored value only (growth in the window unknown)._",
        "stale": "_The latest snapshot is {h} h old — there is a gap in the snapshots._",
        "partial": "_{n} posts have no snapshot before the window — growth may be undercounted._",
        "empty": "_No posts and no snapshots in this window._",
    },
}


def _loc(locale: str) -> str:
    return locale if locale in _TEXT else "uz"


def detect_metric(text_: str, arg: str = "") -> str:
    """Planner argumenti (oq ro'yxatda bo'lsa) > savol so'zlari > `views`."""
    if arg in METRICS:
        return arg
    folded = OF._fold(text_)
    for metric, words in _METRIC_WORDS:
        if any(w in folded for w in words):
            return metric
    return "views"


# ─── SQL ─────────────────────────────────────────────────────────────────────


def delta_sql(col: str) -> str:
    """Har post uchun `start` / `end` / `basis` — modul docstring'idagi semantika.

    `col` faqat `METRICS` qiymatlaridan (SQL in'ektsiyasi yo'q — oq ro'yxat).
    """
    if col not in METRICS.values():
        raise ValueError(f"unknown metric column {col!r}")
    return f"""
WITH win AS (
    SELECT s.message_id, count(*) AS snaps,
           min(s.captured_at) AS first_at, max(s.captured_at) AS end_at
    FROM message_metric_snapshots s
    JOIN messages m ON m.id = s.message_id
    WHERE m.chat_id = ANY(:chat_ids) AND m.deleted_at IS NULL
      AND s.captured_at >= :since AND s.captured_at < :until
      AND s.{col} IS NOT NULL
    GROUP BY s.message_id
),
d AS (
    SELECT m.id AS message_id, m.chat_id, m.tg_msg_id, m.published_at, m.text,
           m.media_type, win.snaps, win.end_at, e.v AS end_v,
           CASE WHEN m.published_at >= :since THEN 0
                WHEN b.v IS NOT NULL THEN b.v
                ELSE f.v END AS start_v,
           CASE WHEN m.published_at >= :since THEN 'born'
                WHEN b.v IS NOT NULL THEN 'baseline'
                ELSE 'partial' END AS basis
    FROM win
    JOIN messages m ON m.id = win.message_id
    JOIN LATERAL (
        SELECT s.{col} AS v FROM message_metric_snapshots s
        WHERE s.message_id = win.message_id AND s.captured_at = win.end_at
          AND s.{col} IS NOT NULL
        ORDER BY s.id DESC LIMIT 1
    ) e ON true
    JOIN LATERAL (
        SELECT s.{col} AS v FROM message_metric_snapshots s
        WHERE s.message_id = win.message_id AND s.captured_at = win.first_at
          AND s.{col} IS NOT NULL
        ORDER BY s.id LIMIT 1
    ) f ON true
    LEFT JOIN LATERAL (
        SELECT s.{col} AS v FROM message_metric_snapshots s
        WHERE s.message_id = win.message_id AND s.captured_at < :since
          AND s.captured_at >= :lookback AND s.{col} IS NOT NULL
        ORDER BY s.captured_at DESC, s.id DESC LIMIT 1
    ) b ON true
),
x AS (SELECT d.*, (d.end_v - d.start_v) AS delta FROM d)
"""


def _params(chat_ids: Sequence[int], since: datetime, until: datetime) -> dict[str, Any]:
    return {
        "chat_ids": [int(c) for c in chat_ids],
        "since": since,
        "until": until,
        "lookback": since - BASELINE_LOOKBACK,
    }


def _stmt(sql: str) -> Any:
    return text(sql).bindparams(bindparam("chat_ids", type_=ARRAY(BigInteger)))


@dataclass(slots=True)
class PostRow:
    message_id: int
    chat_id: int
    tg_msg_id: int
    published_at: datetime | None
    text: str
    media_type: str
    value: int  # delta (snapshot) yoki oxirgi qiymat (zaxira)
    latest: int | None = None
    basis: str = ""
    snaps: int = 0


@dataclass(slots=True)
class WindowTrend:
    since: datetime
    until: datetime
    posts: int = 0  # oynada snapshoti bor postlar
    total_delta: int = 0
    partial: int = 0
    unknown: int = 0  # `partial` + bitta snapshot — o'sish umuman noma'lum
    last_snapshot_at: datetime | None = None
    top: list[PostRow] = field(default_factory=list)
    per_chat: list[tuple[int, int, int]] = field(default_factory=list)  # (chat, delta, posts)


@dataclass(slots=True)
class Published:
    posts: int = 0
    latest_sum: int = 0
    top: list[PostRow] = field(default_factory=list)


async def window_trend(
    session: Any,
    *,
    chat_ids: Sequence[int],
    since: datetime,
    until: datetime,
    metric: str,
    top_n: int = TOP_N_DEFAULT,
    with_top: bool = True,
) -> WindowTrend:
    """Bitta oyna: agregat (+ ixtiyoriy top-N va chat bo'yicha) — uchta so'rov, bitta CTE."""
    col = METRICS[metric]
    base = delta_sql(col)
    params = _params(chat_ids, since, until)
    out = WindowTrend(since=since, until=until)
    agg = (
        await session.execute(
            _stmt(
                base + "SELECT count(*), coalesce(sum(delta), 0), "
                "count(*) FILTER (WHERE basis = 'partial'), "
                "count(*) FILTER (WHERE basis = 'partial' AND snaps < 2), max(end_at) FROM x"
            ),
            params,
        )
    ).one()
    out.posts = int(agg[0] or 0)
    out.total_delta = int(agg[1] or 0)
    out.partial = int(agg[2] or 0)
    out.unknown = int(agg[3] or 0)
    out.last_snapshot_at = agg[4]
    if not with_top or not out.posts:
        return out
    rows = (
        await session.execute(
            _stmt(
                base + "SELECT message_id, chat_id, tg_msg_id, published_at, text, media_type, "
                "delta, end_v, basis, snaps FROM x ORDER BY delta DESC, message_id LIMIT :n"
            ),
            {**params, "n": max(1, min(TOP_N_MAX, int(top_n)))},
        )
    ).all()
    out.top = [
        PostRow(
            message_id=int(r[0]),
            chat_id=int(r[1]),
            tg_msg_id=int(r[2]),
            published_at=r[3],
            text=str(r[4] or ""),
            media_type=str(r[5] or ""),
            value=int(r[6] or 0),
            latest=int(r[7]) if r[7] is not None else None,
            basis=str(r[8] or ""),
            snaps=int(r[9] or 0),
        )
        for r in rows
    ]
    if len(set(chat_ids)) > 1:
        per = (
            await session.execute(
                _stmt(
                    base + "SELECT chat_id, coalesce(sum(delta), 0), count(*) FROM x "
                    "GROUP BY chat_id ORDER BY 2 DESC, chat_id LIMIT :n"
                ),
                {**params, "n": PER_CHAT_MAX},
            )
        ).all()
        out.per_chat = [(int(r[0]), int(r[1] or 0), int(r[2] or 0)) for r in per]
    return out


async def published(
    session: Any,
    *,
    chat_ids: Sequence[int],
    since: datetime,
    until: datetime,
    metric: str,
    top_n: int = 0,
) -> Published:
    """Oynada chop etilgan postlar — `messages.<metric>` **oxirgi** qiymati bo'yicha."""
    col = METRICS[metric]
    params = {"chat_ids": [int(c) for c in chat_ids], "since": since, "until": until}
    where = (
        "FROM messages WHERE chat_id = ANY(:chat_ids) AND deleted_at IS NULL "
        "AND published_at >= :since AND published_at < :until"
    )
    agg = (
        await session.execute(
            _stmt(f"SELECT count(*), coalesce(sum({col}), 0) {where}"),
            params,
        )
    ).one()
    out = Published(posts=int(agg[0] or 0), latest_sum=int(agg[1] or 0))
    if top_n <= 0 or not out.posts:
        return out
    rows = (
        await session.execute(
            _stmt(
                f"SELECT id, chat_id, tg_msg_id, published_at, text, media_type, {col} "
                f"{where} AND {col} IS NOT NULL ORDER BY {col} DESC, id LIMIT :n"
            ),
            {**params, "n": max(1, min(TOP_N_MAX, int(top_n)))},
        )
    ).all()
    out.top = [
        PostRow(
            message_id=int(r[0]),
            chat_id=int(r[1]),
            tg_msg_id=int(r[2]),
            published_at=r[3],
            text=str(r[4] or ""),
            media_type=str(r[5] or ""),
            value=int(r[6] or 0),
            latest=int(r[6] or 0),
            basis="latest",
        )
        for r in rows
    ]
    return out


# ─── sof yordamchilar ────────────────────────────────────────────────────────


def pct_change(cur: float, prev: float) -> float | None:
    """Oldingi > 0 bo'lsagina foiz; aks holda `None` (0 dan foiz — ma'nosiz)."""
    if prev <= 0:
        return None
    return round((cur - prev) / prev * 100.0, 1)


def stale_hours(last: datetime | None, until: datetime, now: datetime) -> int | None:
    """Oxirgi snapshot `min(until, now)` dan necha soat eski (≥ `STALE_SNAPSHOT_H` bo'lsa)."""
    if last is None:
        return None
    edge = min(until, now)
    hours = int((edge - last).total_seconds() // 3600)
    return hours if hours >= STALE_SNAPSHOT_H else None


def resolve_window(
    ctx: Any, args: dict[str, Any], *, now: datetime | None = None
) -> tuple[datetime, datetime, str]:
    """Oyna: `period` argumenti (savolning o'z oynasi bo'lmasa) > reja oynasi > oxirgi 7 kun."""
    from app.services import timewin as TW

    now = now or datetime.now(UTC)
    plan = ctx.plan
    period = str(args.get("period") or "")
    if period and not getattr(plan, "window_from_question", False):
        win = TW.period_window(period, tz=getattr(plan, "window_tz", "") or ctx.tz, now=now)
        if win is not None and win.since is not None:
            return win.since, win.until, ""
    since = getattr(plan, "since", None)
    until = getattr(plan, "until", None) or now
    if isinstance(since, datetime) and until > since:
        return since, until, ""
    return (
        now - timedelta(days=DEFAULT_DAYS),
        now,
        f"trends: no period in the question — used the last {DEFAULT_DAYS} days",
    )


def scope_chats(ctx: Any, args: dict[str, Any]) -> list[int]:
    """`chat_id` > `scope=all` > reja chatlari > tanlangan > hammasi (faqat shu akkaunt)."""
    known = ctx.chats_by_id
    cid = args.get("chat_id")
    if isinstance(cid, int) and cid in known:
        return [cid]
    if str(args.get("scope") or "") == "all":
        return sorted(known)
    for source in (list(ctx.plan.chats or []), list(ctx.selected or [])):
        ids = [int(c) for c in source if int(c) in known]
        if ids:
            return ids
    return sorted(known)


def _date(dt: datetime | None, tz: str) -> str:
    if dt is None:
        return ""
    from app.services import timewin as TW

    return dt.astimezone(TW.zone(tz)).strftime("%Y-%m-%d")


def _title(ctx: Any, chat_id: int) -> str:
    chat = ctx.chats_by_id.get(chat_id)
    return U.attr(chat.title if chat else str(chat_id), max_chars=60)


def _link(ctx: Any, row: PostRow) -> str:
    from app.services.analytics import msg_link

    chat = ctx.chats_by_id.get(row.chat_id)
    return (msg_link(chat, row.tg_msg_id) or "") if chat is not None else ""


def cite_token(chat_id: int, tg_msg_id: int) -> str:
    """Sitata tokeni (`deep_search.to_evidence` bilan bir shakl) — kodda yasaladi."""
    return f"#c{int(chat_id)}m{int(tg_msg_id)}"


def render_block(
    ctx: Any, rows: Sequence[PostRow], *, snapshots: bool, notes: Sequence[str], tz: str
) -> str:
    t = _TEXT[_loc(ctx.locale)]
    parts: list[str] = []
    if rows:
        table_rows: list[tuple[Any, ...]]
        if snapshots:
            table_rows = [
                (
                    i,
                    _title(ctx, r.chat_id),
                    _date(r.published_at, tz),
                    r.value,
                    r.latest if r.latest is not None else "",
                    cite_token(r.chat_id, r.tg_msg_id),
                )
                for i, r in enumerate(rows, 1)
            ]
            parts.append(
                OF.render_table(t["headers"], table_rows, numeric_cols=(0, 3, 4), raw_cols=(5,))
            )
        else:
            table_rows = [
                (
                    i,
                    _title(ctx, r.chat_id),
                    _date(r.published_at, tz),
                    r.value,
                    cite_token(r.chat_id, r.tg_msg_id),
                )
                for i, r in enumerate(rows, 1)
            ]
            parts.append(
                OF.render_table(t["latest_headers"], table_rows, numeric_cols=(0, 3), raw_cols=(4,))
            )
    parts.extend(notes)
    return "\n\n".join(p for p in parts if p)


def to_hits(ctx: Any, rows: Sequence[PostRow]) -> list[Any]:
    from app.services import search as S

    return [
        S.Hit(
            message_id=r.message_id,
            tg_msg_id=r.tg_msg_id,
            score=float(r.value),
            source="trends",
            chat_id=r.chat_id,
            chat_title=ctx.chats_by_id[r.chat_id].title if r.chat_id in ctx.chats_by_id else "",
            date=r.published_at,
            text=U.neutralize(r.text)[:EVIDENCE_TEXT_CHARS],
            media_type=r.media_type or "",
        )
        for r in rows
    ]


# ─── ko'nikma ────────────────────────────────────────────────────────────────


def _no(value: Any) -> bool:
    return str(value or "").strip().lower() in ("no", "false", "0", "yo'q", "нет")


async def run_trends(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    from app.services.skills import SkillResult

    now = datetime.now(UTC)
    metric = detect_metric(ctx.raw_question, str(args.get("metric") or ""))
    raw_n = args.get("top_n")
    top_n = max(1, min(TOP_N_MAX, int(raw_n))) if isinstance(raw_n, int) else TOP_N_DEFAULT
    since, until, wnote = resolve_window(ctx, args, now=now)
    chat_ids = scope_chats(ctx, args)
    notes = [TRENDS_NOTE]
    if wnote:
        notes.append(wnote)
    if not chat_ids:
        return SkillResult(notes=[*notes, "trends: no chats"], summary={"posts": 0})
    tz = str(getattr(ctx.plan, "window_tz", "") or ctx.tz)
    length = until - since
    p_since = since - length
    async with ctx.session_factory() as session:
        cur = await window_trend(
            session, chat_ids=chat_ids, since=since, until=until, metric=metric, top_n=top_n
        )
        prev = await window_trend(
            session, chat_ids=chat_ids, since=p_since, until=since, metric=metric, with_top=False
        )
        pub = await published(
            session,
            chat_ids=chat_ids,
            since=since,
            until=until,
            metric=metric,
            top_n=0 if cur.posts else top_n,
        )
        prev_pub = await published(
            session, chat_ids=chat_ids, since=p_since, until=since, metric=metric
        )

    from app.services import timewin as TW

    window_txt = TW.format_window(TW.TimeWindow(since=since, until=until, kind="", tz=tz))
    prev_txt = TW.format_window(TW.TimeWindow(since=p_since, until=since, kind="", tz=tz))
    t = _TEXT[_loc(ctx.locale)]
    src = "trends"
    facts: list[Fact] = [
        Fact("tr_metric", metric, label="metric of the trend", source=src),
        Fact("tr_window", window_txt, label="trend window", source=src),
        Fact(
            "tr_snapshot_posts",
            cur.posts,
            label=f"posts with {metric} snapshots in the window",
            source=src,
        ),
        Fact(
            "tr_total_delta",
            cur.total_delta,
            label=f"{metric} gained in the window (sum of per-post snapshot growth)",
            source=src,
        ),
        Fact(
            "tr_posts",
            pub.posts,
            label="posts published in the window",
            source=src,
        ),
        Fact(
            "tr_posts_latest_sum",
            pub.latest_sum,
            label=f"latest stored {metric} of posts published in the window",
            source=src,
        ),
    ]
    block_notes: list[str] = []
    rows: list[PostRow] = list(cur.top)
    snapshots = bool(cur.posts)
    if not snapshots:
        rows = list(pub.top)
        notes.append(
            "trends: no metric snapshots in the window — ranking uses the latest stored value "
            "of posts published in the window; growth inside the window is unknown"
        )
        block_notes.append(t["no_snap"] if rows else t["empty"])
    else:
        if cur.partial:
            facts.append(
                Fact(
                    "tr_partial_n",
                    cur.partial,
                    label="posts without a snapshot before the window (growth may be undercounted)",
                    source=src,
                )
            )
            block_notes.append(t["partial"].format(n=cur.partial))
        if cur.unknown:
            facts.append(
                Fact(
                    "tr_unknown_n",
                    cur.unknown,
                    label="posts with a single snapshot and no baseline (growth unknown)",
                    source=src,
                )
            )
        gap = stale_hours(cur.last_snapshot_at, until, now)
        if gap is not None:
            facts.append(
                Fact(
                    "tr_last_snapshot_age_h", gap, label="hours since the last snapshot", source=src
                )
            )
            block_notes.append(t["stale"].format(h=gap))
    for i, r in enumerate(rows, 1):
        link = _link(ctx, r)
        label = (
            f"{'growth' if snapshots else 'latest value'} of #{i} post in "
            f'"{_title(ctx, r.chat_id)}" ({_date(r.published_at, tz)}'
            f"{', ' + link if link else ''})"
        )
        facts.append(
            Fact(
                f"tr_top{i}_{'delta' if snapshots else 'latest'}", r.value, label=label, source=src
            )
        )
    for i, (cid, delta, n) in enumerate(cur.per_chat, 1):
        facts.append(
            Fact(
                f"tr_chat{i}_delta",
                delta,
                label=f'{metric} gained in "{_title(ctx, cid)}" ({n} posts with snapshots)',
                source=src,
            )
        )
    if not _no(args.get("compare")):
        facts.append(Fact("tr_prev_window", prev_txt, label="previous equal window", source=src))
        if prev.posts:
            facts.append(
                Fact(
                    "tr_prev_total_delta",
                    prev.total_delta,
                    label=f"{metric} gained in the previous equal window",
                    source=src,
                )
            )
            pct = pct_change(cur.total_delta, prev.total_delta) if snapshots else None
            if pct is not None:
                facts.append(
                    Fact(
                        "tr_delta_pct_change",
                        pct,
                        unit="%",
                        label="growth vs the previous equal window",
                        expr="(tr_total_delta - tr_prev_total_delta) / tr_prev_total_delta * 100",
                        source=src,
                    )
                )
        else:
            notes.append("trends: no snapshots in the previous window — growth not comparable")
        facts.append(
            Fact(
                "tr_prev_posts",
                prev_pub.posts,
                label="posts published in the previous equal window",
                source=src,
            )
        )
        facts.append(
            Fact(
                "tr_prev_posts_latest_sum",
                prev_pub.latest_sum,
                label=f"latest stored {metric} of posts published in the previous window",
                source=src,
            )
        )
        ppct = pct_change(pub.latest_sum, prev_pub.latest_sum)
        if ppct is not None:
            facts.append(
                Fact(
                    "tr_posts_pct_change",
                    ppct,
                    unit="%",
                    label=f"latest {metric} of posts published: this window vs previous",
                    expr="(tr_posts_latest_sum - tr_prev_posts_latest_sum) / "
                    "tr_prev_posts_latest_sum * 100",
                    source=src,
                )
            )
    hits = to_hits(ctx, rows)
    taint = getattr(ctx, "taint", None)
    if taint is not None and hasattr(taint, "add_hits"):
        taint.add_hits(hits)
    ctx.add_evidence(hits)
    block = render_block(ctx, rows, snapshots=snapshots, notes=block_notes, tz=tz)
    if block:
        ctx.claims.append(OF.wrap_block(block))
        notes.append(OF.FORMAT_NOTE)
    return SkillResult(
        evidence=hits,
        facts=facts,
        notes=notes,
        summary={
            "metric": metric,
            "posts": pub.posts,
            "snapshot_posts": cur.posts,
            "top": len(rows),
        },
    )


__all__ = [
    "BASELINE_LOOKBACK",
    "METRICS",
    "TRENDS_NOTE",
    "PostRow",
    "WindowTrend",
    "cite_token",
    "delta_sql",
    "detect_metric",
    "pct_change",
    "published",
    "render_block",
    "resolve_window",
    "run_trends",
    "scope_chats",
    "stale_hours",
    "window_trend",
]
