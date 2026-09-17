"""Chat ma'lumotlari UI uchun — profil, qidiruv, statistika (ui/R1a, faqat O'QISH).

GET /api/accounts/{id}/chats/search?q=&type=&limit=       server tomonda, uz-fold
GET /api/accounts/{id}/messages/search?q=&chat_id=&limit=  lokal gibrid qidiruv
GET /api/accounts/{id}/chats/{chat_id}                     bitta chat + sync holati
GET /api/accounts/{id}/chats/{chat_id}/stats?days=         analytics.chat_stats + series
GET /api/accounts/{id}/chats/{chat_id}/digests?days=       kunlik digest'lar (level 1)
GET /api/accounts/{id}/chats/{chat_id}/messages/{tg_msg_id} bitta xabar (sitata kartasi)

Router **prefikssiz** va `routers/tg.py` ichidan `include_router` bilan
ulanadi (`/api/accounts` prefiksi o'sha yerdan keladi) — `web/main.py` ga
tegilmaydi. Hammasi GET, shuning uchun CSRF kerak emas; egalik esa har
so'rovda tekshiriladi (`_own_chat` / `_own_account`).

**Bu yo'llardagi matn — Telegram kontenti, ya'ni ishonchsiz ma'lumot.** U
foydalanuvchiga ko'rsatiladi, promptga **hech qachon** bu yerdan kirmaydi
(promptga faqat `search.select_context` / `chat_service.render_context`
`<untrusted_data>` konvertida beradi).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Integer, String, case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import session_scope
from app.db.models import Account, Chat, ChatDigest, ChatType, Message
from app.logging import get_logger
from app.services import avatars as AV
from app.services.textnorm import uz_fold
from app.web.security import RateLimiter, WebIdentity, require_identity

log = get_logger(__name__)

router = APIRouter(tags=["chatdata"])

CHAT_SEARCH_MAX = 100
MSG_SEARCH_MAX = 50
MSG_TEXT_CHARS = 300
DIGEST_MAX = 62  # `tools.WINDOW_DAYS_MAX` bilan bir xil qopqoq
STATS_DAYS_MAX = 365
_TYPES = {str(t) for t in ChatType}

#: Xabar qidiruvi DB'ga (FTS + pgvector) boradi — foydalanuvchi bo'yicha shift.
#: `web/security.RateLimiter` naqshi (jarayon xotirasi, auth bilan bir xil).
_msg_search_limiter = RateLimiter(30, 60.0)


# ─── egalik ──────────────────────────────────────────────────────────────────


async def _own_account(db: AsyncSession, user_id: int, account_id: int) -> Account:
    acc = await db.get(Account, account_id)
    if acc is None or acc.user_id != user_id:
        raise HTTPException(status_code=404, detail={"code": "account.not_found"})
    return acc


async def _own_chat(db: AsyncSession, user_id: int, account_id: int, chat_id: int) -> Chat:
    await _own_account(db, user_id, account_id)
    chat = await AV.get_owned_chat(db, user_id, chat_id)
    if chat is None or chat.account_id != account_id:
        raise HTTPException(status_code=404, detail={"code": "pool.err.no_dialog"})
    return chat


# ─── chat qatori ─────────────────────────────────────────────────────────────


def avatar_url(c: Chat) -> str | None:
    """Avatar havolasi — **faqat fayl mavjud bo'lsa** (aks holda UI inisiallarni chizadi)."""
    if not c.avatar_photo_id or not c.avatar_mime:
        return None
    if not AV.path_for(c.id, c.avatar_mime).exists():
        return None
    return f"/api/accounts/{c.account_id}/chats/{c.id}/avatar?v={c.avatar_photo_id}"


def chat_out(c: Chat) -> dict[str, Any]:
    """`chats` qatorining JSON shakli — `routers/tg.py` ham shu funksiyani ishlatadi."""
    est = c.total_estimate or 0
    if est:
        pct = min(100, round(100 * c.synced_total / est))
    else:
        pct = 100 if c.sync_state == "done" else 0
    return {
        "id": c.id,
        "peer_id": c.tg_peer_id,
        "title": c.title,
        "type": c.type,
        "username": c.username,
        "is_admin": c.is_admin,
        "is_writable": c.is_writable,
        "write_mode": c.write_mode,
        "participants_count": c.participants_count,
        "sync_state": c.sync_state,
        "synced_total": c.synced_total,
        "total_estimate": c.total_estimate,
        "progress": pct,
        "sync_error": c.sync_error,
        "last_sync_at": c.last_sync_at.isoformat() if c.last_sync_at else None,
        "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
        # ── ui/R1a (migratsiya 0017) ──
        # `about` / `last_message_text` — Telegram kontenti: faqat ko'rsatish uchun.
        "about": c.about,
        "avatar_url": avatar_url(c),
        "is_verified": bool(c.is_verified),
        "is_scam": bool(c.is_scam),
        "is_fake": bool(c.is_fake),
        "unread_count": int(c.unread_count or 0),
        "last_message_text": c.last_message_text,
        "link_prefix": c.link_prefix,
    }


# ─── chat qidiruvi (uz-fold) ────────────────────────────────────────────────


def _like_escape(text_: str) -> str:
    return text_.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _rank_py(field: str, q: str) -> int:
    """Python tomonda reyting: prefiks > so'z boshi > ichida > yo'q."""
    if not field or not q:
        return 0
    if field.startswith(q):
        return 3
    if f" {q}" in field:
        return 2
    if q in field:
        return 1
    return 0


@router.get("/{account_id}/chats/search")
async def search_chats(
    account_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    q: Annotated[str, Query(max_length=128)] = "",
    type: Annotated[str, Query(max_length=16)] = "",  # API nomi — shadowing emas
    limit: Annotated[int, Query(ge=1, le=CHAT_SEARCH_MAX)] = 30,
) -> dict[str, Any]:
    """Sarlavha / username / about bo'yicha server tomonda qidiruv.

    Moslik **fold fazosida** (`uz_fold`): apostrofli, apostrofsiz va kirill
    yozuvdagi variantlar bitta so'rov. Postgres'dagi `uz_fold()` (0008) va
    Python'dagi `textnorm.uz_fold` bir xil natija beradi
    (`test_db_integration.py::test_uz_fold_sql_python_parity`), shuning uchun
    funksiya bo'lmagan bazada Python yo'liga tushish xavfsiz.
    """
    folded = uz_fold(q).strip()
    kind = type.strip().lower()
    if kind and kind not in _TYPES:
        raise HTTPException(status_code=400, detail={"code": "chat.err.bad_type"})

    async with session_scope() as db:
        await _own_account(db, ident.user_id, account_id)
        conds: list[Any] = [Chat.account_id == account_id]
        if kind:
            conds.append(Chat.type == kind)
        if not folded:
            rows = (
                (
                    await db.execute(
                        select(Chat)
                        .where(*conds)
                        .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return {"items": [chat_out(c) for c in rows], "q": q, "folded": folded}

        try:
            rows = await _search_sql(db, conds, folded, limit)
        except Exception as exc:  # `uz_fold()` yo'q (eski baza) — Python yo'li
            log.warning("chatdata.uz_fold_sql_failed", error=str(exc)[:200])
            rows = await _search_python(db, conds, folded, limit)
        return {"items": [chat_out(c) for c in rows], "q": q, "folded": folded}


def _fold_col(col: Any) -> Any:
    return func.uz_fold(func.coalesce(col, literal("", String)))


async def _search_sql(db: AsyncSession, conds: list[Any], folded: str, limit: int) -> list[Chat]:
    pat = _like_escape(folded)
    title_f, user_f, about_f = (
        _fold_col(Chat.title),
        _fold_col(Chat.username),
        _fold_col(Chat.about),
    )
    rank = case(
        (title_f.like(f"{pat}%", escape="\\"), literal(3, Integer)),
        (user_f.like(f"{pat}%", escape="\\"), literal(3, Integer)),
        (title_f.like(f"% {pat}%", escape="\\"), literal(2, Integer)),
        (title_f.like(f"%{pat}%", escape="\\"), literal(1, Integer)),
        (user_f.like(f"%{pat}%", escape="\\"), literal(1, Integer)),
        else_=literal(0, Integer),
    ).label("rank")
    stmt = (
        select(Chat, rank)
        .where(
            *conds,
            or_(
                title_f.like(f"%{pat}%", escape="\\"),
                user_f.like(f"%{pat}%", escape="\\"),
                about_f.like(f"%{pat}%", escape="\\"),
            ),
        )
        .order_by(rank.desc(), Chat.last_message_at.desc().nulls_last(), Chat.id)
        .limit(limit)
    )
    return [row[0] for row in (await db.execute(stmt)).all()]


async def _search_python(db: AsyncSession, conds: list[Any], folded: str, limit: int) -> list[Chat]:
    """Zaxira yo'l: `uz_fold()` SQL funksiyasi bo'lmasa fold Python'da."""
    rows = (
        (
            await db.execute(
                select(Chat)
                .where(*conds)
                .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
                .limit(2000)
            )
        )
        .scalars()
        .all()
    )
    scored: list[tuple[int, int, Chat]] = []
    for i, c in enumerate(rows):
        rank = max(
            _rank_py(uz_fold(c.title or ""), folded),
            _rank_py(uz_fold(c.username or ""), folded),
            1 if folded in uz_fold(c.about or "") else 0,
        )
        if rank:
            scored.append((rank, -i, c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [c for _, _, c in scored[:limit]]


# ─── xabar qidiruvi (lokal, LLM'siz) ────────────────────────────────────────


def _msg_link(chat: Chat, tg_msg_id: int) -> str | None:
    if chat.link_prefix:
        return f"{chat.link_prefix}{tg_msg_id}"
    return f"https://t.me/{chat.username}/{tg_msg_id}" if chat.username else None


def _sender_of(m: Message, chat: Chat) -> str:
    if chat.type == ChatType.CHANNEL and not m.sender_id:
        return chat.title
    return f"user:{m.sender_id}" if m.sender_id else "—"


@router.get("/{account_id}/messages/search")
async def search_messages(
    account_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    q: Annotated[str, Query(max_length=256)] = "",
    chat_id: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MSG_SEARCH_MAX)] = 20,
) -> dict[str, Any]:
    """Lokal gibrid qidiruv (FTS + pgvector, RRF) — **LLM'siz, rerank'siz**.

    Natija to'g'ridan-to'g'ri foydalanuvchiga ketadi (promptga emas), shuning
    uchun kengaytirish (`expand`) ham, qayta reyting ham yoqilmaydi: ular
    model chaqiradi va bu yerda ularning qiymati yo'q.
    """
    from app.services import search as S

    question = q.strip()
    if not question:
        return {"items": [], "q": q}
    if not _msg_search_limiter.check(f"msgsearch:{ident.user_id}"):
        raise HTTPException(status_code=429, detail={"code": "chat.err.rate_limited"})

    async with session_scope() as db:
        await _own_account(db, ident.user_id, account_id)
        chat: Chat | None = None
        if chat_id:
            chat = await _own_chat(db, ident.user_id, account_id, chat_id)
            hits = await S.hybrid_search(db, chat.id, question, limit=limit, expand=False)
        else:
            hits = await S.hybrid_search_multi(db, account_id, question, limit=limit, expand=False)
        items = await _hydrate(db, hits[:limit], chat)
    return {"items": items, "q": q}


async def _hydrate(db: AsyncSession, hits: list[Any], chat: Chat | None) -> list[dict[str, Any]]:
    """`Hit` → UI qatori. Matn/sana DB'dan (hit'larda ular to'ldirilmagan)."""
    if not hits:
        return []
    ids = [int(h.message_id) for h in hits]
    rows = (await db.execute(select(Message).where(Message.id.in_(ids)))).scalars().all()
    by_id = {int(m.id): m for m in rows}
    chat_ids = {int(m.chat_id) for m in rows}
    chats: dict[int, Chat] = {chat.id: chat} if chat is not None else {}
    missing = chat_ids - set(chats)
    if missing:
        found = (await db.execute(select(Chat).where(Chat.id.in_(missing)))).scalars().all()
        chats.update({int(c.id): c for c in found})

    out: list[dict[str, Any]] = []
    for h in hits:
        m = by_id.get(int(h.message_id))
        if m is None:
            continue
        c = chats.get(int(m.chat_id))
        if c is None:
            continue
        out.append(
            {
                "chat_id": c.id,
                "chat_title": c.title,
                "tg_msg_id": m.tg_msg_id,
                "date": m.published_at.isoformat() if m.published_at else None,
                "sender": _sender_of(m, c),
                "text": (m.text or "").replace("\n", " ")[:MSG_TEXT_CHARS],
                "link": _msg_link(c, m.tg_msg_id),
            }
        )
    return out


# ─── statistika ──────────────────────────────────────────────────────────────

_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@router.get("/{account_id}/chats/{chat_id}/stats")
async def chat_statistics(
    account_id: int,
    chat_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    days: Annotated[int, Query(ge=1, le=STATS_DAYS_MAX)] = 30,
) -> dict[str, Any]:
    """`analytics.chat_stats` + kunlik qator (grafiklar uchun)."""
    from app.services.analytics import chat_stats

    async with session_scope() as db:
        chat = await _own_chat(db, ident.user_id, account_id, chat_id)
        stats = await chat_stats(db, chat, days=days)
        data = stats.to_dict()
        data["by_hour"] = [int(stats.by_hour.get(f"{h:02d}", 0)) for h in range(24)]
        data["by_weekday"] = [int(stats.by_weekday.get(name, 0)) for name in _WD]
        data["series"] = await _series(db, chat.id, days=days)
    return data


async def spark_series(
    db: AsyncSession, chat_ids: list[int], *, days: int = 14
) -> dict[int, list[int]]:
    """`/chats` qatori sparklayni uchun — **bitta** guruhlangan so'rov, hamma
    chat uchun birdan (M3). `chats.py` `?spark=1` bilan so'raganda chaqiradi.

    Har chat uchun alohida so'rov yubormaymiz — bu 100+ qatorli ro'yxatda
    100+ ta SQL so'rovga aylanardi. `deleted_at IS NULL` — o'chirilgan
    xabarlar hisobga kirmaydi (0015'dagi qoida bilan bir xil).
    """
    if not chat_ids:
        return {}
    until = datetime.now(UTC)
    since = until - timedelta(days=days)
    day = func.date_trunc("day", Message.published_at).label("day")
    rows = (
        await db.execute(
            select(Message.chat_id, day, func.count(Message.id))
            .where(
                Message.chat_id.in_(chat_ids),
                Message.deleted_at.is_(None),
                Message.published_at >= since,
                Message.published_at < until,
            )
            .group_by(Message.chat_id, day)
        )
    ).all()
    per_chat: dict[int, dict[str, int]] = {}
    for chat_id, day_val, cnt in rows:
        per_chat.setdefault(int(chat_id), {})[day_val.date().isoformat()] = int(cnt or 0)
    out: dict[int, list[int]] = {}
    for chat_id in chat_ids:
        found = per_chat.get(int(chat_id), {})
        out[int(chat_id)] = [
            found.get((since + timedelta(days=i + 1)).date().isoformat(), 0) for i in range(days)
        ]
    return out


async def _series(db: AsyncSession, chat_id: int, *, days: int) -> list[dict[str, Any]]:
    """Kun bo'yicha: postlar soni va ko'rishlar yig'indisi (bo'sh kunlar ham bor).

    Ko'rishlar `messages.views` — **oxirgi ma'lum qiymat**, o'sish emas
    (o'sish faqat `message_metric_snapshots` dan, `chat_stats.views_delta_period`).
    """
    until = datetime.now(UTC)
    since = until - timedelta(days=days)
    day = func.date_trunc("day", Message.published_at).label("day")
    rows = (
        await db.execute(
            select(day, func.count(Message.id), func.coalesce(func.sum(Message.views), 0))
            .where(
                Message.chat_id == chat_id,
                Message.published_at >= since,
                Message.published_at < until,
            )
            .group_by(day)
            .order_by(day)
        )
    ).all()
    found = {r[0].date().isoformat(): (int(r[1] or 0), int(r[2] or 0)) for r in rows}
    out: list[dict[str, Any]] = []
    for i in range(days):
        key = (since + timedelta(days=i + 1)).date().isoformat()
        posts, views = found.get(key, (0, 0))
        out.append({"day": key, "posts": posts, "views": views})
    return out


# ─── digest'lar ──────────────────────────────────────────────────────────────


@router.get("/{account_id}/chats/{chat_id}/digests")
async def chat_digests(
    account_id: int,
    chat_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    days: Annotated[int, Query(ge=1, le=DIGEST_MAX)] = 14,
) -> dict[str, Any]:
    """Kunlik digest'lar (level 1), yangisidan eskisiga. Qopqoq — 62 kun."""
    since = datetime.now(UTC) - timedelta(days=min(days, DIGEST_MAX))
    async with session_scope() as db:
        await _own_chat(db, ident.user_id, account_id, chat_id)
        rows = (
            (
                await db.execute(
                    select(ChatDigest)
                    .where(
                        ChatDigest.chat_id == chat_id,
                        ChatDigest.level == 1,
                        ChatDigest.day >= since,
                    )
                    .order_by(ChatDigest.day.desc())
                    .limit(DIGEST_MAX)
                )
            )
            .scalars()
            .all()
        )
    return {
        "items": [
            {
                "day": d.day.date().isoformat() if d.day else None,
                "digest": d.digest,
                "msg_count": int(d.msg_count or 0),
            }
            # Qopqoq ikki joyda: SQL `LIMIT` va shu yerda — javob hajmi
            # so'rov shakli qanday bo'lishidan qat'i nazar chegaralangan.
            for d in list(rows)[:DIGEST_MAX]
        ]
    }


# ─── bitta chat ──────────────────────────────────────────────────────────────


@router.get("/{account_id}/chats/{chat_id}")
async def chat_detail(
    account_id: int, chat_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    """Bitta chat: `chat_out` + `sync` (holat/progress/xato) kichik obyekti."""
    async with session_scope() as db:
        chat = await _own_chat(db, ident.user_id, account_id, chat_id)
        data = chat_out(chat)
        data["sync"] = {
            "state": chat.sync_state,
            "progress": data["progress"],
            "synced_total": chat.synced_total,
            "total_estimate": chat.total_estimate,
            "last_sync_at": data["last_sync_at"],
            "error": chat.sync_error,
        }
    return data


# ─── bitta xabar (sitata hover kartasi, ui/R2b) ──────────────────────────────


#: Hover kartasidagi matn qopqog'i. `MSG_TEXT_CHARS` (300) qidiruv qatori
#: uchun; bu yerda karta biroz kengroq, lekin baribir chegaralangan.
CITATION_TEXT_CHARS = 600


@router.get("/{account_id}/chats/{chat_id}/messages/{tg_msg_id}")
async def message_detail(
    account_id: int,
    chat_id: int,
    tg_msg_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
) -> dict[str, Any]:
    """Bitta xabar — `/ask` dagi sitata chipining hover kartasi uchun.

    **Faqat lokal DB**, MTProto yo'q: karta ochilishi jonli so'rov keltirib
    chiqarmaydi (bir javobda o'nlab sitata bo'lishi mumkin). Sinxronlanmagan
    xabar topilmaydi — UI o'sha holatda faqat "Telegram'da ochish" ni
    ko'rsatadi.

    Matn — Telegram kontenti, ya'ni **ishonchsiz**: faqat ko'rsatish uchun,
    promptga bu yo'ldan hech qachon kirmaydi.
    """
    async with session_scope() as db:
        chat = await _own_chat(db, ident.user_id, account_id, chat_id)
        msg = (
            await db.execute(
                select(Message).where(Message.chat_id == chat.id, Message.tg_msg_id == tg_msg_id)
            )
        ).scalar_one_or_none()
        if msg is None:
            raise HTTPException(status_code=404, detail={"code": "chat.err.not_found"})
        return {
            "chat_id": chat.id,
            "chat_title": chat.title,
            "tg_msg_id": int(msg.tg_msg_id),
            "date": msg.published_at.isoformat() if msg.published_at else None,
            "sender": _sender_of(msg, chat),
            "text": (msg.text or "").replace("\n", " ")[:CITATION_TEXT_CHARS],
            "link": _msg_link(chat, int(msg.tg_msg_id)),
        }
