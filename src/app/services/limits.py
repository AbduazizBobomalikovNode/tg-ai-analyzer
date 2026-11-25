"""Limitlar (8-bosqich): foydalanuvchi bo'yicha chat tezligi va kunlik LLM byudjeti.

* `chat_rate` — daqiqasiga so'rovlar (jarayon xotirasi, api bitta worker).
* `daily_usage` — so'nggi 24 soatdagi tokenlar/xarajat (`conversation_messages`
  assistant qatorlari) **va** shu foydalanuvchining samarali limiti.
* `effective_budget` — samarali limit: foydalanuvchi override'i (`user_budgets`,
  migratsiya 0023) yoki global `LLM_DAILY_TOKEN_BUDGET` /
  `LLM_DAILY_COST_BUDGET_USD` (0 = cheksiz), ustiga faol "bugungi qo'shimcha"
  (boost), keyin ixtiyoriy qattiq shift (`LLM_DAILY_*_MAX`). **Har** byudjet
  tekshiruvi shu yerdan o'tadi — global sozlamani to'g'ridan-to'g'ri
  solishtirish boshqa joyda yo'q.
* `assert_within_budget` — oshsa `ChatError("budget")` → UI 429.
Byudjet **javobdan oldin** tekshiriladi (so'nggi javob biroz oshirishi mumkin — qabul).

Kesh: samarali limitning xom qatori Redis'da 60 s (`limits:budget:<uid>`),
o'zgarganda o'chiriladi — issiq yo'lda DB'ga qo'shimcha so'rov yo'q. Redis
nosoz bo'lsa to'g'ridan-to'g'ri DB'dan o'qiladi, DB ham yiqilsa — global
standart (fail-open, boshqa kesh'lar bilan bir uslub).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Account, Conversation, ConversationMessage, UserBudget
from app.logging import get_logger
from app.web.security import RateLimiter

log = get_logger(__name__)

chat_rate = RateLimiter(get_settings().chat_rate_per_minute, 60)
# Deep search alohida, qattiqroq shift: bitta deep so'rov ≈ 10x oddiy so'rov
# narxi (arxitektura v2, 11-bo'lim), shuning uchun umumiy chat kvotasidan
# alohida hisoblanadi.
deep_rate = RateLimiter(max(get_settings().deep_rate_per_minute, 1), 60)

#: Sarf oynasi — sirpanuvchi 24 soat (kalendar kuni emas).
WINDOW = timedelta(days=1)
#: Redis kesh: samarali limitning xom qatori.
CACHE_PREFIX = "limits:budget:"
CACHE_TTL_S = 60
#: Byudjet oxirgi marta qachon to'sgani (UI "oxirgi marta tugagan" satri).
HIT_PREFIX = "limits:budget_hit:"
HIT_TTL_S = 7 * 86400
#: Validatsiya chegaralari (API ham, servis ham).
MAX_TOKENS = 1_000_000_000
MAX_USD = 1000.0
MAX_BOOST_HOURS = 48
DEFAULT_TZ = "Asia/Tashkent"


# ─── samarali limit ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """`user_budgets` qatorining ORM'siz nusxasi (kesh shu shaklda saqlanadi)."""

    token_budget: int | None = None
    cost_budget_usd: float | None = None
    boost_tokens: int = 0
    boost_usd: float = 0.0
    boost_until: datetime | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "t": self.token_budget,
                "u": self.cost_budget_usd,
                "bt": self.boost_tokens,
                "bu": self.boost_usd,
                "until": self.boost_until.isoformat() if self.boost_until else None,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> BudgetRow | None:
        data = json.loads(raw)
        if not data:
            return None
        until = data.get("until")
        return cls(
            token_budget=None if data.get("t") is None else int(data["t"]),
            cost_budget_usd=None if data.get("u") is None else float(data["u"]),
            boost_tokens=int(data.get("bt") or 0),
            boost_usd=float(data.get("bu") or 0.0),
            boost_until=datetime.fromisoformat(until) if until else None,
        )

    @classmethod
    def of(cls, ub: Any) -> BudgetRow:
        return cls(
            token_budget=None if ub.token_budget is None else int(ub.token_budget),
            cost_budget_usd=None if ub.cost_budget_usd is None else float(ub.cost_budget_usd),
            boost_tokens=int(ub.boost_tokens or 0),
            boost_usd=float(ub.boost_usd or 0.0),
            boost_until=ub.boost_until,
        )

    def boost_active(self, now: datetime) -> bool:
        return (
            self.boost_until is not None
            and self.boost_until > now
            and (self.boost_tokens > 0 or self.boost_usd > 0)
        )


@dataclass(frozen=True, slots=True)
class Budget:
    """Samarali kunlik limit. `tokens` / `usd` = 0 — cheksiz."""

    tokens: int
    usd: float
    source: str = "global"  # "global" | "custom"
    boost_active: bool = False
    boost_until: datetime | None = None
    boost_tokens: int = 0
    boost_usd: float = 0.0
    global_tokens: int = 0
    global_usd: float = 0.0
    custom_tokens: int | None = None
    custom_usd: float | None = None
    ceiling_tokens: int | None = None
    ceiling_usd: float | None = None


def _limit(glob: float, custom: float | None, boost: float, active: bool, ceiling: float) -> float:
    """Bitta o'lcham (token yoki $) uchun samarali qiymat. 0 = cheksiz.

    * baza = override (bo'lsa) yoki global;
    * baza cheksiz (0) bo'lsa boost ma'nosiz — cheksizligicha qoladi, faqat
      foydalanuvchi o'zi 0 (cheksiz) qo'ygan va shift berilgan bo'lsa shiftga;
    * shift faqat foydalanuvchi o'zgartirgan qiymatga (override/boost)
      qo'llanadi — global standart admin qarori.
    """
    base = glob if custom is None else custom
    if base <= 0:
        return ceiling if (custom is not None and ceiling > 0) else 0
    value = base + (boost if active else 0)
    if (custom is not None or active) and ceiling > 0:
        value = min(value, ceiling)
    return value


def compute_budget(row: BudgetRow | None, *, now: datetime | None = None) -> Budget:
    """Sof funksiya: xom qator + joriy sozlamalar → samarali limit."""
    s = get_settings()
    now = now or datetime.now(UTC)
    row = row or BudgetRow()
    active = row.boost_active(now)
    c_tok = int(s.llm_daily_token_budget_max or 0)
    c_usd = float(s.llm_daily_cost_budget_usd_max or 0.0)
    custom = row.token_budget is not None or row.cost_budget_usd is not None
    return Budget(
        tokens=int(
            _limit(s.llm_daily_token_budget, row.token_budget, row.boost_tokens, active, c_tok)
        ),
        usd=round(
            float(
                _limit(
                    s.llm_daily_cost_budget_usd,
                    row.cost_budget_usd,
                    row.boost_usd,
                    active,
                    c_usd,
                )
            ),
            4,
        ),
        source="custom" if custom else "global",
        boost_active=active,
        boost_until=row.boost_until if active else None,
        boost_tokens=row.boost_tokens if active else 0,
        boost_usd=row.boost_usd if active else 0.0,
        global_tokens=int(s.llm_daily_token_budget),
        global_usd=float(s.llm_daily_cost_budget_usd),
        custom_tokens=row.token_budget,
        custom_usd=row.cost_budget_usd,
        ceiling_tokens=c_tok or None,
        ceiling_usd=c_usd or None,
    )


async def _redis() -> Any:
    import redis.asyncio as aioredis

    return aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]


async def _cache_get(user_id: int) -> tuple[bool, BudgetRow | None]:
    """`(topildi, qator)` — Redis nosoz yoki kalit yo'q → `(False, None)`."""
    try:
        r = await _redis()
        try:
            raw = await r.get(f"{CACHE_PREFIX}{int(user_id)}")
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("limits.cache_read_failed", error=str(exc)[:100])
        return False, None
    if raw is None:
        return False, None
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
        return True, BudgetRow.from_json(text)
    except (ValueError, TypeError, KeyError):
        return False, None


async def _cache_put(user_id: int, row: BudgetRow | None) -> None:
    try:
        r = await _redis()
        try:
            await r.set(
                f"{CACHE_PREFIX}{int(user_id)}",
                row.to_json() if row else "{}",
                ex=CACHE_TTL_S,
            )
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("limits.cache_write_failed", error=str(exc)[:100])


async def invalidate(user_id: int) -> None:
    """Keshni o'chiradi (yozuvdan keyin). Redis yo'q — TTL baribir 60 s."""
    try:
        r = await _redis()
        try:
            await r.delete(f"{CACHE_PREFIX}{int(user_id)}")
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("limits.cache_invalidate_failed", error=str(exc)[:100])


async def _read_row(session: AsyncSession, user_id: int) -> BudgetRow | None:
    ub = (
        await session.execute(select(UserBudget).where(UserBudget.user_id == int(user_id)))
    ).scalar_one_or_none()
    return BudgetRow.of(ub) if ub is not None else None


async def budget_row(session: AsyncSession, user_id: int) -> BudgetRow | None:
    """Kesh → DB. DB ham yiqilsa `None` (= global standart, fail-open)."""
    hit, row = await _cache_get(user_id)
    if hit:
        return row
    try:
        row = await _read_row(session, user_id)
    except Exception as exc:
        log.warning("limits.budget_read_failed", user_id=user_id, error=str(exc)[:160])
        return None
    await _cache_put(user_id, row)
    return row


async def effective_budget(session: AsyncSession, user_id: int) -> Budget:
    return compute_budget(await budget_row(session, user_id))


# ─── sarf ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Usage:
    tokens: int
    cost_usd: float
    requests: int
    token_limit: int
    cost_limit: float
    budget: Budget | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 4),
            "requests": self.requests,
            "token_limit": self.token_limit or None,
            "cost_limit": self.cost_limit or None,
            "tokens_pct": round(100 * self.tokens / self.token_limit, 1)
            if self.token_limit
            else None,
            "cost_pct": round(100 * self.cost_usd / self.cost_limit, 1)
            if self.cost_limit
            else None,
        }

    @property
    def exceeded(self) -> str | None:
        if self.token_limit and self.tokens >= self.token_limit:
            return "tokens"
        if self.cost_limit and self.cost_usd >= self.cost_limit:
            return "cost"
        return None


def _tokens_expr() -> Any:
    """Kunlik token hisobi: in + out + **thinking** (migratsiya 0022).

    Fikrlash tokenlari provider tomonida to'lanadi (output narxida); ular
    sanalmasa token byudjeti haqiqiy sarfdan past ko'rinardi. Eski qatorlarda
    ustun `NULL` — `coalesce` bilan 0.
    """
    return (
        ConversationMessage.tokens_in
        + ConversationMessage.tokens_out
        + func.coalesce(ConversationMessage.tokens_thinking, 0)
    )


async def daily_usage(session: AsyncSession, user_id: int) -> Usage:
    since = datetime.now(UTC) - WINDOW
    row = (
        await session.execute(
            select(
                func.coalesce(func.sum(_tokens_expr()), 0),
                func.coalesce(func.sum(ConversationMessage.cost_usd), 0.0),
                func.count(ConversationMessage.id),
            )
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(
                Conversation.user_id == user_id,
                ConversationMessage.role == "assistant",
                ConversationMessage.created_at >= since,
            )
        )
    ).one()
    budget = await effective_budget(session, user_id)
    return Usage(
        tokens=int(row[0] or 0),
        cost_usd=float(row[1] or 0),
        requests=int(row[2] or 0),
        token_limit=budget.tokens,
        cost_limit=budget.usd,
        budget=budget,
    )


async def _load_rows(session: AsyncSession) -> dict[int, BudgetRow]:
    """Barcha override qatorlari (kichik jadval: foydalanuvchi soni qadar).

    Nosozlik — bo'sh lug'at (hamma global standartda, fail-open).
    """
    try:
        rows = (await session.execute(select(UserBudget))).scalars().all()
        return {int(ub.user_id): BudgetRow.of(ub) for ub in rows}
    except Exception as exc:
        log.warning("limits.budget_rows_failed", error=str(exc)[:160])
        return {}


async def blocked_user_ids(session: AsyncSession) -> set[int]:
    """Kunlik byudjeti tugagan foydalanuvchilar (M3) — **bitta** jamlama so'rovda.

    Fon ishlari (embedding, hafta tugunlari) chat so'rovi emas, lekin ular ham
    LLM pulini yeydi. Har akkaunt uchun alohida `daily_usage` chaqirish qimmat,
    shuning uchun jamlama bir marta olinadi va egasi bo'yicha guruhlanadi.
    Limit har foydalanuvchi uchun `compute_budget` dan (override + boost).

    Global limit ham, override ham yo'q bo'lsa — bo'sh to'plam.
    """
    s = get_settings()
    overrides = await _load_rows(session)
    if not s.llm_daily_token_budget and not s.llm_daily_cost_budget_usd and not overrides:
        return set()
    since = datetime.now(UTC) - WINDOW
    rows = (
        await session.execute(
            select(
                Conversation.user_id,
                func.coalesce(func.sum(_tokens_expr()), 0),
                func.coalesce(func.sum(ConversationMessage.cost_usd), 0.0),
            )
            .join(ConversationMessage, ConversationMessage.conversation_id == Conversation.id)
            .where(
                ConversationMessage.role == "assistant",
                ConversationMessage.created_at >= since,
            )
            .group_by(Conversation.user_id)
        )
    ).all()
    now = datetime.now(UTC)
    blocked: set[int] = set()
    for user_id, tokens, cost in rows:
        b = compute_budget(overrides.get(int(user_id)), now=now)
        over_tokens = bool(b.tokens) and int(tokens or 0) >= b.tokens
        over_cost = bool(b.usd) and float(cost or 0) >= b.usd
        if over_tokens or over_cost:
            blocked.add(int(user_id))
    return blocked


async def blocked_account_ids(session: AsyncSession) -> set[int]:
    """Byudjeti tugagan egalarning akkauntlari — fon ishlari shularni chetlab o'tadi."""
    users = await blocked_user_ids(session)
    if not users:
        return set()
    rows = (
        await session.execute(select(Account.id).where(Account.user_id.in_(sorted(users))))
    ).scalars()
    return {int(x) for x in rows}


async def account_budget_blocked(session: AsyncSession, account_id: int) -> str | None:
    """Shu akkaunt egasining byudjet holati — `"tokens"` / `"cost"` / `None`."""
    owner = (
        await session.execute(select(Account.user_id).where(Account.id == account_id))
    ).scalar_one_or_none()
    if owner is None:
        return None
    return await check_daily_budget(session, int(owner))


async def _note_hit(user_id: int, kind: str) -> None:
    """Byudjet to'sgan paytni eslab qoladi (UI uchun; nosozlik — jim)."""
    try:
        r = await _redis()
        try:
            await r.set(
                f"{HIT_PREFIX}{int(user_id)}",
                json.dumps({"at": datetime.now(UTC).isoformat(), "kind": kind}),
                ex=HIT_TTL_S,
            )
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("limits.hit_write_failed", error=str(exc)[:100])


async def last_hit(user_id: int) -> dict[str, Any] | None:
    try:
        r = await _redis()
        try:
            raw = await r.get(f"{HIT_PREFIX}{int(user_id)}")
        finally:
            await r.aclose()
    except Exception as exc:
        log.debug("limits.hit_read_failed", error=str(exc)[:100])
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


async def check_daily_budget(session: AsyncSession, user_id: int) -> str | None:
    """Byudjet holati — **xato ko'tarmaydi**: `"tokens"` / `"cost"` / `None`.

    Deep search uchun kerak: u so'rovni HTTP xatosi bilan emas, `error`
    hodisasi bilan to'xtatadi (SSE oqimi allaqachon ochilgan bo'lishi mumkin),
    shuning uchun tekshiruv va uni qanday e'lon qilish bir-biridan ajratilgan.
    """
    over = (await daily_usage(session, user_id)).exceeded
    if over:
        await _note_hit(user_id, over)
    return over


async def assert_within_budget(session: AsyncSession, user_id: int) -> Usage:
    from app.services.chat_service import ChatError

    u = await daily_usage(session, user_id)
    if u.exceeded:
        await _note_hit(user_id, u.exceeded)
        raise ChatError("budget", u.exceeded)
    return u


# ─── sozlamalar (Sozlamalar → Byudjet) ───────────────────────────────────────


def _zone(tz: str | None) -> ZoneInfo:
    for name in (tz or DEFAULT_TZ, DEFAULT_TZ, "UTC"):
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            continue
    return ZoneInfo("UTC")


def next_local_midnight(tz: str | None, *, now: datetime | None = None) -> datetime:
    """Keyingi lokal yarim tun (UTC da). Boost standart muddati."""
    now = now or datetime.now(UTC)
    zone = _zone(tz)
    local = now.astimezone(zone)
    midnight = datetime.combine(local.date() + timedelta(days=1), time(0, 0), tzinfo=zone)
    return midnight.astimezone(UTC)


async def _get_or_new(session: AsyncSession, user_id: int) -> UserBudget:
    ub = (
        await session.execute(select(UserBudget).where(UserBudget.user_id == int(user_id)))
    ).scalar_one_or_none()
    if ub is None:
        ub = UserBudget(user_id=int(user_id), boost_tokens=0, boost_usd=0.0)
        session.add(ub)
    return ub


_UNSET: Any = object()


async def set_user_budget(
    session: AsyncSession,
    user_id: int,
    *,
    token_budget: int | None = _UNSET,
    cost_budget_usd: float | None = _UNSET,
) -> None:
    """Override'ni yozadi. Berilmagan maydon o'zgarmaydi, `None` — globalga qaytaradi."""
    ub = await _get_or_new(session, user_id)
    if token_budget is not _UNSET:
        ub.token_budget = None if token_budget is None else min(int(token_budget), MAX_TOKENS)
    if cost_budget_usd is not _UNSET:
        ub.cost_budget_usd = (
            None if cost_budget_usd is None else round(min(float(cost_budget_usd), MAX_USD), 4)
        )
    ub.updated_at = datetime.now(UTC)
    await session.flush()
    await invalidate(user_id)
    log.info(
        "limits.budget_updated",
        user_id=user_id,
        action="set",
        token_budget=ub.token_budget,
        cost_budget_usd=ub.cost_budget_usd,
    )


async def add_boost(
    session: AsyncSession, user_id: int, *, tokens: int, usd: float, until: datetime
) -> None:
    """Bugungi qo'shimcha. Faol boost bo'lsa **ustiga qo'shiladi**, muddat — kechrog'i."""
    now = datetime.now(UTC)
    ub = await _get_or_new(session, user_id)
    active = BudgetRow.of(ub).boost_active(now)
    if active:
        ub.boost_tokens = min(int(ub.boost_tokens or 0) + int(tokens), MAX_TOKENS)
        ub.boost_usd = round(min(float(ub.boost_usd or 0) + float(usd), MAX_USD), 4)
        ub.boost_until = max(ub.boost_until or until, until)
    else:
        ub.boost_tokens = min(int(tokens), MAX_TOKENS)
        ub.boost_usd = round(min(float(usd), MAX_USD), 4)
        ub.boost_until = until
    ub.updated_at = now
    await session.flush()
    await invalidate(user_id)
    log.info(
        "limits.budget_updated",
        user_id=user_id,
        action="boost",
        boost_tokens=ub.boost_tokens,
        boost_usd=ub.boost_usd,
        boost_until=ub.boost_until.isoformat() if ub.boost_until else None,
    )


async def clear_boost(session: AsyncSession, user_id: int) -> None:
    ub = (
        await session.execute(select(UserBudget).where(UserBudget.user_id == int(user_id)))
    ).scalar_one_or_none()
    if ub is not None:
        ub.boost_tokens, ub.boost_usd, ub.boost_until = 0, 0.0, None
        ub.updated_at = datetime.now(UTC)
        await session.flush()
    await invalidate(user_id)
    log.info("limits.budget_updated", user_id=user_id, action="boost_clear")


async def _resets_at(session: AsyncSession, user_id: int, usage: Usage) -> datetime | None:
    """To'sib turgan byudjet qachon bo'shashi: eng eski qatorlar 24 soatlik
    oynadan chiqib, sarf limitdan pastga tushadigan payt (sirpanuvchi oyna)."""
    if not usage.exceeded:
        return None
    since = datetime.now(UTC) - WINDOW
    rows = (
        await session.execute(
            select(
                ConversationMessage.created_at,
                _tokens_expr(),
                func.coalesce(ConversationMessage.cost_usd, 0.0),
            )
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(
                Conversation.user_id == user_id,
                ConversationMessage.role == "assistant",
                ConversationMessage.created_at >= since,
            )
            .order_by(ConversationMessage.created_at)
        )
    ).all()
    tokens, cost = usage.tokens, usage.cost_usd
    for created_at, tok, usd in rows:
        tokens -= int(tok or 0)
        cost -= float(usd or 0)
        ok_tokens = not usage.token_limit or tokens < usage.token_limit
        ok_cost = not usage.cost_limit or cost < usage.cost_limit
        if ok_tokens and ok_cost:
            return created_at + WINDOW  # type: ignore[no-any-return]
    return None


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


async def budget_view(session: AsyncSession, user_id: int, *, tz: str | None) -> dict[str, Any]:
    """`GET /api/settings/budget` javobi (UI kartasi shu shakldan chiziladi)."""
    usage = await daily_usage(session, user_id)
    b = usage.budget or compute_budget(None)
    ceiling = (
        {"tokens": b.ceiling_tokens, "usd": b.ceiling_usd}
        if (b.ceiling_tokens or b.ceiling_usd)
        else None
    )
    return {
        "usage_24h": {
            "tokens": usage.tokens,
            "usd": round(usage.cost_usd, 4),
            "requests": usage.requests,
        },
        "limits": {
            "tokens": b.tokens or None,
            "usd": b.usd or None,
            "source": b.source,
            "global": {"tokens": b.global_tokens or None, "usd": b.global_usd or None},
            "custom": {"tokens": b.custom_tokens, "usd": b.custom_usd},
            "ceiling": ceiling,
        },
        "boost": {"tokens": b.boost_tokens, "usd": b.boost_usd, "until": _iso(b.boost_until)}
        if b.boost_active
        else None,
        "blocked": usage.exceeded,
        "last_hit": await last_hit(user_id),
        "resets_hint": {
            "window_hours": int(WINDOW.total_seconds() // 3600),
            "at": _iso(await _resets_at(session, user_id, usage)),
        },
        "tz": tz or DEFAULT_TZ,
        "boost_default_until": _iso(next_local_midnight(tz)),
        "max": {"tokens": MAX_TOKENS, "usd": MAX_USD, "hours": MAX_BOOST_HOURS},
    }


# ─── tezlik ──────────────────────────────────────────────────────────────────


def check_chat_rate(user_id: int) -> bool:
    return chat_rate.check(str(user_id))


def check_deep_rate(user_id: int) -> bool:
    """Deep so'rov shifti. `DEEP_RATE_PER_MINUTE=0` — rejim o'chiq (hech biri o'tmaydi)."""
    if get_settings().deep_rate_per_minute <= 0:
        return False
    return deep_rate.check(str(user_id))


def refund_deep_rate(user_id: int) -> None:
    """Bajarilmagan deep so'rovining shift slotini qaytaradi.

    Tasdiqsiz `exhaustive` — javob emas, **baho**: reja quriladi va yugurish
    to'xtaydi. Uni kvotaga yozish foydalanuvchini o'zi tanlamagan ish uchun
    jazolardi (uch marta baho ko'rsa, tasdiqlashga kvota qolmasdi).
    """
    if get_settings().deep_rate_per_minute <= 0:
        return
    deep_rate.refund(str(user_id))
