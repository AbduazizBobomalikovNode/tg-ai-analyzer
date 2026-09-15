"""3-bosqich/C API: alert qoidalari (CRUD) va topilgan hodisalar.

GET    /api/alerts?account_id=            qoidalar ro'yxati (egasiniki)
POST   /api/alerts                        yangi qoida
PUT    /api/alerts/{id}                   tahrir (pattern o'zgarsa vektor bekor)
DELETE /api/alerts/{id}                   o'chirish
GET    /api/alerts/{id}/events            oxirgi hodisalar (matn snippet'i bilan)

Egalik **har chaqiruvda** tekshiriladi: qoida → akkaunt → `accounts.user_id`.
Qoida id'si taxmin qilinadigan ketma-ket son, shuning uchun "id bilan kirish"
yagona to'siq bo'lmasligi kerak — 404 (403 emas): mavjudlik ham oshkor
qilinmaydi.

O'zgartiruvchi uchta metod `require_csrf` ga bog'langan (`X-Requested-With`).
Hodisa matni `alerts.snippet` orqali chiqadi: `untrusted.neutralize` +
`html.escape` — u ham botga, ham brauzerga bir xil xavfsiz shaklda boradi.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, AlertEvent, AlertRule, Chat, Message
from app.services import alerts as AL
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

EVENTS_LIMIT = 50


class AlertIn(BaseModel):
    account_id: int
    chat_id: int | None = None  # null = barcha chatlar
    kind: str = Field(default="keyword", pattern="^(keyword|semantic)$")
    label: str = Field(default="", max_length=64)
    pattern: str = Field(min_length=1, max_length=AL.PATTERN_MAX_CHARS)
    threshold: float = Field(default=0.80, ge=AL.THRESHOLD_MIN, le=AL.THRESHOLD_MAX)
    enabled: bool = True
    cooldown_minutes: int = Field(default=30, ge=0, le=AL.COOLDOWN_MAX)


class AlertPatch(BaseModel):
    chat_id: int | None = None
    kind: str | None = Field(default=None, pattern="^(keyword|semantic)$")
    label: str | None = Field(default=None, max_length=64)
    pattern: str | None = Field(default=None, max_length=AL.PATTERN_MAX_CHARS)
    threshold: float | None = Field(default=None, ge=AL.THRESHOLD_MIN, le=AL.THRESHOLD_MAX)
    enabled: bool | None = None
    cooldown_minutes: int | None = Field(default=None, ge=0, le=AL.COOLDOWN_MAX)


def _fail(exc: AL.AlertError) -> HTTPException:
    status = {"not_found": 404, "too_many": 409}.get(exc.code, 400)
    return HTTPException(status_code=status, detail={"code": f"alert.err.{exc.code}"})


async def _own_account(db: Any, user_id: int, account_id: int) -> Account:
    acc: Account | None = await db.get(Account, int(account_id))
    if acc is None or acc.user_id != user_id:
        raise HTTPException(status_code=404, detail={"code": "account.not_found"})
    return acc


async def _own_rule(db: Any, user_id: int, rule_id: int) -> AlertRule:
    rule: AlertRule | None = await db.get(AlertRule, int(rule_id))
    if rule is None:
        raise HTTPException(status_code=404, detail={"code": "alert.err.not_found"})
    await _own_account(db, user_id, rule.account_id)
    return rule


async def _check_chat(db: Any, account_id: int, chat_id: int | None) -> None:
    if chat_id is None:
        return
    chat = await db.get(Chat, int(chat_id))
    if chat is None or chat.account_id != int(account_id):
        raise HTTPException(status_code=404, detail={"code": "alert.err.not_found"})


@router.get("")
async def list_alerts(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    account_id: int | None = Query(default=None),
) -> dict[str, Any]:
    async with session_scope() as db:
        stmt = (
            select(AlertRule)
            .join(Account, Account.id == AlertRule.account_id)
            .where(Account.user_id == ident.user_id)
            .order_by(AlertRule.id.desc())
        )
        if account_id is not None:
            await _own_account(db, ident.user_id, account_id)
            stmt = stmt.where(AlertRule.account_id == int(account_id))
        rules = list((await db.execute(stmt)).scalars().all())
        return {"items": [AL.rule_dict(r) for r in rules]}


@router.post("", dependencies=[Depends(require_csrf)], status_code=201)
async def create_alert(
    body: AlertIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    s = get_settings()
    async with session_scope() as db:
        await _own_account(db, ident.user_id, body.account_id)
        await _check_chat(db, body.account_id, body.chat_id)
        try:
            AL.validate(body.kind, body.pattern, body.threshold, body.cooldown_minutes)
        except AL.AlertError as exc:
            raise _fail(exc) from exc
        n = (
            await db.execute(
                select(func.count())
                .select_from(AlertRule)
                .where(AlertRule.account_id == int(body.account_id))
            )
        ).scalar() or 0
        if int(n) >= s.alerts_max_rules_per_account:
            raise _fail(AL.AlertError("too_many"))
        rule = AlertRule(
            account_id=int(body.account_id),
            chat_id=body.chat_id,
            kind=body.kind,
            label=body.label.strip(),
            pattern=body.pattern.strip(),
            threshold=float(body.threshold),
            enabled=bool(body.enabled),
            cooldown_minutes=int(body.cooldown_minutes),
            last_checked_msg_id=0,
        )
        db.add(rule)
        await db.flush()
        return {"rule": AL.rule_dict(rule)}


@router.put("/{rule_id}", dependencies=[Depends(require_csrf)])
async def update_alert(
    rule_id: int, body: AlertPatch, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        rule = await _own_rule(db, ident.user_id, rule_id)
        data = body.model_dump(exclude_unset=True)
        kind = str(data.get("kind", rule.kind))
        pattern = str(data.get("pattern", rule.pattern) or "")
        threshold = float(data.get("threshold", rule.threshold) or 0)
        cooldown = int(data.get("cooldown_minutes", rule.cooldown_minutes) or 0)
        try:
            AL.validate(kind, pattern, threshold, cooldown)
        except AL.AlertError as exc:
            raise _fail(exc) from exc
        if "chat_id" in data:
            await _check_chat(db, rule.account_id, data["chat_id"])
            rule.chat_id = data["chat_id"]
        # Naqsh yoki tur o'zgarsa keshlangan vektor yaroqsiz — skaner qayta embed qiladi.
        if pattern.strip() != (rule.pattern or "").strip() or kind != str(rule.kind):
            rule.embedding = None
        rule.kind = kind
        rule.pattern = pattern.strip()
        rule.threshold = threshold
        rule.cooldown_minutes = cooldown
        if "label" in data and data["label"] is not None:
            rule.label = str(data["label"]).strip()
        if "enabled" in data and data["enabled"] is not None:
            rule.enabled = bool(data["enabled"])
        await db.flush()
        return {"rule": AL.rule_dict(rule)}


@router.delete("/{rule_id}", dependencies=[Depends(require_csrf)])
async def delete_alert(
    rule_id: int, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        rule = await _own_rule(db, ident.user_id, rule_id)
        await db.execute(delete(AlertRule).where(AlertRule.id == rule.id))
        return {"ok": True}


@router.get("/{rule_id}/events")
async def rule_events(
    rule_id: int,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: int = Query(default=20, ge=1, le=EVENTS_LIMIT),
) -> dict[str, Any]:
    async with session_scope() as db:
        rule = await _own_rule(db, ident.user_id, rule_id)
        rows = (
            await db.execute(
                select(AlertEvent, Chat, Message)
                .join(Chat, Chat.id == AlertEvent.chat_id)
                .join(Message, Message.id == AlertEvent.message_id)
                .where(AlertEvent.rule_id == rule.id)
                .order_by(AlertEvent.created_at.desc())
                .limit(limit)
            )
        ).all()
        return {"items": [AL.event_dict(ev, chat, msg) for ev, chat, msg in rows]}


__all__ = ["router"]
