"""Shaxsiy briefing API (3-bosqich/B).

GET /api/briefings?limit=&account_id=     oxirgi briefinglar (ro'yxat)
GET /api/briefings/{day}?account_id=      bir kunning to'liq briefingi
GET /api/settings/briefing?account_id=    joriy sozlama
PUT /api/settings/briefing                {account_id, tz, quiet_from, quiet_to,
                                           enabled, hour, chats}

Egalik har chaqiruvda tekshiriladi: `accounts.user_id == sessiya user_id`.
O'zgartiruvchi endpoint CSRF header'ini (`X-Requested-With: fetch`) talab
qiladi — boshqa router'lar bilan bir xil qoida.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any
from zoneinfo import ZoneInfoNotFoundError, available_timezones

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Account, Briefing, Chat
from app.web.security import WebIdentity, require_csrf, require_identity

router = APIRouter(prefix="/api", tags=["briefings"])

MAX_BRIEFING_CHATS = 100


class BriefingSettingsIn(BaseModel):
    account_id: int
    tz: str = Field(default="Asia/Tashkent", max_length=64)
    quiet_from: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    quiet_to: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    enabled: bool = False
    hour: int = Field(default=8, ge=0, le=23)
    #: null = barcha chatlar; ro'yxat = faqat shu `chats.id` lar
    chats: list[int] | None = None

    @field_validator("tz")
    @classmethod
    def _known_tz(cls, v: str) -> str:
        if v not in available_timezones():
            raise ValueError("unknown timezone")
        return v

    @field_validator("chats")
    @classmethod
    def _cap_chats(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        if len(v) > MAX_BRIEFING_CHATS:
            raise ValueError("too many chats")
        return sorted({int(x) for x in v})


def _parse_time(value: str | None) -> Any:
    if not value:
        return None
    from datetime import time as _time

    hh, mm = value.split(":")
    return _time(hour=int(hh), minute=int(mm))


async def _own_account(db: Any, user_id: int, account_id: int) -> Account:
    acc: Account | None = await db.get(Account, account_id)
    if acc is None or acc.user_id != user_id:
        raise HTTPException(status_code=404, detail={"code": "account.not_found"})
    return acc


async def _account_ids(db: Any, user_id: int, account_id: int | None) -> list[int]:
    if account_id is not None:
        await _own_account(db, user_id, account_id)
        return [account_id]
    rows = (await db.execute(select(Account.id).where(Account.user_id == user_id))).scalars().all()
    return [int(x) for x in rows]


def _row_summary(row: Briefing) -> dict[str, Any]:
    structured = row.structured or {}
    return {
        "account_id": row.account_id,
        "day": row.day.isoformat(),
        "tldr": structured.get("tldr") or "",
        "chats": len(structured.get("chats") or []),
        "tokens": int(row.tokens_in or 0) + int(row.tokens_out or 0),
        "cost_usd": row.cost_usd,
        "model": row.model,
        "delivered_at": row.delivered_at.isoformat() if row.delivered_at else None,
        "delivery_error": row.delivery_error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/briefings")
async def list_briefings(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    limit: Annotated[int, Query(ge=1, le=90)] = 14,
    account_id: Annotated[int | None, Query()] = None,
) -> dict[str, Any]:
    async with session_scope() as db:
        ids = await _account_ids(db, ident.user_id, account_id)
        if not ids:
            return {"items": []}
        rows = (
            (
                await db.execute(
                    select(Briefing)
                    .where(Briefing.account_id.in_(ids))
                    .order_by(Briefing.day.desc(), Briefing.account_id)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return {"items": [_row_summary(r) for r in rows]}


@router.get("/briefings/{day}")
async def get_briefing(
    day: date,
    ident: Annotated[WebIdentity, Depends(require_identity)],
    account_id: Annotated[int | None, Query()] = None,
) -> dict[str, Any]:
    async with session_scope() as db:
        ids = await _account_ids(db, ident.user_id, account_id)
        if not ids:
            raise HTTPException(status_code=404, detail={"code": "briefing.not_found"})
        row = (
            await db.execute(
                select(Briefing)
                .where(Briefing.account_id.in_(ids), Briefing.day == day)
                .order_by(Briefing.account_id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "briefing.not_found"})
        return {**_row_summary(row), "text": row.text, "structured": row.structured or {}}


@router.get("/settings/briefing")
async def get_briefing_settings(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    account_id: Annotated[int | None, Query()] = None,
) -> dict[str, Any]:
    async with session_scope() as db:
        ids = await _account_ids(db, ident.user_id, account_id)
        if not ids:
            raise HTTPException(status_code=404, detail={"code": "account.not_found"})
        acc = await _own_account(db, ident.user_id, ids[0])
        chats = (
            await db.execute(
                select(Chat.id, Chat.title, Chat.type)
                .where(Chat.account_id == acc.id)
                .order_by(Chat.title)
            )
        ).all()
        return {
            "settings": _settings_payload(acc),
            "available_chats": [{"id": int(c[0]), "title": c[1], "type": c[2]} for c in chats],
        }


def _settings_payload(acc: Account) -> dict[str, Any]:
    return {
        "account_id": acc.id,
        "tz": acc.tz,
        "quiet_from": acc.quiet_from.strftime("%H:%M") if acc.quiet_from else None,
        "quiet_to": acc.quiet_to.strftime("%H:%M") if acc.quiet_to else None,
        "enabled": bool(acc.briefing_enabled),
        "hour": int(acc.briefing_hour or 0),
        "chats": acc.briefing_chats,
    }


@router.put("/settings/briefing", dependencies=[Depends(require_csrf)])
async def put_briefing_settings(
    body: BriefingSettingsIn, ident: Annotated[WebIdentity, Depends(require_identity)]
) -> dict[str, Any]:
    async with session_scope() as db:
        acc = await _own_account(db, ident.user_id, body.account_id)
        if body.chats is not None:
            owned = set(
                int(x)
                for x in (await db.execute(select(Chat.id).where(Chat.account_id == acc.id)))
                .scalars()
                .all()
            )
            unknown = [c for c in body.chats if c not in owned]
            if unknown:
                raise HTTPException(status_code=400, detail={"code": "briefing.err.unknown_chat"})
        try:
            acc.tz = body.tz
            acc.quiet_from = _parse_time(body.quiet_from)
            acc.quiet_to = _parse_time(body.quiet_to)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise HTTPException(
                status_code=400, detail={"code": "briefing.err.bad_settings"}
            ) from exc
        acc.briefing_enabled = body.enabled
        acc.briefing_hour = body.hour
        acc.briefing_chats = body.chats
        await db.flush()
        return {"settings": _settings_payload(acc)}
