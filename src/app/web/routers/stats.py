"""Dashboard API.

GET /api/stats/overview?days=30   so'rovlar, tokenlar, xarajat, sifat, model/strategiya kesimi
GET /api/stats/ingestion          chatlar/xabarlar/embedding/snapshot holati
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.db.base import session_scope
from app.services import stats
from app.web.security import WebIdentity, require_identity

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/overview")
async def overview(
    ident: Annotated[WebIdentity, Depends(require_identity)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> dict[str, Any]:
    async with session_scope() as db:
        return await stats.overview(db, ident.user_id, days=days)


@router.get("/system")
async def system(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    """Chuqur sog'liq: DB, Redis, worker heartbeat'lari, kutayotgan amallar, kunlik byudjet."""
    import time

    from sqlalchemy import text

    from app.config import get_settings
    from app.observability import heartbeats
    from app.services import limits

    s = get_settings()
    out: dict[str, Any] = {"version": s.app_version, "env": s.env}
    t0 = time.perf_counter()
    try:
        async with session_scope() as db:
            await db.execute(text("SELECT 1"))
            out["db"] = {"ok": True, "ms": int((time.perf_counter() - t0) * 1000)}
            out["budget"] = (await limits.daily_usage(db, ident.user_id)).to_dict()
    except Exception as exc:
        out["db"] = {"ok": False, "error": str(exc)[:120]}
    t1 = time.perf_counter()
    hb = await heartbeats()
    out["redis"] = {"ok": "_error" not in hb, "ms": int((time.perf_counter() - t1) * 1000)}
    out["heartbeats"] = hb
    return out


@router.get("/ingestion")
async def ingestion(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        return await stats.ingestion_overview(db, ident.user_id)
