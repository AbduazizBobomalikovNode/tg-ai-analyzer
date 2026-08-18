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


@router.get("/ingestion")
async def ingestion(ident: Annotated[WebIdentity, Depends(require_identity)]) -> dict[str, Any]:
    async with session_scope() as db:
        return await stats.ingestion_overview(db, ident.user_id)
