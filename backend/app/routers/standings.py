"""Constructor championship standings (external API)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import constructor_standings as cs

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("")
def get_standings(year: int = Query(2025, ge=2000, le=2035)) -> dict:
    try:
        return cs.fetch_constructors_standings(year)
    except Exception as e:
        raise HTTPException(502, f"Could not load standings: {e}") from e
