"""Constructor championship standings (external API)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import constructor_standings as cs

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("")
def get_standings(
    year: int = Query(2025, ge=2000, le=2035),
    round: int | None = Query(None, ge=1, le=30),
    before_event: bool = Query(False),
) -> dict:
    try:
        if round is not None:
            if before_event:
                if round <= 1:
                    return {
                        "season": year,
                        "event_round": round,
                        "round_used": 0,
                        "source": "ergast-compatible",
                        "entries": [],
                    }
                data = cs.fetch_constructors_standings_for_round(year, round - 1)
                data["event_round"] = round
                data["round_used"] = round - 1
                return data
            data = cs.fetch_constructors_standings_for_round(year, round)
            data["event_round"] = round
            data["round_used"] = round
            return data
        return cs.fetch_constructors_standings(year)
    except Exception as e:
        raise HTTPException(502, f"Could not load standings: {e}") from e
