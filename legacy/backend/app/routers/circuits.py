"""2025 circuit calendar for the UI."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.circuit_calendar import circuits_for_api

router = APIRouter(prefix="/circuits", tags=["circuits"])


@router.get("")
def get_circuits() -> dict:
    return circuits_for_api()
