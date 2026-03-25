"""Fetch F1 constructor championship standings from f1api.dev (cached).

Shared by backend inference and training pipeline (`pipeline/team_features.py`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

# Maps f1api.dev `teamId` -> app dropdown name (see schema_builder.TEAMS).
TEAM_ID_TO_APP_NAME: dict[str, str] = {
    "red_bull": "Red Bull Racing",
    "mclaren": "McLaren",
    "ferrari": "Ferrari",
    "mercedes": "Mercedes",
    "aston_martin": "Aston Martin",
    "alpine": "Alpine",
    "williams": "Williams",
    "rb": "RB",
    "sauber": "Kick Sauber",
    "haas": "Haas F1 Team",
}


@lru_cache(maxsize=16)
def _fetch_year_json(year: int) -> dict[str, Any]:
    url = f"https://f1api.dev/api/{year}/constructors-championship"
    req = urllib.request.Request(url, headers={"User-Agent": "f1-overtake-prediction/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_constructors_standings(year: int) -> dict[str, Any]:
    """Return normalized standings for API/UI. Raises on network/parse errors."""
    raw = _fetch_year_json(year)
    rows = raw.get("constructors_championship") or []
    entries: list[dict[str, Any]] = []
    for row in rows:
        tid = str(row.get("teamId") or "")
        app_team = TEAM_ID_TO_APP_NAME.get(tid)
        if app_team is None:
            continue
        team = row.get("team") or {}
        entries.append(
            {
                "position": int(row.get("position") or 0),
                "team_id": tid,
                "points": float(row.get("points") or 0),
                "wins": int(row.get("wins") or 0),
                "app_team": app_team,
                "team_name": team.get("teamName") if isinstance(team, dict) else None,
            }
        )
    entries.sort(key=lambda e: e["position"])
    return {
        "season": int(raw.get("season") or year),
        "source": "f1api.dev",
        "entries": entries,
    }


def constructor_position_for_team(year: int, app_team_name: str) -> int | None:
    """Championship position (1 = best) for an app team label, or None if unknown."""
    try:
        data = fetch_constructors_standings(year)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None
    for e in data.get("entries") or []:
        if e.get("app_team") == app_team_name:
            return int(e["position"])
    return None


def standings_positions_by_year_team(year: int) -> dict[str, int]:
    """Map app team name -> championship position (1 = best). Empty dict on failure."""
    try:
        data = fetch_constructors_standings(year)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return {}
    return {str(e["app_team"]): int(e["position"]) for e in data.get("entries") or []}


def clear_standings_cache() -> None:
    """Test helper."""
    _fetch_year_json.cache_clear()
