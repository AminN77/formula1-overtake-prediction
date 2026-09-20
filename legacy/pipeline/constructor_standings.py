"""Fetch F1 constructor standings for pipeline and backend use.

Shared by backend inference and training pipeline (`pipeline/team_features.py`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

import requests

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

# FastF1/app team naming aliases to canonical app labels.
APP_TEAM_ALIASES: dict[str, str] = {
    "red bull": "Red Bull Racing",
    "red bull racing": "Red Bull Racing",
    "oracle red bull racing": "Red Bull Racing",
    "mclaren": "McLaren",
    "mclaren f1 team": "McLaren",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "mercedes": "Mercedes",
    "mercedes-amg petronas f1 team": "Mercedes",
    "aston martin": "Aston Martin",
    "aston martin aramco f1 team": "Aston Martin",
    "alpine": "Alpine",
    "alpine f1 team": "Alpine",
    "bwt alpine f1 team": "Alpine",
    "williams": "Williams",
    "williams racing": "Williams",
    "rb": "RB",
    "racing bulls": "RB",
    "visa cash app racing bulls f1 team": "RB",
    "visa cash app rb f1 team": "RB",
    "alphatauri": "RB",
    "scuderia alphatauri": "RB",
    "kick sauber": "Kick Sauber",
    "stake f1 team kick sauber": "Kick Sauber",
    "alfa romeo": "Kick Sauber",
    "alfa romeo f1 team": "Kick Sauber",
    "alfa romeo racing": "Kick Sauber",
    "sauber": "Kick Sauber",
    "haas f1 team": "Haas F1 Team",
    "haas": "Haas F1 Team",
    "alphatauri": "RB",
    "rb f1 team": "RB",
}

ERGAST_BASE_URLS: tuple[str, ...] = (
    "https://api.jolpi.ca/ergast/f1",
    "https://ergast.com/api/f1",
)


def normalize_app_team_name(team_name: str) -> str:
    raw = str(team_name or "").strip()
    if not raw:
        return raw
    key = raw.lower()
    return APP_TEAM_ALIASES.get(key, raw)


@lru_cache(maxsize=16)
def _fetch_year_json(year: int) -> dict[str, Any]:
    url = f"https://f1api.dev/api/{year}/constructors-championship"
    req = urllib.request.Request(url, headers={"User-Agent": "f1-overtake-prediction/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


@lru_cache(maxsize=256)
def _fetch_round_json(year: int, round_number: int) -> dict[str, Any]:
    last_exc: Exception | None = None
    for base in ERGAST_BASE_URLS:
        url = f"{base}/{year}/{round_number}/constructorStandings.json"
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "f1-overtake-prediction/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if lists:
                return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError(f"No constructor standings data for {year} round {round_number}")


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


def fetch_constructors_standings_for_round(year: int, round_number: int) -> dict[str, Any]:
    """Return normalized standings for a specific season round (Ergast/Jolpica)."""
    if round_number < 1:
        raise ValueError("round_number must be >= 1")
    raw = _fetch_round_json(year, round_number)
    lists = raw.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    entries: list[dict[str, Any]] = []
    if lists:
        rows = lists[0].get("ConstructorStandings", [])
        for row in rows:
            c = row.get("Constructor", {}) if isinstance(row.get("Constructor"), dict) else {}
            name = str(c.get("name") or "").strip()
            app_team = normalize_app_team_name(name)
            entries.append(
                {
                    "position": int(row.get("position") or 0),
                    "team_id": str(c.get("constructorId") or ""),
                    "points": float(row.get("points") or 0),
                    "wins": int(row.get("wins") or 0),
                    "app_team": app_team,
                    "team_name": name or None,
                }
            )
    entries.sort(key=lambda e: e["position"])
    return {
        "season": int(year),
        "round": int(round_number),
        "source": "ergast-compatible",
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
    return {normalize_app_team_name(str(e["app_team"])): int(e["position"]) for e in data.get("entries") or []}


def standings_positions_by_year_round(year: int, round_number: int) -> dict[str, int]:
    """Map app team -> constructor standing at the specified season round."""
    try:
        data = fetch_constructors_standings_for_round(year, round_number)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return {}
    return {normalize_app_team_name(str(e["app_team"])): int(e["position"]) for e in data.get("entries") or []}


def standings_positions_before_event(year: int, event_round: int) -> dict[str, int]:
    """Map app team -> standings before event round (same season, round-1).

    For round 1 there is no prior same-season standing; returns empty dict so caller can fallback.
    """
    if event_round <= 1:
        return {}
    return standings_positions_by_year_round(year, event_round - 1)


def clear_standings_cache() -> None:
    """Test helper."""
    _fetch_year_json.cache_clear()
    _fetch_round_json.cache_clear()
