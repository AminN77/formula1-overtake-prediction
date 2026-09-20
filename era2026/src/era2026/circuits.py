"""
Circuit character, as physics rather than as outcome.

An earlier design collapsed `track` into a historical **overtake rate** per
circuit. That failed its validation: Spearman rho +0.09 between the 2022-2025
per-circuit rate and 2026 actuals. Circuit *outcomes* did not survive the
regulation change.

Circuit *physics* does. Measured across the 13 circuits raced in both eras:

    mid-sector speed trap (i2)   rho = +0.974   p < 0.0001
    straight dominance           rho = +0.967   p < 0.0001
    sector-1 speed trap (i1)     rho = +0.966   p < 0.0001
    race distance                rho = +0.999   p < 0.0001
    top speed (SpeedST)          rho = +0.461   p = 0.11      EXCLUDED

Top speed is deliberately excluded. It is the one property the 2026 power unit
and active aero actually changed, and including it would reintroduce exactly
the kind of stale quantity the overtake-rate prior failed on.

Why this is not the same mistake twice: a historical overtake rate encodes what
*happened* under a ruleset that no longer exists, while a speed trap encodes the
shape of the track, which the cars still drive. The distinction is testable, and
it was tested before being used.

Circuits with no prior history (Madrid, new for 2026) receive the cross-circuit
median and are flagged, which is what a production system must do for any new
venue.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from era2026.features import TIER_1

#: Added to the registry by :func:`register`.
CIRCUIT_FEATURES: dict[str, str] = {
    "circuit_speed_i1": TIER_1,
    "circuit_speed_i2": TIER_1,
    "circuit_straight_dominance": TIER_1,
    "circuit_known": TIER_1,
}

#: 2026 event names to the legacy `track` label. Barcelona keeps its own name in
#: 2026 because Madrid took over the "Spanish Grand Prix" title.
EVENT_TO_TRACK = {
    "Australian Grand Prix": "MELBOURNE",
    "Chinese Grand Prix": "SHANGHAI",
    "Japanese Grand Prix": "SUZUKA",
    "Miami Grand Prix": "MIAMI",
    "Canadian Grand Prix": "MONTRÉAL",
    "Monaco Grand Prix": "MONACO",
    "Barcelona Grand Prix": "BARCELONA",
    "Austrian Grand Prix": "SPIELBERG",
    "British Grand Prix": "SILVERSTONE",
    "Belgian Grand Prix": "SPA-FRANCORCHAMPS",
    "Hungarian Grand Prix": "BUDAPEST",
    "Dutch Grand Prix": "ZANDVOORT",
    "Italian Grand Prix": "MONZA",
    "Azerbaijan Grand Prix": "BAKU",
    "Singapore Grand Prix": "MARINA BAY",
    "United States Grand Prix": "AUSTIN",
    "Mexico City Grand Prix": "MEXICO CITY",
    "São Paulo Grand Prix": "SÃO PAULO",
    "Las Vegas Grand Prix": "LAS VEGAS",
    "Qatar Grand Prix": "LUSAIL",
    "Abu Dhabi Grand Prix": "YAS ISLAND",
    "Bahrain Grand Prix": "SAKHIR",
    # "Spanish Grand Prix" is Madrid from 2026 and has no prior history.
}

LEGACY_YEARS = (2022, 2023, 2024, 2025)


def _legacy_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "legacy" / "data" / "v6"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("legacy/data/v6 not found")


@lru_cache(maxsize=1)
def circuit_table() -> pd.DataFrame:
    """Per-circuit physical character, fitted once on the pre-2026 era."""
    columns = ["track", "attacker_speed_i1", "attacker_speed_i2", "attacker_straight_speed"]
    frames = []
    for year in LEGACY_YEARS:
        path = _legacy_dir() / f"scenarios_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=columns))
    if not frames:
        raise FileNotFoundError("no legacy scenario files found")

    legacy = pd.concat(frames, ignore_index=True)
    table = legacy.groupby("track").agg(
        circuit_speed_i1=("attacker_speed_i1", "median"),
        circuit_speed_i2=("attacker_speed_i2", "median"),
        _top=("attacker_straight_speed", "median"),
    )
    # Ratio, not the raw top speed: the ratio transfers (rho 0.967), the
    # absolute value does not (rho 0.461).
    table["circuit_straight_dominance"] = table["_top"] / table["circuit_speed_i2"]
    return table.drop(columns=["_top"])


def attach_circuit_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add circuit character to hazard rows, by event name."""
    if rows is None or rows.empty:
        return rows

    table = circuit_table()
    defaults = table.median()
    out = rows.copy()
    tracks = out["event_name"].map(EVENT_TO_TRACK)

    for column in ("circuit_speed_i1", "circuit_speed_i2", "circuit_straight_dominance"):
        out[column] = tracks.map(table[column]).astype(float).fillna(float(defaults[column]))
    out["circuit_known"] = tracks.map(table.index.to_series()).notna().astype(int)
    return out


def register() -> None:
    from era2026.features import FEATURE_TIERS

    FEATURE_TIERS.update(CIRCUIT_FEATURES)
