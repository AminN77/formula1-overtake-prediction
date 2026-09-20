"""2025 F1 calendar metadata for UI (circuit picker, round/lap auto-fill)."""

from __future__ import annotations

from typing import Any

# Keys must match `race_name` strings used in schema_builder.RACES / training data.
# Round order = 2025 season.
CIRCUIT_CALENDAR_2025: dict[str, dict[str, Any]] = {
    "Australian Grand Prix": {
        "round": 1,
        "total_laps": 58,
        "country": "AU",
        "city": "Melbourne",
        "drs_zones": 4,
    },
    "Chinese Grand Prix": {
        "round": 2,
        "total_laps": 56,
        "country": "CN",
        "city": "Shanghai",
        "drs_zones": 2,
    },
    "Japanese Grand Prix": {
        "round": 3,
        "total_laps": 53,
        "country": "JP",
        "city": "Suzuka",
        "drs_zones": 1,
    },
    "Bahrain Grand Prix": {
        "round": 4,
        "total_laps": 57,
        "country": "BH",
        "city": "Sakhir",
        "drs_zones": 3,
    },
    "Saudi Arabian Grand Prix": {
        "round": 5,
        "total_laps": 50,
        "country": "SA",
        "city": "Jeddah",
        "drs_zones": 1,
    },
    "Miami Grand Prix": {
        "round": 6,
        "total_laps": 57,
        "country": "US",
        "city": "Miami",
        "drs_zones": 3,
    },
    "Emilia Romagna Grand Prix": {
        "round": 7,
        "total_laps": 63,
        "country": "IT",
        "city": "Imola",
        "drs_zones": 1,
    },
    "Monaco Grand Prix": {
        "round": 8,
        "total_laps": 78,
        "country": "MC",
        "city": "Monte Carlo",
        "drs_zones": 1,
    },
    "Canadian Grand Prix": {
        "round": 9,
        "total_laps": 70,
        "country": "CA",
        "city": "Montreal",
        "drs_zones": 3,
    },
    "Spanish Grand Prix": {
        "round": 10,
        "total_laps": 66,
        "country": "ES",
        "city": "Barcelona",
        "drs_zones": 2,
    },
    "Austrian Grand Prix": {
        "round": 11,
        "total_laps": 71,
        "country": "AT",
        "city": "Spielberg",
        "drs_zones": 3,
    },
    "British Grand Prix": {
        "round": 12,
        "total_laps": 52,
        "country": "GB",
        "city": "Silverstone",
        "drs_zones": 2,
    },
    "Belgian Grand Prix": {
        "round": 13,
        "total_laps": 44,
        "country": "BE",
        "city": "Spa",
        "drs_zones": 2,
    },
    "Hungarian Grand Prix": {
        "round": 14,
        "total_laps": 70,
        "country": "HU",
        "city": "Budapest",
        "drs_zones": 1,
    },
    "Dutch Grand Prix": {
        "round": 15,
        "total_laps": 72,
        "country": "NL",
        "city": "Zandvoort",
        "drs_zones": 2,
    },
    "Italian Grand Prix": {
        "round": 16,
        "total_laps": 53,
        "country": "IT",
        "city": "Monza",
        "drs_zones": 3,
    },
    "Azerbaijan Grand Prix": {
        "round": 17,
        "total_laps": 51,
        "country": "AZ",
        "city": "Baku",
        "drs_zones": 2,
    },
    "Singapore Grand Prix": {
        "round": 18,
        "total_laps": 62,
        "country": "SG",
        "city": "Marina Bay",
        "drs_zones": 1,
    },
    "United States Grand Prix": {
        "round": 19,
        "total_laps": 56,
        "country": "US",
        "city": "Austin",
        "drs_zones": 1,
    },
    "Mexico City Grand Prix": {
        "round": 20,
        "total_laps": 71,
        "country": "MX",
        "city": "Mexico City",
        "drs_zones": 1,
    },
    "São Paulo Grand Prix": {
        "round": 21,
        "total_laps": 71,
        "country": "BR",
        "city": "São Paulo",
        "drs_zones": 2,
    },
    "Las Vegas Grand Prix": {
        "round": 22,
        "total_laps": 50,
        "country": "US",
        "city": "Las Vegas",
        "drs_zones": 2,
    },
    "Qatar Grand Prix": {
        "round": 23,
        "total_laps": 57,
        "country": "QA",
        "city": "Lusail",
        "drs_zones": 1,
    },
    "Abu Dhabi Grand Prix": {
        "round": 24,
        "total_laps": 58,
        "country": "AE",
        "city": "Yas Marina",
        "drs_zones": 2,
    },
}


def circuits_for_api() -> dict[str, Any]:
    """Payload for GET /api/circuits — sorted by round."""
    ordered = sorted(
        CIRCUIT_CALENDAR_2025.items(),
        key=lambda kv: int(kv[1]["round"]),
    )
    return {
        "season": 2025,
        "circuits": [
            {"race_name": name, **meta}
            for name, meta in ordered
        ],
    }
