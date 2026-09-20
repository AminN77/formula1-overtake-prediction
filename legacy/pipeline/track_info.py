from typing import Tuple

DRS_ZONES = {
    'bahrain': [1100],
    'saudi arabia': [660],
    'australia': [330],
    'azerbaijan': [433, 419],
    'miami': [320],
    'monaco': [0],
    'spain': [655],
    'canada': [350],
    'austria': [435, 300, 250],
    'great britain': [164, 335],
    'hungary': [0],
    'belgium': [1900, 1300],
    'netherlands': [290],
    'italy': [365, 370, 370],
    'singapore': [0],
    'japan': [200],
    'qatar': [350],
    'united states': [0],
    'mexico': [0],
    'brazil': [650],
    'abu dhabi': [1053],
    'france': [900],
    'china': [1170],
    'turkey': [400],
    'portugal': [460],
}

SECTOR_TYPES = {
    'speed': [
        'italy', 'belgium', 'great britain',
        'azerbaijan', 'abu dhabi', 'austria'
    ],
    'technical': [
        'monaco', 'singapore', 'hungary', 'turkey'
    ],
    'mixed': [
        'bahrain', 'spain', 'france', 'japan',
        'united states', 'brazil', 'portugal',
        'china', 'qatar', 'australia', 'canada',
        'netherlands', 'mexico', 'miami'
    ]
}

# IP03 §3.2: explicit mapping for all 2022–2025 calendar circuits (no silent "street" fallback).
# Order of keys is not used; each substring should match at most one category (first match wins).
TRACK_TYPES = {
    'high-speed': [
        'italy', 'belgium', 'great britain', 'austria',
        'qatar', 'saudi arabia',
    ],
    'medium-speed': [
        'spain', 'japan', 'abu dhabi', 'bahrain', 'china',
        'emilia', 'french', 'france', 'netherlands', 'dutch',
        'canada', 'mexico', 'united states', 'usa', 'brazil',
        'portugal', 'turkey',
    ],
    'low-speed': [
        'monaco', 'hungary', 'singapore',
    ],
    'street': [
        'azerbaijan', 'miami', 'las vegas', 'australia',
    ],
}


def get_sector_type(track: str) -> str:
    track_lower = track.lower()
    for sector_type, tracks in SECTOR_TYPES.items():
        if any(t in track_lower for t in tracks):
            return sector_type
    return 'mixed'


def get_track_type(track: str) -> str:
    track_lower = track.lower()
    for track_type, tracks in TRACK_TYPES.items():
        if any(t in track_lower for t in tracks):
            return track_type
    # Unknown / new circuit: avoid labelling everything as "street" (IP03 §3.2).
    return 'medium-speed'


def get_drs_zone_info(track: str, sector: int) -> Tuple[bool, int]:
    track_lower = track.lower()
    for track_name, zones in DRS_ZONES.items():
        if track_name in track_lower and sector <= len(zones):
            return True, zones[sector - 1]
    return False, 0
