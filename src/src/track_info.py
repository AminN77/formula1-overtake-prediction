from typing import Tuple

# DRS zones per track (approximate lengths in meters)
DRS_ZONES = {
    'monza': [365, 370, 370],
    'silverstone': [164, 335],
    'spa': [1900, 1300],
    'baku': [433, 419],
    'monte carlo': [0],
    'barcelona': [0],
    'bahrain': [0],
    'saudi arabia': [660],
    'miami': [0],
    'imola': [0],
    'canada': [0],
    'budapest': [0],
    'zandvoort': [290],
    'singapore': [0],
    'suzuka': [200],
    'qatar': [350],
    'cota': [0],
    'mexico': [0],
    'brazil': [0],
    'abu dhabi': [1053],
}

SECTOR_TYPES = {
    'speed': ['monza', 'spa', 'silverstone', 'baku', 'abu dhabi'],
    'technical': ['monaco', 'singapore', 'imola'],
    'mixed': ['budapest', 'barcelona', 'paul ricard']
}

TRACK_TYPES = {
    'high-speed': ['monza', 'spa', 'silverstone'],
    'medium-speed': ['budapest', 'paul ricard'],
    'low-speed': ['monaco', 'singapore'],
    'street': ['baku', 'miami', 'saudi arabia'],
    'ovals': ['cota', 'mexico', 'brazil']
}


def get_sector_type(track: str) -> str:
    """Determine sector type based on track characteristics."""
    track_lower = track.lower()
    for sector_type, tracks in SECTOR_TYPES.items():
        if any(t in track_lower for t in tracks):
            return sector_type
    return 'mixed'


def get_track_type(track: str) -> str:
    """Determine track type."""
    track_lower = track.lower()
    for track_type, tracks in TRACK_TYPES.items():
        if any(t in track_lower for t in tracks):
            return track_type
    return 'street'


def get_drs_zone_info(track: str, sector: int) -> Tuple[bool, int]:
    """Check if in DRS zone and get zone length."""
    track_lower = track.lower()
    for track_name, zones in DRS_ZONES.items():
        if track_name in track_lower and sector <= len(zones):
            return True, zones[sector - 1]
    return False, 0