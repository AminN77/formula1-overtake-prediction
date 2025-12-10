from dataclasses import dataclass
from datetime import datetime


@dataclass
class BattleRecord:
    """Typed class representing a battle record in ClickHouse."""

    # Driver identifiers
    attacker: int
    defender: int

    # Battle outcome
    overtake: bool
    time_stamp: datetime

    # Car state features
    attacker_speed: int  # Note: keeping typo from table schema
    defender_speed: int
    speed_difference: int

    # Race state features
    lap_number: int
    safety_car: bool
    yellow_flag: bool

    # Tyre features
    attacker_tyre_compound: str
    defender_tyre_compound: str
    attacker_tyre_age: int
    defender_tyre_age: int
    tyre_age_difference: int

    # Track features
    track: str
    sector: int
    sector_type: str
    is_in_drs_zone: bool
    drs_zone_length: int
    track_type: str

    # Qualification features
    attacker_qualification_rank: int
    defender_qualification_rank: int

    def to_dict(self) -> dict:
        """Convert to dictionary for ClickHouse insertion."""
        return {
            'attacker': self.attacker,
            'defender': self.defender,
            'overtake': self.overtake,
            'time_stamp': self.time_stamp,
            'attacker_speed': self.attacker_speed,
            'defender_speed': self.defender_speed,
            'speed_difference': self.speed_difference,
            'lap_number': self.lap_number,
            'safety_car': self.safety_car,
            'yellow_flag': self.yellow_flag,
            'attacker_tyre_compound': self.attacker_tyre_compound,
            'defender_tyre_compound': self.defender_tyre_compound,
            'attacker_tyre_age': self.attacker_tyre_age,
            'defender_tyre_age': self.defender_tyre_age,
            'tyre_age_difference': self.tyre_age_difference,
            'track': self.track,
            'sector': self.sector,
            'sector_type': self.sector_type,
            'is_in_drs_zone': self.is_in_drs_zone,
            'drs_zone_length': self.drs_zone_length,
            'track_type': self.track_type,
            'attacker_qualification_rank': self.attacker_qualification_rank,
            'defender_qualification_rank': self.defender_qualification_rank,
        }

    def to_list(self) -> list:
        """Convert to list for ClickHouse batch insertion."""
        return [
            self.attacker,
            self.defender,
            self.overtake,
            self.time_stamp,
            self.attacker_speed,
            self.defender_speed,
            self.speed_difference,
            self.lap_number,
            self.safety_car,
            self.yellow_flag,
            self.attacker_tyre_compound,
            self.defender_tyre_compound,
            self.attacker_tyre_age,
            self.defender_tyre_age,
            self.tyre_age_difference,
            self.track,
            self.sector,
            self.sector_type,
            self.is_in_drs_zone,
            self.drs_zone_length,
            self.track_type,
            self.attacker_qualification_rank,
            self.defender_qualification_rank,
        ]

    @staticmethod
    def column_names() -> list:
        """Get list of column names for ClickHouse."""
        return [
            'attacker', 'defender', 'overtake', 'time_stamp',
            'attacker_speed', 'defender_speed', 'speed_difference',
            'lap_number', 'safety_car', 'yellow_flag',
            'attacker_tyre_compound', 'defender_tyre_compound',
            'attacker_tyre_age', 'defender_tyre_age', 'tyre_age_difference',
            'track', 'sector', 'sector_type', 'is_in_drs_zone', 'drs_zone_length',
            'track_type', 'attacker_qualification_rank', 'defender_qualification_rank'
        ]