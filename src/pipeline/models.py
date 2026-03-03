from dataclasses import dataclass, fields, asdict


@dataclass
class BattleRecord:
    # ── identifiers ──────────────────────────────────────────────
    attacker: str
    defender: str
    overtake: bool

    # ── race context ─────────────────────────────────────────────
    year: int
    race_name: str
    lap_number: int
    total_laps: int
    race_progress: float
    attacker_position: int
    defender_position: int

    # ── lap-time gap ─────────────────────────────────────────────
    attacker_lap_time: float
    defender_lap_time: float
    gap_ahead: float

    # ── speed traps (km/h) ──────────────────────────────────────
    attacker_speed_i1: float
    defender_speed_i1: float
    attacker_speed_i2: float
    defender_speed_i2: float
    attacker_finish_line_speed: float
    defender_finish_line_speed: float
    attacker_straight_speed: float
    defender_straight_speed: float

    # ── flags ────────────────────────────────────────────────────
    safety_car: bool
    yellow_flag: bool

    # ── tyres ────────────────────────────────────────────────────
    attacker_tyre_compound: str
    defender_tyre_compound: str
    attacker_tyre_age: int
    defender_tyre_age: int
    tyre_age_difference: int
    attacker_stint: int
    defender_stint: int
    attacker_fresh_tyre: bool
    defender_fresh_tyre: bool

    # ── track / circuit ──────────────────────────────────────────
    track: str
    sector: int
    sector_type: str
    is_in_drs_zone: bool
    drs_zone_length: int
    track_type: str

    # ── qualification ────────────────────────────────────────────
    attacker_qualification_rank: int
    defender_qualification_rank: int

    # ── weather ──────────────────────────────────────────────────
    air_temp: float
    track_temp: float
    humidity: float
    rainfall: bool
    wind_speed: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def column_names() -> list[str]:
        return [f.name for f in fields(BattleRecord)]
