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
    round_number: int
    event_date: str
    lap_number: int
    total_laps: int
    race_progress: float
    attacker_position: int
    defender_position: int

    # ── gap & pace ───────────────────────────────────────────────
    attacker_lap_time: float
    defender_lap_time: float
    gap_ahead: float            # actual inter-car gap (seconds) from LapStartTime
    pace_delta: float           # defender_lap_time - attacker_lap_time (positive = attacker faster)

    # ── speed traps (km/h) ──────────────────────────────────────
    attacker_speed_i1: float
    defender_speed_i1: float
    attacker_speed_i2: float
    defender_speed_i2: float
    attacker_finish_line_speed: float
    defender_finish_line_speed: float
    attacker_straight_speed: float
    defender_straight_speed: float

    # ── speed deltas (attacker − defender, km/h) ────────────────
    speed_i1_delta: float
    speed_i2_delta: float
    speed_fl_delta: float
    speed_st_delta: float

    # ── flags ────────────────────────────────────────────────────
    safety_car: bool
    yellow_flag: bool

    # ── tyres ────────────────────────────────────────────────────
    attacker_tyre_compound: str
    defender_tyre_compound: str
    attacker_tyre_age: int
    defender_tyre_age: int
    tyre_age_difference: int    # signed: attacker − defender (negative = attacker on fresher tyres)
    attacker_stint: int
    defender_stint: int
    attacker_fresh_tyre: bool
    defender_fresh_tyre: bool

    # ── pit stop ─────────────────────────────────────────────────
    pit_stop_involved: bool     # either driver pits on current or next lap

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

    # ── IP03 / v4: multi-horizon labels & sector micro-features ──
    overtake_within_2: bool
    overtake_within_3: bool
    sector1_delta: float
    sector2_delta: float
    sector3_delta: float
    strongest_sector: int  # 0=S1, 1=S2, 2=S3; -1 if no valid sector times
    compound_mismatch: bool

    # ── IP04 / v5: team, situation, race-phase ──
    attacker_team: str
    defender_team: str
    same_team: bool
    gap_to_car_ahead: float       # gap from defender to the car in front (P-2)
    gap_to_car_behind: float      # gap from attacker to the car behind (P+2)
    drs_train_size: int           # number of cars within 1 s chain around the battle
    race_phase: str               # 'opening' / 'middle' / 'closing'
    stint_phase: str              # 'fresh' / 'mid' / 'degraded' / 'cliff'

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def column_names() -> list[str]:
        return [f.name for f in fields(BattleRecord)]
