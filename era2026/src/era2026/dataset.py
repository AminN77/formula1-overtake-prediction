"""Assemble the modelling table once and cache it to parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from era2026.episodes import DEFAULT_GAP_THRESHOLD, hazard_rows_from_session, trainable
from era2026.features import build_features
from era2026.form import attach_form, register as register_form
from era2026.sessions import completed_rounds, load_race

register_form()


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


DATA_DIR = _project_root() / "data"


def build_season(year: int, rounds: list[int] | None = None,
                 gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> pd.DataFrame:
    if rounds is None:
        rounds = [int(r) for r in completed_rounds(year)["RoundNumber"]]
    frames, sessions = [], {}
    for round_number in rounds:
        session = load_race(year, round_number)
        sessions[round_number] = session
        rows = hazard_rows_from_session(session, year, round_number, gap_threshold)
        if len(rows):
            frames.append(build_features(session, rows))
    if not frames:
        return pd.DataFrame()
    # Form needs the whole season in one pass, because a round-k row reads
    # rounds 1..k-1. The prior-rounds-only rule keeps it leak-free per fold.
    return attach_form(pd.concat(frames, ignore_index=True), sessions)


def season_path(year: int) -> Path:
    return DATA_DIR / f"hazard_rows_{year}.parquet"


def load_or_build(year: int, refresh: bool = False) -> pd.DataFrame:
    path = season_path(year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = build_season(year)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def cli() -> None:
    import sys

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    frame = load_or_build(year, refresh=True)
    train = trainable(frame)
    print(f"{year}: {len(frame)} rows, {len(train)} trainable, "
          f"{int(train['label'].sum())} events, {train['label'].mean():.2%} positive")
    print(f"wrote {season_path(year)}")
