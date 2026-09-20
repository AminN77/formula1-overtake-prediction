"""
Warm the FastF1 cache for every completed round of a season.

The F1 live timing endpoint is not reachable from the Claude sandboxes, so the
cache is the shared substrate: warm it once on a machine with network access and
the extraction layer can then be developed and run entirely offline.

Car telemetry is not fetched by default. The speed traps the feature set needs
(SpeedI1, SpeedI2, SpeedFL, SpeedST) are lap-level columns, so laps, weather and
race control are sufficient and far smaller.

    cd era2026
    uv run era2026-warm            # current season
    uv run era2026-warm 2026 2025  # specific seasons
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


CACHE = _project_root() / ".fastf1_cache"


def warm_season(year: int, telemetry: bool = False) -> int:
    import fastf1
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    sched = fastf1.get_event_schedule(year, include_testing=False)
    sched = sched[sched["RoundNumber"] > 0]
    now = pd.Timestamp.utcnow().tz_localize(None)
    done = sched[pd.to_datetime(sched["EventDate"]) < now]

    print(f"\n{year}: {len(done)} completed rounds of {len(sched)}")
    failed = 0
    for row in done.itertuples():
        rnd, name = int(row.RoundNumber), row.EventName
        try:
            s = fastf1.get_session(year, rnd, "R")
            s.load(telemetry=telemetry, weather=True, messages=True)
            rc = 0 if s.race_control_messages is None else len(s.race_control_messages)
            print(f"  R{rnd:<3} {name[:30]:32s} laps={len(s.laps):<5} "
                  f"weather={len(s.weather_data):<5} rc={rc}")
        except Exception as exc:
            failed += 1
            print(f"  R{rnd:<3} {name[:30]:32s} FAILED {type(exc).__name__}: {exc}")
    return failed


def cli() -> None:
    years = [int(a) for a in sys.argv[1:]] or [2026]
    telemetry = "--telemetry" in sys.argv
    failed = sum(warm_season(y, telemetry) for y in years)
    size = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file())
    print(f"\nCache at {CACHE} is now {size / 1e6:.0f} MB")
    if failed:
        print(f"{failed} round(s) failed. Re-run to retry; cached rounds are skipped.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    cli()
