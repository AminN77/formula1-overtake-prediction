"""
Generate battle CSVs from FastF1 race data.

Usage (from project root, PYTHONPATH=. or pip install -e .):
    python -m pipeline.main                           # 2022-2024 → data/battles_2022_2023_2024.csv
    python -m pipeline.main --years 2022              # single season
    python -m pipeline.main --output ../data/v2/b.csv # custom output path
"""

import argparse
import sys
from pathlib import Path
from typing import List

import fastf1
import pandas as pd

from .battle_detector import detect_races_battles
from .models import BattleRecord


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


def collect_battles(years: List[int], cache_path: str = None) -> List[BattleRecord]:
    all_battles: List[BattleRecord] = []

    for year in years:
        print(f"\nProcessing season {year}")
        schedule = fastf1.get_event_schedule(year)

        for _, event in schedule.iterrows():
            event_name = event["EventName"]
            if event["EventFormat"] not in ("conventional",):
                continue

            print(f"  {event_name}")
            try:
                battles = detect_races_battles(
                    year=year,
                    gp=event["Country"],
                    identifier="R",
                    cache_path=cache_path,
                )
                if not battles:
                    print("    No battles found")
                    continue

                all_battles.extend(battles)
                print(f"    {len(battles)} battles")

            except Exception as exc:
                print(f"    Skipped: {exc}")

    return all_battles


def battles_to_dataframe(battles: List[BattleRecord]) -> pd.DataFrame:
    return pd.DataFrame([b.to_dict() for b in battles], columns=BattleRecord.column_names())


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df):,} battles to {path}")


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate F1 battle dataset as CSV")
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2022, 2023, 2024],
        help="Season years to process (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: notebooks/battles_<years>.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Write battles_<year>.csv per season into this directory (adds driver rolling features).",
    )
    parser.add_argument(
        "--cache", type=str, default=None,
        help="FastF1 cache directory",
    )
    args = parser.parse_args(argv)

    battles = collect_battles(args.years, args.cache)
    if not battles:
        print("No battles collected.")
        sys.exit(1)

    df = battles_to_dataframe(battles)

    if args.output_dir:
        from .driver_features import enrich_driver_features
        from .team_features import enrich_team_features

        df = enrich_driver_features(df)
        df = enrich_team_features(df)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for y in sorted(set(df["year"].tolist())):
            sub = df[df["year"] == y]
            save_csv(sub, out_dir / f"battles_{y}.csv")
        return

    if args.output:
        out_path = Path(args.output)
    else:
        year_tag = "_".join(str(y) for y in sorted(args.years))
        out_path = DEFAULT_OUTPUT_DIR / f"battles_{year_tag}.csv"

    save_csv(df, out_path)


if __name__ == "__main__":
    main()
