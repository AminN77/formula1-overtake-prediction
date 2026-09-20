"""
Export everything the dashboard renders, as one JSON file.

The UI is static by design. The legacy app needed a backend because it scored
arbitrary hand-entered inputs; this one shows analysis of completed races, a
backtest trajectory and calibration, all of which are fixed once computed. A
server would add deployment surface and nothing else.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


UI_DIR = _project_root() / "ui"
DATA_PATH = UI_DIR / "data.json"

#: Episodes kept per race for the replay view. The longest and the most
#: eventful ones carry the interesting battles; a full dump is mostly noise.
EPISODES_PER_RACE = 12


def _clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def build_export(year: int = 2026) -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    from era2026.backtest import pooled_metrics, run_backtest
    from era2026.calibration import BootstrapEnsemble, Calibrated, reliability
    from era2026.dataset import load_or_build
    from era2026.episodes import OUTCOME_EVENT, trainable
    from era2026.features import FEATURE_TIERS
    from era2026.models import FIVE, BaseRate, GradientBoosting, LogisticBaseline

    rows = load_or_build(year)
    pool = trainable(rows)
    columns = list(FEATURE_TIERS)

    def factory():
        return GradientBoosting(name="lgbm", columns=columns)

    ladder = [
        BaseRate(),
        LogisticBaseline(name="gap_only", columns=["gap_ahead"]),
        LogisticBaseline(name="logistic_5", columns=FIVE),
        GradientBoosting(name="lightgbm", columns=columns),
        Calibrated(name="calibrated", factory=factory, method="isotonic"),
    ]
    result = run_backtest(rows, ladder)
    pooled = pooled_metrics(result.predictions)

    best = result.predictions[result.predictions["model"] == "calibrated"]
    reliability_table = reliability(best["label"].to_numpy(), best["hazard"].to_numpy())

    # interval width by round, relative to the predicted hazard
    # Restricted to the backtest range. Rounds 2-4 train on one or two races and
    # produce degenerate near-constant predictions, whose bootstrap spread is
    # tiny for the wrong reason. Charting them reverses the trend the analysis
    # actually found.
    from era2026.backtest import DEFAULT_START_ROUND

    widths = []
    for k in sorted(pool["round_number"].unique()):
        if k < DEFAULT_START_ROUND:
            continue
        train, test = pool[pool["round_number"] < k], pool[pool["round_number"] == k]
        if train.empty or test.empty or train["label"].nunique() < 2:
            continue
        ensemble = BootstrapEnsemble(name="b", factory=factory, n_bootstraps=25)
        interval = ensemble.fit(train, train["label"]).predict_interval(test)
        widths.append({
            "round": int(k),
            "train_rounds": int(train["round_number"].nunique()),
            "train_rows": int(len(train)),
            "width": float(interval["width"].mean()),
            "relative_width": float((interval["width"] / interval["hazard"].clip(lower=1e-4)).median()),
        })

    # per-race summary
    races = (rows.groupby(["round_number", "event_name"], as_index=False)
             .agg(episodes=("episode_id", "nunique"), rows=("lap_number", "size"),
                  events=("outcome", lambda s: int((s == OUTCOME_EVENT).sum())),
                  censored=("censored", "sum"), total_laps=("total_laps", "max")))
    races["positive_rate"] = races["events"] / races["rows"]

    # episodes for the replay, with the hazard the model gave each lap
    hazards = best.set_index(["episode_id", "lap_number"])["hazard"].to_dict()
    # Only rounds the backtest actually predicted. Earlier rounds have no model
    # output at all, and drawing them as a flat zero would claim the model
    # predicted "no chance" where it in fact predicted nothing.
    predicted_rounds = set(best["round_number"].unique()) if "round_number" in best.columns \
        else set(result.per_round["round_number"].unique())
    episodes = []
    for (round_number, event), group in rows.groupby(["round_number", "event_name"]):
        if round_number not in predicted_rounds:
            continue
        ranked = (group.groupby("episode_id")
                  .agg(laps=("lap_number", "size"),
                       events=("outcome", lambda s: int((s == OUTCOME_EVENT).sum())))
                  .sort_values(["events", "laps"], ascending=False)
                  .head(EPISODES_PER_RACE))
        for episode_id in ranked.index:
            legs = group[group["episode_id"] == episode_id].sort_values("lap_number")
            episodes.append({
                "round": int(round_number),
                "event": event,
                "episode_id": episode_id,
                "attacker": legs["attacker"].iloc[0],
                "defender": legs["defender"].iloc[0],
                "laps": [{
                    "lap": int(r.lap_number),
                    "gap": _clean(r.gap_ahead),
                    "outcome": r.outcome,
                    "hazard": _clean(hazards.get((episode_id, int(r.lap_number)))),
                } for r in legs.itertuples()],
            })

    feature_signal = []
    for name in columns:
        values = pool[name]
        if values.nunique() < 3:
            continue
        try:
            bucket = pd.qcut(values, 3, labels=False, duplicates="drop")
        except ValueError:
            continue
        grouped = pool.groupby(bucket)["label"].mean()
        if len(grouped) < 2:
            continue
        feature_signal.append({
            "feature": name, "tier": FEATURE_TIERS[name],
            "low": float(grouped.iloc[0]), "high": float(grouped.iloc[-1]),
            "spread": float(abs(grouped.iloc[-1] - grouped.iloc[0])),
        })
    feature_signal.sort(key=lambda d: d["spread"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "year": year,
        "totals": {
            "rounds": int(rows["round_number"].nunique()),
            "episodes": int(rows["episode_id"].nunique()),
            "rows": int(len(rows)),
            "trainable": int(len(pool)),
            "events": int(pool["label"].sum()),
            "positive_rate": float(pool["label"].mean()),
            "features": len(columns),
            "backtest_rounds": int(result.per_round["round_number"].nunique()),
        },
        "ladder": _records(pooled),
        "per_round": _records(result.per_round),
        "reliability": _records(reliability_table),
        "intervals": widths,
        "races": _records(races),
        "episodes": episodes,
        "feature_signal": feature_signal[:20],
    }


def cli() -> None:
    import sys

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    payload = build_export(year)
    UI_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size = DATA_PATH.stat().st_size / 1024
    print(f"wrote {DATA_PATH} ({size:.0f} KB)")
    print(f"  {payload['totals']['rounds']} rounds, {payload['totals']['episodes']} episodes, "
          f"{len(payload['episodes'])} exported, {payload['totals']['events']} events")
