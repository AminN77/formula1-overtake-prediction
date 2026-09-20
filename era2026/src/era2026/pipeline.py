"""
The race-triggered retraining loop.

    ingest round k -> features -> score round k with the INCUMBENT
                   -> retrain challenger -> promotion gate -> promote or archive

Step three is the load-bearing one. The incumbent scores the new race *before
anything is retrained*, so one computation produces both the out-of-sample
result and the drift signal, and it is the same computation the backtest
performs. The number reported and the behaviour deployed are one measurement.

On the promotion gate
---------------------
The obvious gate is wrong. After round k the challenger has trained on rounds
1..k while the incumbent trained on 1..k-1, so scoring both on recent rounds
compares a model that has seen those races against one that has not. The
challenger wins that every time, and the gate becomes a rubber stamp.

The gate here instead re-runs the expanding-origin backtest over the trailing
window for *both configurations*, each trained only on rounds prior to each
test round. That compares recipes rather than fitted objects, which is the
question actually being asked: would this configuration have served us better
over the last five races.

A single race cannot separate a better model from a luckier one at this sample
size (per-round PR-AUC standard deviation is 0.22), so the window is five.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from era2026.backtest import pooled_metrics, run_backtest
from era2026.drift import base_rate_shift, drift_report
from era2026.episodes import trainable
from era2026.features import FEATURE_TIERS
from era2026.registry import (
    ModelVersion,
    data_fingerprint,
    incumbent,
    new_version_id,
    promote,
    register,
)

#: Rounds the promotion gate reads. One race is noise; five is a signal.
GATE_WINDOW = 5

#: The challenger must beat the incumbent by more than this to be promoted.
#: Zero would promote on noise every other week.
PROMOTION_MARGIN = 0.005


def default_model(params: dict | None = None, columns: list[str] | None = None):
    from era2026.calibration import Calibrated
    from era2026.models import GradientBoosting

    cols = columns or list(FEATURE_TIERS)

    def factory():
        return GradientBoosting(name="lgbm", columns=cols, params=params)

    return Calibrated(name="calibrated_lgbm", factory=factory, method="isotonic")


@dataclass
class RoundOutcome:
    round_number: int
    incumbent_version: str | None
    challenger_version: str
    incumbent_score: float
    challenger_score: float
    margin: float
    promoted: bool
    reason: str
    live_metrics: dict = field(default_factory=dict)
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_rate: dict = field(default_factory=dict)


def score_live_round(rows: pd.DataFrame, round_number: int, params: dict | None) -> dict:
    """What the incumbent scored on a race it had never seen. Evaluation and
    monitoring in the same step."""
    pool = trainable(rows)
    train = pool[pool["round_number"] < round_number]
    test = pool[pool["round_number"] == round_number]
    if train.empty or test.empty or train["label"].nunique() < 2:
        return {}

    model = default_model(params).fit(train, train["label"])
    p = np.asarray(model.predict_proba(test), dtype=float)

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out = {
        "rows": int(len(test)),
        "events": int(test["label"].sum()),
        "positive_rate": float(test["label"].mean()),
        "brier": float(brier_score_loss(test["label"], p)),
    }
    if test["label"].nunique() > 1:
        out["pr_auc"] = float(average_precision_score(test["label"], p))
        out["roc_auc"] = float(roc_auc_score(test["label"], p))
    return out


def gate(rows: pd.DataFrame, round_number: int, incumbent_params: dict | None,
         challenger_params: dict | None, window: int = GATE_WINDOW) -> tuple[float, float, pd.DataFrame]:
    """Walk both configurations forward over the trailing window."""
    start = max(2, round_number - window + 1)
    models = [
        default_model(incumbent_params),
        default_model(challenger_params),
    ]
    models[0].name, models[1].name = "incumbent", "challenger"
    result = run_backtest(rows, models, start_round=start)
    if result.predictions.empty:
        return float("nan"), float("nan"), result.per_round
    pooled = pooled_metrics(result.predictions).set_index("model")
    return (
        float(pooled.loc["incumbent", "pr_auc"]),
        float(pooled.loc["challenger", "pr_auc"]),
        result.per_round,
    )


def run_round(
    rows: pd.DataFrame,
    round_number: int,
    challenger_params: dict | None = None,
    window: int = GATE_WINDOW,
    margin: float = PROMOTION_MARGIN,
) -> RoundOutcome:
    """One turn of the loop, for the race that just happened."""
    from era2026.tuning import load_frozen

    current = incumbent()
    incumbent_params = current["params"] if current else load_frozen()
    challenger_params = challenger_params or load_frozen()

    live = score_live_round(rows, round_number, incumbent_params)

    pool = trainable(rows)
    # Drift reads a trailing window rather than the single new race. Race-level
    # features (weather, lap count) have one value per race, so a one-race
    # comparison cannot distinguish a shifting world from a different circuit.
    window_start = max(1, round_number - GATE_WINDOW + 1)
    history = pool[pool["round_number"] < window_start]
    recent = pool[pool["round_number"].between(window_start, round_number)]
    fresh = pool[pool["round_number"] == round_number]
    drift = (drift_report(history, recent, list(FEATURE_TIERS))
             if len(history) and len(recent) else pd.DataFrame())
    rates = (base_rate_shift(pool[pool["round_number"] < round_number], fresh)
             if len(fresh) else {})

    incumbent_score, challenger_score, _ = gate(
        rows, round_number, incumbent_params, challenger_params, window
    )

    train = pool[pool["round_number"] <= round_number]
    version = ModelVersion(
        version=new_version_id(round_number),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        trained_on_rounds=sorted(int(r) for r in train["round_number"].unique()),
        train_rows=int(len(train)),
        train_events=int(train["label"].sum()),
        data_fingerprint=data_fingerprint(train),
        params=challenger_params or {},
        features=list(FEATURE_TIERS),
        metrics={"gate_pr_auc": challenger_score, "live_round": live},
    )

    improvement = challenger_score - incumbent_score
    if current is None:
        promoted, reason = True, "no incumbent: first model is promoted by default"
    elif not np.isfinite(improvement):
        promoted, reason = False, "gate produced no comparable result"
    elif improvement > margin:
        promoted, reason = True, f"challenger beat incumbent by {improvement:+.4f} over {window} rounds"
    else:
        promoted, reason = False, f"improvement {improvement:+.4f} did not clear the {margin} margin"

    version.status = "incumbent" if promoted else "archived"
    version.note = reason
    register(version)
    if promoted:
        promote(version.version, over=current["version"] if current else None)

    return RoundOutcome(
        round_number=round_number,
        incumbent_version=current["version"] if current else None,
        challenger_version=version.version,
        incumbent_score=incumbent_score,
        challenger_score=challenger_score,
        margin=improvement,
        promoted=promoted,
        reason=reason,
        live_metrics=live,
        drift=drift,
        base_rate=rates,
    )


def cli() -> None:
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    from era2026.dataset import load_or_build

    parser = argparse.ArgumentParser(description="Run one turn of the retraining loop.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--window", type=int, default=GATE_WINDOW)
    args = parser.parse_args()

    rows = load_or_build(args.year, refresh=args.refresh)
    outcome = run_round(rows, args.round, window=args.window)

    print(f"\nRound {outcome.round_number}")
    print(f"  incumbent scored live: {outcome.live_metrics or 'n/a'}")
    print(f"  gate over last {args.window} rounds: "
          f"incumbent={outcome.incumbent_score:.4f} challenger={outcome.challenger_score:.4f} "
          f"({outcome.margin:+.4f})")
    print(f"  {'PROMOTED' if outcome.promoted else 'NOT PROMOTED'}: {outcome.reason}")
    if outcome.base_rate:
        print(f"  base rate {outcome.base_rate['reference_rate']:.2%} -> "
              f"{outcome.base_rate['current_rate']:.2%}")
    if len(outcome.drift):
        worst = outcome.drift.head(5)
        print("\n  largest feature drift (PSI):")
        for row in worst.itertuples():
            print(f"    {row.feature:28s} {row.psi:6.3f}  {row.severity}")
