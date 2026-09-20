"""
Weights & Biases tracking for the backtest.

Structure: one W&B **group** per backtest sweep, one **run** per model inside it.
Each run logs its metrics per round as a step series, so the per-round
trajectory (does the model improve as the era accumulates data) is a native W&B
line chart rather than something to reconstruct later. That trajectory is the
project's headline result, not a diagnostic.

Offline by default. ``api.wandb.ai`` is not reachable from the Claude sandboxes,
so runs are written to ``era2026/wandb/`` and synced later with ``wandb sync``.
Nothing about the pipeline depends on the network.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from era2026.backtest import BacktestResult, pooled_metrics


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


WANDB_DIR = _project_root() / "wandb"
DEFAULT_PROJECT = "era2026-overtake"

#: Metrics logged per round, in the order they appear on the run page.
ROUND_METRICS = ["pr_auc", "roc_auc", "brier", "log_loss", "positive_rate", "events", "n"]


def credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return netrc.is_file() and "api.wandb.ai" in netrc.read_text(encoding="utf-8")
    except OSError:
        return False


def resolve_mode(requested: str | None = None) -> str:
    """Online only when credentials exist and the caller did not force otherwise."""
    if requested:
        return requested
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if credentials_available() else "offline"


def log_backtest(
    result: BacktestResult,
    rows: pd.DataFrame,
    config: dict | None = None,
    project: str = DEFAULT_PROJECT,
    group: str | None = None,
    mode: str | None = None,
    dataset_path: Path | None = None,
) -> dict[str, str]:
    """Log one backtest sweep. Returns each model's run directory or URL."""
    import wandb

    if result.per_round.empty:
        return {}

    mode = resolve_mode(mode)
    group = group or f"backtest-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    WANDB_DIR.mkdir(parents=True, exist_ok=True)

    train_pool = rows[~rows["censored"]] if "censored" in rows.columns else rows
    shared = {
        "rounds_available": int(rows["round_number"].nunique()),
        "rows_total": int(len(rows)),
        "rows_trainable": int(len(train_pool)),
        "events": int(train_pool["label"].sum()),
        "base_rate": float(train_pool["label"].mean()),
        **(config or {}),
    }

    pooled = pooled_metrics(result.predictions).set_index("model")
    handles: dict[str, str] = {}

    for model_name, group_rows in result.per_round.groupby("model", sort=False):
        run = wandb.init(
            project=project,
            group=group,
            name=model_name,
            job_type="backtest",
            mode=mode,
            dir=str(WANDB_DIR),
            config={**shared, "model": model_name},
            reinit=True,
        )

        # Declare `round` as the x-axis rather than driving wandb's implicit
        # `_step` with explicit step= values. Explicit steps are fragile: a
        # sparse or offset step sequence (ours starts at 5) is easy to get wrong
        # and leaves charts keyed to a counter nobody cares about. define_metric
        # makes the round number the actual axis, which is what the chart means.
        run.define_metric("round")
        run.define_metric("round/*", step_metric="round")

        for record in group_rows.sort_values("round_number").to_dict("records"):
            payload = {f"round/{k}": record[k] for k in ROUND_METRICS if k in record}
            payload["round/train_rows"] = record["train_rows"]
            payload["round"] = int(record["round_number"])
            run.log(payload)

        summary = pooled.loc[model_name]
        for key in ("pr_auc", "roc_auc", "brier", "log_loss", "pr_auc_lift"):
            if key in summary:
                run.summary[f"pooled/{key}"] = float(summary[key])
        run.summary["per_round/pr_auc_mean"] = float(group_rows["pr_auc"].mean())
        run.summary["per_round/pr_auc_sd"] = float(group_rows["pr_auc"].std())

        # The table is not a step series, so it is logged on its own commit.
        run.log({"per_round_table": wandb.Table(dataframe=group_rows.reset_index(drop=True))},
                commit=True)

        if dataset_path and Path(dataset_path).exists():
            artifact = wandb.Artifact(f"hazard-rows-{shared['rounds_available']}r", type="dataset")
            artifact.add_file(str(dataset_path))
            run.log_artifact(artifact)

        handles[model_name] = run.url if mode == "online" else str(run.dir)
        run.finish()

    return handles


def cli() -> None:
    """``era2026-backtest``: build the table, run the ladder, log, print."""
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    from era2026.backtest import DEFAULT_START_ROUND, run_backtest
    from era2026.dataset import load_or_build, season_path
    from era2026.models import ladder

    parser = argparse.ArgumentParser(description="Run the expanding-origin backtest.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--start-round", type=int, default=DEFAULT_START_ROUND)
    parser.add_argument("--refresh", action="store_true", help="rebuild the feature table")
    parser.add_argument("--mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()

    rows = load_or_build(args.year, refresh=args.refresh)
    result = run_backtest(rows, ladder(), start_round=args.start_round)
    pooled = pooled_metrics(result.predictions)

    print("\nPooled over backtest rounds\n")
    print(pooled[["model", "n", "events", "positive_rate", "pr_auc", "pr_auc_lift",
                  "roc_auc", "brier"]].round(4).to_string(index=False))
    print("\nPer-round mean and spread\n")
    print(result.summary.round(4).to_string(index=False))

    if args.mode != "disabled":
        mode = resolve_mode(args.mode)
        if mode == "offline":
            # wandb creates its own `wandb/` inside the dir it is given, so the
            # runs land one level deeper than WANDB_DIR.
            print("\nW&B: offline (no credentials found). Sync later with:")
            print(f"  wandb login && wandb sync {WANDB_DIR / 'wandb'}/offline-run-*")
        handles = log_backtest(
            result, rows,
            config={"year": args.year, "start_round": args.start_round},
            project=args.project, mode=mode, dataset_path=season_path(args.year),
        )
        for name, handle in handles.items():
            print(f"  {name}: {handle}")
