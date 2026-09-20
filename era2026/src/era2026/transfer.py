"""
Layer 1: the pre-2026 era, compressed to a scalar per row.

Eight seasons of old data never reach the 2026 model as rows. An old-era model
is fitted once on 2022-2025, frozen, and used to score 2026 rows. The in-era
model then only has to learn the *correction*, which is a far smaller function
than the whole problem.

Three properties make this safe and measurable:

* The old model never sees 2026 data, so it cannot leak into any backtest fold
  and does not need refitting per fold.
* It is restricted to Tier 1 and Tier 2 features, so no DRS, no absolute car
  speeds, no team identity crosses the era boundary.
* It is one column. Switch it off and the comparison is exact.

It also degrades gracefully: if the old era is useless, the 2026 model ignores
the feature and nothing is lost but a column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from era2026.features import TIER_1

#: Legacy target. Matches the in-era hazard question (a pass on the next lap)
#: rather than the legacy default of a pass within three laps.
LEGACY_TARGET = "overtake_next_lap"

LEGACY_YEARS = (2022, 2023, 2024, 2025)

#: Feature added to the registry by :func:`register`.
TRANSFER_FEATURES: dict[str, str] = {"old_era_score": TIER_1}


def _legacy_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "legacy" / "data" / "v6"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("legacy/data/v6 not found")


def shared_features(columns: list[str]) -> list[str]:
    """Features present in both eras with the same name and meaning.

    The 2026 feature builder deliberately reuses the legacy naming, so the
    intersection is large. Anything era-specific is absent by construction.
    """
    legacy_columns = set(pd.read_csv(_legacy_dir() / "scenarios_2024.csv", nrows=1).columns)
    return [c for c in columns if c in legacy_columns]


def load_legacy(years=LEGACY_YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        path = _legacy_dir() / f"scenarios_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("no legacy scenario files found")
    return pd.concat(frames, ignore_index=True)


@dataclass
class OldEraModel:
    """Fitted once on the pre-2026 era, then frozen."""

    columns: list[str]
    years: tuple[int, ...] = LEGACY_YEARS
    model_: object | None = field(default=None, repr=False)
    train_rows_: int = 0
    train_rate_: float = 0.0

    def fit(self) -> "OldEraModel":
        import lightgbm as lgb

        legacy = load_legacy(self.years)
        if LEGACY_TARGET not in legacy.columns:
            raise KeyError(f"legacy data has no {LEGACY_TARGET!r}")

        # Pit-involved candidates are censored in the 2026 framing, so excluding
        # them here keeps the two eras describing the same situation.
        if "pit_stop_involved" in legacy.columns:
            legacy = legacy[~legacy["pit_stop_involved"].astype(bool)]

        X = legacy[self.columns].astype(float).fillna(0.0)
        y = legacy[LEGACY_TARGET].astype(int).to_numpy()
        self.train_rows_ = int(len(X))
        self.train_rate_ = float(y.mean())

        self.model_ = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
            min_child_samples=50, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        self.model_.fit(X, y)
        return self

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        X = rows[self.columns].astype(float).fillna(0.0)
        return self.model_.predict_proba(X)[:, 1]


def attach_old_era_score(rows: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add ``old_era_score``: one frozen scalar per row from the pre-2026 era."""
    if rows is None or rows.empty:
        return rows
    if columns is None:
        # Intersect the declared feature registry, never the raw column list:
        # the frame also carries identifiers such as `attacker` and `defender`,
        # which exist in the legacy files too and would be pulled in as
        # "shared" before failing on the first driver code.
        from era2026.features import FEATURE_TIERS

        declared = [c for c in FEATURE_TIERS if c not in TRANSFER_FEATURES]
        columns = shared_features(declared)
    model = OldEraModel(columns=columns).fit()
    out = rows.copy()
    out["old_era_score"] = model.score(out)
    out.attrs["old_era_train_rows"] = model.train_rows_
    out.attrs["old_era_train_rate"] = model.train_rate_
    out.attrs["old_era_columns"] = columns
    return out


def register() -> None:
    from era2026.features import FEATURE_TIERS

    FEATURE_TIERS.update(TRANSFER_FEATURES)
