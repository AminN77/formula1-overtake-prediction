"""
Distribution drift between the training pool and a new round.

A regulation era is not stationary. Teams develop, and 2027 will not resemble
2026. Drift is tracked per round so a promotion failure can be told apart from
the world having moved.

Population Stability Index, the standard measure: under 0.1 is stable, 0.1 to
0.25 is a moderate shift, above 0.25 is a material one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

#: Race-level features carry one value per race, so a distribution test on them
#: needs tens of races on each side. In the first season of an era there are
#: nowhere near enough, and the honest report is "not testable yet" rather than
#: a large number that means nothing. This threshold deliberately silences
#: race-level drift until 2027.
MIN_RACES_FOR_RACE_LEVEL_DRIFT = 8

#: Features that increase every race by construction. A driver's career round
#: count always "drifts"; reporting it is a permanent false positive.
MONOTONE_BY_CONSTRUCTION = frozenset({
    "attacker_prior_rounds", "defender_prior_rounds",
})


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    floor = 1e-4
    ref_share = np.clip(ref_share, floor, None)
    cur_share = np.clip(cur_share, floor, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def race_constant_features(rows: pd.DataFrame, columns: list[str],
                           group_column: str = "round_number") -> list[str]:
    """Features that barely vary inside a race.

    These need different treatment. Weather, total laps and a driver's career
    round count are fixed for a whole grand prix, so comparing one race's rows
    against many races' rows registers enormous PSI every single week, whatever
    the world is doing. A monitor that fires every week is not a monitor. They
    are compared race-to-race instead, one value per race.
    """
    constant = []
    for column in columns:
        if column not in rows.columns:
            continue
        within = rows.groupby(group_column)[column].std(ddof=0)
        overall = rows[column].std(ddof=0)
        if overall <= 0:
            continue
        if within.mean() / overall < 0.25:
            constant.append(column)
    return constant


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str],
                 group_column: str = "round_number") -> pd.DataFrame:
    """Row-level PSI for features that vary within a race, race-level for the rest."""
    combined = pd.concat([reference, current], ignore_index=True)
    constant = set(race_constant_features(combined, columns, group_column))

    records = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        if column in MONOTONE_BY_CONSTRUCTION:
            records.append({
                "feature": column, "scope": "counter", "psi": float("nan"),
                "reference_mean": float(reference[column].mean()),
                "current_mean": float(current[column].mean()),
                "severity": "by_construction",
            })
            continue
        if column in constant:
            # One value per race on both sides, so a shift means the calendar
            # moved, not that the rows did. This needs several races to mean
            # anything: a distribution shift cannot be measured from a single
            # observation, and forcing it produces an identical maximal PSI for
            # every race-level feature at once, which is noise dressed as alarm.
            ref_values = reference.groupby(group_column)[column].mean().to_numpy()
            cur_values = current.groupby(group_column)[column].mean().to_numpy()
            scope = "per_race"
            psi = (population_stability_index(ref_values, cur_values, bins=5)
                   if len(cur_values) >= MIN_RACES_FOR_RACE_LEVEL_DRIFT else float("nan"))
        else:
            psi = population_stability_index(reference[column].to_numpy(),
                                             current[column].to_numpy())
            scope = "per_row"
        records.append({
            "feature": column,
            "scope": scope,
            "psi": psi,
            "reference_mean": float(reference[column].mean()),
            "current_mean": float(current[column].mean()),
            "severity": (
                "stable" if psi < PSI_STABLE
                else "moderate" if psi < PSI_MODERATE
                else "material"
            ) if np.isfinite(psi) else "insufficient_data",
        })
    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


def base_rate_shift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """The label rate is the drift that matters most: it moves the intercept."""
    ref = float(reference["label"].mean()) if len(reference) else float("nan")
    cur = float(current["label"].mean()) if len(current) else float("nan")
    return {
        "reference_rate": ref,
        "current_rate": cur,
        "ratio": cur / ref if ref else float("nan"),
        "absolute_change": cur - ref,
    }
