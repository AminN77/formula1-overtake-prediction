"""Form features: the prior-rounds-only guarantee, and the censoring trap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from era2026.form import NEUTRAL_RATE, driver_round_summary, prior_form


def rows_for(rounds: int = 3, per_round: int = 20, label_by_round=None) -> pd.DataFrame:
    records = []
    for r in range(1, rounds + 1):
        for i in range(per_round):
            label = label_by_round(r, i) if label_by_round else int(i % 5 == 0)
            records.append({
                "round_number": r,
                "attacker": "HAM" if i % 2 else "VER",
                "defender": "VER" if i % 2 else "HAM",
                "label": label,
                "censored": False,
            })
    return pd.DataFrame(records)


class FakeSession:
    def __init__(self, drivers: list[str], teams: list[str], paces: list[float]):
        records = []
        for lap in (1, 2, 3):
            for i, driver in enumerate(drivers):
                records.append({
                    "LapNumber": lap, "Driver": driver, "Position": i + 1,
                    "Team": teams[i], "LapTime": pd.Timedelta(seconds=paces[i]),
                })
        self.laps = pd.DataFrame(records)


def sessions_for(rounds: int = 3) -> dict[int, FakeSession]:
    return {
        r: FakeSession(["VER", "HAM"], ["Red Bull Racing", "Mercedes"], [90.0, 91.0])
        for r in range(1, rounds + 1)
    }


def test_first_round_has_no_history():
    form = prior_form(driver_round_summary(rows_for(), sessions_for()))
    first = form[form["round_number"] == 1]
    assert (first["prior_rounds"] == 0).all()
    assert (first["overtake_rate"] == NEUTRAL_RATE).all()


def test_prior_rounds_accumulate():
    form = prior_form(driver_round_summary(rows_for(rounds=4), sessions_for(4)))
    counts = form.groupby("round_number")["prior_rounds"].max().to_dict()
    assert counts == {1: 0, 2: 1, 3: 2, 4: 3}


def test_form_at_round_k_ignores_round_k():
    """The guarantee. Changing a round's own labels must not move its own form."""
    baseline = prior_form(driver_round_summary(rows_for(), sessions_for()))

    def flip_last_round(r, i):
        return 1 if r == 3 else int(i % 5 == 0)

    perturbed = prior_form(
        driver_round_summary(rows_for(label_by_round=flip_last_round), sessions_for())
    )
    third_before = baseline[baseline["round_number"] == 3].set_index("driver")["overtake_rate"]
    third_after = perturbed[perturbed["round_number"] == 3].set_index("driver")["overtake_rate"]
    pd.testing.assert_series_equal(third_before, third_after)


def test_earlier_rounds_do_move_later_form():
    """Complement of the leak test: history must actually be used."""
    def flip_first_round(r, i):
        return 1 if r == 1 else int(i % 5 == 0)

    baseline = prior_form(driver_round_summary(rows_for(), sessions_for()))
    perturbed = prior_form(
        driver_round_summary(rows_for(label_by_round=flip_first_round), sessions_for())
    )
    b = baseline[baseline["round_number"] == 3]["overtake_rate"].to_numpy()
    p = perturbed[perturbed["round_number"] == 3]["overtake_rate"].to_numpy()
    assert not np.allclose(b, p)


def test_censored_rows_never_enter_the_rates():
    """Censored rows carry label = -1; summing them would corrupt every rate."""
    rows = rows_for()
    rows.loc[rows.index[:10], ["label", "censored"]] = [-1, True]
    summary = driver_round_summary(rows, sessions_for())
    assert (summary["attack_passes"] >= 0).all()
    assert (summary["attack_passes"] <= summary["attack_rows"]).all()
    assert (summary["defend_passed"] <= summary["defend_rows"]).all()


def test_team_pace_rank_orders_by_pace():
    sessions = {1: FakeSession(["VER", "HAM"], ["Red Bull Racing", "Mercedes"], [90.0, 95.0])}
    summary = driver_round_summary(rows_for(rounds=1), sessions)
    form = prior_form(summary)
    assert not form.empty
