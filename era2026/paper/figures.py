"""Generate every figure and number the paper cites, from the real pipeline."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated print palette (light surface #fcfcfb); single theme by design.
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, GOOD = "#d03b3b", "#0ca30c"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


def main() -> dict:
    from era2026.analysis import error_by_group, learning_curve, permutation_importance
    from era2026.backtest import run_backtest, pooled_metrics
    from era2026.calibration import BootstrapEnsemble, Calibrated, reliability
    from era2026.dataset import load_or_build
    from era2026.episodes import trainable
    from era2026.features import FEATURE_TIERS, model_features
    from era2026.models import (BaseRate, GradientBoosting, LogisticBaseline,
                                NeuralNet, RandomForest, SupportVector, FIVE)

    rows = load_or_build(2026)
    pool = trainable(rows)
    cols = model_features()
    mk = lambda: GradientBoosting(name="g", columns=cols)
    numbers: dict = {}

    print("running the full ladder...")
    models = [BaseRate(),
              LogisticBaseline(name="gap_only", columns=["gap_ahead"]),
              LogisticBaseline(name="logistic_5", columns=FIVE),
              NeuralNet(name="mlp", columns=cols),
              SupportVector(name="svm_rbf", columns=cols),
              LogisticBaseline(name="logistic_all", columns=cols, C=0.1),
              RandomForest(name="random_forest", columns=cols),
              GradientBoosting(name="lightgbm", columns=cols),
              Calibrated(name="shipped", factory=mk, method="isotonic")]
    res = run_backtest(rows, models)
    pooled = pooled_metrics(res.predictions)
    numbers["ladder"] = pooled.round(4).to_dict("records")

    LABEL = {"base_rate": "base rate", "gap_only": "gap alone", "logistic_5": "logistic (5)",
             "mlp": "MLP", "svm_rbf": "SVM (RBF)", "logistic_all": "logistic (all)",
             "random_forest": "random forest", "lightgbm": "LightGBM",
             "shipped": "LightGBM + isotonic"}

    # --- fig 1: ladder ------------------------------------------------
    d = pooled.dropna(subset=["pr_auc"]).sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    best = d["pr_auc"].max()
    ax.barh([LABEL.get(m, m) for m in d["model"]], d["pr_auc"],
            color=[S1 if v == best else "#9ec5f4" for v in d["pr_auc"]], height=.62)
    for y, v in enumerate(d["pr_auc"]):
        ax.text(v + .006, y, f"{v:.3f}", va="center", fontsize=8, fontweight="semibold")
    ax.set_xlabel("PR-AUC (base rate $=0.066$)")
    ax.set_xlim(0, d["pr_auc"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    save(fig, "ladder")

    # --- fig 2: per-round trajectory ----------------------------------
    rounds = sorted(res.per_round["round_number"].unique())
    def series(m):
        return [res.per_round.query("round_number==@r and model==@m")["pr_auc"].iloc[0]
                if len(res.per_round.query("round_number==@r and model==@m")) else np.nan
                for r in rounds]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for name, colour, label in [("shipped", S1, "LightGBM + isotonic"),
                                ("random_forest", S2, "random forest"),
                                ("gap_only", S3, "gap alone")]:
        ax.plot(rounds, series(name), marker="o", ms=4, lw=1.8, color=colour, label=label)
    ax.plot(rounds, series("base_rate"), ls="--", lw=1.2, color=MUTED, label="base rate")
    sd = np.nanstd(series("shipped"))
    numbers["per_round_sd"] = float(sd)
    ax.set_xlabel("round"); ax.set_ylabel("PR-AUC"); ax.set_xticks(rounds)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    save(fig, "per_round")

    # --- fig 3: calibration -------------------------------------------
    best_pred = res.predictions.query("model=='shipped'")
    rel = reliability(best_pred["label"].to_numpy(), best_pred["hazard"].to_numpy())
    numbers["reliability"] = rel.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    top = max(rel["predicted"].max(), rel["observed"].max()) * 1.12
    ax.plot([0, top], [0, top], ls="--", lw=1.2, color=MUTED, label="perfect")
    ax.plot(rel["predicted"], rel["observed"], marker="o", ms=5, lw=1.8, color=S1)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7.5)
    save(fig, "calibration")

    # --- fig 4: interval width ----------------------------------------
    widths = []
    for k in rounds:
        tr, te = pool[pool.round_number < k], pool[pool.round_number == k]
        if tr.empty or te.empty or tr["label"].nunique() < 2:
            continue
        iv = BootstrapEnsemble(name="b", factory=mk, n_bootstraps=25).fit(tr, tr["label"]).predict_interval(te)
        widths.append({"round": int(k), "rel": float((iv["width"] / iv["hazard"].clip(lower=1e-4)).median())})
    w = pd.DataFrame(widths)
    from scipy.stats import spearmanr
    rho = spearmanr(w["round"], w["rel"])
    numbers["interval"] = {"rho": float(rho.statistic), "p": float(rho.pvalue),
                           "first_half": float(w["rel"][:5].mean()), "second_half": float(w["rel"][5:].mean())}
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot(w["round"], w["rel"], marker="o", ms=5, lw=1.8, color=S1)
    z = np.polyfit(w["round"], w["rel"], 1)
    ax.plot(w["round"], np.poly1d(z)(w["round"]), ls="--", lw=1.2, color=MUTED)
    ax.set_xlabel("round"); ax.set_ylabel("interval width / hazard"); ax.set_xticks(rounds)
    save(fig, "intervals")

    # --- fig 5: learning curve ----------------------------------------
    lc = learning_curve(rows, mk, cols, test_round=14)
    numbers["learning_curve"] = lc.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(lc["train_rounds"], lc["train_pr_auc"], marker="o", ms=4, lw=1.8, color=S2, label="training")
    ax.plot(lc["train_rounds"], lc["test_pr_auc"], marker="o", ms=4, lw=1.8, color=S1, label="held-out (round 14)")
    ax.fill_between(lc["train_rounds"], lc["test_pr_auc"], lc["train_pr_auc"], color=S2, alpha=.08)
    ax.set_xlabel("training rounds"); ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5)
    save(fig, "learning_curve")

    # --- fig 6: error by circuit --------------------------------------
    err = error_by_group(best_pred, rows, "event_name").sort_values("bias")
    numbers["error_by_circuit"] = err.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    names = [e.replace(" Grand Prix", "") for e in err["event_name"]]
    ax.barh(names, err["bias"] * 100,
            color=[CRIT if abs(b) > .03 else "#9ec5f4" for b in err["bias"]], height=.62)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("predicted minus actual pass rate (percentage points)")
    ax.grid(axis="y", visible=False)
    save(fig, "error_by_circuit")

    # --- fig 7: permutation importance --------------------------------
    tr, te = pool[pool.round_number < 14], pool[pool.round_number == 14]
    fitted = GradientBoosting(name="g", columns=cols).fit(tr, tr["label"])
    pi = permutation_importance(fitted, te, cols, repeats=5)
    numbers["permutation_importance"] = pi.head(15).round(4).to_dict("records")
    numbers["zero_importance"] = int((pi["importance"] <= 0).sum())
    numbers["n_features_model"] = len(cols)
    numbers["n_features_registry"] = len(FEATURE_TIERS)
    topk = pi.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(topk["feature"], topk["importance"],
            xerr=topk["sd"], error_kw={"ecolor": MUTED, "elinewidth": .8},
            color=[S1 if t == "tier1" else S2 for t in topk["tier"]], height=.64)
    ax.set_xlabel("PR-AUC drop when shuffled (held-out round)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)]
    ax.legend(handles, ["tier 1", "tier 2"], fontsize=7.5, loc="lower right")
    save(fig, "permutation_importance")

    # --- fig 8: gap at pass -------------------------------------------
    events = pool[pool["label"] == 1]["gap_ahead"]
    numbers["gap_at_pass"] = {q: float(np.percentile(events, q)) for q in (50, 90, 95, 99)}
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.hist(events, bins=30, color=S1, edgecolor="white", linewidth=.4)
    ax.axvline(np.median(events), color=CRIT, lw=1.5, ls="--")
    ax.text(np.median(events) + .06, ax.get_ylim()[1] * .9,
            f"median {np.median(events):.2f}s", color=CRIT, fontsize=7.5)
    ax.set_xlabel("gap the lap before a pass (s)"); ax.set_ylabel("passes")
    save(fig, "gap_at_pass")

    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {HERE / 'numbers.json'}")
    return numbers


if __name__ == "__main__":
    main()
