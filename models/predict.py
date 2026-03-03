"""
Overtake probability inference script.

Usage (run from models/):
    # Score a CSV of battles
    python predict.py --input ../data/v2/battles_2022.csv

    # Score and filter to likely overtakes only
    python predict.py --input ../data/v2/battles_2022.csv --threshold 0.15

    # Use a specific saved model
    python predict.py --input battles_new.csv --model artifacts/overtake_model_v2.pkl

Output: prints a summary and saves *_predictions.csv alongside the input file.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: Path):
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train and save a model first by running all cells in model_testing_2.ipynb")
        sys.exit(1)
    return joblib.load(model_path)


def load_meta(model_path: Path) -> dict:
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def build_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Reproduce the feature engineering from the training notebook."""
    df = df.copy()

    if "qualification_rank_difference" not in df.columns:
        if "attacker_qualification_rank" in df.columns and "defender_qualification_rank" in df.columns:
            df["qualification_rank_difference"] = (
                df["attacker_qualification_rank"] - df["defender_qualification_rank"]
            )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"WARNING: these feature columns are missing and will be set to 0: {missing}")
        for c in missing:
            df[c] = 0

    return df[feature_cols]


def predict(input_path: Path, model_path: Path, threshold: float = None) -> pd.DataFrame:
    model = load_model(model_path)
    meta  = load_meta(model_path)

    feature_cols = meta.get("features", [])
    if not feature_cols:
        print("Metadata not found — could not determine feature list.")
        sys.exit(1)

    t = threshold if threshold is not None else meta.get("threshold", 0.5)

    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"Loaded {len(df):,} battles from {input_path}")

    X = build_features(df, feature_cols)
    probas = model.predict_proba(X)[:, 1]

    df["overtake_probability"] = probas.round(4)
    df["overtake_predicted"]   = (probas >= t).astype(int)

    predicted_count = df["overtake_predicted"].sum()
    print(f"\nResults (threshold = {t:.3f}):")
    print(f"  Predicted overtakes : {predicted_count:,} / {len(df):,}  ({predicted_count/len(df):.2%})")
    if "overtake" in df.columns:
        actual_count = df["overtake"].astype(int).sum()
        print(f"  Actual  overtakes   : {actual_count:,} / {len(df):,}  ({actual_count/len(df):.2%})")

    if "race_name" in df.columns:
        print("\nTop-5 races by mean overtake probability:")
        print(
            df.groupby("race_name")["overtake_probability"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .to_string()
        )

    out_path = input_path.with_name(input_path.stem + "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Score battles with the saved overtake model")
    parser.add_argument("--input",  required=True,  help="Path to battle CSV (v2 schema)")
    parser.add_argument("--model",  default=str(Path(__file__).parent / "artifacts" / "overtake_model_v2.pkl"),
                        help="Path to saved model .pkl")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: use value from model metadata)")
    args = parser.parse_args()

    predict(
        input_path  = Path(args.input),
        model_path  = Path(args.model),
        threshold   = args.threshold,
    )


if __name__ == "__main__":
    main()
