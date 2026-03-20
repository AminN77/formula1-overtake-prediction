"""
F1 Overtake Probability — Gradio Web UI

Combines single-battle scoring (score_battle.py) and batch CSV
prediction (predict.py) into one interactive application.

Launch:
    cd models/
    python app.py
"""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_V4_MODEL = _HERE / "artifacts" / "overtake_model_v4.pkl"
_V4_META = _HERE / "artifacts" / "overtake_model_v4_meta.json"
_V3_MODEL = _HERE / "artifacts" / "overtake_model_v3.pkl"
_V3_META = _HERE / "artifacts" / "overtake_model_v3_meta.json"
_V2_MODEL = _HERE / "artifacts" / "overtake_model_v2.pkl"
_V2_META = _HERE / "artifacts" / "overtake_model_v2_meta.json"

sys.path.insert(0, str(_HERE.parent / "src"))
try:
    from pipeline.track_info import get_sector_type, get_track_type, get_drs_zone_info
except ImportError:
    def get_sector_type(track):
        return "mixed"
    def get_track_type(track):
        return "street"
    def get_drs_zone_info(track, sector):
        return False, 0


# ── Model loading ────────────────────────────────────────────────────────────

def _load_model_and_meta():
    """Prefer v4; fall back to v3; then v2."""
    for model_p, meta_p, version in [
        (_V4_MODEL, _V4_META, "v4"),
        (_V3_MODEL, _V3_META, "v3"),
        (_V2_MODEL, _V2_META, "v2"),
    ]:
        if model_p.exists() and meta_p.exists():
            pipeline = joblib.load(model_p)
            meta = json.loads(meta_p.read_text())
            return pipeline, meta, version
    raise FileNotFoundError(
        "No model found in artifacts/. Run model_testing_4.ipynb first."
    )


_PIPELINE, _META, _MODEL_VERSION = _load_model_and_meta()
_FEATURES = _META["features"]
_THRESHOLD = _META.get("threshold", 0.5)


# ── Track / speed helpers ────────────────────────────────────────────────────

RACES = [
    "Abu Dhabi Grand Prix", "Australian Grand Prix", "Austrian Grand Prix",
    "Azerbaijan Grand Prix", "Bahrain Grand Prix", "Belgian Grand Prix",
    "British Grand Prix", "Canadian Grand Prix", "Chinese Grand Prix",
    "Dutch Grand Prix", "Emilia Romagna Grand Prix", "French Grand Prix",
    "Hungarian Grand Prix", "Italian Grand Prix", "Japanese Grand Prix",
    "Las Vegas Grand Prix", "Mexico City Grand Prix", "Miami Grand Prix",
    "Monaco Grand Prix", "Qatar Grand Prix",
    "Saudi Arabian Grand Prix", "Singapore Grand Prix",
    "Spanish Grand Prix", "United States Grand Prix",
]

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

TYRE_PACE_RANK = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2,
    "INTERMEDIATE": 1.5, "WET": 2, "UNKNOWN": 1,
}
TYRE_CLIFF = {"SOFT": 18, "MEDIUM": 28, "HARD": 38, "INTERMEDIATE": 30, "WET": 25}

_CIRCUIT_SPEEDS = {
    "abu dhabi":     {"i1": 227.9, "i2": 297.7, "fl": 210.1, "st": 304.4},
    "australian":    {"i1": 227.1, "i2": 294.9, "fl": 287.3, "st": 256.3},
    "austrian":      {"i1": 240.0, "i2": 270.0, "fl": 260.0, "st": 290.0},
    "azerbaijan":    {"i1": 182.8, "i2": 206.2, "fl": 321.0, "st": 318.3},
    "bahrain":       {"i1": 171.6, "i2": 249.6, "fl": 279.8, "st": 255.3},
    "belgian":       {"i1": 284.7, "i2": 194.1, "fl": 214.9, "st": 237.1},
    "british":       {"i1": 244.1, "i2": 251.5, "fl": 242.1, "st": 301.9},
    "canadian":      {"i1": 199.5, "i2": 271.8, "fl": 278.6, "st": 305.9},
    "chinese":       {"i1": 210.0, "i2": 280.0, "fl": 275.0, "st": 310.0},
    "dutch":         {"i1": 204.0, "i2": 263.9, "fl": 298.8, "st": 259.2},
    "emilia":        {"i1": 220.0, "i2": 280.0, "fl": 270.0, "st": 310.0},
    "french":        {"i1": 239.0, "i2": 274.8, "fl": 290.1, "st": 310.7},
    "hungarian":     {"i1": 217.5, "i2": 235.8, "fl": 244.9, "st": 218.3},
    "italian":       {"i1": 249.3, "i2": 316.6, "fl": 308.5, "st": 282.2},
    "japanese":      {"i1": 209.4, "i2": 297.2, "fl": 253.8, "st": 292.9},
    "las vegas":     {"i1": 220.0, "i2": 280.0, "fl": 290.0, "st": 320.0},
    "mexico":        {"i1": 255.4, "i2": 279.1, "fl": 243.6, "st": 259.3},
    "miami":         {"i1": 210.0, "i2": 270.0, "fl": 280.0, "st": 300.0},
    "monaco":        {"i1": 166.0, "i2": 176.4, "fl": 257.2, "st": 273.9},
    "qatar":         {"i1": 220.0, "i2": 280.0, "fl": 275.0, "st": 310.0},
    "saudi":         {"i1": 209.6, "i2": 303.3, "fl": 295.2, "st": 308.5},
    "singapore":     {"i1": 249.9, "i2": 260.8, "fl": 243.4, "st": 228.1},
    "spanish":       {"i1": 240.3, "i2": 268.1, "fl": 276.8, "st": 252.3},
    "united states": {"i1": 184.7, "i2": 179.2, "fl": 199.3, "st": 300.9},
}
_FALLBACK_SPEEDS = {"i1": 220.0, "i2": 265.0, "fl": 270.0, "st": 285.0}
_COMPOUND_DELTA = {
    "SOFT": 1.005, "MEDIUM": 1.000, "HARD": 0.995,
    "INTERMEDIATE": 0.925, "WET": 0.870,
}


def _circuit_base(race_name: str) -> dict:
    name_lower = race_name.lower()
    for key, speeds in _CIRCUIT_SPEEDS.items():
        if key in name_lower:
            return speeds.copy()
    return _FALLBACK_SPEEDS.copy()


def _estimate_speeds(race: str, compound: str, lap_time: float, ref_time: float):
    base = _circuit_base(race)
    c_mult = _COMPOUND_DELTA.get(compound.upper(), 1.0)
    pace = (ref_time / lap_time) if lap_time and ref_time else 1.0
    return {k: round(v * c_mult * pace, 1) for k, v in base.items()}


def _interpret(proba: float) -> tuple[str, str]:
    """Return (label, colour) for the probability."""
    if proba < 0.05:
        return "Very unlikely", "#6c757d"
    elif proba < 0.15:
        return "Possible", "#ffc107"
    elif proba < 0.30:
        return "Likely", "#fd7e14"
    elif proba < 0.50:
        return "Very likely", "#dc3545"
    else:
        return "Highly likely", "#b5179e"


# ── Single-battle scoring ────────────────────────────────────────────────────

def score_single_battle(
    race, year, lap, total_laps,
    attacker_pos, defender_pos, gap,
    attacker_compound, attacker_tyre_age,
    defender_compound, defender_tyre_age,
    attacker_lap_time, defender_lap_time,
    sector, safety_car, yellow_flag,
    attacker_fresh_tyre, defender_fresh_tyre,
    attacker_stint, defender_stint,
    attacker_qual_rank, defender_qual_rank,
    air_temp, track_temp, humidity, rainfall, wind_speed,
    gap_delta_1, battle_duration, overtakes_this_race,
):
    try:
        year = int(year)
        lap = int(lap)
        total_laps = int(total_laps)
        attacker_pos = int(attacker_pos)
        defender_pos = int(defender_pos)
        gap = float(gap)
        attacker_tyre_age = int(attacker_tyre_age)
        defender_tyre_age = int(defender_tyre_age)
        attacker_lap_time = float(attacker_lap_time)
        defender_lap_time = float(defender_lap_time)
        sector = int(sector)
        attacker_stint = int(attacker_stint)
        defender_stint = int(defender_stint)
        attacker_qual_rank = int(attacker_qual_rank)
        defender_qual_rank = int(defender_qual_rank)
        air_temp = float(air_temp)
        track_temp = float(track_temp)
        humidity = float(humidity)
        wind_speed = float(wind_speed)
        gap_delta_1 = float(gap_delta_1)
        battle_duration = int(battle_duration)
        overtakes_this_race = int(overtakes_this_race)
    except (ValueError, TypeError) as e:
        return _error_html(f"Invalid input: {e}"), ""

    ref_time = (attacker_lap_time + defender_lap_time) / 2
    att_s = _estimate_speeds(race, attacker_compound, attacker_lap_time, ref_time)
    def_s = _estimate_speeds(race, defender_compound, defender_lap_time, ref_time)
    in_drs, drs_len = get_drs_zone_info(race, sector)

    pace_delta = defender_lap_time - attacker_lap_time
    is_closing = 1 if gap_delta_1 < 0 else 0
    tyre_age_diff = attacker_tyre_age - defender_tyre_age

    att_pace_rank = TYRE_PACE_RANK.get(attacker_compound.upper(), 1)
    def_pace_rank = TYRE_PACE_RANK.get(defender_compound.upper(), 1)
    compound_advantage = def_pace_rank - att_pace_rank
    cliff_thresh = TYRE_CLIFF.get(defender_compound.upper(), 28)
    tyre_cliff_risk = 1 if defender_tyre_age > cliff_thresh else 0
    attacker_on_newer_stint = 1 if attacker_stint > defender_stint else 0

    row = {
        "year": year,
        "race_name": race,
        "lap_number": lap,
        "total_laps": total_laps,
        "race_progress": round(lap / total_laps, 4) if total_laps > 0 else 0,
        "attacker_position": attacker_pos,
        "defender_position": defender_pos,
        "attacker_lap_time": attacker_lap_time,
        "defender_lap_time": defender_lap_time,
        "gap_ahead": gap,
        "pace_delta": pace_delta,
        "attacker_speed_i1": att_s["i1"],
        "defender_speed_i1": def_s["i1"],
        "attacker_speed_i2": att_s["i2"],
        "defender_speed_i2": def_s["i2"],
        "attacker_finish_line_speed": att_s["fl"],
        "defender_finish_line_speed": def_s["fl"],
        "attacker_straight_speed": att_s["st"],
        "defender_straight_speed": def_s["st"],
        "speed_i1_delta": att_s["i1"] - def_s["i1"],
        "speed_i2_delta": att_s["i2"] - def_s["i2"],
        "speed_fl_delta": att_s["fl"] - def_s["fl"],
        "speed_st_delta": att_s["st"] - def_s["st"],
        "safety_car": safety_car,
        "yellow_flag": yellow_flag,
        "attacker_tyre_compound": attacker_compound.upper(),
        "defender_tyre_compound": defender_compound.upper(),
        "attacker_tyre_age": attacker_tyre_age,
        "defender_tyre_age": defender_tyre_age,
        "tyre_age_difference": tyre_age_diff,
        "attacker_stint": attacker_stint,
        "defender_stint": defender_stint,
        "attacker_fresh_tyre": attacker_fresh_tyre,
        "defender_fresh_tyre": defender_fresh_tyre,
        "sector": sector,
        "sector_type": get_sector_type(race),
        "is_in_drs_zone": in_drs,
        "drs_zone_length": drs_len,
        "track_type": get_track_type(race),
        "air_temp": air_temp,
        "track_temp": track_temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "gap_delta_1": gap_delta_1,
        "gap_delta_3": gap_delta_1,  # best single-lap estimate
        "is_closing": is_closing,
        "closing_laps": is_closing,
        "pace_delta_avg_3": pace_delta,
        "battle_duration": battle_duration,
        "attempted_before": 0,
        "overtakes_this_race": overtakes_this_race,
        "compound_advantage": compound_advantage,
        "tyre_cliff_risk": tyre_cliff_risk,
        "attacker_on_newer_stint": attacker_on_newer_stint,
        "qualification_rank_difference": attacker_qual_rank - defender_qual_rank,
    }

    df = pd.DataFrame([row])

    missing = [c for c in _FEATURES if c not in df.columns]
    for c in missing:
        df[c] = 0

    proba = float(_PIPELINE.predict_proba(df[_FEATURES])[:, 1][0])
    label, colour = _interpret(proba)
    decision = "OVERTAKE" if proba >= _THRESHOLD else "No overtake"

    html = _result_html(proba, label, colour, decision, row)
    label_output = f"{proba:.1%} — {label}"
    return html, label_output


def _error_html(msg: str) -> str:
    return f"""
    <div style="padding:20px; background:#2d1117; border:1px solid #f85149;
                border-radius:12px; color:#f85149; font-size:15px;">
      <strong>Error:</strong> {msg}
    </div>"""


def _result_html(proba, label, colour, decision, row) -> str:
    pct = proba * 100
    bar_w = max(2, min(100, pct))

    decision_bg = "#22863a" if "OVERTAKE" in decision else "#586069"
    decision_text = decision

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif; max-width:560px; margin:auto;">

      <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                  border-radius:16px; padding:28px 32px; color:#e6e6e6;
                  border:1px solid #30363d;">

        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:18px;">
          <div>
            <div style="font-size:13px; color:#8b949e; text-transform:uppercase;
                        letter-spacing:1px;">Race</div>
            <div style="font-size:18px; font-weight:600; color:#fff;">{row.get('race_name','—')}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:13px; color:#8b949e;">Lap</div>
            <div style="font-size:18px; font-weight:600; color:#fff;">
              {row.get('lap_number','?')} / {row.get('total_laps','?')}
            </div>
          </div>
        </div>

        <div style="display:flex; gap:16px; margin-bottom:18px; flex-wrap:wrap;">
          <div style="background:#0d1117; border-radius:10px; padding:10px 16px; flex:1;
                      min-width:120px;">
            <div style="font-size:11px; color:#8b949e;">Positions</div>
            <div style="font-size:16px; color:#fff;">
              P{row.get('attacker_position','?')}
              <span style="color:#8b949e;"> vs </span>
              P{row.get('defender_position','?')}
            </div>
          </div>
          <div style="background:#0d1117; border-radius:10px; padding:10px 16px; flex:1;
                      min-width:120px;">
            <div style="font-size:11px; color:#8b949e;">Gap</div>
            <div style="font-size:16px; color:#fff;">{row.get('gap_ahead',0):.3f}s</div>
          </div>
          <div style="background:#0d1117; border-radius:10px; padding:10px 16px; flex:1;
                      min-width:120px;">
            <div style="font-size:11px; color:#8b949e;">Tyres</div>
            <div style="font-size:14px; color:#fff;">
              {row.get('attacker_tyre_compound','?')} ({row.get('attacker_tyre_age','?')}L)
              vs {row.get('defender_tyre_compound','?')} ({row.get('defender_tyre_age','?')}L)
            </div>
          </div>
        </div>

        <div style="text-align:center; margin:24px 0 12px;">
          <div style="font-size:48px; font-weight:700; color:{colour};">{pct:.1f}%</div>
          <div style="font-size:15px; color:{colour}; font-weight:500;">{label}</div>
        </div>

        <div style="background:#0d1117; border-radius:8px; height:14px; overflow:hidden;
                    margin:12px 0;">
          <div style="height:100%; width:{bar_w}%;
                      background:linear-gradient(90deg,#238636,{colour});
                      border-radius:8px; transition:width 0.4s;"></div>
        </div>

        <div style="text-align:center; margin-top:14px;">
          <span style="display:inline-block; padding:6px 18px; border-radius:20px;
                       background:{decision_bg}; color:#fff; font-size:14px;
                       font-weight:600; letter-spacing:0.5px;">
            {decision_text}
          </span>
          <div style="font-size:11px; color:#8b949e; margin-top:8px;">
            Model: {_MODEL_VERSION} &nbsp;|&nbsp; Threshold: {_THRESHOLD:.3f}
          </div>
        </div>

      </div>
    </div>"""


# ── Batch CSV scoring ────────────────────────────────────────────────────────

def _engineer_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce model_testing_3 feature engineering for batch data."""
    df = df.copy()

    if "qualification_rank_difference" not in df.columns:
        if "attacker_qualification_rank" in df.columns and "defender_qualification_rank" in df.columns:
            df["qualification_rank_difference"] = (
                df["attacker_qualification_rank"] - df["defender_qualification_rank"]
            )

    if "pace_delta" not in df.columns:
        if "defender_lap_time" in df.columns and "attacker_lap_time" in df.columns:
            df["pace_delta"] = df["defender_lap_time"] - df["attacker_lap_time"]

    for delta, att, dfn in [
        ("speed_i1_delta", "attacker_speed_i1", "defender_speed_i1"),
        ("speed_i2_delta", "attacker_speed_i2", "defender_speed_i2"),
        ("speed_fl_delta", "attacker_finish_line_speed", "defender_finish_line_speed"),
        ("speed_st_delta", "attacker_straight_speed", "defender_straight_speed"),
    ]:
        if delta not in df.columns and att in df.columns and dfn in df.columns:
            df[delta] = df[att] - df[dfn]

    # Battle sequence identification
    if "year" in df.columns and "race_name" in df.columns:
        df = df.sort_values(["year", "race_name", "attacker", "defender", "lap_number"]).copy()
        df["_pair"] = (
            df["attacker"].astype(str) + "_vs_" + df["defender"].astype(str)
            + "_" + df["race_name"].astype(str) + "_" + df["year"].astype(str)
        )
        df["_lap_gap"] = df.groupby("_pair")["lap_number"].diff().fillna(99)
        df["_break"] = (df["_lap_gap"] != 1).astype(int)
        df["_seq"] = df.groupby("_pair")["_break"].cumsum()
        df["_bg"] = df["_pair"] + "_" + df["_seq"].astype(str)
    else:
        df["_bg"] = range(len(df))

    # Gap trend features
    if "gap_ahead" in df.columns:
        grp = df.groupby("_bg")["gap_ahead"]
        df["gap_delta_1"] = grp.diff(1).fillna(0)
        df["gap_delta_3"] = grp.diff(3).fillna(0)
        df["is_closing"] = (df["gap_delta_1"] < 0).astype(int)
        df["closing_laps"] = (
            df.groupby("_bg")["is_closing"]
            .transform(lambda s: s.rolling(3, min_periods=1).sum())
        )

    if "pace_delta" in df.columns:
        df["pace_delta_avg_3"] = (
            df.groupby("_bg")["pace_delta"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )

    # Battle context
    df["battle_duration"] = df.groupby("_bg").cumcount() + 1

    if "overtake" in df.columns:
        race_pair = (
            df["attacker"].astype(str) + "_" + df["defender"].astype(str)
            + "_" + df["race_name"].astype(str) + "_" + df["year"].astype(str)
        )
        df["_rp"] = race_pair
        df["attempted_before"] = (
            df.sort_values("lap_number")
            .groupby("_rp")["overtake"]
            .transform(lambda s: s.shift(1).cummax().fillna(0))
            .astype(int)
        )
        race_key = df["race_name"].astype(str) + "_" + df["year"].astype(str)
        df["_rk"] = race_key
        df["overtakes_this_race"] = (
            df.sort_values("lap_number")
            .groupby("_rk")["overtake"]
            .transform(lambda s: s.shift(1).cumsum().fillna(0))
            .astype(int)
        )
        df.drop(columns=["_rp", "_rk"], inplace=True, errors="ignore")
    else:
        df["attempted_before"] = 0
        df["overtakes_this_race"] = 0

    # Tyre features
    att_pace = df["attacker_tyre_compound"].map(TYRE_PACE_RANK).fillna(1)
    def_pace = df["defender_tyre_compound"].map(TYRE_PACE_RANK).fillna(1)
    df["compound_advantage"] = def_pace - att_pace
    cliff = df["defender_tyre_compound"].map(TYRE_CLIFF).fillna(28)
    df["tyre_cliff_risk"] = (df["defender_tyre_age"] > cliff).astype(int)
    df["attacker_on_newer_stint"] = 0
    if "attacker_stint" in df.columns and "defender_stint" in df.columns:
        df["attacker_on_newer_stint"] = (df["attacker_stint"] > df["defender_stint"]).astype(int)

    df.drop(columns=["_pair", "_lap_gap", "_break", "_seq", "_bg"],
            inplace=True, errors="ignore")
    return df


def score_batch(file, threshold, filter_pits):
    if file is None:
        return None, "Please upload a CSV file.", None

    try:
        file_path = file if isinstance(file, str) else file.name
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        return None, f"**Error:** Could not read CSV: {e}", None

    original_len = len(df)

    if filter_pits and "pit_stop_involved" in df.columns:
        df = df[~df["pit_stop_involved"]].reset_index(drop=True)

    df = _engineer_batch_features(df)

    missing = [c for c in _FEATURES if c not in df.columns]
    for c in missing:
        df[c] = 0

    probas = _PIPELINE.predict_proba(df[_FEATURES])[:, 1]
    df["overtake_probability"] = probas.round(4)
    df["overtake_predicted"] = (probas >= threshold).astype(int)

    n_pred = int(df["overtake_predicted"].sum())
    n_total = len(df)
    has_actual = "overtake" in df.columns

    summary_parts = [
        f"**Scored {n_total:,} battles** (from {original_len:,} rows)",
        f"**Threshold:** {threshold:.3f}",
        f"**Predicted overtakes:** {n_pred:,} / {n_total:,} ({n_pred/max(n_total,1):.1%})",
    ]
    if has_actual:
        n_actual = int(df["overtake"].astype(int).sum())
        summary_parts.append(f"**Actual overtakes:** {n_actual:,} ({n_actual/max(n_total,1):.1%})")

        from sklearn.metrics import roc_auc_score, average_precision_score
        y = df["overtake"].astype(int).values
        if y.sum() > 0 and y.sum() < len(y):
            roc = roc_auc_score(y, probas)
            pr = average_precision_score(y, probas)
            summary_parts.append(f"**ROC-AUC:** {roc:.4f} &nbsp; **PR-AUC:** {pr:.4f}")

    if "race_name" in df.columns:
        top5 = (
            df.groupby("race_name")["overtake_probability"]
            .mean().sort_values(ascending=False).head(5)
        )
        race_lines = " | ".join(f"{r}: {v:.3f}" for r, v in top5.items())
        summary_parts.append(f"**Top races by mean P(overtake):** {race_lines}")

    if filter_pits and original_len > n_total:
        summary_parts.append(
            f"*Filtered {original_len - n_total:,} pit-stop-involved rows*"
        )

    summary_md = "\n\n".join(summary_parts)

    show_cols = ["overtake_probability", "overtake_predicted"]
    for c in ["race_name", "lap_number", "attacker", "defender", "overtake"]:
        if c in df.columns:
            show_cols.insert(0, c)
    preview = df[show_cols].sort_values("overtake_probability", ascending=False).head(50)

    with tempfile.NamedTemporaryFile(suffix="_predictions.csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        download_path = f.name

    return preview, summary_md, download_path


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="F1 Overtake Predictor") as app:

    gr.Markdown(
        """
        # F1 Overtake Predictor
        Predict the probability of an overtake during an F1 battle using the latest v4 model
        (IP03 improvements, trained on 2022-2024).
        """
    )

    with gr.Tabs():

        # ── Tab 1: Single Battle ─────────────────────────────────
        with gr.TabItem("Single Battle"):
            gr.Markdown("### Score a single battle scenario")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Race & lap**")
                    race = gr.Dropdown(choices=RACES, value="Italian Grand Prix", label="Grand Prix")
                    with gr.Row():
                        year = gr.Number(value=2025, label="Year", precision=0)
                        lap = gr.Number(value=35, label="Lap", precision=0)
                        total_laps = gr.Number(value=53, label="Total laps", precision=0)

                    gr.Markdown("**Positions & gap**")
                    with gr.Row():
                        attacker_pos = gr.Number(value=8, label="Attacker pos", precision=0)
                        defender_pos = gr.Number(value=7, label="Defender pos", precision=0)
                        gap = gr.Number(value=0.56, label="Gap ahead (s)", precision=3)

                    gr.Markdown("**Attacker**")
                    with gr.Row():
                        att_compound = gr.Dropdown(choices=COMPOUNDS, value="HARD", label="Tyre compound")
                        att_tyre_age = gr.Slider(0, 50, value=20, step=1, label="Tyre age (laps)")
                        att_lap_time = gr.Number(value=92.1, label="Lap time (s)", precision=1)

                    gr.Markdown("**Defender**")
                    with gr.Row():
                        def_compound = gr.Dropdown(choices=COMPOUNDS, value="HARD", label="Tyre compound")
                        def_tyre_age = gr.Slider(0, 50, value=22, step=1, label="Tyre age (laps)")
                        def_lap_time = gr.Number(value=92.8, label="Lap time (s)", precision=1)

                with gr.Column(scale=1):
                    gr.Markdown("**Advanced options**")
                    sector = gr.Number(value=1, label="Sector", precision=0)
                    with gr.Row():
                        safety_car = gr.Checkbox(value=False, label="Safety car")
                        yellow_flag = gr.Checkbox(value=False, label="Yellow flag")
                    with gr.Row():
                        att_fresh = gr.Checkbox(value=False, label="Att. fresh tyre")
                        def_fresh = gr.Checkbox(value=False, label="Def. fresh tyre")
                    with gr.Row():
                        att_stint = gr.Number(value=2, label="Att. stint", precision=0)
                        def_stint = gr.Number(value=2, label="Def. stint", precision=0)
                    with gr.Row():
                        att_qual = gr.Number(value=8, label="Att. quali", precision=0)
                        def_qual = gr.Number(value=7, label="Def. quali", precision=0)

                    gr.Markdown("**Weather**")
                    air_temp = gr.Slider(10, 45, value=25, step=0.5, label="Air temp (C)")
                    track_temp = gr.Slider(15, 65, value=36, step=0.5, label="Track temp (C)")
                    humidity_sl = gr.Slider(0, 100, value=50, step=1, label="Humidity (%)")
                    rainfall = gr.Checkbox(value=False, label="Rainfall")
                    wind_speed = gr.Slider(0, 15, value=2, step=0.5, label="Wind speed (m/s)")

                    gr.Markdown("**Battle context (v4)**")
                    gap_delta_1 = gr.Number(value=-0.05, label="Gap change last lap (s)",
                                            info="Negative = closing")
                    battle_dur = gr.Slider(1, 20, value=3, step=1, label="Battle duration (laps)")
                    ot_race = gr.Number(value=5, label="Overtakes in race so far", precision=0)

            predict_btn = gr.Button("Predict", variant="primary", size="lg")
            result_html = gr.HTML()
            result_label = gr.Textbox(label="Result", visible=False)

            predict_btn.click(
                fn=score_single_battle,
                inputs=[
                    race, year, lap, total_laps,
                    attacker_pos, defender_pos, gap,
                    att_compound, att_tyre_age,
                    def_compound, def_tyre_age,
                    att_lap_time, def_lap_time,
                    sector, safety_car, yellow_flag,
                    att_fresh, def_fresh,
                    att_stint, def_stint,
                    att_qual, def_qual,
                    air_temp, track_temp, humidity_sl, rainfall, wind_speed,
                    gap_delta_1, battle_dur, ot_race,
                ],
                outputs=[result_html, result_label],
            )

        # ── Tab 2: Batch CSV ─────────────────────────────────────
        with gr.TabItem("Batch CSV Scoring"):
            gr.Markdown(
                "### Upload a battle CSV and score all rows\n"
                "Accepts v4 data (from `data/v4/`) or v3/v2 data — missing features "
                "are engineered automatically."
            )

            with gr.Row():
                csv_input = gr.File(label="Upload battle CSV", file_types=[".csv"])
                with gr.Column():
                    threshold_sl = gr.Slider(
                        0.01, 0.95, value=round(_THRESHOLD, 2), step=0.01,
                        label="Decision threshold",
                    )
                    filter_pits_cb = gr.Checkbox(
                        value=True,
                        label="Filter out pit-stop-involved rows",
                    )
                    batch_btn = gr.Button("Score", variant="primary", size="lg")

            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(label="Top 50 predictions (by probability)", wrap=True)
            batch_download = gr.File(label="Download full predictions CSV")

            batch_btn.click(
                fn=score_batch,
                inputs=[csv_input, threshold_sl, filter_pits_cb],
                outputs=[batch_table, batch_summary, batch_download],
            )

    gr.Markdown(
        f"<center style='color:#8b949e; font-size:12px;'>"
        f"Model: {_MODEL_VERSION} &nbsp;|&nbsp; "
        f"Threshold: {_THRESHOLD:.3f} &nbsp;|&nbsp; "
        f"Trained on: {_META.get('train_years', '?')} &nbsp;|&nbsp; "
        f"Features: {len(_FEATURES)}"
        f"</center>"
    )


if __name__ == "__main__":
    app.launch(share=False, show_error=True, server_port=7865)
