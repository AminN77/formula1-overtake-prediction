# 🏎️ F1 Overtake Prediction — 1-Month Roadmap

## **🎯 Vision**

Build an end-to-end pipeline that predicts overtaking probability in Formula One using telemetry, tire, race, and track context data.

---

# **📅 High-Level Roadmap (4 Weeks)**

## **Week 1 — Data & Overtake Detection**

**Goals:** Collect data, detect overtakes, define dataset schema.

**Tasks**

* Set up repo structure + Kanban board.
* Collect data from FastF1, Kaggle, FIA reports, and tire datasets.
* Implement automatic overtaking detection (exclude pit stops).
* Validate detected overtakes using telemetry (speed/DRS).
* Define dataset schema for attacker–defender pairs.

---

## **Week 2 — Feature Engineering**

**Goals:** Build rich car, race, track, and driver features.

**Tasks**

* Car state: speed delta, throttle/brake patterns, DRS status.
* Race state: lap number, gaps, pit status, safety cars.
* Track features: sector type, DRS zone, segment metadata.
* Tire data: compound, age, degradation metrics.
* Driver stats: experience, historical overtakes.
* Label generation: overtake within next N laps.
* Validate sample dataset rows.

---

## **Week 3 — Modeling & Evaluation**

**Goals:** Train models, handle imbalance, produce metrics & insights.

**Tasks**

* Train baseline models: Logistic Regression, Random Forest.
* Train boosted models: XGBoost / LightGBM / CatBoost.
* Perform cross-race generalization tests.
* Compute metrics: F1, ROC-AUC, Precision@Top10%.
* Run feature importance analysis (SHAP).
* Save models + evaluation plots.

---

## **Week 4 — Insights, Final Pipeline & Deliverables**

**Goals:** Produce interpretable results, finalize pipeline, prepare presentation.

**Tasks**

* Analyze 2–3 races as case studies (predictions vs reality).
* Convert notebooks into clean scripts (ETL → features → model).
* Document entire pipeline (README + comments).
* Optional: Build Streamlit/Notebook dashboard for visualization.
* Prepare final report summarizing methodology + insights.

---

# **📌 Success Criteria**

* Accurate overtake detection pipeline.
* Structured dataset with contextual features.
* Working ML model with measurable performance.
* Clear insights on what drives overtaking (DRS, tires, speed delta).
* Reproducible code and documented workflow.