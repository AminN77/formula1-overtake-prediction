# 🏁 **F1 Overtake Prediction — Concrete Task List**

---

# **📌 Phase 1 — Problem Definition**

### **1.1 Finalize Prediction Problem**

* Write 1–2 sentence definition of the prediction task
* Choose horizon = 1 lap
* Confirm gap threshold < 2 seconds
* Document exclusions (pit, lapped cars, DNF)

### **1.2 Define Dataset Structure**

* Decide on row = (attacker, defender) at start of lap
* Define schema for attacker, defender, labels, features

**Deliverable:** `/docs/problem_definition.md`

---

# **📌 Phase 2 — Raw Data Acquisition**

### **2.1 Gather Telemetry & Timing Data**

* Download race sessions using FastF1
* Cache all sessions locally
* Extract lap times, gaps, positions, DRS, speed, etc.

### **2.2 Gather Tire Data**

* Collect tire compounds + life from Pirelli PDFs or APIs
* Link tire data to lap numbers

### **2.3 Gather Race Control Data**

* Extract pit-in/pit-out
* Extract SC/VSC periods
* Extract DNF status

### **2.4 Gather Track Metadata**

* DRS zone lengths
* Sector types
* Track categories (street/permanent)

**Deliverable:** `/data/raw/<race>/`

---

# **📌 Phase 3 — Labeling (Overtake Extraction)**

### **3.1 Detect Raw Position Changes**

* Compare positions for each driver between lap L and lap L+1
* Generate initial list of position swaps

### **3.2 Filter Out Non-Overtakes**

* Exclude pit-stop-related swaps
* Exclude lapped/unlapped events
* Exclude DNFs
* Exclude SC restarts (optional)

### **3.3 Validate Overtake Events**

* Manually sample 10–20 detected overtakes
* Ensure correctness of rules

**Deliverable:** `overtakes.csv` (driverA, driverB, lap, race_id)

---

# **📌 Phase 4 — Candidate Battle Generation**

### **4.1 Extract Race Order Per Lap**

* For each lap, build list of drivers sorted by position

### **4.2 Generate Attacker–Defender Pairs**

* For each lap L:

  * For each consecutive pair P, P+1:

    * Ensure same lap
    * Ensure not in pit
    * Ensure gap < 2s

### **4.3 Link Candidate to Overtake Label**

* For each candidate at lap L, set label = 1 if:

  * A true overtake occurs between L and L+1 with same pair
* Else label = 0

### **4.4 Store All Candidates**

**Deliverable:**
`candidates.csv` with fields:

* race_id
* lap
* attacker_id
* defender_id
* gap
* label

---

# **📌 Phase 5 — Feature Engineering**

### **5.1 Car State Features**

* Speed metrics
* Throttle/Brake
* DRS availability
* Speed differences (A-B)

### **5.2 Race State Features**

* Lap number
* SC/VSC
* Gap to next cars ahead/behind
* Pit stop count

### **5.3 Tire Features**

* Tire compound (one-hot)
* Tire age
* Tire age difference
* Tire performance proxies

### **5.4 Track Features**

* Track type (street/normal)
* DRS zone length (if standardized)
* Sector type next/previous lap

### **5.5 Battle Geometry Features**

* Gap trend last lap
* Speed delta trend
* Acceleration differences
* Overtake attempts in previous laps (optional)

### **5.6 Feature Cleaning**

* Normalize or scale needed features
* Validate no leakage from future laps
* Handle missing values

**Deliverable:** `features.csv`

---

# **📌 Phase 6 — Data Splitting**

### **6.1 Define Splitting Strategy**

* Split by race, not by rows
* Decide:

  * Train: 70–80% races
  * Val: 10%
  * Test: 10–20%

### **6.2 Create Train/Val/Test Index Files**

* `train_races.txt`
* `val_races.txt`
* `test_races.txt`

### **6.3 Verify No Leakage**

* Ensure no race appears in more than one split
* Ensure no future-derived features leak into lap L

**Deliverable:** `/splits/`

---

# **📌 Phase 7 — Baseline Modeling**

### **7.1 Train Logistic Regression Baseline**

* Use minimal feature set
* Ensure basic sanity

### **7.2 Evaluate Baseline**

* Compute AUC
* Compute F1-score
* Compute Precision@10%
* Interpret coefficients

### **7.3 Debug Failures**

* Inspect false positives
* Inspect false negatives
* Check data cleanliness

**Deliverable:** `baseline_report.md`

---

# **📌 Phase 8 — Boosted Models (XGBoost/LightGBM)**

### **8.1 Train XGBoost Model**

* Use full feature set
* Tune basic params (max_depth, learning_rate)

### **8.2 Evaluate Model on Validation and Test Races**

* AUC
* F1
* Precision@10%
* SHAP feature importance

### **8.3 Compare with Baseline**

* Identify which features improved
* Identify new failure modes

**Deliverable:** `model_report.md`

---

# **📌 Phase 9 — Error Analysis & Insights**

### **9.1 Analyze False Positives**

* Identify common reasons (e.g., defender strong top speed)

### **9.2 Analyze False Negatives**

* Missing features?
* Labeling issues?
* Rare events?

### **9.3 Race-by-Race Analysis**

* Evaluate per-track performance
* Detect distribution mismatch

### **9.4 Extract Racing Insights**

* Dominant features
* Track dependency of overtakes
* Tire deltas that trigger overtakes
* DRS impact

**Deliverable:** `/reports/insights.md`

---

# **📌 Phase 10 — Final Documentation & Packaging**

### **10.1 Create README Overview**

* Project motivation
* Dataset pipeline
* Modeling approach
* Key insights
* How to run

### **10.2 Clean Final Scripts**

* `extract_overtakes.py`
* `generate_candidates.py`
* `engineer_features.py`
* `train_model.py`
* `evaluate.py`

### **10.3 Create Notebooks for Presentation**

* EDA notebook
* Feature importance notebook
* Race case study notebook

### **10.4 (Optional) Build Visualization Tool**

* Streamlit dashboard
* Lap-by-lap battle prediction charts