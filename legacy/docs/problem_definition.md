# Overtake Probability Prediction — Problem Definition

## 1. Overview
This document defines the problem statement, labeling rules, and scope for predicting the probability of an overtaking event in Formula One. The goal is to formalize how battles, overtakes, and labels are defined so that data preparation and modeling can be performed consistently.

---

## 2. Prediction Task

**Objective:**  
Predict whether an attacker will overtake a defender within the next lap.

**Formal Definition:**  
At the start of lap *L*, for every pair of consecutive cars in race order where the time gap between them is **less than 1.0 second**, predict whether the attacker will perform an on-track overtake between lap *L* and lap *L+1*.

The model outputs a probability:

`P(overtake_next_lap | state_at_lap_L)`

---

## 3. Battle Definition (Candidate Situations)

A **battle** is defined as a pair of cars at the beginning of lap *L* that satisfies all of the following conditions:

1. **Consecutive positions**  
   - Defender at position *P*  
   - Attacker at position *P+1*

2. **Gap threshold**  
   - Time gap < 1.0 second

3. **Track and race constraints**  
   - Both cars on track (not in pit lane)  
   - Both cars on the same lap

Each battle forms a single input instance for the prediction model.

---

## 4. Overtake Definition (Positive Label)

A **true overtake** occurs when all the following conditions are met:

1. **Position change**  
   - Between lap *L* and lap *L+1*:  
     - Attacker was behind defender at lap *L*  
     - Attacker is ahead of defender at lap *L+1*

2. **Exclusions**  
   The position change must **not** be caused by:
   - A pit stop of the defender  
   - A DNF or retirement of the defender

Only competitive, on-track passes are considered overtakes.

---

## 5. Label Assignment

For each battle at lap *L*:

- **Label = 1** if a valid overtake (as defined above) occurs between lap *L* and *L+1* for the same attacker–defender pair.
- **Label = 0** otherwise.

---

## 6. Scope and Assumptions

- Only adjacent attacker–defender pairs are included.  
- Only battles with gap < 1.0 second are included.  
- Prediction horizon is exactly one lap.  
- All features used for prediction must be available at the start of lap *L*.  
- Pit stop–induced and mechanical-failure–induced position changes are excluded from labeling.

---

## 7. Expected Output Format

For each battle instance:
- race_id
- lap
- attacker_id
- defender_id
- label
- p_overtake_next_lap