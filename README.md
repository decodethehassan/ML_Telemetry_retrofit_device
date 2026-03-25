# ⚡ PV Fault Detection System
### Single-Phase Photovoltaic Fault Diagnosis — Final Year Project

> A complete end-to-end pipeline for real-time fault detection in single-phase PV systems, built with XGBoost, rolling-window feature engineering, rule-based hysteresis logic, and hardware-deployable output.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Fault Classes](#fault-classes)
- [Dataset Sources](#dataset-sources)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Model Architecture](#model-architecture)
- [Hardware Deployment](#hardware-deployment)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

---

## Project Overview

This project develops a **real-time fault detection and classification system** for single-phase photovoltaic (PV) systems. The system:

- Reads voltage, current, and power sensor data in a streaming fashion
- Classifies the PV system state into one of three classes: **Normal**, **Open Circuit**, or **Shading**
- Uses a two-stage XGBoost classifier augmented with rule-based gates and hysteresis logic
- Produces a deployment-ready hardware profile (JSON + C header file) for embedded integration

---

## Problem Statement

Two PV datasets were available:

| Dataset | Type | Source |
|---|---|---|
| GPVS-Faults | 3-phase, public | Academic benchmark |
| Single-phase dataset | Single-phase | Provided by project supervisor |

**The core challenge:** Can a 3-phase public dataset be used to improve single-phase fault detection on the target deployment domain?

**Conclusion reached:** Direct 3-phase → single-phase conversion is not scientifically valid due to phase-relationship dependencies. Instead, a **common-fault mapping + feature alignment + target-domain-first training** strategy was adopted.

---

## Fault Classes

Only faults common to both datasets were retained:

| Class | GPVS Label | Description |
|---|---|---|
| `normal` | F0 | Healthy system operation |
| `open_circuit` | F7 | One or more modules disconnected |
| `shading` | F5 | Partial shading on panel surface |

All other GPVS fault types were excluded to maintain label validity for the single-phase deployment target.

---

## Dataset Sources

### GPVS-Faults (3-phase public dataset)
- F0 → normal
- F5 → shading
- F7 → open_circuit
- Fault files segmented: only the clearly-faulted portion retained (transition region removed)

### Single-Phase Dataset (supervisor-provided)
- Multiple normal operating files
- One shading file
- Multiple open-circuit files
- Power unit consistency verified against `V × I` product

---

## Project Structure

```
pv-fault-detection/
│
├── data/
│   ├── raw/
│   │   ├── gpvs/               # Raw GPVS-Faults dataset files
│   │   └── single_phase/       # Raw single-phase dataset files
│   └── processed/
│       ├── master_table.csv         # Unified, label-mapped dataset
│       ├── windows_table.csv        # Rolling-window feature dataset
│       └── healthy_references.csv   # Healthy median reference values
│
├── src/
│   ├── 01_build_master_table.py     # Data ingestion, label mapping, unit correction
│   ├── 02_make_windows.py           # Rolling window feature engineering
│   ├── 03_train_xgb.py              # Two-stage XGBoost training & evaluation
│   ├── 04_streaming_hysteresis_eval.py   # Streaming simulation with hysteresis
│   └── 05_finalize_hardware_profile.py   # Final hardened pipeline + deployment export
│
├── models/
│   ├── stage1_model.json            # Stage 1: Normal vs Fault classifier
│   └── stage2_model.json            # Stage 2: Open Circuit vs Shading classifier
│
├── reports/
│   ├── sanity_report.txt
│   ├── training_results.txt
│   └── final_evaluation.txt
│
├── deployment/
│   ├── deployment_profile.json      # Feature list, thresholds, fill medians
│   └── deployment_profile.h         # C header file for firmware integration
│
└── README.md
```

---

## Pipeline Architecture

```
Raw Data (GPVS + Single-Phase)
         │
         ▼
[01] Build Master Table
  • Label mapping
  • Unit correction (verify P = V × I)
  • GPVS fault segmentation
  • Unified CSV with: timestamp, voltage, current, power,
    temperature, class_label, source_domain
         │
         ▼
[02] Rolling Window Features
  • Window size: 5 samples
  • Features per signal: mean, std, min, max,
    first, last, slope, range, delta
  • Relative features: V/I/P vs healthy reference
  • Healthy reference medians saved per domain
         │
         ▼
[03] Two-Stage XGBoost Training
  • Experiment A: target_only
  • Experiment B: transfer_weighted (GPVS + target)
  • Winner: target_only
         │
         ▼
[04] Streaming + Hysteresis Logic
  • Probability thresholds
  • Consecutive confirmation (k_on / k_off)
  • Hysteresis enter/exit logic
  • Open-circuit hard rule
         │
         ▼
[05] Hardware-Oriented Hardening
  • Conservative shading gate (multi-condition)
  • Strong open-circuit rule (current + power fraction)
  • Reduced false alarms on healthy data
  • Export: JSON profile + C header
```

---

## Key Design Decisions

### 1. Why Not LSTM?
The sensor data is **tabular electrical measurements** (voltage, current, power). For this:
- XGBoost is typically stronger for tabular data with small target datasets
- Lower computational cost → suitable for embedded deployment
- More interpretable for debugging
- LSTM preserved as optional benchmark comparison

### 2. Why Two-Stage Classification?
- Easier debugging: isolate normal-vs-fault from fault-type decisions
- More realistic for hardware: Stage 1 acts as an alert gate
- Extensible: new fault types can be added to Stage 2 without retraining Stage 1

### 3. Why Relative Features?
GPVS and single-phase data have different absolute electrical scales. Relative features (e.g., `power / healthy_reference_power`) normalize across domains and make transfer more meaningful.

### 4. Why target_only Won Over transfer_weighted?
The professor's single-phase data is already the correct deployment domain. Forcing GPVS data into training with high weight introduced noise from scale and phase mismatches. GPVS is useful context, but the model must stay **target-domain driven**.

### 5. Why Rule-Based Gates?
The main failure mode of the raw model was **normal samples being predicted as shading**. Pure ML probability was insufficient. Conservative multi-condition gates (relative power, relative current, probability together) significantly reduced false alarms.

---

## Model Architecture

### Stage 1 — Normal vs Fault (Binary)

| Input | Window-based features (voltage, current, power + relative) |
|---|---|
| Output | `normal` or `fault` |
| Algorithm | XGBoost binary classifier |

### Stage 2 — Fault Type (Conditional)

Triggered only when Stage 1 outputs `fault`.

| Input | Same window features |
|---|---|
| Output | `open_circuit` or `shading` |
| Algorithm | XGBoost multi-class classifier |

### Final Decision Logic

```
Stage 1 → FAULT?
   │
   NO  → Report: NORMAL
   │
   YES → Apply open-circuit hard rule:
            low current fraction AND low power fraction?
            → YES: Report OPEN CIRCUIT
            → NO:  Apply shading gate:
                     fault prob + shading prob + low rel. power + low rel. current?
                     → YES: Report SHADING
                     → NO:  Report NORMAL (conservative fallback)
```

Hysteresis wraps the entire decision: state only changes after `k_on` consecutive confirmations, and only clears after `k_off` consecutive normal windows.

---

## Hardware Deployment

Two deployment artifacts are generated by `05_finalize_hardware_profile.py`:

### `deployment_profile.json`
```json
{
  "feature_list": [...],
  "fill_medians": {...},
  "thresholds": {
    "fault_enter": ...,
    "fault_exit": ...,
    "shading_prob": ...,
    "oc_current_fraction": ...,
    "oc_power_fraction": ...
  },
  "hysteresis": {
    "k_on": ...,
    "k_off": ...
  }
}
```

### `deployment_profile.h`
C header file containing:
- `#define WINDOW_SIZE`
- `#define FAULT_ENTER_THRESHOLD`
- `#define SHADING_PROB_THRESHOLD`
- `#define OC_CURRENT_FRACTION`
- `#define OC_POWER_FRACTION`
- Feature count and feature name array
- Fill median values for missing-data handling

---

## Results Summary

| Metric | Observation |
|---|---|
| Best training strategy | `target_only` (outperformed `transfer_weighted`) |
| Weakest component (initial) | Stage 1 normal-vs-fault detection |
| Most common initial error | Normal predicted as Shading |
| After hardening | Normal recall improved significantly |
| False shading alarms | Reduced substantially after multi-condition gate |
| Deployment readiness | JSON + C header exported |

---

## How to Run

### Step 1: Build the master dataset
```bash
python src/01_build_master_table.py
```

### Step 2: Create rolling window features
```bash
python src/02_make_windows.py
```

### Step 3: Train the XGBoost models
```bash
python src/03_train_xgb.py
```

### Step 4: Evaluate streaming + hysteresis behavior
```bash
python src/04_streaming_hysteresis_eval.py
```

### Step 5: Generate final hardware profile
```bash
python src/05_finalize_hardware_profile.py
```

Deployment artifacts will be saved in the `deployment/` folder.

---

## Dependencies

```
python >= 3.8
xgboost
scikit-learn
pandas
numpy
matplotlib
```

Install with:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib
```

---

## Future Work

| Option | Description |
|---|---|
| LSTM benchmark | Train an LSTM model for comparison; not intended to replace XGBoost |
| Expanded target data | Collect more single-phase shading and open-circuit samples |
| Additional fault types | Extend Stage 2 with line-line fault, bypass diode fault |
| MCU integration | Deploy `.h` profile on STM32 / Arduino with ADC sampling |
| Real-time dashboard | Serial monitor + live prediction display |

---

## Summary

> We started from two mismatched PV datasets, selected only the common fault classes, unified them into one structured pipeline, built a window-based XGBoost fault diagnosis system, improved it with hardware-style rules and hysteresis, and finally produced a deployable fault detection profile for real-time embedded use.

---

*Final Year Project — Electrical/Electronics Engineering*
