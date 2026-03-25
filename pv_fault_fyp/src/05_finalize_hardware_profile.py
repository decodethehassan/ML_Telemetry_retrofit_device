from pathlib import Path
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows_table.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "xgb" / "target_only"

OUT_DIR = PROJECT_ROOT / "reports" / "final_hw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "final_hw_summary.txt"
OUT_PARAMS = OUT_DIR / "best_hw_params.json"
OUT_VAL_PRED = OUT_DIR / "val_predictions_hw.csv"
OUT_TEST_PRED = OUT_DIR / "test_predictions_hw.csv"
OUT_EXT_PRED = OUT_DIR / "external_predictions_hw.csv"

OUT_DEPLOY_JSON = PROJECT_ROOT / "models" / "xgb" / "deployment_profile.json"
OUT_DEPLOY_HEADER = PROJECT_ROOT / "models" / "xgb" / "deployment_profile.h"


# =========================
# Config
# =========================
BOUNDARY_GAP = 5
EXTERNAL_FILE = "solar_data_log_OpenCircuit_threshold100"
LABELS = ["normal", "open_circuit", "shading"]


# =========================
# Helpers
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(path: Path, lines: list[str]):
    path.write_text("\n".join(lines), encoding="utf-8")


def contiguous_split(df_file: pd.DataFrame, train_frac=0.60, val_frac=0.20, gap=5):
    df_file = df_file.sort_values(["window_center_index", "window_start_index"]).reset_index(drop=True)
    n = len(df_file)

    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + val_frac))

    gap = min(gap, max(1, n // 50))

    train_end = max(cut1 - gap, 1)
    val_start = min(cut1 + gap, n)
    val_end = max(cut2 - gap, val_start)
    test_start = min(cut2 + gap, n)

    train_df = df_file.iloc[:train_end].copy()
    val_df = df_file.iloc[val_start:val_end].copy()
    test_df = df_file.iloc[test_start:].copy()

    return train_df, val_df, test_df


def build_target_splits(df: pd.DataFrame):
    target = df[df["source_domain"] == "single_phase"].copy()

    external_df = target[target["source_file"] == EXTERNAL_FILE].copy()
    core_df = target[target["source_file"] != EXTERNAL_FILE].copy()

    train_parts = []
    val_parts = []
    test_parts = []
    split_log = []

    for file_name in sorted(core_df["source_file"].unique()):
        part = core_df[core_df["source_file"] == file_name].copy()
        tr, va, te = contiguous_split(part, gap=BOUNDARY_GAP)

        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

        split_log.append(
            f"{file_name}: total={len(part)}, train={len(tr)}, val={len(va)}, test={len(te)}"
        )

    target_train = pd.concat(train_parts, ignore_index=True)
    target_val = pd.concat(val_parts, ignore_index=True)
    target_test = pd.concat(test_parts, ignore_index=True)

    return target_train, target_val, target_test, external_df, split_log


def eval_multiclass(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": report,
    }


def prepare_features(df: pd.DataFrame, feature_cols: list[str], fill_values: dict):
    X = df[feature_cols].copy()
    fill_series = pd.Series(fill_values)
    return X.fillna(fill_series)


def add_model_outputs(df: pd.DataFrame, stage1_model, stage2_model, feature_cols, med1, med2):
    out = df.copy()

    X1 = prepare_features(out, feature_cols, med1)
    X2 = prepare_features(out, feature_cols, med2)

    out["fault_prob"] = stage1_model.predict_proba(X1)[:, 1]
    out["shade_prob"] = stage2_model.predict_proba(X2)[:, 1]

    return out


def shading_gate(row, params):
    fault_prob = float(row["fault_prob"])
    shade_prob = float(row["shade_prob"])

    power_rel = row.get("power_W_rel_mean", np.nan)
    current_rel = row.get("current_A_rel_mean", np.nan)
    low_power_fraction = row.get("low_power_fraction", np.nan)

    if pd.isna(power_rel) or pd.isna(current_rel):
        return False

    cond = (
        fault_prob >= params["shade_fault_prob_min"]
        and shade_prob >= params["shade_th"]
        and power_rel <= params["shade_power_rel_max"]
        and current_rel <= params["shade_current_rel_max"]
    )

    if pd.notna(low_power_fraction):
        cond = cond and (low_power_fraction >= params["shade_low_power_min"])

    return cond


def open_circuit_rule(row, params):
    low_current_fraction = row.get("low_current_fraction", np.nan)
    low_power_fraction = row.get("low_power_fraction", np.nan)

    if pd.isna(low_current_fraction) or pd.isna(low_power_fraction):
        return False

    return (
        low_current_fraction >= params["oc_current_frac_th"]
        and low_power_fraction >= params["oc_power_frac_th"]
    )


def simulate_stream_file(df_file: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Stricter hardware-oriented logic:
    - open circuit enters through hard rule
    - shading enters only if stricter shading gate passes
    - hysteresis is applied to state changes
    """

    df_file = df_file.sort_values(["window_center_index", "window_start_index"]).reset_index(drop=True).copy()

    enter_th = params["enter_th"]
    exit_th = params["exit_th"]
    k_on = params["k_on"]
    k_off = params["k_off"]

    state = "normal"
    current_fault_type = "open_circuit"

    on_count = 0
    off_count = 0

    preds = []

    for _, row in df_file.iterrows():
        fault_prob = float(row["fault_prob"])

        oc_gate = open_circuit_rule(row, params)
        sh_gate = shading_gate(row, params)

        if oc_gate:
            candidate_fault = True
            candidate_type = "open_circuit"
        elif sh_gate:
            candidate_fault = True
            candidate_type = "shading"
        else:
            candidate_fault = False
            candidate_type = "normal"

        if state == "normal":
            # More conservative fault entry
            if candidate_fault and fault_prob >= enter_th:
                on_count += 1
            else:
                on_count = 0

            if on_count >= k_on:
                state = "fault"
                current_fault_type = candidate_type
                on_count = 0

            preds.append("normal" if state == "normal" else current_fault_type)

        else:
            # While in fault state, update current fault type if supported
            if oc_gate:
                current_fault_type = "open_circuit"
            elif sh_gate:
                current_fault_type = "shading"

            # Exit condition
            if (not candidate_fault) and (fault_prob <= exit_th):
                off_count += 1
            else:
                off_count = 0

            if off_count >= k_off:
                state = "normal"
                current_fault_type = "open_circuit"
                off_count = 0
                preds.append("normal")
            else:
                preds.append(current_fault_type)

    df_file["pred_label"] = preds
    return df_file


def simulate_stream_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    parts = []
    for file_name in sorted(df["source_file"].unique()):
        part = df[df["source_file"] == file_name].copy()
        sim = simulate_stream_file(part, params)
        parts.append(sim)
    return pd.concat(parts, ignore_index=True)


def candidate_grid():
    # Conservative search focused on reducing false shading alarms
    enter_th_vals = [0.45, 0.50, 0.55]
    exit_th_vals = [0.25, 0.30]
    k_on_vals = [3, 4]
    k_off_vals = [3, 5]

    shade_th_vals = [0.55, 0.60, 0.65]
    shade_fault_prob_min_vals = [0.55, 0.60, 0.65]
    shade_power_rel_max_vals = [0.80, 0.85, 0.90]
    shade_current_rel_max_vals = [0.85, 0.95, 1.00]
    shade_low_power_min_vals = [0.0, 0.2]

    oc_current_frac_th_vals = [0.80]
    oc_power_frac_th_vals = [0.80]

    for vals in product(
        enter_th_vals,
        exit_th_vals,
        k_on_vals,
        k_off_vals,
        shade_th_vals,
        shade_fault_prob_min_vals,
        shade_power_rel_max_vals,
        shade_current_rel_max_vals,
        shade_low_power_min_vals,
        oc_current_frac_th_vals,
        oc_power_frac_th_vals,
    ):
        (
            enter_th,
            exit_th,
            k_on,
            k_off,
            shade_th,
            shade_fault_prob_min,
            shade_power_rel_max,
            shade_current_rel_max,
            shade_low_power_min,
            oc_current_frac_th,
            oc_power_frac_th,
        ) = vals

        if exit_th >= enter_th:
            continue

        yield {
            "enter_th": enter_th,
            "exit_th": exit_th,
            "k_on": k_on,
            "k_off": k_off,
            "shade_th": shade_th,
            "shade_fault_prob_min": shade_fault_prob_min,
            "shade_power_rel_max": shade_power_rel_max,
            "shade_current_rel_max": shade_current_rel_max,
            "shade_low_power_min": shade_low_power_min,
            "oc_current_frac_th": oc_current_frac_th,
            "oc_power_frac_th": oc_power_frac_th,
        }


def tune_params(val_df: pd.DataFrame):
    best = None
    all_results = []

    for params in candidate_grid():
        pred_df = simulate_stream_dataset(val_df, params)
        metrics = eval_multiclass(pred_df["class_label"], pred_df["pred_label"])

        report = metrics["classification_report"]
        normal_recall = float(report.get("normal", {}).get("recall", 0.0))
        normal_precision = float(report.get("normal", {}).get("precision", 0.0))
        shading_precision = float(report.get("shading", {}).get("precision", 0.0))

        # Hardware-oriented score:
        # prioritize macro F1, then keeping healthy cases healthy, then reducing false shading
        score = (
            metrics["macro_f1"]
            + 0.20 * normal_recall
            + 0.10 * normal_precision
            + 0.05 * shading_precision
        )

        record = {
            "params": params,
            "score": score,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "normal_recall": normal_recall,
            "normal_precision": normal_precision,
            "shading_precision": shading_precision,
            "confusion_matrix": metrics["confusion_matrix"],
        }
        all_results.append(record)

        if best is None or record["score"] > best["score"]:
            best = record

    return best, all_results


def build_header_text(feature_cols, med1, med2, params):
    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// Auto-generated deployment profile")
    lines.append("// Final hardware-oriented thresholds and feature metadata")
    lines.append("")
    lines.append("#define PV_WINDOW_SIZE 5")
    lines.append(f"#define PV_ENTER_TH {params['enter_th']:.6f}f")
    lines.append(f"#define PV_EXIT_TH {params['exit_th']:.6f}f")
    lines.append(f"#define PV_K_ON {int(params['k_on'])}")
    lines.append(f"#define PV_K_OFF {int(params['k_off'])}")
    lines.append("")
    lines.append(f"#define PV_SHADE_TH {params['shade_th']:.6f}f")
    lines.append(f"#define PV_SHADE_FAULT_PROB_MIN {params['shade_fault_prob_min']:.6f}f")
    lines.append(f"#define PV_SHADE_POWER_REL_MAX {params['shade_power_rel_max']:.6f}f")
    lines.append(f"#define PV_SHADE_CURRENT_REL_MAX {params['shade_current_rel_max']:.6f}f")
    lines.append(f"#define PV_SHADE_LOW_POWER_MIN {params['shade_low_power_min']:.6f}f")
    lines.append("")
    lines.append(f"#define PV_OC_CURRENT_FRAC_TH {params['oc_current_frac_th']:.6f}f")
    lines.append(f"#define PV_OC_POWER_FRAC_TH {params['oc_power_frac_th']:.6f}f")
    lines.append("")
    lines.append(f"#define PV_FEATURE_COUNT {len(feature_cols)}")
    lines.append("")
    lines.append("static const char* PV_FEATURE_NAMES[] = {")
    for f in feature_cols:
        lines.append(f'    "{f}",')
    lines.append("};")
    lines.append("")
    lines.append("static const float PV_STAGE1_FILL_MEDIANS[] = {")
    for f in feature_cols:
        lines.append(f"    {float(med1[f]):.9f}f,")
    lines.append("};")
    lines.append("")
    lines.append("static const float PV_STAGE2_FILL_MEDIANS[] = {")
    for f in feature_cols:
        lines.append(f"    {float(med2[f]):.9f}f,")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


# =========================
# Main
# =========================
def main():
    if not IN_WINDOWS.exists():
        raise FileNotFoundError(f"windows_table.csv not found: {IN_WINDOWS}")

    print("Reading windows table...")
    df = pd.read_csv(IN_WINDOWS, low_memory=False)

    stage1_model = joblib.load(MODEL_ROOT / "stage1_fault_detector.joblib")
    stage2_model = joblib.load(MODEL_ROOT / "stage2_fault_classifier.joblib")

    med1 = load_json(MODEL_ROOT / "stage1_fill_medians.json")
    med2 = load_json(MODEL_ROOT / "stage2_fill_medians.json")
    feature_cols = load_json(MODEL_ROOT / "feature_columns.json")["feature_cols"]

    # Rebuild same target split
    _, target_val, target_test, external_df, split_log = build_target_splits(df)

    print("Preparing model probabilities...")
    target_val = add_model_outputs(target_val, stage1_model, stage2_model, feature_cols, med1, med2)
    target_test = add_model_outputs(target_test, stage1_model, stage2_model, feature_cols, med1, med2)
    external_df = add_model_outputs(external_df, stage1_model, stage2_model, feature_cols, med1, med2)

    print("Tuning stricter hardware parameters on validation split...")
    best, all_results = tune_params(target_val)
    best_params = best["params"]

    print("Best hardware params:")
    print(best_params)
    print(f"Best score: {best['score']:.4f}")
    print(f"Best validation macro_f1: {best['macro_f1']:.4f}")
    print(f"Best validation normal_recall: {best['normal_recall']:.4f}")

    # Apply best params
    val_pred = simulate_stream_dataset(target_val, best_params)
    test_pred = simulate_stream_dataset(target_test, best_params)
    ext_pred = simulate_stream_dataset(external_df, best_params)

    val_pred.to_csv(OUT_VAL_PRED, index=False)
    test_pred.to_csv(OUT_TEST_PRED, index=False)
    ext_pred.to_csv(OUT_EXT_PRED, index=False)

    val_metrics = eval_multiclass(val_pred["class_label"], val_pred["pred_label"])
    test_metrics = eval_multiclass(test_pred["class_label"], test_pred["pred_label"])
    ext_metrics = eval_multiclass(ext_pred["class_label"], ext_pred["pred_label"])

    # Save params and deployment profile
    save_json(OUT_PARAMS, best_params)

    deploy_profile = {
        "model_root": str(MODEL_ROOT),
        "feature_cols": feature_cols,
        "stage1_fill_medians": med1,
        "stage2_fill_medians": med2,
        "window_size": 5,
        "best_hw_params": best_params,
        "notes": [
            "Use target_only XGBoost models for deployment.",
            "Open-circuit is protected by hard rule + hysteresis.",
            "Shading entry is stricter to reduce normal->shading false alarms."
        ]
    }
    save_json(OUT_DEPLOY_JSON, deploy_profile)

    header_text = build_header_text(feature_cols, med1, med2, best_params)
    OUT_DEPLOY_HEADER.write_text(header_text, encoding="utf-8")

    # Summary
    lines = []
    lines.append("FINAL HARDWARE-ORIENTED XGBOOST PROFILE")
    lines.append("=" * 80)
    lines.append("Target split summary:")
    lines.extend(split_log)
    lines.append("")
    lines.append("Best params:")
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Best tuning score: {best['score']:.4f}")
    lines.append(f"Best validation macro_f1 during tuning: {best['macro_f1']:.4f}")
    lines.append(f"Best validation normal_recall during tuning: {best['normal_recall']:.4f}")
    lines.append("")

    def add_metrics_block(name, metrics):
        lines.append(name)
        lines.append(f"  accuracy={metrics['accuracy']:.4f}")
        lines.append(f"  macro_f1={metrics['macro_f1']:.4f}")
        lines.append(f"  confusion_matrix={metrics['confusion_matrix']}")
        lines.append("")

    add_metrics_block("Validation", val_metrics)
    add_metrics_block("Test", test_metrics)
    add_metrics_block("External", ext_metrics)

    lines.append("Artifacts:")
    lines.append(f"  Params JSON: {OUT_PARAMS}")
    lines.append(f"  Deployment JSON: {OUT_DEPLOY_JSON}")
    lines.append(f"  Deployment header: {OUT_DEPLOY_HEADER}")
    lines.append(f"  Validation predictions: {OUT_VAL_PRED}")
    lines.append(f"  Test predictions: {OUT_TEST_PRED}")
    lines.append(f"  External predictions: {OUT_EXT_PRED}")

    save_text(OUT_SUMMARY, lines)

    print("=" * 80)
    print("DONE")
    print(f"Summary saved to: {OUT_SUMMARY}")
    print(f"Best params saved to: {OUT_PARAMS}")
    print(f"Deployment JSON saved to: {OUT_DEPLOY_JSON}")
    print(f"Deployment header saved to: {OUT_DEPLOY_HEADER}")
    print("=" * 80)


if __name__ == "__main__":
    main()