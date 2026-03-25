from pathlib import Path
import json
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

OUT_DIR = PROJECT_ROOT / "reports" / "streaming"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "streaming_summary.txt"
OUT_VAL_PRED = OUT_DIR / "val_predictions.csv"
OUT_TEST_PRED = OUT_DIR / "test_predictions.csv"
OUT_EXT_PRED = OUT_DIR / "external_predictions.csv"
OUT_PARAMS = OUT_DIR / "best_streaming_params.json"


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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
        ),
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


def simulate_stream_file(df_file: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Streaming simulation with:
    - enter threshold
    - exit threshold
    - consecutive confirmations
    - open-circuit hard rule
    """

    df_file = df_file.sort_values(["window_center_index", "window_start_index"]).reset_index(drop=True).copy()

    enter_th = params["enter_th"]
    exit_th = params["exit_th"]
    k_on = params["k_on"]
    k_off = params["k_off"]
    shade_th = params["shade_th"]
    oc_current_frac_th = params["oc_current_frac_th"]
    oc_power_frac_th = params["oc_power_frac_th"]

    state = "normal"
    current_fault_type = "open_circuit"

    on_count = 0
    off_count = 0

    preds = []

    for _, row in df_file.iterrows():
        fault_prob = float(row["fault_prob"])
        shade_prob = float(row["shade_prob"])

        low_current_fraction = row.get("low_current_fraction", np.nan)
        low_power_fraction = row.get("low_power_fraction", np.nan)

        oc_rule = False
        if pd.notna(low_current_fraction) and pd.notna(low_power_fraction):
            if low_current_fraction >= oc_current_frac_th and low_power_fraction >= oc_power_frac_th:
                oc_rule = True

        predicted_fault_type = "shading" if shade_prob >= shade_th else "open_circuit"
        if oc_rule:
            predicted_fault_type = "open_circuit"

        if state == "normal":
            if fault_prob >= enter_th:
                on_count += 1
            else:
                on_count = 0

            if on_count >= k_on:
                state = "fault"
                current_fault_type = predicted_fault_type
                on_count = 0

            if state == "normal":
                preds.append("normal")
            else:
                preds.append(current_fault_type)

        else:
            if fault_prob <= exit_th:
                off_count += 1
            else:
                off_count = 0

            current_fault_type = predicted_fault_type

            if off_count >= k_off:
                state = "normal"
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


def grid_search_params(val_df: pd.DataFrame):
    candidates = []

    for enter_th in [0.45, 0.50, 0.55, 0.60]:
        for exit_th in [0.20, 0.25, 0.30, 0.35]:
            if exit_th >= enter_th:
                continue
            for k_on in [2, 3]:
                for k_off in [3, 5]:
                    params = {
                        "enter_th": enter_th,
                        "exit_th": exit_th,
                        "k_on": k_on,
                        "k_off": k_off,
                        "shade_th": 0.50,
                        "oc_current_frac_th": 0.80,
                        "oc_power_frac_th": 0.80,
                    }

                    pred_df = simulate_stream_dataset(val_df, params)
                    metrics = eval_multiclass(pred_df["class_label"], pred_df["pred_label"])

                    candidates.append({
                        "params": params,
                        "macro_f1": metrics["macro_f1"],
                        "accuracy": metrics["accuracy"],
                    })

    best = sorted(candidates, key=lambda x: (x["macro_f1"], x["accuracy"]), reverse=True)[0]
    return best["params"], candidates


def summarize_results(name: str, metrics: dict, lines: list[str]):
    lines.append(name)
    lines.append(f"  accuracy={metrics['accuracy']:.4f}")
    lines.append(f"  macro_f1={metrics['macro_f1']:.4f}")
    lines.append(f"  confusion_matrix={metrics['confusion_matrix']}")
    lines.append("")


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

    # Rebuild the same target split
    _, target_val, target_test, external_df, split_log = build_target_splits(df)

    print("Preparing model probabilities...")
    target_val = add_model_outputs(target_val, stage1_model, stage2_model, feature_cols, med1, med2)
    target_test = add_model_outputs(target_test, stage1_model, stage2_model, feature_cols, med1, med2)
    external_df = add_model_outputs(external_df, stage1_model, stage2_model, feature_cols, med1, med2)

    print("Tuning streaming parameters on validation split...")
    best_params, all_candidates = grid_search_params(target_val)

    print("Best params found:")
    print(best_params)

    # Apply best params
    val_pred = simulate_stream_dataset(target_val, best_params)
    test_pred = simulate_stream_dataset(target_test, best_params)
    ext_pred = simulate_stream_dataset(external_df, best_params)

    # Save predictions
    val_pred.to_csv(OUT_VAL_PRED, index=False)
    test_pred.to_csv(OUT_TEST_PRED, index=False)
    ext_pred.to_csv(OUT_EXT_PRED, index=False)

    # Metrics
    val_metrics = eval_multiclass(val_pred["class_label"], val_pred["pred_label"])
    test_metrics = eval_multiclass(test_pred["class_label"], test_pred["pred_label"])
    ext_metrics = eval_multiclass(ext_pred["class_label"], ext_pred["pred_label"])

    # Save params
    with open(OUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Report
    lines = []
    lines.append("STREAMING / HYSTERESIS EVALUATION")
    lines.append("=" * 80)
    lines.append("Target split summary:")
    lines.extend(split_log)
    lines.append("")
    lines.append("Best params:")
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    summarize_results("Validation", val_metrics, lines)
    summarize_results("Test", test_metrics, lines)
    summarize_results("External", ext_metrics, lines)

    lines.append("Notes:")
    lines.append("- External set is the threshold100 open-circuit file.")
    lines.append("- For external set, accuracy and open-circuit behavior are more informative than macro F1.")
    lines.append("- This script evaluates streaming logic closer to deployment behavior.")

    save_text(OUT_SUMMARY, lines)

    print("=" * 80)
    print("DONE")
    print(f"Summary saved to: {OUT_SUMMARY}")
    print(f"Best params saved to: {OUT_PARAMS}")
    print(f"Validation predictions saved to: {OUT_VAL_PRED}")
    print(f"Test predictions saved to: {OUT_TEST_PRED}")
    print(f"External predictions saved to: {OUT_EXT_PRED}")
    print("=" * 80)


if __name__ == "__main__":
    main()