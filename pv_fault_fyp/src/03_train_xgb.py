from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows_table.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "xgb"
REPORT_DIR = PROJECT_ROOT / "reports" / "xgb"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================
RANDOM_STATE = 42
BOUNDARY_GAP = 5
GPVS_PER_CLASS_CAP = 20000

EXTERNAL_FILE = "solar_data_log_OpenCircuit_threshold100"

# Compact hardware-friendly feature set
FEATURE_COLS = [
    "voltage_V_std",
    "voltage_V_slope",
    "voltage_V_range",
    "voltage_V_delta",
    "voltage_V_rel_mean",
    "voltage_V_rel_last",
    "voltage_V_pct_change_from_ref",

    "current_A_std",
    "current_A_slope",
    "current_A_range",
    "current_A_delta",
    "current_A_rel_mean",
    "current_A_rel_last",
    "current_A_pct_change_from_ref",

    "power_W_std",
    "power_W_slope",
    "power_W_range",
    "power_W_delta",
    "power_W_rel_mean",
    "power_W_rel_last",
    "power_W_pct_change_from_ref",

    "mean_power_over_voltage",
    "mean_current_over_voltage",
    "low_current_fraction",
    "low_power_fraction",
]

FAULT_MAP = {
    "open_circuit": 0,
    "shading": 1,
}
INV_FAULT_MAP = {v: k for k, v in FAULT_MAP.items()}


# =========================
# Helpers
# =========================
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


def sample_gpvs_train(df: pd.DataFrame, per_class_cap=20000):
    gpvs = df[df["source_domain"] == "gpvs"].copy()

    sampled_parts = []
    log_lines = []

    for cls in ["normal", "open_circuit", "shading"]:
        part = gpvs[gpvs["class_label"] == cls].copy()
        n_take = min(per_class_cap, len(part))

        if len(part) > n_take:
            part = part.sample(n=n_take, random_state=RANDOM_STATE)

        sampled_parts.append(part)
        log_lines.append(f"GPVS sampled for {cls}: {len(part)}")

    gpvs_sample = pd.concat(sampled_parts, ignore_index=True)
    return gpvs_sample, log_lines


def prepare_X_y_stage1(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    y = df["fault_binary"].astype(int).copy()
    return X, y


def prepare_X_y_stage2(df: pd.DataFrame, feature_cols: list[str]):
    fault_df = df[df["class_label"].isin(["open_circuit", "shading"])].copy()
    X = fault_df[feature_cols].copy()
    y = fault_df["class_label"].map(FAULT_MAP).astype(int).copy()
    return X, y, fault_df


def fill_with_train_medians(X_train, X_other_list):
    med = X_train.median(numeric_only=True)

    X_train_filled = X_train.fillna(med)
    X_others_filled = [x.fillna(med) for x in X_other_list]

    return X_train_filled, X_others_filled, med.to_dict()


def get_stage1_model():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def get_stage2_model():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def eval_binary(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], zero_division=0, output_dict=True
        ),
    }


def eval_multiclass(y_true, y_pred):
    labels = ["normal", "open_circuit", "shading"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def pipeline_predict(stage1_model, stage2_model, X: pd.DataFrame):
    pred_fault = stage1_model.predict(X)

    final_pred = np.array(["normal"] * len(X), dtype=object)

    fault_mask = pred_fault == 1
    if np.any(fault_mask):
        pred_fault_type = stage2_model.predict(X.loc[fault_mask])
        final_pred[fault_mask] = [INV_FAULT_MAP[int(v)] for v in pred_fault_type]

    return final_pred


def top_features(model, feature_cols, top_n=15):
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def run_experiment(
    tag: str,
    train_df: pd.DataFrame,
    target_val: pd.DataFrame,
    target_test: pd.DataFrame,
    external_df: pd.DataFrame,
    feature_cols: list[str],
    sample_weight_single=1.0,
    sample_weight_gpvs=0.3,
):
    report_lines = []
    report_lines.append(f"EXPERIMENT: {tag}")
    report_lines.append("=" * 80)

    # -------------------------
    # Stage 1: Normal vs Fault
    # -------------------------
    X1_train, y1_train = prepare_X_y_stage1(train_df, feature_cols)
    X1_val, y1_val = prepare_X_y_stage1(target_val, feature_cols)
    X1_test, y1_test = prepare_X_y_stage1(target_test, feature_cols)
    X1_ext, y1_ext = prepare_X_y_stage1(external_df, feature_cols)

    X1_train, [X1_val, X1_test, X1_ext], med1 = fill_with_train_medians(
        X1_train, [X1_val, X1_test, X1_ext]
    )

    w1 = np.where(train_df["source_domain"] == "gpvs", sample_weight_gpvs, sample_weight_single)

    stage1_model = get_stage1_model()
    stage1_model.fit(X1_train, y1_train, sample_weight=w1, verbose=False)

    pred1_val = stage1_model.predict(X1_val)
    pred1_test = stage1_model.predict(X1_test)
    pred1_ext = stage1_model.predict(X1_ext)

    stage1_val_metrics = eval_binary(y1_val, pred1_val)
    stage1_test_metrics = eval_binary(y1_test, pred1_test)
    stage1_ext_metrics = eval_binary(y1_ext, pred1_ext)

    report_lines.append("Stage 1 metrics")
    report_lines.append(f"Validation accuracy={stage1_val_metrics['accuracy']:.4f}, macro_f1={stage1_val_metrics['macro_f1']:.4f}")
    report_lines.append(f"Test accuracy={stage1_test_metrics['accuracy']:.4f}, macro_f1={stage1_test_metrics['macro_f1']:.4f}")
    report_lines.append(f"External accuracy={stage1_ext_metrics['accuracy']:.4f}, macro_f1={stage1_ext_metrics['macro_f1']:.4f}")
    report_lines.append("")

    # -------------------------
    # Stage 2: Fault type
    # -------------------------
    X2_train, y2_train, train_fault_df = prepare_X_y_stage2(train_df, feature_cols)
    X2_val, y2_val, val_fault_df = prepare_X_y_stage2(target_val, feature_cols)
    X2_test, y2_test, test_fault_df = prepare_X_y_stage2(target_test, feature_cols)
    X2_ext, y2_ext, ext_fault_df = prepare_X_y_stage2(external_df, feature_cols)

    X2_train, [X2_val, X2_test, X2_ext], med2 = fill_with_train_medians(
        X2_train, [X2_val, X2_test, X2_ext]
    )

    w2 = np.where(train_fault_df["source_domain"] == "gpvs", sample_weight_gpvs, sample_weight_single)

    stage2_model = get_stage2_model()
    stage2_model.fit(X2_train, y2_train, sample_weight=w2, verbose=False)

    pred2_val = stage2_model.predict(X2_val)
    pred2_test = stage2_model.predict(X2_test)
    pred2_ext = stage2_model.predict(X2_ext)

    stage2_val_metrics = eval_binary(y2_val, pred2_val)
    stage2_test_metrics = eval_binary(y2_test, pred2_test)
    stage2_ext_metrics = eval_binary(y2_ext, pred2_ext)

    report_lines.append("Stage 2 metrics (fault-type only)")
    report_lines.append(f"Validation accuracy={stage2_val_metrics['accuracy']:.4f}, macro_f1={stage2_val_metrics['macro_f1']:.4f}")
    report_lines.append(f"Test accuracy={stage2_test_metrics['accuracy']:.4f}, macro_f1={stage2_test_metrics['macro_f1']:.4f}")
    report_lines.append(f"External accuracy={stage2_ext_metrics['accuracy']:.4f}, macro_f1={stage2_ext_metrics['macro_f1']:.4f}")
    report_lines.append("")

    # -------------------------
    # End-to-end pipeline
    # -------------------------
    Xp_val = target_val[feature_cols].fillna(pd.Series(med1))
    Xp_test = target_test[feature_cols].fillna(pd.Series(med1))
    Xp_ext = external_df[feature_cols].fillna(pd.Series(med1))

    pred_pipe_val = pipeline_predict(stage1_model, stage2_model, Xp_val)
    pred_pipe_test = pipeline_predict(stage1_model, stage2_model, Xp_test)
    pred_pipe_ext = pipeline_predict(stage1_model, stage2_model, Xp_ext)

    y_pipe_val = target_val["class_label"].to_numpy()
    y_pipe_test = target_test["class_label"].to_numpy()
    y_pipe_ext = external_df["class_label"].to_numpy()

    pipe_val_metrics = eval_multiclass(y_pipe_val, pred_pipe_val)
    pipe_test_metrics = eval_multiclass(y_pipe_test, pred_pipe_test)
    pipe_ext_metrics = eval_multiclass(y_pipe_ext, pred_pipe_ext)

    report_lines.append("End-to-end 3-class pipeline metrics")
    report_lines.append(f"Validation accuracy={pipe_val_metrics['accuracy']:.4f}, macro_f1={pipe_val_metrics['macro_f1']:.4f}")
    report_lines.append(f"Test accuracy={pipe_test_metrics['accuracy']:.4f}, macro_f1={pipe_test_metrics['macro_f1']:.4f}")
    report_lines.append(f"External accuracy={pipe_ext_metrics['accuracy']:.4f}, macro_f1={pipe_ext_metrics['macro_f1']:.4f}")
    report_lines.append("")

    # -------------------------
    # Feature importance
    # -------------------------
    report_lines.append("Top Stage 1 features:")
    for name, score in top_features(stage1_model, feature_cols):
        report_lines.append(f"  {name}: {score:.6f}")

    report_lines.append("")
    report_lines.append("Top Stage 2 features:")
    for name, score in top_features(stage2_model, feature_cols):
        report_lines.append(f"  {name}: {score:.6f}")

    # -------------------------
    # Save outputs
    # -------------------------
    exp_model_dir = MODEL_DIR / tag
    exp_model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(stage1_model, exp_model_dir / "stage1_fault_detector.joblib")
    joblib.dump(stage2_model, exp_model_dir / "stage2_fault_classifier.joblib")

    save_json(exp_model_dir / "stage1_fill_medians.json", med1)
    save_json(exp_model_dir / "stage2_fill_medians.json", med2)
    save_json(exp_model_dir / "feature_columns.json", {"feature_cols": feature_cols})

    save_json(exp_model_dir / "metrics_stage1_val.json", stage1_val_metrics)
    save_json(exp_model_dir / "metrics_stage1_test.json", stage1_test_metrics)
    save_json(exp_model_dir / "metrics_stage1_external.json", stage1_ext_metrics)

    save_json(exp_model_dir / "metrics_stage2_val.json", stage2_val_metrics)
    save_json(exp_model_dir / "metrics_stage2_test.json", stage2_test_metrics)
    save_json(exp_model_dir / "metrics_stage2_external.json", stage2_ext_metrics)

    save_json(exp_model_dir / "metrics_pipeline_val.json", pipe_val_metrics)
    save_json(exp_model_dir / "metrics_pipeline_test.json", pipe_test_metrics)
    save_json(exp_model_dir / "metrics_pipeline_external.json", pipe_ext_metrics)

    save_text(REPORT_DIR / f"{tag}_summary.txt", report_lines)

    return {
        "tag": tag,
        "stage1_val": stage1_val_metrics,
        "stage1_test": stage1_test_metrics,
        "stage1_external": stage1_ext_metrics,
        "stage2_val": stage2_val_metrics,
        "stage2_test": stage2_test_metrics,
        "stage2_external": stage2_ext_metrics,
        "pipeline_val": pipe_val_metrics,
        "pipeline_test": pipe_test_metrics,
        "pipeline_external": pipe_ext_metrics,
    }


def main():
    if not IN_WINDOWS.exists():
        raise FileNotFoundError(f"windows_table.csv not found: {IN_WINDOWS}")

    print("Reading windows table...")
    df = pd.read_csv(IN_WINDOWS, low_memory=False)

    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    print(f"Total windows loaded: {len(df)}")

    # Build target splits
    target_train, target_val, target_test, external_df, split_log = build_target_splits(df)

    print("Target split summary:")
    for line in split_log:
        print("  ", line)

    print(f"target_train={len(target_train)}, target_val={len(target_val)}, target_test={len(target_test)}, external={len(external_df)}")

    # GPVS sample for transfer training
    gpvs_sample, gpvs_log = sample_gpvs_train(df, per_class_cap=GPVS_PER_CLASS_CAP)

    print("GPVS sampling summary:")
    for line in gpvs_log:
        print("  ", line)

    # Baseline: target-only
    print("\nRunning target_only experiment...")
    result_target_only = run_experiment(
        tag="target_only",
        train_df=target_train,
        target_val=target_val,
        target_test=target_test,
        external_df=external_df,
        feature_cols=FEATURE_COLS,
        sample_weight_single=1.0,
        sample_weight_gpvs=0.3,
    )

    # Transfer: target + sampled GPVS
    transfer_train = pd.concat([target_train, gpvs_sample], ignore_index=True)

    print("\nRunning transfer_weighted experiment...")
    result_transfer = run_experiment(
        tag="transfer_weighted",
        train_df=transfer_train,
        target_val=target_val,
        target_test=target_test,
        external_df=external_df,
        feature_cols=FEATURE_COLS,
        sample_weight_single=1.0,
        sample_weight_gpvs=0.3,
    )

    # Compare
    compare_lines = []
    compare_lines.append("XGBOOST EXPERIMENT COMPARISON")
    compare_lines.append("=" * 80)
    compare_lines.append("Target split summary:")
    compare_lines.extend(split_log)
    compare_lines.append("")
    compare_lines.append("GPVS sampling summary:")
    compare_lines.extend(gpvs_log)
    compare_lines.append("")

    def add_result_block(name, res):
        compare_lines.append(f"{name}")
        compare_lines.append(f"  pipeline_val_accuracy={res['pipeline_val']['accuracy']:.4f}")
        compare_lines.append(f"  pipeline_val_macro_f1={res['pipeline_val']['macro_f1']:.4f}")
        compare_lines.append(f"  pipeline_test_accuracy={res['pipeline_test']['accuracy']:.4f}")
        compare_lines.append(f"  pipeline_test_macro_f1={res['pipeline_test']['macro_f1']:.4f}")
        compare_lines.append(f"  pipeline_external_accuracy={res['pipeline_external']['accuracy']:.4f}")
        compare_lines.append(f"  pipeline_external_macro_f1={res['pipeline_external']['macro_f1']:.4f}")
        compare_lines.append("")

    add_result_block("target_only", result_target_only)
    add_result_block("transfer_weighted", result_transfer)

    best_tag = "transfer_weighted" if result_transfer["pipeline_val"]["macro_f1"] >= result_target_only["pipeline_val"]["macro_f1"] else "target_only"
    compare_lines.append(f"Best model by validation macro F1: {best_tag}")

    save_text(REPORT_DIR / "comparison_summary.txt", compare_lines)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Reports saved to: {REPORT_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Best model by validation macro F1: {best_tag}")
    print("=" * 80)


if __name__ == "__main__":
    main()