from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_MASTER = PROJECT_ROOT / "data" / "processed" / "master_table.csv"
OUT_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows_table.csv"
OUT_HEALTHY_REFS = PROJECT_ROOT / "data" / "processed" / "healthy_references.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "window_report.txt"


# =========================
# Config
# =========================
WINDOW_SIZE = 5
STEP_SIZE = 1

RAW_COLS = ["voltage_V", "current_A", "power_W"]
GROUP_COLS = ["source_domain", "source_file", "mode", "subsystem", "class_label"]


# =========================
# Helpers
# =========================
def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def slope_feature(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)

    if len(x) < 2 or np.isnan(x).all():
        return np.nan

    idx = np.arange(len(x), dtype=float)
    mask = ~np.isnan(x)

    if mask.sum() < 2:
        return np.nan

    x_valid = x[mask]
    idx_valid = idx[mask]

    m, _ = np.polyfit(idx_valid, x_valid, 1)
    return float(m)


def rolling_features(arr: np.ndarray, prefix: str, refs: dict) -> dict:
    arr = np.asarray(arr, dtype=float)

    mean_val = float(np.nanmean(arr))
    std_val = float(np.nanstd(arr))
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    first_val = float(arr[0])
    last_val = float(arr[-1])
    slope_val = slope_feature(arr)

    ref = refs.get(prefix, np.nan)

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_first": first_val,
        f"{prefix}_last": last_val,
        f"{prefix}_slope": slope_val,
        f"{prefix}_range": max_val - min_val,
        f"{prefix}_delta": last_val - first_val,
        f"{prefix}_rel_mean": safe_div(mean_val, ref),
        f"{prefix}_rel_last": safe_div(last_val, ref),
        f"{prefix}_pct_change_from_ref": safe_div(mean_val - ref, ref),
    }


def make_reference_key(source_domain, mode, subsystem):
    mode_val = "" if pd.isna(mode) else str(mode)
    subsystem_val = "" if pd.isna(subsystem) else str(subsystem)
    return (str(source_domain), mode_val, subsystem_val)


def build_healthy_references(df: pd.DataFrame) -> pd.DataFrame:
    healthy = df[df["class_label"] == "normal"].copy()

    refs = (
        healthy
        .groupby(["source_domain", "mode", "subsystem"], dropna=False)[RAW_COLS]
        .median()
        .reset_index()
        .rename(columns={
            "voltage_V": "ref_voltage_V",
            "current_A": "ref_current_A",
            "power_W": "ref_power_W"
        })
    )

    return refs


def refs_to_dict(ref_df: pd.DataFrame) -> dict:
    out = {}
    for _, row in ref_df.iterrows():
        key = make_reference_key(row["source_domain"], row["mode"], row["subsystem"])
        out[key] = {
            "voltage_V": float(row["ref_voltage_V"]),
            "current_A": float(row["ref_current_A"]),
            "power_W": float(row["ref_power_W"]),
        }
    return out


def build_windows_for_group(group_df: pd.DataFrame, ref_lookup: dict) -> pd.DataFrame:
    group_df = group_df.sort_values("sample_index").reset_index(drop=True)

    if len(group_df) < WINDOW_SIZE:
        return pd.DataFrame()

    source_domain = group_df.loc[0, "source_domain"]
    source_file = group_df.loc[0, "source_file"]
    mode = group_df.loc[0, "mode"]
    subsystem = group_df.loc[0, "subsystem"]
    class_label = group_df.loc[0, "class_label"]

    key = make_reference_key(source_domain, mode, subsystem)
    refs = ref_lookup.get(key, {
        "voltage_V": np.nan,
        "current_A": np.nan,
        "power_W": np.nan,
    })

    rows = []

    v_all = group_df["voltage_V"].to_numpy(dtype=float)
    i_all = group_df["current_A"].to_numpy(dtype=float)
    p_all = group_df["power_W"].to_numpy(dtype=float)
    idx_all = group_df["sample_index"].to_numpy()
    ts_all = group_df["timestamp"].astype(str).to_numpy()

    n = len(group_df)

    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        v = v_all[start:end]
        i = i_all[start:end]
        p = p_all[start:end]

        row = {
            "source_domain": source_domain,
            "source_file": source_file,
            "mode": mode,
            "subsystem": subsystem,
            "class_label": class_label,
            "fault_binary": 0 if class_label == "normal" else 1,

            "window_size": WINDOW_SIZE,
            "window_start_index": int(idx_all[start]),
            "window_end_index": int(idx_all[end - 1]),
            "window_center_index": int(idx_all[start + WINDOW_SIZE // 2]),

            "timestamp_start": str(ts_all[start]),
            "timestamp_end": str(ts_all[end - 1]),

            "ref_voltage_V": refs["voltage_V"],
            "ref_current_A": refs["current_A"],
            "ref_power_W": refs["power_W"],
        }

        row.update(rolling_features(v, "voltage_V", refs))
        row.update(rolling_features(i, "current_A", refs))
        row.update(rolling_features(p, "power_W", refs))

        row["mean_power_over_voltage"] = safe_div(row["power_W_mean"], row["voltage_V_mean"])
        row["mean_power_over_current"] = safe_div(row["power_W_mean"], row["current_A_mean"])
        row["mean_current_over_voltage"] = safe_div(row["current_A_mean"], row["voltage_V_mean"])

        row["neg_voltage_count"] = int(np.sum(v < 0))
        row["neg_current_count"] = int(np.sum(i < 0))
        row["neg_power_count"] = int(np.sum(p < 0))

        if pd.notna(refs["current_A"]) and refs["current_A"] != 0:
            row["low_current_fraction"] = float(np.mean(i < 0.2 * refs["current_A"]))
        else:
            row["low_current_fraction"] = np.nan

        if pd.notna(refs["power_W"]) and refs["power_W"] != 0:
            row["low_power_fraction"] = float(np.mean(p < 0.2 * refs["power_W"]))
        else:
            row["low_power_fraction"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not IN_MASTER.exists():
        raise FileNotFoundError(f"Master table not found: {IN_MASTER}")

    print("Reading master table...")
    df = pd.read_csv(
        IN_MASTER,
        dtype={"timestamp": "string"},
        low_memory=False
    )

    required_cols = [
        "timestamp", "sample_index", "voltage_V", "current_A", "power_W",
        "temperature_C", "class_label", "source_domain", "source_file", "mode", "subsystem"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master table: {missing}")

    print(f"Master rows loaded: {len(df)}")

    # Healthy references
    print("Building healthy references...")
    ref_df = build_healthy_references(df)
    ref_df.to_csv(OUT_HEALTHY_REFS, index=False)
    ref_lookup = refs_to_dict(ref_df)

    # Remove old windows file if exists
    if OUT_WINDOWS.exists():
        OUT_WINDOWS.unlink()

    grouped = list(df.groupby(GROUP_COLS, dropna=False))
    total_groups = len(grouped)

    report_lines = []
    report_lines.append("WINDOW GENERATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Input master table: {IN_MASTER}")
    report_lines.append(f"Output windows table: {OUT_WINDOWS}")
    report_lines.append(f"Output healthy references: {OUT_HEALTHY_REFS}")
    report_lines.append(f"Window size: {WINDOW_SIZE}")
    report_lines.append(f"Step size: {STEP_SIZE}")
    report_lines.append("")

    report_lines.append("Healthy reference values:")
    for _, row in ref_df.iterrows():
        report_lines.append(
            f"source_domain={row['source_domain']}, mode={row['mode']}, subsystem={row['subsystem']} | "
            f"V_ref={row['ref_voltage_V']:.6f}, I_ref={row['ref_current_A']:.6f}, P_ref={row['ref_power_W']:.6f}"
        )

    report_lines.append("")
    report_lines.append("Group summaries:")

    total_windows = 0
    first_write = True

    print(f"Processing {total_groups} groups...")

    for idx, (group_key, g) in enumerate(grouped, start=1):
        print(f"[{idx}/{total_groups}] Processing group: {group_key} | raw_rows={len(g)}")

        wdf = build_windows_for_group(g, ref_lookup)
        window_count = len(wdf)
        total_windows += window_count

        report_lines.append(
            f"group={group_key}, raw_rows={len(g)}, windows_created={window_count}"
        )

        if not wdf.empty:
            wdf.to_csv(
                OUT_WINDOWS,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )
            first_write = False

    if not OUT_WINDOWS.exists():
        raise RuntimeError("No windows were created. windows_table.csv was not generated.")

    print("Reloading windows table for summary...")
    windows_df = pd.read_csv(OUT_WINDOWS, low_memory=False)

    report_lines.append("")
    report_lines.append("Final windows summary:")
    report_lines.append(f"Total windows: {len(windows_df)}")

    report_lines.append("Class counts:")
    for k, v in windows_df["class_label"].value_counts().items():
        report_lines.append(f"  {k}: {v}")

    report_lines.append("")
    report_lines.append("Counts by source_domain and class_label:")
    combo = windows_df.groupby(["source_domain", "class_label"]).size()
    for idx_key, val in combo.items():
        report_lines.append(f"  {idx_key}: {val}")

    report_lines.append("")
    report_lines.append("Columns:")
    for c in windows_df.columns:
        report_lines.append(f"  {c}")

    report_lines.append("")
    report_lines.append("Missing values:")
    for c, cnt in windows_df.isna().sum().items():
        report_lines.append(f"  {c}: {cnt}")

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("DONE")
    print(f"Windows table saved to: {OUT_WINDOWS}")
    print(f"Healthy references saved to: {OUT_HEALTHY_REFS}")
    print(f"Window report saved to: {OUT_REPORT}")
    print("Window class counts:")
    print(windows_df["class_label"].value_counts())
    print("=" * 80)


if __name__ == "__main__":
    main()