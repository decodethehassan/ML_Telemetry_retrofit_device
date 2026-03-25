from pathlib import Path
import pandas as pd
import numpy as np
import re


# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_SINGLE = PROJECT_ROOT / "data" / "raw" / "single_phase"
RAW_GPVS = PROJECT_ROOT / "data" / "raw" / "gpvs"

OUT_MASTER = PROJECT_ROOT / "data" / "processed" / "master_table.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "sanity_report.txt"


# =========================
# Helpers
# =========================
def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read file: {path}\nLast error: {last_error}")


def to_float_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-+eE]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def pick_col(df: pd.DataFrame, exact=None, contains=None):
    exact = exact or []
    contains = contains or []

    norm_map = {normalize_name(c): c for c in df.columns}

    for e in exact:
        e_norm = normalize_name(e)
        if e_norm in norm_map:
            return norm_map[e_norm]

    for c in df.columns:
        c_norm = normalize_name(c)
        for token in contains:
            if normalize_name(token) in c_norm:
                return c

    return None


def choose_single_phase_label(file_stem: str) -> str:
    name = file_stem.lower()
    if "opencircuit" in name:
        return "open_circuit"
    if "shading" in name:
        return "shading"
    return "normal"


def choose_gpvs_label(file_stem: str):
    """
    Returns:
        scenario_num, mode, class_label
    """
    m = re.match(r"F(\d)([LM])$", file_stem, re.IGNORECASE)
    if not m:
        return None, None, None

    scenario = int(m.group(1))
    mode = m.group(2).upper()

    if scenario == 0:
        return scenario, mode, "normal"
    elif scenario == 5:
        return scenario, mode, "shading"
    elif scenario == 7:
        return scenario, mode, "open_circuit"
    else:
        return scenario, mode, None


def median_abs_error(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if x.empty:
        return np.nan
    return float(np.median(np.abs(x.iloc[:, 0] - x.iloc[:, 1])))


def ensure_output_dirs():
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Single-phase processing
# =========================
def process_single_phase_file(path: Path, report_lines: list) -> pd.DataFrame:
    df = read_csv_flexible(path)
    original_columns = list(df.columns)

    timestamp_col = pick_col(df, exact=["Timestamp", "Time", "DateTime"], contains=["timestamp", "time"])
    voltage_col = pick_col(df, exact=["Voltage (V)", "Voltage"], contains=["voltage", "volt"])
    current_col = pick_col(df, exact=["Current (mA)", "Current (A)", "Current"], contains=["current"])
    power_col = pick_col(df, exact=["Power (W)", "Power (mW)", "Power"], contains=["power"])
    temp_col = pick_col(df, exact=["Temperature (C)", "Temperature"], contains=["temperature", "temp"])

    if voltage_col is None or current_col is None:
        raise ValueError(
            f"[SINGLE_PHASE] Missing required voltage/current column in {path.name}. "
            f"Found columns: {original_columns}"
        )

    voltage_V = to_float_series(df[voltage_col])
    current_raw = to_float_series(df[current_col])

    # Detect current unit
    current_header = normalize_name(current_col)
    current_median = np.nanmedian(current_raw)

    if "ma" in current_header or current_median > 20:
        current_A = current_raw / 1000.0
        current_unit_note = "Current treated as mA and converted to A"
    else:
        current_A = current_raw.copy()
        current_unit_note = "Current treated as already in A"

    predicted_power_W = voltage_V * current_A

    power_note = ""
    if power_col is not None:
        power_raw = to_float_series(df[power_col])

        err_if_W = median_abs_error(power_raw, predicted_power_W)
        err_if_mW = median_abs_error(power_raw / 1000.0, predicted_power_W)

        if np.isnan(err_if_W) and np.isnan(err_if_mW):
            power_W = predicted_power_W
            power_note = "Power missing/invalid -> computed as V*I"
        elif err_if_W <= err_if_mW:
            power_W = power_raw
            power_note = f"Power kept as W (median |P_reported - V*I| = {err_if_W:.6f})"
        else:
            power_W = power_raw / 1000.0
            power_note = f"Power treated as mW and converted to W (median |P_reported/1000 - V*I| = {err_if_mW:.6f})"
    else:
        power_W = predicted_power_W
        power_note = "No power column -> computed as V*I"

    if temp_col is not None:
        temperature_C = to_float_series(df[temp_col])
    else:
        temperature_C = pd.Series(np.nan, index=df.index, dtype="float64")

    if timestamp_col is not None:
        timestamp = df[timestamp_col].astype(str)
    else:
        timestamp = pd.Series(np.arange(len(df)).astype(str), index=df.index)

    class_label = choose_single_phase_label(path.stem)

    out = pd.DataFrame({
        "timestamp": timestamp,
        "sample_index": np.arange(len(df)),
        "voltage_V": voltage_V,
        "current_A": current_A,
        "power_W": power_W,
        "temperature_C": temperature_C,
        "class_label": class_label,
        "source_domain": "single_phase",
        "source_file": path.stem,
        "mode": "",
        "subsystem": ""
    })

    before = len(out)
    out = out.dropna(subset=["voltage_V", "current_A", "power_W"]).copy()
    after = len(out)

    report_lines.append(f"\n[SINGLE_PHASE] {path.name}")
    report_lines.append(f"Original columns: {original_columns}")
    report_lines.append(f"Chosen columns -> timestamp={timestamp_col}, voltage={voltage_col}, current={current_col}, power={power_col}, temp={temp_col}")
    report_lines.append(current_unit_note)
    report_lines.append(power_note)
    report_lines.append(f"Rows before cleaning: {before}")
    report_lines.append(f"Rows after dropping NaNs in V/I/P: {after}")
    report_lines.append(f"Class label: {class_label}")
    report_lines.append(
        f"Ranges -> V:[{out['voltage_V'].min():.4f}, {out['voltage_V'].max():.4f}], "
        f"I:[{out['current_A'].min():.6f}, {out['current_A'].max():.6f}], "
        f"P:[{out['power_W'].min():.6f}, {out['power_W'].max():.6f}]"
    )

    return out


# =========================
# GPVS processing
# =========================
def pick_gpvs_pairs(df: pd.DataFrame):
    """
    Try to detect subsystem column pairs.

    S1 candidates are preferred for open-circuit.
    S2 candidates are preferred for shading.
    """

    # Time
    time_col = pick_col(df, exact=["Time", "Timestamp"], contains=["time"])

    # Subsystem 1 candidates
    s1_current = pick_col(
        df,
        exact=["Ipv", "Ipv0", "IpvS1", "Ipv_0"],
        contains=["ipv"]
    )
    s1_voltage = pick_col(
        df,
        exact=["Vpv", "UpvIst", "Upv", "Vpv0", "Upv_Ist"],
        contains=["vpv", "upv"]
    )

    # Subsystem 2 candidates
    s2_current = pick_col(
        df,
        exact=["Ipv1", "Ipv_1", "IpvS2", "Ipv2", "IpvIst1"],
        contains=["ipv1", "ipv2"]
    )
    s2_voltage = pick_col(
        df,
        exact=["Vpv1", "Vpv_1", "UpvIst1", "Upv1", "Upv_Ist1", "Vpv2"],
        contains=["vpv1", "upv1", "vpv2", "upv2"]
    )

    pairs = []

    if s1_current is not None and s1_voltage is not None:
        pairs.append(("S1", s1_voltage, s1_current))

    if s2_current is not None and s2_voltage is not None:
        if (s2_voltage != s1_voltage) or (s2_current != s1_current):
            pairs.append(("S2", s2_voltage, s2_current))

    # Fallback if only one generic DC pair exists
    if not pairs:
        generic_current = pick_col(df, exact=["Ipv"], contains=["ipv", "current"])
        generic_voltage = pick_col(df, exact=["Vpv", "Upv", "UpvIst"], contains=["vpv", "upv", "voltage"])
        if generic_current is not None and generic_voltage is not None:
            pairs.append(("GENERIC", generic_voltage, generic_current))

    return time_col, pairs


def build_gpvs_block(df: pd.DataFrame, time_col: str, voltage_col: str, current_col: str,
                     class_label: str, source_file: str, mode: str, subsystem: str,
                     fault_keep_last_45: bool, report_lines: list) -> pd.DataFrame:
    voltage_V = to_float_series(df[voltage_col])
    current_A = to_float_series(df[current_col])

    timestamp = df[time_col].astype(str) if time_col is not None else pd.Series(np.arange(len(df)).astype(str), index=df.index)
    sample_index = np.arange(len(df))

    out = pd.DataFrame({
        "timestamp": timestamp,
        "sample_index": sample_index,
        "voltage_V": voltage_V,
        "current_A": current_A,
        "power_W": voltage_V * current_A,
        "temperature_C": np.nan,
        "class_label": class_label,
        "source_domain": "gpvs",
        "source_file": source_file,
        "mode": mode,
        "subsystem": subsystem
    })

    before_segment = len(out)

    # For GPVS fault files, faults were introduced halfway.
    # We keep only the last 45% and drop the transition around the middle.
    if fault_keep_last_45:
        start_fault = int(len(out) * 0.55)
        out = out.iloc[start_fault:].copy()
        report_lines.append(
            f"GPVS fault segmentation for {source_file}/{subsystem}: kept rows from index {start_fault} to end "
            f"(last 45%), dropped first 55%"
        )

    before_clean = len(out)
    out = out.dropna(subset=["voltage_V", "current_A", "power_W"]).copy()
    after_clean = len(out)

    report_lines.append(
        f"GPVS block -> file={source_file}, mode={mode}, subsystem={subsystem}, "
        f"V_col={voltage_col}, I_col={current_col}, rows_before_segment={before_segment}, "
        f"rows_before_clean={before_clean}, rows_after_clean={after_clean}"
    )

    return out


def process_gpvs_file(path: Path, report_lines: list) -> list:
    df = read_csv_flexible(path)
    original_columns = list(df.columns)

    scenario, mode, class_label = choose_gpvs_label(path.stem)
    if class_label is None:
        # Skip non-common faults
        report_lines.append(f"\n[GPVS] {path.name} -> skipped (not in common fault set)")
        return []

    time_col, pairs = pick_gpvs_pairs(df)

    report_lines.append(f"\n[GPVS] {path.name}")
    report_lines.append(f"Original columns: {original_columns}")
    report_lines.append(f"Scenario={scenario}, mode={mode}, class={class_label}")
    report_lines.append(f"Detected time column: {time_col}")
    report_lines.append(f"Detected DC pairs: {pairs}")

    if not pairs:
        raise ValueError(
            f"[GPVS] Could not detect usable DC voltage/current columns in {path.name}. "
            f"Found columns: {original_columns}"
        )

    blocks = []

    if scenario == 0:
        # Healthy file -> if both S1 and S2 exist, keep both
        for subsystem, v_col, i_col in pairs:
            blocks.append(
                build_gpvs_block(
                    df=df,
                    time_col=time_col,
                    voltage_col=v_col,
                    current_col=i_col,
                    class_label="normal",
                    source_file=path.stem,
                    mode=mode,
                    subsystem=subsystem,
                    fault_keep_last_45=False,
                    report_lines=report_lines
                )
            )

    elif scenario == 5:
        # Shading -> prefer S2, else fallback to first available pair
        chosen = None
        for p in pairs:
            if p[0] == "S2":
                chosen = p
                break
        if chosen is None:
            chosen = pairs[0]

        subsystem, v_col, i_col = chosen
        blocks.append(
            build_gpvs_block(
                df=df,
                time_col=time_col,
                voltage_col=v_col,
                current_col=i_col,
                class_label="shading",
                source_file=path.stem,
                mode=mode,
                subsystem=subsystem,
                fault_keep_last_45=True,
                report_lines=report_lines
            )
        )

    elif scenario == 7:
        # Open-circuit -> prefer S1, else fallback to first available pair
        chosen = None
        for p in pairs:
            if p[0] == "S1":
                chosen = p
                break
        if chosen is None:
            chosen = pairs[0]

        subsystem, v_col, i_col = chosen
        blocks.append(
            build_gpvs_block(
                df=df,
                time_col=time_col,
                voltage_col=v_col,
                current_col=i_col,
                class_label="open_circuit",
                source_file=path.stem,
                mode=mode,
                subsystem=subsystem,
                fault_keep_last_45=True,
                report_lines=report_lines
            )
        )

    return blocks


# =========================
# Main
# =========================
def main():
    ensure_output_dirs()

    report_lines = []
    all_frames = []

    report_lines.append("UNIFIED DATASET BUILD REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Project root: {PROJECT_ROOT}")
    report_lines.append(f"Single-phase folder: {RAW_SINGLE}")
    report_lines.append(f"GPVS folder: {RAW_GPVS}")
    report_lines.append("")

    # Process single-phase files
    single_files = sorted(RAW_SINGLE.glob("*.csv"))
    if not single_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_SINGLE}")

    report_lines.append("PROCESSING SINGLE-PHASE FILES")
    report_lines.append("-" * 80)

    for path in single_files:
        out = process_single_phase_file(path, report_lines)
        all_frames.append(out)

    # Process GPVS files
    gpvs_files = sorted(RAW_GPVS.glob("*.csv"))
    if not gpvs_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_GPVS}")

    report_lines.append("\nPROCESSING GPVS FILES")
    report_lines.append("-" * 80)

    for path in gpvs_files:
        blocks = process_gpvs_file(path, report_lines)
        all_frames.extend(blocks)

    if not all_frames:
        raise RuntimeError("No data frames were created. Check file structure and column names.")

    master = pd.concat(all_frames, ignore_index=True)

    # Final cleaning
    master = master.dropna(subset=["voltage_V", "current_A", "power_W"]).copy()
    master = master.reset_index(drop=True)

    # Standard column order
    master = master[
        [
            "timestamp",
            "sample_index",
            "voltage_V",
            "current_A",
            "power_W",
            "temperature_C",
            "class_label",
            "source_domain",
            "source_file",
            "mode",
            "subsystem",
        ]
    ]

    # Save master table
    master.to_csv(OUT_MASTER, index=False)

    # Global report
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("FINAL SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Master table saved to: {OUT_MASTER}")
    report_lines.append(f"Total rows: {len(master)}")
    report_lines.append("")

    report_lines.append("Class counts:")
    class_counts = master["class_label"].value_counts(dropna=False)
    for k, v in class_counts.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.append("")
    report_lines.append("Counts by source_domain and class_label:")
    combo = master.groupby(["source_domain", "class_label"]).size()
    for idx, val in combo.items():
        report_lines.append(f"  {idx}: {val}")

    report_lines.append("")
    report_lines.append("Missing values:")
    missing = master.isna().sum()
    for col, cnt in missing.items():
        report_lines.append(f"  {col}: {cnt}")

    report_lines.append("")
    report_lines.append("Numeric ranges:")
    for col in ["voltage_V", "current_A", "power_W", "temperature_C"]:
        report_lines.append(
            f"  {col}: min={master[col].min(skipna=True)}, max={master[col].max(skipna=True)}"
        )

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=" * 80)
    print("DONE")
    print(f"Master table saved to: {OUT_MASTER}")
    print(f"Sanity report saved to: {OUT_REPORT}")
    print("Class counts:")
    print(master["class_label"].value_counts())
    print("=" * 80)


if __name__ == "__main__":
    main()