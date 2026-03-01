from pathlib import Path
import argparse
import pandas as pd
import numpy as np

HISTORY_PATH = Path("data/processed/fci_history.parquet")
OUT_PATH = Path("data/processed/fci_labeled.parquet")

LABEL_TO_NAME = {
    0: "NO_ACTION",
    1: "RECONFIGURE",
    2: "RELOCATE",
    3: "REPLACE",
}


def _coalesce_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_device_type_value(x):
    """Normalize messy device types to canonical: ZM1, MM3, UM3+."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ("", "NONE", "NAN", "NULL"):
        return np.nan
    if s == "M3":
        return "MM3"
    if s == "UM3":
        return "UM3+"

    s2 = (
        s.replace(" ", "")
         .replace("-", "")
         .replace("_", "")
         .replace("/", "")
    )
    if "ZM1" in s2:
        return "ZM1"
    if "MM3" in s2 or s2 == "M3":
        return "MM3"
    if "UM3" in s2:
        return "UM3+"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(HISTORY_PATH))
    p.add_argument("--output", default=str(OUT_PATH))

    # thresholds
    p.add_argument("--battery_low", type=float, default=20.0)
    p.add_argument("--inactive_days", type=int, default=180)
    p.add_argument("--zero_current_threshold", type=float, default=1.0)

    # strictness knobs
    p.add_argument("--reconf_intermittent_days", type=int, default=14)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run pipelines/update_history.py first.")

    df = pd.read_parquet(in_path)

    if "Serial" not in df.columns:
        raise KeyError("Missing required column: Serial")

    # --- Canonical device_type and DROP None ---
    if "Device_Type" in df.columns:
        dt = df["Device_Type"]
        if "device_type" in df.columns:
            dt = dt.where(~dt.isna(), df["device_type"])
    elif "device_type" in df.columns:
        dt = df["device_type"]
    else:
        dt = pd.Series([np.nan] * len(df))

    df["device_type"] = dt.map(normalize_device_type_value)

    # ✅ drop rows with missing/unknown device_type
    before = len(df)
    df = df[df["device_type"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"🧹 Dropped {dropped:,} rows with missing device_type")

    # --- detect columns ---
    battery_col = _coalesce_col(df, ["BatteryLevel", "Battery_Level", "Battery", "battery_level"])
    status_col = _coalesce_col(df, ["Status", "status"])
    ts_col = _coalesce_col(df, ["BatteryLatestReport", "Last_Heard", "LastHeard", "last_heard"])
    lc_col = _coalesce_col(df, ["LineCurrent", "Line_Current", "line_current"])
    lat_col = _coalesce_col(df, ["Latitude", "lat", "LATITUDE"])
    lon_col = _coalesce_col(df, ["Longitude", "lon", "LONGITUDE"])

    if ts_col is None:
        raise KeyError("Missing timestamp column (expected BatteryLatestReport or Last_Heard).")

    # timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[~df[ts_col].isna()].copy()

    if ts_col != "BatteryLatestReport":
        df["BatteryLatestReport"] = df[ts_col]
    else:
        df["BatteryLatestReport"] = df["BatteryLatestReport"]

    # days since last report (relative to latest in dataset)
    latest_ts = df["BatteryLatestReport"].max()
    df["days_since_last_report"] = (latest_ts - df["BatteryLatestReport"]).dt.total_seconds() / 86400.0
    df["days_since_last_report"] = df["days_since_last_report"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # battery low flag
    if battery_col is not None:
        df[battery_col] = pd.to_numeric(df[battery_col], errors="coerce")
        df["battery_low_flag"] = (df[battery_col] < args.battery_low).astype(int)
    else:
        df["battery_low_flag"] = 0

    # status flags (strict parsing)
    if status_col is not None:
        s = df[status_col].astype(str).str.upper().str.strip()
        df["offline_flag"] = s.str.contains(r"\bOFF\b|\bOFFLINE\b", regex=True).astype(int)
        df["online_flag"] = s.str.contains(r"\bON\b|\bONLINE\b", regex=True).astype(int)
        df["intermittent_flag"] = s.str.contains(r"\bINTER\b|\bINTERMITTENT\b", regex=True).astype(int)
        df["standby_flag"] = s.str.contains(r"\bSTANDBY\b", regex=True).astype(int)
    else:
        df["offline_flag"] = 0
        df["online_flag"] = 0
        df["intermittent_flag"] = 0
        df["standby_flag"] = 0

    # zero current flag
    if lc_col is not None:
        df[lc_col] = pd.to_numeric(df[lc_col], errors="coerce")
        df["zero_current_flag"] = ((df[lc_col].fillna(0).abs()) <= args.zero_current_threshold).astype(int)
    else:
        df["zero_current_flag"] = 0

    # coordinate missing flag
    if lat_col is not None and lon_col is not None:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df["coord_missing_flag"] = (df[lat_col].isna() | df[lon_col].isna()).astype(int)
    else:
        df["coord_missing_flag"] = 0

    # default labels
    df["action_name"] = "NO_ACTION"
    df["action_label"] = 0

    # ============================================================
    # ACTION RULES (priority order)
    # 1) REPLACE
    # 2) RELOCATE
    # 3) RECONFIGURE (strict)
    # ============================================================

    # 1) REPLACE
    replace_mask = (df["days_since_last_report"] >= args.inactive_days)
    if battery_col is not None:
        replace_mask = replace_mask | (
            (df["battery_low_flag"] == 1) & ((df["offline_flag"] == 1) | (df["intermittent_flag"] == 1))
        )

    df.loc[replace_mask, "action_name"] = "REPLACE"
    df.loc[replace_mask, "action_label"] = 3

    # 2) RELOCATE (only if not replace)
    not_replace = df["action_label"] != 3

    # online + zero current
    relocate_a = not_replace & (df["online_flag"] == 1) & (df["zero_current_flag"] == 1)

    # coords missing but alive
    relocate_b = not_replace & (df["coord_missing_flag"] == 1) & (
        (df["online_flag"] == 1) | (df["intermittent_flag"] == 1) | (df["standby_flag"] == 1)
    )

    relocate_mask = relocate_a | relocate_b
    df.loc[relocate_mask, "action_name"] = "RELOCATE"
    df.loc[relocate_mask, "action_label"] = 2

    # 3) RECONFIGURE (strict intermittent only, recent, not replace/relocate)
    not_replace_or_relocate = ~df["action_name"].isin(["REPLACE", "RELOCATE"])
    reconf_mask = (
        not_replace_or_relocate
        & (df["intermittent_flag"] == 1)
        & (df["days_since_last_report"] <= args.reconf_intermittent_days)
        & (df["battery_low_flag"] == 0)
        & (df["zero_current_flag"] == 0)
    )
    df.loc[reconf_mask, "action_name"] = "RECONFIGURE"
    df.loc[reconf_mask, "action_label"] = 1

    # save labeled
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"✅ Labeled actions saved -> {out_path}")
    print("\ndevice_type counts:")
    print(df["device_type"].value_counts(dropna=False))
    print("\naction_name counts:")
    print(df["action_name"].value_counts(dropna=False))


if __name__ == "__main__":
    main()