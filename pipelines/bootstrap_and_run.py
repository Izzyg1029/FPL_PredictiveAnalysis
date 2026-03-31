# bootstrap_and_run.py
#
# Run this ONCE from inside the pipelines/ folder when you don't have raw
# daily CSV files.  It builds fci_history.parquet from your existing
# powerbi_exports/predictions_latest.csv, then runs the three steps that
# actually matter:
#   1. label_actions.py      - label all 4 action classes
#   2. train_action_models_rf.py - retrain MM3/ZM1 models
#   3. predict_daily_actions.py  - generate fresh predictions
#
# Usage (from inside pipelines/):
#   python bootstrap_and_run.py

import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# -- Paths (relative to pipelines/) ------------------------------------------
PREDICTIONS_CSV  = Path("../powerbi_exports/predictions_latest.csv")
HISTORY_PARQUET  = Path("data/processed/fci_history.parquet")
RECONFIGURE_CSV  = Path("../state/reconfigure_attempts.csv")
# Search for install_dates.csv dynamically
def _find_install_dates() -> Path:
    script_dir = Path(__file__).resolve().parent
    base = script_dir.parent
    candidates = [
        base / "data" / "clean" / "install_dates.csv",
        base / "data" / "data" / "clean" / "install_dates.csv",
        base / "data" / "data" / "data" / "clean" / "install_dates.csv",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Found install_dates.csv at: {p}")
            return p
    print(f"  WARNING: install_dates.csv not found. Searched: {[str(p) for p in candidates]}")
    return candidates[0]

INSTALL_DATES_CSV = _find_install_dates()  # Serial -> InstallDate
EXPECTED_LIFETIME_YEARS = 10  # default device lifetime

LABEL_TO_NAME = {0: "NO_ACTION", 1: "RECONFIGURE", 2: "RELOCATE", 3: "REPLACE"}

# Only process these device types - add "UM3+" here when ready
ACTIVE_DEVICE_TYPES = ["MM3", "ZM1"]


# -- Step 0: Build fci_history.parquet from predictions_latest.csv -----------
def bootstrap_history():
    print("=" * 65)
    print("BOOTSTRAP: Building fci_history.parquet from predictions_latest.csv")
    print("=" * 65)

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_CSV}. "
            "Make sure you are running from inside the pipelines/ folder."
        )

    df = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    print(f"Loaded {len(df):,} rows from {PREDICTIONS_CSV.name}")
    print(f"Device types: {df['Device_Type'].value_counts().to_dict()}")

    # -- Normalise device type ------------------------------------------------
    def norm(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().upper()
        if s in ("", "NONE", "NAN", "NULL"): return np.nan
        if s == "M3":  return "MM3"
        if s == "UM3": return "UM3+"
        s2 = s.replace(" ","").replace("-","").replace("_","").replace("/","")
        if "ZM1" in s2:          return "ZM1"
        if "MM3" in s2 or s2=="M3": return "MM3"
        if "UM3" in s2:          return "UM3+"
        return s

    # Use device_type (lowercase) first - it is always fully populated.
    # Device_Type (mixed-case) is often mostly null in exported files.
    if "device_type" in df.columns and df["device_type"].notna().sum() >= df.get("Device_Type", pd.Series(dtype=str)).notna().sum():
        dt_col = "device_type"
    elif "Device_Type" in df.columns:
        dt_col = "Device_Type"
    else:
        dt_col = "device_type"
    df["Device_Type"] = df[dt_col].map(norm)
    df["device_type"] = df["Device_Type"]   # keep both consistent

    # Filter to active device types only
    before = len(df)
    df = df[df["device_type"].isin(ACTIVE_DEVICE_TYPES)].copy()
    print(f"  Filtered to {ACTIVE_DEVICE_TYPES}: {len(df):,} rows (dropped {before-len(df):,} non-active)")

    # -- Timestamps ----------------------------------------------------------
    for col in ["BatteryLatestReport", "Last_Heard"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # -- Inject reconfigure history from state/reconfigure_attempts.csv ------
    df["reconfigure_count"]        = 0
    df["reconfigure_attempted"]    = False
    df["last_reconfigure_time"]    = pd.NaT

    if RECONFIGURE_CSV.exists():
        tracker = pd.read_csv(RECONFIGURE_CSV)
        tracker["attempt_date"] = pd.to_datetime(tracker["attempt_date"], errors="coerce")
        counts = tracker.groupby("Serial").size().to_dict()
        last_t = tracker.groupby("Serial")["attempt_date"].max().to_dict()
        now = pd.Timestamp.now()
        df["reconfigure_count"] = df["Serial"].map(counts).fillna(0).astype(int)
        df["last_reconfigure_time"] = df["Serial"].map(last_t)
        df["reconfigure_attempted"] = df["reconfigure_count"] > 0
        print(f"Injected reconfigure history for "
              f"{(df['reconfigure_count']>0).sum()} devices")

    today = pd.Timestamp.now().normalize()

    # -- Merge install dates and compute age/lifetime features ---------------
    if INSTALL_DATES_CSV.exists():
        inst = pd.read_csv(INSTALL_DATES_CSV)
        inst["InstallDate"] = pd.to_datetime(inst["InstallDate"], errors="coerce")
        inst = inst.dropna(subset=["InstallDate"])
        inst_map = inst.set_index("Serial")["InstallDate"].to_dict()
        df["InstallDate"] = pd.to_datetime(df["Serial"].map(inst_map), errors="coerce")
        df["device_age_days"] = (today - df["InstallDate"]).dt.days.clip(lower=0)
        df["device_age_years"] = (df["device_age_days"] / 365).round(2)
        df["expected_lifetime_days"] = EXPECTED_LIFETIME_YEARS * 365
        df["pct_life_used"] = (df["device_age_days"] / df["expected_lifetime_days"] * 100).round(1)
        matched = df["InstallDate"].notna().sum()
        over90 = (df["pct_life_used"] > 90).sum()
        print(f"  Install dates matched: {matched:,} devices")
        print(f"  Devices past 90% lifetime: {over90:,}")
    else:
        print(f"  WARNING: install_dates.csv not found at {INSTALL_DATES_CSV}")
        print(f"  Age-based replacement will not be available")
        df["device_age_days"] = np.nan
        df["device_age_years"] = np.nan
        df["expected_lifetime_days"] = EXPECTED_LIFETIME_YEARS * 365
        df["pct_life_used"] = np.nan

    # -- Ensure required flag columns exist ----------------------------------
    flag_defaults = {
        "battery_low_flag": 0,  "offline_flag": 0,     "online_flag": 0,
        "intermittent_flag": 0, "standby_flag": 0,     "zero_current_flag": 0,
        "coord_missing_flag": 0,"overheat_flag": 0,    "gps_jump_flag": 0,
        "critical_current_flag": 0, "high_current_flag": 0, "low_current_flag": 0,
        "comm_age_days": 0,
    }
    for col, default in flag_defaults.items():
        if col not in df.columns:
            df[col] = default

    # -- Recompute comm_age_days from raw timestamps --------------------------
    ts = None
    for c in ["BatteryLatestReport", "Last_Heard"]:
        if c in df.columns:
            t = pd.to_datetime(df[c], errors="coerce")
            ts = t if ts is None else ts.combine_first(t)
    if ts is not None:
        df["comm_age_days"] = (today - ts).dt.days.clip(lower=0).fillna(0)
        print(f"comm_age_days recomputed: mean={df['comm_age_days'].mean():.0f}d, "
              f">14d: {(df['comm_age_days']>14).sum()}, "
              f">90d: {(df['comm_age_days']>90).sum()}")

    # -- Save parquet ---------------------------------------------------------
    HISTORY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HISTORY_PARQUET, index=False)
    print(f"\n   Saved fci_history.parquet -- {len(df):,} rows, {len(df.columns)} columns")
    print(f"Path: {HISTORY_PARQUET.resolve()}")
    return True


# -- Helper: run a pipeline script -------------------------------------------
def run_script(script_name: str, step: int, total: int) -> bool:
    print(f"\n{'='*65}")
    print(f"STEP {step}/{total}: {script_name}")
    print(f"{'='*65}")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True, text=True, encoding="utf-8"
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {script_name}:")
        if result.stderr:
            print(result.stderr)
        return False
    print(f"Completed: {script_name}")
    return True


# -- Main ---------------------------------------------------------------------
def main():
    print("=" * 65)
    print("FPL PREDICTIVE ANALYSIS -- BOOTSTRAP + PREDICT")
    print("=" * 65)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 0 -- build the parquet
    try:
        bootstrap_history()
    except Exception as e:
        print(f"\n Bootstrap failed: {e}")
        sys.exit(1)

    # Steps 1-3 -- label -> train -> predict
    scripts = [
        ("label_actions.py",           1, 3),
        ("train_action_models_rf.py",  2, 3),
        ("predict_daily_actions.py",   3, 3),
    ]

    for script, step, total in scripts:
        if not run_script(script, step, total):
            print(f"\n Pipeline stopped at step {step} ({script})")
            print("Check the error above and re-run this script after fixing it.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("ALL STEPS COMPLETE")
    print("=" * 65)
    print("Fresh predictions saved to:")
    print("powerbi_exports/predictions_latest.csv")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
