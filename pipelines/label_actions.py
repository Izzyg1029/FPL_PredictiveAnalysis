# label_actions.py - COMPLETE VERSION with 48-hour reconfigure window
# FIXED: Consistent column naming (Device_Type), comm_age_days, and all rules

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

# Only label these device types
ACTIVE_DEVICE_TYPES = ["MM3", "ZM1"]

def _coalesce_col(df, candidates):
    """Find first existing column from list of candidates"""
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
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").replace("/", "")
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
    p.add_argument("--battery_low", type=float, default=20.0, help="Battery low threshold percentage")
    p.add_argument("--inactive_days", type=int, default=180, help="Days without communication to be considered inactive")
    p.add_argument("--zero_current_threshold", type=float, default=1.0, help="Threshold for zero current detection")
    p.add_argument("--reconf_intermittent_days", type=int, default=14, help="Days for reconfigure intermittent flag")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run pipelines/update_history.py first.")

    print("Loading history data...")
    df = pd.read_parquet(in_path)
    print(f"Total rows: {len(df)}")
    if 'Device_Type' in df.columns:
        print(f"Device_Type counts: {df['Device_Type'].value_counts().to_dict()}")
        print(f"First few Device_Type values: {df['Device_Type'].head(10).tolist()}")
        print(f"Loaded {len(df):,} rows from history")

    # --- Normalize device types (create lowercase version for display, but keep original for masks) ---
    if "Device_Type" in df.columns:
        df["device_type_display"] = df["Device_Type"].map(normalize_device_type_value)
    else:
        df["device_type_display"] = np.nan

    # ===== DEBUG: Check device types =====
    print("\nDevice types in history data:")
    print(df["device_type_display"].value_counts(dropna=False))
    print(f"Total rows: {len(df):,}")
    # ====================================

    # --- Detect columns based on actual data ---
    battery_col = _coalesce_col(df, ['BatteryLevel', 'Battery_Level', 'Battery', 'battery_level'])
    status_col = _coalesce_col(df, ['Status', 'status', 'DeviceStatus', 'Profile_Status'])
    ts_col = _coalesce_col(df, ['BatteryLatestReport', 'Last_Heard', 'LastHeard', 'last_heard', 'Date', 'date'])
    lc_col = _coalesce_col(df, ['LineCurrent', 'Line_Current', 'line_current', 'Current'])
    temp_col = _coalesce_col(df, ['LineTemperatrue', 'LineTemperature', 'Line_Temperature', 'Temperature'])
    lat_col = _coalesce_col(df, ['Latitude', 'lat', 'LATITUDE'])
    lon_col = _coalesce_col(df, ['Longitude', 'lon', 'LONGITUDE'])
    temp_col = _coalesce_col(df, ['LineTemperatrue', 'LineTemperature', 'Line_Temperature', 'Temperature'])

    # Filter to active device types
    before = len(df)
    df = df[df["device_type_display"].isin(ACTIVE_DEVICE_TYPES)].copy()
    print(f"Filtered to {ACTIVE_DEVICE_TYPES}: {len(df):,} rows (dropped {before-len(df):,})")

    print(f"\nColumn mapping:")
    print(f"battery_col: {battery_col}")
    print(f"status_col: {status_col}")
    print(f"timestamp_col: {ts_col}")
    print(f"current_col: {lc_col}")
    print(f"temp_col: {temp_col}")
    print(f"lat_col: {lat_col}")
    print(f"lon_col: {lon_col}")

    print(f"\nDEBUG - Checking temperature column:")
    if temp_col is not None:
        print(f"temp_col exists: {temp_col}")
        print(f"Sample values: {df[temp_col].head(10).tolist()}")
        print(f"Data type: {df[temp_col].dtype}")
    else:
        print(f"temp_col is None - no temperature data!")

    if ts_col is None:
        raise KeyError("Missing timestamp column!")

    # timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    print(f"\n Missing timestamp before handling: {df[ts_col].isna().sum()} rows")

    # Get max timestamp for filling missing values
    max_timestamp = df[ts_col].max()
    df[ts_col].fillna(max_timestamp, inplace=True)

    # Mark devices that had missing timestamps
    df['timestamp_was_missing'] = df[ts_col].isna()  # This will be False after fillna, so we need a different approach

    # Better: Mark before filling
    df['timestamp_missing'] = df[ts_col].isna()
    df[ts_col].fillna(max_timestamp, inplace=True)

    print(f"After handling: {df[ts_col].isna().sum()} rows still missing")

    if temp_col is not None:
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df["overheat_flag"] = (df[temp_col] > 85).astype(int)
        print(f"overheat_flag: {df['overheat_flag'].sum()} devices (temp > 85C)")

    print("\n Missing timestamp by device type:")
    print(df.groupby('Device_Type')[ts_col].apply(lambda x: x.isna().sum()))
    #df = df[~df[ts_col].isna()].copy()
    df["BatteryLatestReport"] = df[ts_col]

    print(f"After timestamp filter: {len(df)} rows, ZM1: {(df['Device_Type']=='ZM1').sum()}, MM3: {(df['Device_Type']=='MM3').sum()}")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    print(f"After timestamp conversion: {len(df)} rows, ZM1: {(df['Device_Type']=='ZM1').sum()}, MM3: {(df['Device_Type']=='MM3').sum()}")

    # After temperature flag creation
    if temp_col is not None:
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df["overheat_flag"] = (df[temp_col] > 85).astype(int)
        print(f"After temp flags: {len(df)} rows, ZM1: {(df['Device_Type']=='ZM1').sum()}, MM3: {(df['Device_Type']=='MM3').sum()}")

    # After timestamp filter - THIS IS THE KEY ONE
    #df = df[~df[ts_col].isna()].copy()
    df["BatteryLatestReport"] = df[ts_col]
    print(f"After timestamp filter: {len(df)} rows, ZM1: {(df['Device_Type']=='ZM1').sum()}, MM3: {(df['Device_Type']=='MM3').sum()}")
    # battery low flag
    if battery_col is not None:
        df[battery_col] = pd.to_numeric(df[battery_col], errors="coerce")
        df["battery_low_flag"] = (df[battery_col] < args.battery_low).astype(int)
        print(f"battery_low_flag: {df['battery_low_flag'].sum()} devices flagged")
    else:
        df["battery_low_flag"] = 0
        print("No battery column - battery_low_flag = 0 for all")

    # status flags
    if status_col is not None:
        s = df[status_col].astype(str).str.upper().str.strip()
        
        df["offline_flag"] = (s == "OFFLINE").astype(int)
        df["online_flag"] = (s == "ONLINE").astype(int)
        df["intermittent_flag"] = (s == "INTERMITTENT").astype(int)
        df["standby_flag"] = (s == "STANDBY").astype(int)
        
        print(f"offline_flag: {df['offline_flag'].sum()} devices")
        print(f"online_flag: {df['online_flag'].sum()} devices")
        print(f"intermittent_flag: {df['intermittent_flag'].sum()} devices")
        print(f"standby_flag: {df['standby_flag'].sum()} devices")
    else:
        df["offline_flag"] = 0
        df["online_flag"] = 0
        df["intermittent_flag"] = 0
        df["standby_flag"] = 0
        print("No status column - all status flags = 0")

    # zero current flag
    if lc_col is not None:
        df[lc_col] = pd.to_numeric(df[lc_col], errors="coerce")
        df["zero_current_flag"] = ((df[lc_col].fillna(0).abs()) <= args.zero_current_threshold).astype(int)
        print(f"zero_current_flag: {df['zero_current_flag'].sum()} devices")
    else:
        df["zero_current_flag"] = 0
        print("No current column - zero_current_flag = 0")

    # coordinate missing flag + location change (relocation indicator)
    if lat_col is not None and lon_col is not None:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        missing_lat = df[lat_col].isna() | (df[lat_col] == 0)
        missing_lon = df[lon_col].isna() | (df[lon_col] == 0)
        df["coord_missing_flag"] = (missing_lat | missing_lon).astype(int)
        print(f"coord_missing_flag: {df['coord_missing_flag'].sum()} devices")

        # Detect latitude/longitude drift vs previous reading (relocate trigger)
        # Sort by Serial + timestamp to compute per-device previous coordinates
        if ts_col is not None:
            df_sorted = df.sort_values([ts_col])
            df["prev_lat"] = df_sorted.groupby("Serial")[lat_col].shift(1)
            df["prev_lon"] = df_sorted.groupby("Serial")[lon_col].shift(1)
        else:
            df["prev_lat"] = np.nan
            df["prev_lon"] = np.nan

        lat_changed = (~df["prev_lat"].isna()) & (~df[lat_col].isna()) & (
            (df[lat_col] - df["prev_lat"]).abs() > 0.001  # ~100m threshold
        )
        lon_changed = (~df["prev_lon"].isna()) & (~df[lon_col].isna()) & (
            (df[lon_col] - df["prev_lon"]).abs() > 0.001
        )
        df["coord_changed_flag"] = (lat_changed | lon_changed).astype(int)
        print(f"coord_changed_flag: {df['coord_changed_flag'].sum()} devices with location change")
    else:
        df["coord_missing_flag"] = 0
        df["coord_changed_flag"] = 0
        df["prev_lat"] = np.nan
        df["prev_lon"] = np.nan
        print("No coordinate columns - coord_missing_flag/coord_changed_flag = 0")
    
    # Ensure all flag columns exist that might be used in rules
    required_flags = ['gps_jump_flag', 'coord_missing_flag', 'coord_changed_flag', 'zero_current_flag', 
                    'overheat_flag', 'online_flag', 'intermittent_flag',
                    'critical_current_flag', 'high_current_flag', 'low_current_flag']

    for flag in required_flags:
        if flag not in df.columns:
            print(f"{flag} not found - creating with default 0")
            df[flag] = 0
    
    # --- ACTION RULES ---
    df["action_name"] = "NO_ACTION"
    df["action_label"] = 0

    # Create masks for different device types - using original Device_Type column
    print("\n Creating device type masks...")
    print(f"Device_Type column exists: {'Device_Type' in df.columns}")
    print(f"Sample Device_Type values: {df['Device_Type'].head(5).tolist()}")

    # Create masks for different device types - ensure string comparison
    is_zm1 = df["Device_Type"].astype(str).str.strip() == "ZM1"
    is_mm3 = df["Device_Type"].astype(str).str.strip() == "MM3"
    is_um3 = df["Device_Type"].astype(str).str.strip() == "UM3+"
    

    print(f"ZM1 devices: {is_zm1.sum()}")
    print(f"MM3 devices: {is_mm3.sum()}")
    print(f"UM3+ devices: {is_um3.sum()}")

    print(f"\n Total rows in dataframe: {len(df)}")
    print(f"Device_Type value counts at this point:")
    print(df['Device_Type'].value_counts())

    # ===== TRACKING VARIABLES for 48-hour rule =====
    if 'reconfigure_count' not in df.columns:
        df['reconfigure_count'] = 0
    if 'last_reconfigure_time' not in df.columns:
        df['last_reconfigure_time'] = pd.NaT
    if 'hours_since_reconfigure' not in df.columns:
        df['hours_since_reconfigure'] = 999999  # Large number if never reconfigured
    if 'reconfigure_attempted' not in df.columns:
        df['reconfigure_attempted'] = False
    if 'battery_level' not in df.columns:
        df['battery_level'] = np.nan

    # ===== ZM1 RULES with 48-hour reconfigure window =====

    # FIRST TIME ISSUES - try RECONFIGURE
    zm1_first_time = is_zm1 & (df["reconfigure_count"] == 0) & (
        (df["coord_missing_flag"] == 1) |
        (df["gps_jump_flag"] == 1) |
        (pd.isna(df.get("BatteryLevel", pd.Series(index=df.index)))) |
        ((df["comm_age_days"] > 14) & (df["comm_age_days"] <= 90))
    )
    df.loc[zm1_first_time, "action_name"] = "RECONFIGURE"
    df.loc[zm1_first_time, "action_label"] = 1
    print(f"ZM1 first time RECONFIGURE: {zm1_first_time.sum()}")

    # RETRY RECONFIGURE - if last attempt was >48h ago and issue persists
    zm1_retry = is_zm1 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 48) & (
        (df["coord_missing_flag"] == 1) |
        (df["gps_jump_flag"] == 1) |
        (pd.isna(df.get("BatteryLevel", pd.Series(index=df.index)))) |
        ((df["comm_age_days"] > 30))
    )
    df.loc[zm1_retry, "action_name"] = "RECONFIGURE"
    df.loc[zm1_retry, "action_label"] = 1
    print(f"ZM1 retry RECONFIGURE: {zm1_retry.sum()}")

    # AFTER RECONFIGURE FAILED (48+ hours later) - now consider other actions
    zm1_after_reconfigure = is_zm1 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 48)

    # Still having coordinate issues after reconfigure -> RELOCATE
    zm1_relocate = zm1_after_reconfigure & (df["coord_missing_flag"] == 1)
    df.loc[zm1_relocate, "action_name"] = "RELOCATE"
    df.loc[zm1_relocate, "action_label"] = 2
    print(f"ZM1 RELOCATE: {zm1_relocate.sum()}")

    # Battery still low after reconfigure -> REPLACE
    zm1_replace = zm1_after_reconfigure & (df["battery_low_flag"] == 1)
    df.loc[zm1_replace, "action_name"] = "REPLACE"
    df.loc[zm1_replace, "action_label"] = 3
    print(f"ZM1 REPLACE (battery): {zm1_replace.sum()}")

    # Still no communication after reconfigure -> REPLACE
    zm1_replace_comms = zm1_after_reconfigure & (df["comm_age_days"] > 90)
    df.loc[zm1_replace_comms, "action_name"] = "REPLACE"
    df.loc[zm1_replace_comms, "action_label"] = 3
    print(f"ZM1 REPLACE (comms): {zm1_replace_comms.sum()}")

    # URGENT REPLACE - battery critically low (<10%) and recent comms (skip reconfigure)
    if 'battery_level' in df.columns:
        zm1_replace_urgent = is_zm1 & (
            (df["battery_level"] < 10) & 
            (df["comm_age_days"] < 7)
        )
        df.loc[zm1_replace_urgent, "action_name"] = "REPLACE"
        df.loc[zm1_replace_urgent, "action_label"] = 3
        print(f"ZM1 URGENT REPLACE: {zm1_replace_urgent.sum()}")

    # ===== MM3 RULES with 48-hour reconfigure window =====
    
    print("\n MM3 RULE COUNTS:")
    
    # RECONFIGURE for MM3 - FIRST TIME for ANY issue
    mm3_first_time = is_mm3 & (df["reconfigure_count"] == 0) & (
        (df["critical_current_flag"] == 1) |
        (df["overheat_flag"] == 1) |
        (df["high_current_flag"] == 1) |
        (df["zero_current_flag"] == 1) |
        (df["intermittent_flag"] == 1) |
        ((df["comm_age_days"] > 7) & (df["comm_age_days"] <= 90)) |
        (df["coord_missing_flag"] == 1)
    )
    df.loc[mm3_first_time, "action_name"] = "RECONFIGURE"
    df.loc[mm3_first_time, "action_label"] = 1
    print(f"MM3 first time RECONFIGURE: {mm3_first_time.sum()}")

    # RETRY RECONFIGURE for MM3 - if last attempt >48h ago and issue persists
    mm3_retry = is_mm3 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 48) & (
        (df["critical_current_flag"] == 1) |
        (df["overheat_flag"] == 1) |
        (df["high_current_flag"] == 1) |
        (df["zero_current_flag"] == 1) |
        ((df["comm_age_days"] > 30))
    )
    df.loc[mm3_retry, "action_name"] = "RECONFIGURE"
    df.loc[mm3_retry, "action_label"] = 1
    print(f"MM3 retry RECONFIGURE: {mm3_retry.sum()}")

    # REPLACE for MM3 - issue persists 48+ hours after reconfigure
    mm3_replace = is_mm3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 48) & (
        (df["critical_current_flag"] == 1) |
        (df["overheat_flag"] == 1) |
        (df["zero_current_flag"] == 1) |
        ((df["comm_age_days"] > 90))
    )
    df.loc[mm3_replace, "action_name"] = "REPLACE"
    df.loc[mm3_replace, "action_label"] = 3
    print(f"MM3 REPLACE: {mm3_replace.sum()}")

    # RELOCATE for MM3 - location issues:
    #   (a) coord missing 48h after reconfigure, OR
    #   (b) latitude/longitude has changed (device moved), OR
    #   (c) coord missing with no prior reconfigure attempt (device never had valid coords)
    mm3_relocate_after_reconf = is_mm3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 48) & (
        (df["coord_missing_flag"] == 1)
    )
    mm3_relocate_coord_change = is_mm3 & (df["coord_changed_flag"] == 1)
    mm3_relocate_no_coords = is_mm3 & (df["coord_missing_flag"] == 1) & (df["reconfigure_count"] == 0) & (
        (df["comm_age_days"] > 14)
    )
    mm3_relocate = mm3_relocate_after_reconf | mm3_relocate_coord_change | mm3_relocate_no_coords
    df.loc[mm3_relocate, "action_name"] = "RELOCATE"
    df.loc[mm3_relocate, "action_label"] = 2
    print(f"MM3 RELOCATE (after reconf): {mm3_relocate_after_reconf.sum()}")
    print(f"MM3 RELOCATE (coord change): {mm3_relocate_coord_change.sum()}")
    print(f"MM3 RELOCATE (no coords yet): {mm3_relocate_no_coords.sum()}")
    print(f"MM3 RELOCATE total: {mm3_relocate.sum()}")

    # ===== UM3+ RULES with 48-hour reconfigure window =====
    
    print("\n UM3+ RULE COUNTS:")

    # RECONFIGURE for UM3+ - FIRST TIME for ANY issue
    um3_first_time = is_um3 & (df["reconfigure_count"] == 0) & (
        ((df["comm_age_days"] > 7) & (df["comm_age_days"] <= 90)) |
        (df["intermittent_flag"] == 1) |
        (df["coord_missing_flag"] == 1) |
        (df["offline_flag"] == 1)
    )
    df.loc[um3_first_time, "action_name"] = "RECONFIGURE"
    df.loc[um3_first_time, "action_label"] = 1
    print(f"UM3+ first time RECONFIGURE: {um3_first_time.sum()}")

    # RETRY RECONFIGURE for UM3+ - if last attempt >48h ago
    um3_retry = is_um3 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 48) & (
        ((df["comm_age_days"] > 30)) |
        (df["intermittent_flag"] == 1)
    )
    df.loc[um3_retry, "action_name"] = "RECONFIGURE"
    df.loc[um3_retry, "action_label"] = 1
    print(f"UM3+ retry RECONFIGURE: {um3_retry.sum()}")

    # REPLACE for UM3+ - issue persists 48+ hours after reconfigure
    um3_replace = is_um3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 48) & (
        ((df["comm_age_days"] > 90)) |
        (df["offline_flag"] == 1)
    )
    df.loc[um3_replace, "action_name"] = "REPLACE"
    df.loc[um3_replace, "action_label"] = 3
    print(f"UM3+ REPLACE: {um3_replace.sum()}")

    # RELOCATE for UM3+ - location issues only
    um3_relocate = is_um3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 48) & (
        (df["coord_missing_flag"] == 1)
    )
    df.loc[um3_relocate, "action_name"] = "RELOCATE"
    df.loc[um3_relocate, "action_label"] = 2
    print(f"UM3+ RELOCATE: {um3_relocate.sum()}")

    # ===== FINAL SUMMARY =====
    print("\n" + "="*60)
    print("FINAL ACTION COUNTS BY DEVICE TYPE:")
    print("="*60)
    
    # Use the display column for final summary
    df['device_type_for_display'] = df['device_type_display']
    summary = df.groupby('device_type_for_display')['action_name'].value_counts()
    print(summary)
    
    # Also show raw counts
    print("\n RAW ACTION COUNTS:")
    print(df["action_name"].value_counts())

    # Save labeled
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop temporary columns before saving
    cols_to_drop = ['device_type_display', 'device_type_for_display']
    df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df_to_save.to_parquet(out_path, index=False)

    print(f"\n Labeled actions saved -> {out_path}")
    print(f"Total rows: {len(df_to_save):,}")

if __name__ == "__main__":
    main()
