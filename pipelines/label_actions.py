# label_actions.py - UPDATED VERSION (90-day rule, RECONFIGURE focus)

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

HISTORY_PATH = Path("data/processed/fci_history.parquet")
OUT_PATH = Path("data/processed/fci_labeled.parquet")

LABEL_TO_NAME = {
    0: "NO_ACTION",
    1: "RECONFIGURE",
    2: "REPLACE",
}

# Only label these device types
ACTIVE_DEVICE_TYPES = ["MM3", "ZM1", "UM3+"]

def _coalesce_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_device_type_value(x):
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
    p.add_argument("--battery_low", type=float, default=20.0)
    p.add_argument("--inactive_days", type=int, default=180)
    p.add_argument("--zero_current_threshold", type=float, default=1.0)
    p.add_argument("--reconf_intermittent_days", type=int, default=14)
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

    if "Device_Type" in df.columns:
        df["device_type_display"] = df["Device_Type"].map(normalize_device_type_value)
    else:
        df["device_type_display"] = np.nan

    print("\nDevice types in history data:")
    print(df["device_type_display"].value_counts(dropna=False))
    print(f"Total rows: {len(df):,}")

    battery_col = _coalesce_col(df, ['BatteryLevel', 'Battery_Level', 'Battery', 'battery_level'])
    status_col = _coalesce_col(df, ['Status', 'status', 'DeviceStatus', 'Profile_Status'])
    ts_col = _coalesce_col(df, ['BatteryLatestReport', 'Last_Heard', 'LastHeard', 'last_heard', 'Date', 'date'])
    lc_col = _coalesce_col(df, ['LineCurrent', 'Line_Current', 'line_current', 'Current'])
    temp_col = _coalesce_col(df, ['LineTemperatrue', 'LineTemperature', 'Line_Temperature', 'Temperature'])
    lat_col = _coalesce_col(df, ['Latitude', 'lat', 'LATITUDE'])
    lon_col = _coalesce_col(df, ['Longitude', 'lon', 'LONGITUDE'])

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

    if ts_col is None:
        raise KeyError("Missing timestamp column!")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    print(f"\n Missing timestamp before handling: {df[ts_col].isna().sum()} rows")

    max_timestamp = df[ts_col].max()
    df['timestamp_missing'] = df[ts_col].isna()
    df[ts_col] = df[ts_col].fillna(max_timestamp)
    print(f"After handling: {df[ts_col].isna().sum()} rows still missing")

    if temp_col is not None:
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df["overheat_flag"] = (df[temp_col] > 85).astype(int)
        print(f"overheat_flag: {df['overheat_flag'].sum()} devices (temp > 85C)")

    df["BatteryLatestReport"] = df[ts_col]

    if battery_col is not None:
        df[battery_col] = pd.to_numeric(df[battery_col], errors="coerce")
        df["battery_low_flag"] = (df[battery_col] < args.battery_low).astype(int)
        print(f"battery_low_flag: {df['battery_low_flag'].sum()} devices flagged")
    else:
        df["battery_low_flag"] = 0

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

    if lc_col is not None:
        df[lc_col] = pd.to_numeric(df[lc_col], errors="coerce")
        df["zero_current_flag"] = ((df[lc_col].fillna(0).abs()) <= args.zero_current_threshold).astype(int)
        print(f"zero_current_flag: {df['zero_current_flag'].sum()} devices")
    else:
        df["zero_current_flag"] = 0

    if lat_col is not None and lon_col is not None:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        missing_lat = df[lat_col].isna() | (df[lat_col] == 0)
        missing_lon = df[lon_col].isna() | (df[lon_col] == 0)
        df["coord_missing_flag"] = (missing_lat | missing_lon).astype(int)
        print(f"coord_missing_flag: {df['coord_missing_flag'].sum()} devices")
    else:
        df["coord_missing_flag"] = 0

    required_flags = ['gps_jump_flag', 'coord_missing_flag', 'zero_current_flag',
                      'overheat_flag', 'online_flag', 'intermittent_flag',
                      'critical_current_flag', 'high_current_flag', 'low_current_flag']
    for flag in required_flags:
        if flag not in df.columns:
            df[flag] = 0

    df["action_name"] = "NO_ACTION"
    df["action_label"] = 0

    print("\n Creating device type masks...")
    is_zm1 = df["device_type_display"] == "ZM1"
    is_mm3 = df["device_type_display"] == "MM3"
    is_um3 = df["device_type_display"] == "UM3+"

    print(f"ZM1 devices: {is_zm1.sum()}")
    print(f"MM3 devices: {is_mm3.sum()}")
    print(f"UM3+ devices: {is_um3.sum()}")

    if 'reconfigure_count' not in df.columns:
        df['reconfigure_count'] = 0
    if 'last_reconfigure_time' not in df.columns:
        df['last_reconfigure_time'] = pd.NaT
    if 'hours_since_reconfigure' not in df.columns:
        df['hours_since_reconfigure'] = 999999
    if 'reconfigure_attempted' not in df.columns:
        df['reconfigure_attempted'] = False
    if 'battery_level' not in df.columns:
        df['battery_level'] = np.nan

    # ===== ZM1 RULES =====
    zm1_first_time = is_zm1 & (df["reconfigure_count"] == 0) & (
        (df["coord_missing_flag"] == 1) |
        (df["gps_jump_flag"] == 1) |
        (pd.isna(df.get("BatteryLevel", pd.Series(index=df.index)))) |
        ((df["comm_age_days"] > 14) & (df["comm_age_days"] <= 90))
    )
    df.loc[zm1_first_time, "action_name"] = "RECONFIGURE"
    df.loc[zm1_first_time, "action_label"] = 1
    print(f"ZM1 first time RECONFIGURE: {zm1_first_time.sum()}")

    zm1_retry = is_zm1 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 2160) & (
        (df["coord_missing_flag"] == 1) |
        (df["gps_jump_flag"] == 1) |
        (pd.isna(df.get("BatteryLevel", pd.Series(index=df.index)))) |
        ((df["comm_age_days"] > 90))
    )
    df.loc[zm1_retry, "action_name"] = "RECONFIGURE"
    df.loc[zm1_retry, "action_label"] = 1
    print(f"ZM1 retry RECONFIGURE: {zm1_retry.sum()}")

    zm1_after_reconfigure = is_zm1 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 2160)

    zm1_replace = zm1_after_reconfigure & (df["battery_low_flag"] == 1)
    df.loc[zm1_replace, "action_name"] = "REPLACE"
    df.loc[zm1_replace, "action_label"] = 2
    print(f"ZM1 REPLACE (battery): {zm1_replace.sum()}")

    zm1_replace_comms = zm1_after_reconfigure & (df["comm_age_days"] > 90)
    df.loc[zm1_replace_comms, "action_name"] = "REPLACE"
    df.loc[zm1_replace_comms, "action_label"] = 2
    print(f"ZM1 REPLACE (comms): {zm1_replace_comms.sum()}")

    if 'battery_level' in df.columns:
        zm1_replace_urgent = is_zm1 & (
            (df["battery_level"] < 10) &
            (df["comm_age_days"] < 7)
        )
        df.loc[zm1_replace_urgent, "action_name"] = "REPLACE"
        df.loc[zm1_replace_urgent, "action_label"] = 2
        print(f"ZM1 URGENT REPLACE: {zm1_replace_urgent.sum()}")

    # ===== MM3 RULES =====
    print("\n MM3 RULE COUNTS:")

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

    mm3_retry = is_mm3 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 2160) & (
        (df["critical_current_flag"] == 1) |
        (df["overheat_flag"] == 1) |
        (df["high_current_flag"] == 1) |
        (df["zero_current_flag"] == 1) |
        ((df["comm_age_days"] > 90))
    )
    df.loc[mm3_retry, "action_name"] = "RECONFIGURE"
    df.loc[mm3_retry, "action_label"] = 1
    print(f"MM3 retry RECONFIGURE: {mm3_retry.sum()}")

    mm3_replace = is_mm3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 2160) & (
        (df["critical_current_flag"] == 1) |
        (df["overheat_flag"] == 1) |
        (df["zero_current_flag"] == 1) |
        ((df["comm_age_days"] > 90))
    )
    df.loc[mm3_replace, "action_name"] = "REPLACE"
    df.loc[mm3_replace, "action_label"] = 2
    print(f"MM3 REPLACE: {mm3_replace.sum()}")

    # ===== UM3+ RULES =====
    print("\n UM3+ RULE COUNTS:")

    um3_first_time = is_um3 & (df["reconfigure_count"] == 0) & (
        ((df["comm_age_days"] > 7) & (df["comm_age_days"] <= 90)) |
        (df["intermittent_flag"] == 1) |
        (df["coord_missing_flag"] == 1) |
        (df["offline_flag"] == 1)
    )
    df.loc[um3_first_time, "action_name"] = "RECONFIGURE"
    df.loc[um3_first_time, "action_label"] = 1
    print(f"UM3+ first time RECONFIGURE: {um3_first_time.sum()}")

    um3_retry = is_um3 & (df["reconfigure_count"] == 1) & (df["hours_since_reconfigure"] > 2160) & (
        ((df["comm_age_days"] > 90)) |
        (df["intermittent_flag"] == 1)
    )
    df.loc[um3_retry, "action_name"] = "RECONFIGURE"
    df.loc[um3_retry, "action_label"] = 1
    print(f"UM3+ retry RECONFIGURE: {um3_retry.sum()}")

    um3_replace = is_um3 & (df["reconfigure_count"] >= 1) & (df["hours_since_reconfigure"] > 2160) & (
        ((df["comm_age_days"] > 90)) |
        (df["offline_flag"] == 1)
    )
    df.loc[um3_replace, "action_name"] = "REPLACE"
    df.loc[um3_replace, "action_label"] = 2
    print(f"UM3+ REPLACE: {um3_replace.sum()}")

    # ===== FINAL SUMMARY =====
    print("\n" + "="*60)
    print("FINAL ACTION COUNTS BY DEVICE TYPE:")
    print("="*60)

    df['device_type_for_display'] = df['device_type_display']
    summary = df.groupby('device_type_for_display')['action_name'].value_counts()
    print(summary)

    print("\n RAW ACTION COUNTS:")
    print(df["action_name"].value_counts())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols_to_drop = ['device_type_display', 'device_type_for_display']
    df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df_to_save.to_parquet(out_path, index=False)

    print(f"\n Labeled actions saved -> {out_path}")
    print(f"Total rows: {len(df_to_save):,}")

if __name__ == "__main__":
    main()
