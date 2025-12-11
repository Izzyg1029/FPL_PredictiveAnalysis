import pandas as pd
import numpy as np

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def clamp(x, low=0, high=100):
    """
    Restrict a number to stay between 'low' and 'high'.
    Example: clamp(120) -> 100
    """
    return max(low, min(high, x))


def normalize(series):
    """
    Convert values to a 0–1 range.
    Example: min becomes 0, max becomes 1, middle values scale accordingly.
    Protects against division by zero if all numbers are the same.
    """
    s = series.astype(float)

    min_val = s.min()
    max_val = s.max()

    # If data is constant or invalid, return zeros
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=s.index)

    return (s - min_val) / (max_val - min_val)


# ----------------------------------------------------
# Communication Age Feature (Now in DAYS)
# ----------------------------------------------------

def add_common_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 'comm_age_days' = time since last communication (in days),
    for every device.
    """
    df = df.copy()

    df["Last_Heard_dt"] = pd.to_datetime(df["Last_Heard"])
    snapshot_time = df["Last_Heard_dt"].max()

    time_diff_seconds = (snapshot_time - df["Last_Heard_dt"]).dt.total_seconds()

    df["comm_age_days"] = time_diff_seconds / (3600.0 * 24)

    return df


# ----------------------------------------------------
# Install Date / Aging Features
# ----------------------------------------------------

def add_install_age_features(df: pd.DataFrame, install_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aging features using FPL install-date sheet.
    """
    df = df.copy()
    install_df = install_df.copy()

    install_df["InstallDate_dt"] = pd.to_datetime(install_df["InstallDate"], errors="coerce")

    df = df.merge(
        install_df[["Serial", "InstallDate_dt"]],
        on="Serial",
        how="left"
    )

    snapshot = df["Last_Heard_dt"].max()
    df["device_age_days"] = (
        (snapshot - df["InstallDate_dt"]).dt.total_seconds() / (3600 * 24)
    )

    df["expected_lifetime_days"] = np.where(
        df["Device_Type"].str.contains("ZM1", case=False, na=False), 5 * 365,
        np.where(
            df["Device_Type"].str.contains("UM3", case=False, na=False), 10 * 365,
            8 * 365
        )
    )

    df["pct_life_used"] = (df["device_age_days"] / df["expected_lifetime_days"]).clip(lower=0)

    return df


# ----------------------------------------------------
# GPS Drift
# ----------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def add_gps_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df = df.sort_values(["Serial", "Last_Heard_dt"])

    df["gps_prev_lat"] = df.groupby("Serial")["Latitude"].shift(1)
    df["gps_prev_lon"] = df.groupby("Serial")["Longitude"].shift(1)

    df["distance_drift_m"] = df.apply(
        lambda row: haversine_distance(
            row["gps_prev_lat"], row["gps_prev_lon"],
            row["Latitude"], row["Longitude"]
        ) if pd.notna(row["gps_prev_lat"]) else 0,
        axis=1
    )

    df["gps_jump_flag"] = (df["distance_drift_m"] > 30).astype(int)
    return df


# ----------------------------------------------------
# Placeholder Variance & Frequency Features
# ----------------------------------------------------

def add_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["current_variance"] = 0.0
    df["temp_variance"] = 0.0
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zero_current_frequency"] = 0.0
    df["overheat_frequency"] = 0.0
    df["comm_fail_frequency"] = 0.0
    df["gps_jump_frequency"] = 0.0
    return df


# ----------------------------------------------------
# Device-Specific Risk Scoring
# ----------------------------------------------------

def compute_zm1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["battery_level"] = pd.to_numeric(df["BatteryLevel"], errors="coerce")
    df["battery_low_flag"] = (df["battery_level"] < 20).astype(int)

    df["BatteryLatestReport_dt"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")
    snapshot_time = df["Last_Heard_dt"].max()
    df["battery_report_age_days"] = (
        (snapshot_time - df["BatteryLatestReport_dt"]).dt.total_seconds() / (3600 * 24)
    )

    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)
    df["overheat_flag"] = (df["LineTemperatrue_val"] > 45).astype(int)

    risk = (
        0.4 * normalize(df["comm_age_days"]) +
        0.3 * normalize(df["battery_report_age_days"].fillna(0)) +
        0.2 * normalize(df["LineTemperatrue_val"].fillna(0)) +
        0.1 * normalize(df["zero_current_flag"])
    ) * 100

    risk += 20 * df["battery_low_flag"]
    risk += 10 * df["overheat_flag"]

    df["risk_score_zm1"] = risk.apply(clamp)
    return df


def compute_um3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)
    df["high_temp_flag"] = (df["LineTemperatrue_val"] > 40).astype(int)

    risk = (
        0.5 * normalize(df["comm_age_days"]) +
        0.3 * normalize(df["LineTemperatrue_val"].fillna(0)) +
        0.2 * normalize(df["zero_current_flag"])
    ) * 100

    risk += 10 * df["high_temp_flag"]

    df["risk_score_um3"] = risk.apply(clamp)
    return df


def compute_mm3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)
    df["overheat_flag"] = (df["LineTemperatrue_val"] > 50).astype(int)

    risk = (
        0.6 * normalize(df["comm_age_days"]) +
        0.2 * normalize(df["LineCurrent_val"].fillna(0)) +
        0.2 * normalize(df["zero_current_flag"])
    ) * 100

    risk += 10 * df["overheat_flag"]

    df["risk_score_mm3"] = risk.apply(clamp)
    return df


# ----------------------------------------------------
# Main Router
# ----------------------------------------------------

def build_health_features(df_devices: pd.DataFrame, install_df=None) -> pd.DataFrame:
    df = df_devices.copy()

    df = add_common_time_features(df)

    if install_df is not None:
        df = add_install_age_features(df, install_df)

    df = add_gps_drift_features(df)
    df = add_variance_features(df)
    df = add_frequency_features(df)

    mask_zm1 = df["Device_Type"].str.contains("ZM1", case=False, na=False)
    mask_um3 = df["Device_Type"].str.contains("UM3", case=False, na=False)
    mask_mm3 = df["Device_Type"].str.contains("MM3", case=False, na=False)

    if mask_zm1.any():
        df.loc[mask_zm1] = compute_zm1_features(df.loc[mask_zm1])

    if mask_um3.any():
        df.loc[mask_um3] = compute_um3_features(df.loc[mask_um3])

    if mask_mm3.any():
        df.loc[mask_mm3] = compute_mm3_features(df.loc[mask_mm3])

    return df
