import pandas as pd
import numpy as np
from typing import Optional, List, Dict

# ====================================================
# Configuration (Thresholds, Lifetimes, etc.)
# ====================================================

DEVICE_LIFETIMES_DAYS: Dict[str, int] = {
    "ZM1": 5 * 365,   # battery-powered overhead
    "UM3": 10 * 365,  # underground
    "MM3": 8 * 365,   # line-powered overhead
}

TEMP_LIMITS_C: Dict[str, float] = {
    "ZM1": 45.0,   # ZM1 overheats above 45°C
    "UM3": 40.0,   # UM3 overheats above 40°C
    "MM3": 50.0,   # MM3 overheats above 50°C
}

GPS_JUMP_THRESHOLD_M: float = 30.0  # meters


# ====================================================
# Helper Functions
# ====================================================

def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    """Bound a number between low and high."""
    return max(low, min(high, x))


def normalize(series: pd.Series) -> pd.Series:
    """
    Scale values to 0–1 safely.
    If all values are equal or invalid, return zeros.
    """
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=s.index)

    return (s - min_val) / (max_val - min_val)


def _expected_lifetime_for_type(device_type: str) -> int:
    """Return expected life in days based on device type."""
    if not isinstance(device_type, str):
        return DEVICE_LIFETIMES_DAYS["MM3"]
    upper = device_type.upper()
    if "ZM1" in upper:
        return DEVICE_LIFETIMES_DAYS["ZM1"]
    if "UM3" in upper:
        return DEVICE_LIFETIMES_DAYS["UM3"]
    return DEVICE_LIFETIMES_DAYS["MM3"]


# ====================================================
# Core Feature Builders
# ====================================================

def add_common_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - Last_Heard_dt (datetime)
      - comm_age_days = days since last communication
    """
    df = df.copy()
    df["Last_Heard_dt"] = pd.to_datetime(df["Last_Heard"], errors="coerce")
    snapshot = df["Last_Heard_dt"].max()

    if pd.isna(snapshot):
        df["comm_age_days"] = np.nan
        return df

    df["comm_age_days"] = (
        (snapshot - df["Last_Heard_dt"]).dt.total_seconds() / (3600.0 * 24.0)
    )
    return df


def add_install_age_features(df: pd.DataFrame, install_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - InstallDate_dt
      - device_age_days
      - expected_lifetime_days
      - pct_life_used
    Using install_df with columns: Serial, InstallDate.
    """
    df = df.copy()
    install_df = install_df.copy()

    install_df["InstallDate_dt"] = pd.to_datetime(
        install_df["InstallDate"], errors="coerce"
    )

    df = df.merge(
        install_df[["Serial", "InstallDate_dt"]],
        on="Serial",
        how="left",
    )

    snapshot = df["Last_Heard_dt"].max()
    df["expected_lifetime_days"] = df["Device_Type"].apply(_expected_lifetime_for_type)

    if pd.isna(snapshot):
        df["device_age_days"] = np.nan
        df["pct_life_used"] = np.nan
        return df

    df["device_age_days"] = (
        (snapshot - df["InstallDate_dt"]).dt.total_seconds() / (3600.0 * 24.0)
    )
    df["pct_life_used"] = (
        df["device_age_days"] / df["expected_lifetime_days"]
    ).clip(lower=0.0)

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute Haversine distance (meters) between two lat/lon pairs."""
    R = 6371000.0  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(a))


def add_gps_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GPS drift features:
      - distance_drift_m
      - gps_jump_flag (1 if jump > GPS_JUMP_THRESHOLD_M)
    """
    df = df.copy()

    df["Latitude"] = pd.to_numeric(df.get("Latitude", 0.0), errors="coerce")
    df["Longitude"] = pd.to_numeric(df.get("Longitude", 0.0), errors="coerce")

    df = df.sort_values(["Serial", "Last_Heard_dt"])

    df["prev_lat"] = df.groupby("Serial")["Latitude"].shift(1)
    df["prev_lon"] = df.groupby("Serial")["Longitude"].shift(1)

    def _calc(row):
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_lon"]):
            return 0.0
        return haversine_distance(
            row["prev_lat"],
            row["prev_lon"],
            row["Latitude"],
            row["Longitude"],
        )

    df["distance_drift_m"] = df.apply(_calc, axis=1)
    df["gps_jump_flag"] = (df["distance_drift_m"] > GPS_JUMP_THRESHOLD_M).astype(int)

    return df


def add_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for future variance features.
    For now:
      - current_variance
      - temp_variance
    are set to 0.0
    """
    df = df.copy()
    df["current_variance"] = 0.0
    df["temp_variance"] = 0.0
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for future frequency-of-failure features.
    For now all are 0.0:
      - zero_current_frequency
      - overheat_frequency
      - comm_fail_frequency
      - gps_jump_frequency
    """
    df = df.copy()
    df["zero_current_frequency"] = 0.0
    df["overheat_frequency"] = 0.0
    df["comm_fail_frequency"] = 0.0
    df["gps_jump_frequency"] = 0.0
    return df


# ====================================================
# Device-Specific Risk Scoring
# ====================================================

def compute_zm1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features & risk score for ZM1 devices (battery-powered).
    Returns a DataFrame with (at least):
      - battery_level
      - battery_low_flag
      - battery_report_age_days
      - LineCurrent_val
      - LineTemperature_val
      - zero_current_flag
      - overheat_flag
      - risk_score_zm1
    """
    df = df.copy()

    # Battery
    df["battery_level"] = pd.to_numeric(df.get("BatteryLevel", 100.0), errors="coerce")
    df["battery_low_flag"] = (df["battery_level"] < 20.0).astype(int)

    # Battery report recency
    df["BatteryLatestReport_dt"] = pd.to_datetime(
        df.get("BatteryLatestReport"), errors="coerce"
    )
    snapshot = df["Last_Heard_dt"].max()

    if pd.isna(snapshot):
        df["battery_report_age_days"] = 0.0
    else:
        df["battery_report_age_days"] = (
            (snapshot - df["BatteryLatestReport_dt"]).dt.total_seconds()
            / (3600.0 * 24.0)
        ).fillna(0.0)

    # Line current & temperature
    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce")
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    )

    # Flags
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["overheat_flag"] = (
        df["LineTemperature_val"] > TEMP_LIMITS_C["ZM1"]
    ).astype(int)

    # Risk formula (ZM1)
    risk = (
        0.4 * normalize(df["comm_age_days"]) +
        0.3 * normalize(df["battery_report_age_days"]) +
        0.2 * normalize(df["LineTemperature_val"]) +
        0.1 * normalize(df["zero_current_flag"])
    ) * 100.0

    # Penalties
    risk += 20.0 * df["battery_low_flag"]
    risk += 10.0 * df["overheat_flag"]

    df["risk_score_zm1"] = risk.apply(clamp)
    return df


def compute_um3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features & risk score for UM3 devices (underground).
    Returns a DataFrame with:
      - LineCurrent_val
      - LineTemperature_val
      - zero_current_flag
      - overheat_flag
      - risk_score_um3
    """
    df = df.copy()

    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce")
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    )

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["overheat_flag"] = (
        df["LineTemperature_val"] > TEMP_LIMITS_C["UM3"]
    ).astype(int)

    risk = (
        0.5 * normalize(df["comm_age_days"]) +
        0.3 * normalize(df["LineTemperature_val"]) +
        0.2 * normalize(df["zero_current_flag"])
    ) * 100.0

    risk += 10.0 * df["overheat_flag"]

    df["risk_score_um3"] = risk.apply(clamp)
    return df


def compute_mm3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features & risk score for MM3 devices (line-powered).
    Returns a DataFrame with:
      - LineCurrent_val
      - LineTemperature_val
      - zero_current_flag
      - overheat_flag
      - risk_score_mm3
    """
    df = df.copy()

    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce")
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    )

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["overheat_flag"] = (
        df["LineTemperature_val"] > TEMP_LIMITS_C["MM3"]
    ).astype(int)

    risk = (
        0.6 * normalize(df["comm_age_days"]) +
        0.2 * normalize(df["LineCurrent_val"]) +
        0.2 * normalize(df["LineTemperature_val"])
    ) * 100.0

    # Extra penalty for severe overheat
    risk += 15.0 * df["overheat_flag"]

    df["risk_score_mm3"] = risk.apply(clamp)
    return df

def explain_risk(row):
    reasons = []

    # --- Communication gap ---
    if pd.notna(row.get("comm_age_days", None)):
        if row["comm_age_days"] > 7:
            reasons.append("Long communication gap")
        elif row["comm_age_days"] > 3:
            reasons.append("Moderate communication delay")

    # --- Zero current (global) ---
    if row.get("zero_current_flag", 0) == 1:
        reasons.append("Zero current detected")

    # --- Overheat (global) ---
    if row.get("overheat_flag", 0) == 1:
        reasons.append("Over temperature condition")

    # --- ZM1-specific ---
    if "ZM1" in str(row.get("Device_Type", "")):
        if row.get("battery_low_flag", 0) == 1:
            reasons.append("Battery critically low")
        if row.get("battery_report_age_days", 0) > 180:
            reasons.append("Battery report outdated")

    # --- UM3-specific ---
    if "UM3" in str(row.get("Device_Type", "")):
        if row.get("overheat_flag", 0) == 1:
            reasons.append("High underground temperature")

    # --- MM3-specific (cleaned logic) ---
    if "MM3" in str(row.get("Device_Type", "")):

        if row.get("zero_current_flag", 0) == 1:
            reasons.append("Zero current (possible feeder fault)")

        if row.get("overheat_flag", 0) == 1:
            reasons.append("Line temperature exceeds MM3 threshold")

    # No issues
    if not reasons:
        return "Normal operating conditions"

    return ", ".join(reasons)


# ====================================================
# Main Router
# ====================================================

def build_health_features(
    df_devices: pd.DataFrame,
    install_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Main entry point for the Feature Health Tree.

    Steps:
      1. Normalize Device_Type & temperature column names
      2. Add comm_age_days
      3. Add install-age features (optional)
      4. Add GPS drift, variance, frequency placeholders
      5. Compute device-type-specific risk scores
      6. Combine them into a single 'risk_score'
    """
    df = df_devices.copy()

    # --- Normalize Device_Type ---
    if "Device_Type" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Device_Type' column.")
    df["Device_Type"] = df["Device_Type"].astype(str).str.strip().str.upper()

    # --- Fix common temperature typos -> LineTemperature ---
    rename_map = {
        "LineTemperatrue": "LineTemperature",
        "LineTemperture": "LineTemperature",
        "LineTemp": "LineTemperature",
        "Line_Temperature": "LineTemperature",
    }
    df = df.rename(columns=rename_map)

    # --- Ensure GPS columns exist ---
    if "Latitude" not in df.columns:
        df["Latitude"] = 0.0
    if "Longitude" not in df.columns:
        df["Longitude"] = 0.0

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # --- Base time feature ---
    df = add_common_time_features(df)

    # --- Optional install features ---
    if install_df is not None:
        df = add_install_age_features(df, install_df)

    # --- GPS / variance / frequency ---
    df = add_gps_drift_features(df)
    df = add_variance_features(df)
    df = add_frequency_features(df)

    # --- Initialize shared flags & risk columns with defaults ---
    for col in [
        "battery_level",
        "battery_low_flag",
        "battery_report_age_days",
        "LineCurrent_val",
        "LineTemperature_val",
        "zero_current_flag",
        "overheat_flag",
        "risk_score_zm1",
        "risk_score_um3",
        "risk_score_mm3",
    ]:
        if col not in df.columns:
            # Start with neutral defaults
            df[col] = 0.0 if "flag" not in col else 0

    # --- Device-Type Masks ---
    mask_zm1 = df["Device_Type"] == "ZM1"
    mask_um3 = df["Device_Type"] == "UM3"
    mask_mm3 = df["Device_Type"] == "MM3"

    # --- ZM1 ---
    if mask_zm1.any():
        sub = compute_zm1_features(df.loc[mask_zm1].copy())
        for col in [
            "battery_level",
            "battery_low_flag",
            "battery_report_age_days",
            "LineCurrent_val",
            "LineTemperature_val",
            "zero_current_flag",
            "overheat_flag",
            "risk_score_zm1",
        ]:
            if col in sub.columns:
                df.loc[mask_zm1, col] = sub[col].values

    # --- UM3 ---
    if mask_um3.any():
        sub = compute_um3_features(df.loc[mask_um3].copy())
        for col in [
            "LineCurrent_val",
            "LineTemperature_val",
            "zero_current_flag",
            "overheat_flag",
            "risk_score_um3",
        ]:
            if col in sub.columns:
                df.loc[mask_um3, col] = sub[col].values

    # --- MM3 ---
    if mask_mm3.any():
        sub = compute_mm3_features(df.loc[mask_mm3].copy())
        for col in [
            "LineCurrent_val",
            "LineTemperature_val",
            "zero_current_flag",
            "overheat_flag",
            "risk_score_mm3",
        ]:
            if col in sub.columns:
                df.loc[mask_mm3, col] = sub[col].values

    # --- Final combined risk score ---
    df["risk_score"] = df[["risk_score_zm1", "risk_score_um3", "risk_score_mm3"]].max(axis=1)
    df["risk_score"] = df["risk_score"].round(2)
    df["comm_age_days"] = df["comm_age_days"].round(1)

    df["risk_reason"] = df.apply(explain_risk, axis=1)

    return df


# ====================================================
# Utility
# ====================================================

def get_top_risk_devices(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Return the top N devices with risk_score > 0, sorted high → low.
    """
    if "risk_score" not in df.columns:
        raise ValueError("risk_score not found. Did you call build_health_features()?")

    df_positive = df[df["risk_score"] > 0.0]
    return df_positive.sort_values("risk_score", ascending=False).head(n)


# ====================================================
# Suggested Output Schema for Other Trees
# ====================================================

FEATURE_HEALTH_OUTPUT_COLUMNS: List[str] = [
    "Serial",
    "Device_Type",
    "Last_Heard",
    "Last_Heard_dt",
    "comm_age_days",
    "LineCurrent",
    "LineTemperature",
    "LineCurrent_val",
    "LineTemperature_val",
    "battery_level",
    "battery_low_flag",
    "battery_report_age_days",
    "device_age_days",
    "expected_lifetime_days",
    "pct_life_used",
    "Latitude",
    "Longitude",
    "distance_drift_m",
    "gps_jump_flag",
    "current_variance",
    "temp_variance",
    "zero_current_flag",
    "overheat_flag",
    "zero_current_frequency",
    "overheat_frequency",
    "comm_fail_frequency",
    "gps_jump_frequency",
    "risk_score_zm1",
    "risk_score_um3",
    "risk_score_mm3",
    "risk_score",
]

