import pandas as pd
import numpy as np
from typing import Optional, List, Dict

# ====================================================
# Configuration (Thresholds, Lifetimes, etc.)
# ====================================================

DEVICE_LIFETIMES_DAYS: Dict[str, int] = {
    "ZM1": 10 * 365,   # battery-powered overhead
    "UM3": 10 * 365,  # # Conservative estimate (underground typically lasts long)
    "MM3": 10 * 365,   # line-powered overhead
}

TEMP_LIMITS_C: Dict[str, float] = {
    "ZM1": 70.0,   # From spec sheet: -40°C to +70°C
    "UM3": 85.0,   # From spec sheet: -40°C to +85°C
    "MM3": 85.0,   # From spec sheet: -40°C to +85°C
}

# ====================================================
# CURRENT LIMITS Configuration
# ====================================================

# Normal operating current ranges (A)
CURRENT_LIMITS_A: Dict[str, Dict[str, float]] = {
    "ZM1": {
        "min_normal": 1.0,      # Below this might be problematic
        "max_normal": 800.0,    # Max operating per spec
        "warning_threshold": 700.0,  # Warn before hitting max
        "critical_threshold": 850.0  # Above operating range
    },
    "UM3": {
        "min_normal": 1.0,
        "max_normal": 600.0,
        "warning_threshold": 550.0,
        "critical_threshold": 650.0
    },
    "MM3": {
        "min_normal": 1.0,
        "max_normal": 800.0,
        "warning_threshold": 700.0,
        "critical_threshold": 850.0,
        "off_peak_max": 12.0    # For mesh operation per spec
    }
}

GPS_JUMP_THRESHOLD_M: float = 30.0  # meters


# ====================================================
# Helper Functions
# ====================================================

def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    """Bound a number between low and high."""
    return max(low, min(high, x))


def normalize(series: pd.Series, cap: Optional[float] = None) -> pd.Series:
    """
    Scale values to 0–1 safely.
    If all values are equal or invalid, return zeros.
    """
    s = pd.to_numeric(series, errors="coerce")
    
    if cap is not None:
        s = s.clip(upper=cap)

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
    
    # Handle ZM1 variations
    if "ZM1" in upper:
        return DEVICE_LIFETIMES_DAYS["ZM1"]
    
    # Handle UM3 variations (including UM3+)
    if any(pattern in upper for pattern in ["UM3", "UM3+", "U-M3", "U M3"]):
        return DEVICE_LIFETIMES_DAYS["UM3"]
    
    # Handle MM3 variations
    if any(pattern in upper for pattern in ["MM3", "M-M3", "M M3"]):
        return DEVICE_LIFETIMES_DAYS["MM3"]
    
    # Default to MM3
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
    ).clip(lower=0.0, upper=1.0)  # Added upper=1.0

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
      - NEW: age_adjusted_battery_risk
      - NEW: old_and_hot_flag  
      - NEW: maintenance_urgency_score
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

    # Line current & temperature - CAP THESE VALUES
    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce").fillna(0.0)
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    ).fillna(0.0)
    
    # CAP VALUES
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)

    # Flags
    limits = CURRENT_LIMITS_A["ZM1"]
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["low_current_flag"] = ((df["LineCurrent_val"] > 0) & 
                              (df["LineCurrent_val"] < limits["min_normal"])).astype(int)
    df["high_current_flag"] = (df["LineCurrent_val"] > limits["warning_threshold"]).astype(int)
    df["critical_current_flag"] = (df["LineCurrent_val"] > limits["critical_threshold"]).astype(int)
    df["overheat_flag"] = (df["LineTemperature_val"] > TEMP_LIMITS_C["ZM1"]).astype(int)  # 70°C

    # 1. Calculate device age in months (for new features)
    # Note: device_age_days comes from add_install_age_features()
    if "device_age_days" in df.columns:
        df["device_age_months"] = df["device_age_days"] / 30.44  # Convert days to months
        df["device_age_months"] = df["device_age_months"].fillna(0)
    else:
        df["device_age_months"] = 0.0
    # 2. Calculate battery drain rate (simplified: % per year)
    # Assuming battery starts at 100% and declines linearly over expected life
    if "expected_lifetime_days" in df.columns:
        df["battery_drain_rate"] = (100.0 / (df["expected_lifetime_days"] / 365.0))  # % per year
    else:
        df["battery_drain_rate"] = 10.0  # Default: 10% per year for 10-year life
    
    # 3. Temperature exceedence count (placeholder - need historical data)
    # For now, use current overheat flag as proxy
    df["temperature_exceedence_count"] = df["overheat_flag"]  # Will be 0 or 1
    
    # 4. AGE ADJUSTED BATTERY RISK = battery_drain_rate × (device_age_months/24)
    df["age_adjusted_battery_risk"] = df["battery_drain_rate"] * (df["device_age_months"] / 24.0)
    df["age_adjusted_battery_risk"] = df["age_adjusted_battery_risk"].fillna(0).clip(upper=100)
    
    # 5. OLD AND HOT FLAG = 1 if device_age_months > 48 AND temperature_exceedence_count > 2
    # Since we only have current temp, using overheat_flag > 0 as proxy for "has had issues"
    df["old_and_hot_flag"] = ((df["device_age_months"] > 48) & 
                              (df["overheat_flag"] == 1)).astype(int)
    
    # 6. MAINTENANCE URGENCY SCORE = (device_age_months/60)*0.3 + battery_drain_rate*0.4 + temperature_exceedence_count*0.3
    # Normalize components first
    # 6. MAINTENANCE URGENCY SCORE = (device_age_months/60)*0.3 + battery_drain_rate*0.4 + temperature_exceedence_count*0.3
    # Normalize components
    age_component = (df["device_age_months"] / 60.0).fillna(0).clip(upper=1.0)
    battery_component = (df["battery_drain_rate"] / 20.0).clip(upper=1.0)  # Max 20%/year
    temp_component = df["temperature_exceedence_count"].fillna(0).clip(upper=1)
    
    df["maintenance_urgency_score"] = (
        age_component * 0.3 + 
        battery_component * 0.4 + 
        temp_component * 0.3
    )
    df["maintenance_urgency_score"] = df["maintenance_urgency_score"].clip(upper=1.0).round(3)
    # Risk formula (ZM1)
    risk = (
        0.20 * normalize(df["comm_age_days"], cap=365) +           # Reduced from 0.25
        0.20 * normalize(df["battery_report_age_days"], cap=180) + # Reduced from 0.25
        0.10 * normalize(df["LineTemperature_val"], cap=100) +     # Reduced from 0.15
        0.05 * normalize(df["zero_current_flag"]) +
        0.15 * normalize(df["pct_life_used"], cap=1.0) +           # Reduced from 0.20
        0.10 * normalize(df["high_current_flag"] + df["critical_current_flag"]) +
        0.10 * normalize(df["age_adjusted_battery_risk"], cap=100) +  # NEW
        0.10 * normalize(df["maintenance_urgency_score"], cap=1.0)    # NEW
    ) * 100.0

    # Enhanced penalties including new features
    risk += 20.0 * df["battery_low_flag"]
    risk += 10.0 * df["overheat_flag"]
    risk += 15.0 * df["critical_current_flag"]
    risk += 10.0 * df["high_current_flag"]
    risk += 5.0 * df["low_current_flag"]
    risk += 25.0 * df["old_and_hot_flag"]  # NEW: High penalty for old & hot devices
    risk += np.where(df["pct_life_used"] > 0.9, 15.0, 0.0)  # 90%+ life used
    
    df["risk_score_zm1"] = risk.apply(clamp)
    return df


def compute_um3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features & risk score for UM3 devices (underground).
    Returns a DataFrame with:
      - LineCurrent_val
      - LineTemperature_val
      - zero_current_flag
      - low_current_flag
      - high_current_flag
      - critical_current_flag
      - overheat_flag
      - risk_score_um3
    """
    df = df.copy()

    # Line current & temperature - CAP THESE VALUES
    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce").fillna(0.0)
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    ).fillna(0.0)
    
    # CAP VALUES
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)

    # Current risk flags for UM3
    limits = CURRENT_LIMITS_A["UM3"]
    
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["low_current_flag"] = ((df["LineCurrent_val"] > 0) & 
                              (df["LineCurrent_val"] < limits["min_normal"])).astype(int)
    df["high_current_flag"] = (df["LineCurrent_val"] > limits["warning_threshold"]).astype(int)
    df["critical_current_flag"] = (df["LineCurrent_val"] > limits["critical_threshold"]).astype(int)
    df["overheat_flag"] = (df["LineTemperature_val"] > TEMP_LIMITS_C["UM3"]).astype(int)  # 85°C

    # UM3 Risk Formula (with current magnitude)
    risk = (
        0.3 * normalize(df["comm_age_days"], cap=365) +
        0.25 * normalize(df["LineTemperature_val"], cap=100) +
        0.05 * normalize(df["zero_current_flag"]) +
        0.2 * normalize(df["pct_life_used"], cap=1.0) +
        0.2 * normalize(df["high_current_flag"] + df["critical_current_flag"])
    ) * 100.0

    # Enhanced penalties
    risk += 10.0 * df["overheat_flag"]
    risk += 15.0 * df["critical_current_flag"]
    risk += 10.0 * df["high_current_flag"]
    risk += 5.0 * df["low_current_flag"]
    risk += np.where(df["pct_life_used"] > 0.9, 15.0, 0.0)

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

    # Line current & temperature - CAP THESE VALUES
    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce").fillna(0.0)
    df["LineTemperature_val"] = pd.to_numeric(
        df.get("LineTemperature", 0.0), errors="coerce"
    ).fillna(0.0)
    
    # CAP VALUES
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)

    # Current risk flags for MM3
    limits = CURRENT_LIMITS_A["MM3"]
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["low_current_flag"] = ((df["LineCurrent_val"] > 0) & 
                              (df["LineCurrent_val"] < limits["min_normal"])).astype(int)
    df["high_current_flag"] = (df["LineCurrent_val"] > limits["warning_threshold"]).astype(int)
    df["critical_current_flag"] = (df["LineCurrent_val"] > limits["critical_threshold"]).astype(int)
    df["high_offpeak_flag"] = (df["LineCurrent_val"] > limits["off_peak_max"]).astype(int)  # MM3 specific
    df["overheat_flag"] = (df["LineTemperature_val"] > TEMP_LIMITS_C["MM3"]).astype(int)  # 85°C

    # MM3 Risk Formula
    risk = (
        0.35 * normalize(df["comm_age_days"], cap=365) +
        0.15 * normalize(df["LineCurrent_val"], cap=1000) +
        0.10 * normalize(df["LineTemperature_val"], cap=100) +
        0.20 * normalize(df["pct_life_used"], cap=1.0) +
        0.20 * normalize(df["high_current_flag"] + df["critical_current_flag"])
    ) * 100.0

    # Enhanced penalties
    risk += 15.0 * df["overheat_flag"]
    risk += 15.0 * df["critical_current_flag"]
    risk += 10.0 * df["high_current_flag"]
    risk += 5.0 * df["low_current_flag"]
    risk += 8.0 * df["high_offpeak_flag"]  # MM3 specific
    risk += np.where(df["pct_life_used"] > 0.9, 20.0, 0.0)
    
    df["risk_score_mm3"] = risk.apply(clamp)
    return df


def explain_risk(row):
    reasons = []

    if pd.isna(row.get("pct_life_used", None)):
        reasons.append("Unknown device age (install date missing)")
    
    # --- Communication gap ---
    if pd.notna(row.get("comm_age_days", None)):
        comm_age = row["comm_age_days"]
        if comm_age > 14:
            reasons.append(f"Long communication gap ({comm_age:.1f} days)")
        elif comm_age > 7:
            reasons.append(f"Moderate communication delay ({comm_age:.1f} days)")
    
    # --- Zero current ---
    if row.get("zero_current_flag", 0) == 1:
        reasons.append("Zero current detected")

    # --- Overheat ---
    if row.get("overheat_flag", 0) == 1:
        reasons.append("Over temperature condition")

    # --- Current magnitude conditions ---
    current = row.get("LineCurrent_val", 0.0)
    device_type = str(row.get("Device_Type", ""))
    
    if device_type in CURRENT_LIMITS_A:
        limits = CURRENT_LIMITS_A[device_type]
        
        if current > 0 and current < limits["min_normal"]:
            reasons.append(f"Low current ({current:.1f}A)")
        elif current > limits["warning_threshold"]:
            reasons.append(f"High current ({current:.1f}A)")
        elif current > limits["critical_threshold"]:
            reasons.append(f"CRITICAL: Current above operating range ({current:.1f}A)")

    # --- Device age ---
    if pd.notna(row.get("pct_life_used", None)):
        if row["pct_life_used"] > 0.9:
            reasons.append("Device past 90% expected life")
        elif row["pct_life_used"] > 0.7:
            reasons.append("Device aging (70%+ life used)")
    
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

    # --- MM3-specific ---
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

    # --- Create standardized version for easier matching ---
    def standardize_device_type(dev_type):
        dev_type = str(dev_type).upper()
        if "ZM1" in dev_type:
            return "ZM1"
        elif "UM3" in dev_type:  # Will catch UM3+
            return "UM3"
        elif "MM3" in dev_type:
            return "MM3"
        else:
            return dev_type
    
    df["Device_Type_Standardized"] = df["Device_Type"].apply(standardize_device_type)

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

    # --- Apply capping at the dataframe level ---
    # Cap communication age
    df["comm_age_days"] = df["comm_age_days"].clip(upper=365)
    
    # Cap device life percentage
    if "pct_life_used" in df.columns:
        df["pct_life_used"] = df["pct_life_used"].fillna(0).clip(upper=1.0)

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
        "low_current_flag",     
        "high_current_flag",     
        "critical_current_flag", 
        "high_offpeak_flag",     # (MM3 specific)
        "overheat_flag",
        "device_age_months",            # NEW
        "battery_drain_rate",           # NEW  
        "temperature_exceedence_count", # NEW
        "age_adjusted_battery_risk",    # NEW
        "old_and_hot_flag",             # NEW
        "maintenance_urgency_score",    # NEW
        "risk_score_zm1",
        "risk_score_um3",
        "risk_score_mm3",
    ]:
        if col not in df.columns:
            # Start with neutral defaults
            df[col] = 0.0 if "flag" not in col else 0

    # --- Apply additional capping after initialization ---
    # Cap temperature and current for all devices
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    
    # Cap battery report age for ZM1
    zm1_mask = df["Device_Type_Standardized"] == "ZM1"
    if zm1_mask.any() and "battery_report_age_days" in df.columns:
        df.loc[zm1_mask, "battery_report_age_days"] = df.loc[zm1_mask, "battery_report_age_days"].clip(upper=180)

    # --- Device-Type Masks ---
    mask_zm1 = df["Device_Type_Standardized"] == "ZM1"
    mask_um3 = df["Device_Type_Standardized"] == "UM3"
    mask_mm3 = df["Device_Type_Standardized"] == "MM3"

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
            "low_current_flag",     
            "high_current_flag",    
            "critical_current_flag",
            "overheat_flag",
            "device_age_months",            # NEW
            "battery_drain_rate",           # NEW  
            "temperature_exceedence_count", # NEW
            "age_adjusted_battery_risk",    # NEW
            "old_and_hot_flag",             # NEW
            "maintenance_urgency_score",    # NEW
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
            "low_current_flag",     
            "high_current_flag",    
            "critical_current_flag",
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
            "low_current_flag",     
            "high_current_flag",    
            "critical_current_flag",
            "high_offpeak_flag",    
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
    "device_age_months",           # NEW
    "expected_lifetime_days",
    "pct_life_used",
    "battery_drain_rate",          # NEW
    "temperature_exceedence_count", # NEW
    "age_adjusted_battery_risk",   # NEW
    "old_and_hot_flag",            # NEW
    "maintenance_urgency_score",   # NEW
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