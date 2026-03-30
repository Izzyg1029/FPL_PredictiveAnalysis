# feature_health/health_features.py - UPDATED VERSION
# Uses observed battery readings for drain rates, not assumptions

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import reconfigure tracker
try:
    from pipelines.reconfigure_tracker import get_reconfigure_attempts
except ImportError:
    def get_reconfigure_attempts(serial):
        return 0
    print("Note: reconfigure_tracker not found - reconfigure history will not be tracked")

# ====================================================
# Configuration
# ====================================================

DEVICE_LIFETIMES_DAYS: Dict[str, int] = {
    "ZM1": 10 * 365,
    "UM3": 10 * 365,
    "MM3": 10 * 365,
}

TEMP_LIMITS_C: Dict[str, float] = {
    "ZM1": 70.0,
    "UM3": 85.0,
    "MM3": 85.0,
}

DEVICE_CAPABILITIES = {
    "ZM1": {"measures_current": False, "measures_temp": False, "measures_battery": True, "rechargeable": False},
    "UM3": {"measures_current": False, "measures_temp": False, "measures_battery": False, "rechargeable": False},
    "MM3": {"measures_current": True, "measures_temp": True, "measures_battery": False, "rechargeable": False},
}

CURRENT_LIMITS_A: Dict[str, Dict[str, float]] = {
    "ZM1": {"min_normal": 0, "max_normal": 0, "warning_threshold": 0, "critical_threshold": 0},
    "UM3": {"min_normal": 0, "max_normal": 0, "warning_threshold": 0, "critical_threshold": 0},
    "MM3": {
        "min_normal": 1.0,
        "max_normal": 800.0,
        "warning_threshold": 700.0,
        "critical_threshold": 850.0,
        "off_peak_max": 12.0
    }
}

BATTERY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ZM1": {"critical": 20, "warning": 30},
}

GPS_JUMP_THRESHOLD_M: float = 30.0

# ====================================================
# Helper Functions
# ====================================================

def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, x))

def normalize(series: pd.Series, cap: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if cap is not None:
        s = s.clip(upper=cap)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=s.index)
    return (s - min_val) / (max_val - min_val)

def _expected_lifetime_for_type(device_type: str) -> int:
    if not isinstance(device_type, str):
        return DEVICE_LIFETIMES_DAYS["MM3"]
    upper = device_type.upper()
    if "ZM1" in upper:
        return DEVICE_LIFETIMES_DAYS["ZM1"]
    if any(pattern in upper for pattern in ["UM3", "UM3+", "U-M3", "U M3"]):
        return DEVICE_LIFETIMES_DAYS["UM3"]
    if any(pattern in upper for pattern in ["MM3", "M-M3", "M M3"]):
        return DEVICE_LIFETIMES_DAYS["MM3"]
    return DEVICE_LIFETIMES_DAYS["MM3"]

# ====================================================
# Core Feature Builders
# ====================================================

def add_common_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Last_Heard_dt"] = pd.to_datetime(df["Last_Heard"], errors="coerce")
    snapshot = df["Last_Heard_dt"].max()
    if pd.isna(snapshot):
        df["comm_age_days"] = np.nan
        return df
    df["comm_age_days"] = (snapshot - df["Last_Heard_dt"]).dt.total_seconds() / (3600.0 * 24.0)
    return df

def add_install_age_features(df: pd.DataFrame, install_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    install_df = install_df.copy()
    install_df["InstallDate_dt"] = pd.to_datetime(install_df["InstallDate"], errors="coerce")
    df = df.merge(install_df[["Serial", "InstallDate_dt"]], on="Serial", how="left")
    snapshot = df["Last_Heard_dt"].max()
    df["expected_lifetime_days"] = df["Device_Type_Standardized"].apply(_expected_lifetime_for_type)
    if pd.isna(snapshot):
        df["device_age_days"] = np.nan
        df["pct_life_used"] = np.nan
        return df
    df["device_age_days"] = (snapshot - df["InstallDate_dt"]).dt.total_seconds() / (3600.0 * 24.0)
    df["pct_life_used"] = (df["device_age_days"] / df["expected_lifetime_days"]).clip(lower=0.0, upper=1.0)
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def add_gps_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Latitude"] = pd.to_numeric(df.get("Latitude", 0.0), errors="coerce")
    df["Longitude"] = pd.to_numeric(df.get("Longitude", 0.0), errors="coerce")
    df = df.sort_values(["Serial", "Last_Heard_dt"])
    df["prev_lat"] = df.groupby("Serial")["Latitude"].shift(1)
    df["prev_lon"] = df.groupby("Serial")["Longitude"].shift(1)
    def _calc(row):
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_lon"]):
            return 0.0
        return haversine_distance(row["prev_lat"], row["prev_lon"], row["Latitude"], row["Longitude"])
    df["distance_drift_m"] = df.apply(_calc, axis=1)
    df["gps_jump_flag"] = (df["distance_drift_m"] > GPS_JUMP_THRESHOLD_M).astype(int)
    return df

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

# ====================================================
# Device-Specific Risk Scoring
# ====================================================

def compute_zm1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features & risk score for ZM1 devices (battery-powered).
    Uses observed battery readings to calculate drain rates, not assumptions.
    """
    df = df.copy()

    # Battery level with safe handling
    df["battery_level"] = pd.to_numeric(df.get("BatteryLevel", np.nan), errors="coerce")
    
    thresholds = BATTERY_THRESHOLDS.get("ZM1", {"critical": 20, "warning": 30})
    
    df["battery_low_flag"] = np.where(
        df["battery_level"].notna(),
        (df["battery_level"] < thresholds["critical"]).astype(int),
        0
    )
    df["battery_warning_flag"] = np.where(
        df["battery_level"].notna(),
        (df["battery_level"] < thresholds["warning"]).astype(int),
        0
    )

    # Battery report recency
    df["BatteryLatestReport_dt"] = pd.to_datetime(df.get("BatteryLatestReport"), errors="coerce")
    snapshot = df["Last_Heard_dt"].max()

    if pd.isna(snapshot):
        df["battery_report_age_days"] = 0.0
    else:
        df["battery_report_age_days"] = ((snapshot - df["BatteryLatestReport_dt"]).dt.total_seconds() / (3600.0 * 24.0)).fillna(0.0)

    # ZM1 does NOT measure current or temperature
    df["LineCurrent_val"] = 0.0
    df["LineTemperature_val"] = 0.0
    df["zero_current_flag"] = 0
    df["low_current_flag"] = 0
    df["high_current_flag"] = 0
    df["critical_current_flag"] = 0
    df["overheat_flag"] = 0

    # Calculate device age in months
    if "device_age_days" in df.columns:
        df["device_age_months"] = df["device_age_days"] / 30.44
        df["device_age_months"] = df["device_age_months"].fillna(0)
    else:
        df["device_age_months"] = 0.0
        
    # ===== BATTERY DRAIN RATE - USE OBSERVED DATA =====
    # Check if we have drain rate from device profiles (observed data)
    if "battery_drain_rate_per_day" in df.columns and df["battery_drain_rate_per_day"].notna().any():
        # Use the observed drain rate from device profiles
        # This is already calculated from actual battery readings over time
        print("Using observed battery drain rates from device profiles")
        # Keep existing values, just ensure no NaN
        df["battery_drain_rate_per_day"] = df["battery_drain_rate_per_day"].fillna(0.027)
        df["battery_drain_rate"] = df["battery_drain_rate_per_day"] * 365.0
    else:
        # Fallback calculation if no observed data available
        # This uses the device's expected lifetime
        ZM1_DESIGN_LIFETIME_DAYS = 10 * 365
        
        if "expected_lifetime_days" in df.columns:
            safe_lifetime = df["expected_lifetime_days"].fillna(ZM1_DESIGN_LIFETIME_DAYS)
            safe_lifetime = safe_lifetime.clip(lower=365, upper=36500)
            base_yearly_rate = 100.0 / (safe_lifetime / 365.0)
            if "pct_life_used" in df.columns:
                age_factor = df["pct_life_used"].fillna(0)
                aging_multiplier = 1 + (age_factor ** 2) * 2
            else:
                aging_multiplier = 1.0
            df["battery_drain_rate_per_day"] = (base_yearly_rate / 365.0) * aging_multiplier
            df["battery_drain_rate"] = df["battery_drain_rate_per_day"] * 365.0
        else:
            # Last resort fallback
            base_yearly_rate = 10.0
            if "pct_life_used" in df.columns:
                age_factor = df["pct_life_used"].fillna(0)
                aging_multiplier = 1 + (age_factor ** 2) * 2
            else:
                aging_multiplier = 1.0
            df["battery_drain_rate_per_day"] = (base_yearly_rate / 365.0) * aging_multiplier
            df["battery_drain_rate"] = base_yearly_rate * aging_multiplier
    
    # Ensure minimum positive drain rate
    MIN_DAILY_DRAIN = 0.1 / 365.0
    df["battery_drain_rate_per_day"] = df["battery_drain_rate_per_day"].clip(lower=MIN_DAILY_DRAIN)
    df["battery_drain_rate"] = df["battery_drain_rate"].clip(lower=0.1)

    # Add battery acceleration factor
    if "pct_life_used" in df.columns:
        df["aging_acceleration_factor"] = 1 + (df["pct_life_used"] ** 2) * 2
        df["battery_degradation_factor"] = df["aging_acceleration_factor"]
    else:
        df["aging_acceleration_factor"] = 1.0
        df["battery_degradation_factor"] = 1.0
        
    df["temperature_exceedence_count"] = 0
    
    # AGE ADJUSTED BATTERY RISK
    df["age_adjusted_battery_risk"] = df["battery_drain_rate"] * (df["device_age_months"] / 24.0)
    df["age_adjusted_battery_risk"] = df["age_adjusted_battery_risk"].fillna(0).clip(upper=100)
    df["old_and_hot_flag"] = 0
    
    # Maintenance urgency
    age_component = (df["device_age_months"] / 120.0).fillna(0).clip(upper=1.0)
    battery_component = (df["battery_drain_rate"] / 20.0).clip(upper=1.0)
    df["maintenance_urgency_score"] = (age_component * 0.4 + battery_component * 0.6).clip(upper=1.0).round(3)
    
    # ===== RISK CALCULATION =====
    risk = (
        0.25 * normalize(df["comm_age_days"], cap=365) +
        0.25 * normalize(df["battery_report_age_days"], cap=180) +
        0.20 * normalize(df["pct_life_used"], cap=1.0) +
        0.20 * normalize(df["age_adjusted_battery_risk"], cap=100) +
        0.10 * normalize(df["maintenance_urgency_score"], cap=1.0)
    ) * 100.0

    # Penalties
    risk += 30.0 * df["battery_low_flag"]
    risk += np.where(df["pct_life_used"] > 0.9, 20.0, 0.0)
    risk += np.where(df["battery_report_age_days"] > 180, 15.0, 0.0)
    
    # Missing data penalties
    if "Last_Heard_dt" in df.columns:
        risk += np.where(df["Last_Heard_dt"].isna(), 30.0, 0.0)
    if "InstallDate_dt" in df.columns:
        risk += np.where(df["InstallDate_dt"].isna(), 25.0, 0.0)
    if "BatteryLevel" in df.columns:
        risk += np.where(df["BatteryLevel"].isna(), 20.0, 0.0)
    
    df["risk_score_zm1"] = risk.apply(clamp)
    
    return df

def compute_um3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LineCurrent_val"] = 0.0
    df["LineTemperature_val"] = 0.0
    df["zero_current_flag"] = 0
    df["low_current_flag"] = 0
    df["high_current_flag"] = 0
    df["critical_current_flag"] = 0
    df["overheat_flag"] = 0
    df["battery_level"] = np.nan
    df["battery_low_flag"] = 0
    df["battery_warning_flag"] = 0
    
    risk = (0.70 * normalize(df["comm_age_days"], cap=365) + 0.30 * normalize(df["pct_life_used"], cap=1.0)) * 100.0
    risk += np.where(df["pct_life_used"] > 0.9, 25.0, 0.0)
    risk += np.where(df["comm_age_days"] > 7, 20.0, 0.0)
    risk += np.where(df["comm_age_days"] > 30, 40.0, 0.0)
    if "Last_Heard_dt" in df.columns:
        risk += np.where(df["Last_Heard_dt"].isna(), 50.0, 0.0)
    if "InstallDate_dt" in df.columns:
        risk += np.where(df["InstallDate_dt"].isna(), 20.0, 0.0)
    df["risk_score_um3"] = risk.apply(clamp)
    return df

def compute_mm3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LineCurrent_val"] = pd.to_numeric(df.get("LineCurrent", 0.0), errors="coerce").fillna(0.0)
    df["LineTemperature_val"] = pd.to_numeric(df.get("LineTemperature", 0.0), errors="coerce").fillna(0.0)
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)
    
    limits = CURRENT_LIMITS_A["MM3"]
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0.0).astype(int)
    df["low_current_flag"] = ((df["LineCurrent_val"] > 0) & (df["LineCurrent_val"] < limits["min_normal"])).astype(int)
    df["high_current_flag"] = (df["LineCurrent_val"] > limits["warning_threshold"]).astype(int)
    df["critical_current_flag"] = (df["LineCurrent_val"] > limits["critical_threshold"]).astype(int)
    df["high_offpeak_flag"] = (df["LineCurrent_val"] > limits["off_peak_max"]).astype(int)
    df["overheat_flag"] = (df["LineTemperature_val"] > TEMP_LIMITS_C["MM3"]).astype(int)
    df["battery_level"] = np.nan
    df["battery_low_flag"] = 0
    df["battery_warning_flag"] = 0
    
    risk = (0.30 * normalize(df["comm_age_days"], cap=365) +
            0.20 * normalize(df["LineCurrent_val"], cap=1000) +
            0.15 * normalize(df["LineTemperature_val"], cap=100) +
            0.20 * normalize(df["pct_life_used"], cap=1.0) +
            0.15 * normalize(df["high_current_flag"] + df["critical_current_flag"] + df["overheat_flag"])) * 100.0
    risk += 20.0 * df["overheat_flag"]
    risk += 25.0 * df["critical_current_flag"]
    risk += 15.0 * df["high_current_flag"]
    risk += 10.0 * df["low_current_flag"]
    risk += 12.0 * df["high_offpeak_flag"]
    risk += np.where(df["pct_life_used"] > 0.9, 25.0, 0.0)
    risk += 30.0 * df["zero_current_flag"]
    if "Last_Heard_dt" in df.columns:
        risk += np.where(df["Last_Heard_dt"].isna(), 30.0, 0.0)
    if "InstallDate_dt" in df.columns:
        risk += np.where(df["InstallDate_dt"].isna(), 20.0, 0.0)
    if "LineCurrent" in df.columns:
        risk += np.where(df["LineCurrent"].isna(), 15.0, 0.0)
    if "LineTemperature" in df.columns:
        risk += np.where(df["LineTemperature"].isna(), 10.0, 0.0)
    df["risk_score_mm3"] = risk.apply(clamp)
    return df

# ====================================================
# explain_risk - No RELOCATE, all RECONFIGURE
# ====================================================

def explain_risk(row):
    reasons = []
    device_type = str(row.get("Device_Type", "")).upper()
    
    # Get reconfigure history
    reconfigure_count = 0
    if 'Serial' in row.index and pd.notna(row.get('Serial')):
        try:
            reconfigure_count = get_reconfigure_attempts(row['Serial'])
        except:
            pass
    
    # Add reconfigure context
    if reconfigure_count > 0:
        if reconfigure_count == 1:
            reasons.append(f"Reconfigure attempted {reconfigure_count} time")
        else:
            reasons.append(f"Reconfigure attempted {reconfigure_count} times")
        if reconfigure_count >= 2:
            reasons.append("Multiple reconfigure attempts - consider replacement after 90 days")

    # Critical missing data checks
    if pd.isna(row.get("Last_Heard_dt")) or pd.isna(row.get("Last_Heard")):
        reasons.append("Missing Last_Heard date")
    
    # GPS/Sensor Issues - all lead to RECONFIGURE
    if row.get('gps_jump_flag', 0) == 1:
        reasons.append("GPS drift detected - reconfigure recommended")
    
    if row.get('coord_missing_flag', 0) == 1:
        reasons.append("GPS coordinates missing - reconfigure recommended")
    
    # ZM1-specific
    if "ZM1" in device_type:
        # Check drain rate if available
        drain_rate = row.get("battery_drain_rate", 0)
        if drain_rate > 40:  # >40% per year is fast
            reasons.append(f"Battery draining fast ({drain_rate:.0f}%/year)")
        elif drain_rate > 20:
            reasons.append(f"Battery draining faster than expected ({drain_rate:.0f}%/year)")
        
        if pd.isna(row.get("BatteryLevel")):
            reasons.append("Missing battery data - check telemetry")
        elif row.get("battery_low_flag", 0) == 1:
            if reconfigure_count == 0:
                reasons.append("Battery critically low - reconfigure to verify")
            else:
                reasons.append("Battery still low after reconfigure - consider replacement after 90 days")
    
    # MM3-specific
    elif "MM3" in device_type:
        if row.get("critical_current_flag", 0) == 1:
            current = row.get("LineCurrent_val", 0)
            reasons.append(f"CRITICAL: Current {current:.1f}A exceeds max (850A)")
        elif row.get("overheat_flag", 0) == 1:
            temp = row.get("LineTemperature_val", 0)
            if reconfigure_count == 0:
                reasons.append(f"Overheating ({temp:.1f}°C) - check ventilation")
            else:
                reasons.append(f"Still overheating after reconfigure - consider replacement after 90 days")
        elif row.get("high_current_flag", 0) == 1:
            current = row.get("LineCurrent_val", 0)
            reasons.append(f"High current ({current:.1f}A) - investigate")
        elif row.get("low_current_flag", 0) == 1:
            current = row.get("LineCurrent_val", 0)
            reasons.append(f"Low current ({current:.1f}A) - check connection")
        
        if row.get("zero_current_flag", 0) == 1 and len(reasons) == 0:
            reasons.append("Zero current detected - reconfigure sensor")
        
        if row.get("high_offpeak_flag", 0) == 1 and len(reasons) == 0:
            reasons.append("High current during off-peak - investigate")
        
        if row.get("pct_life_used", 0) > 0.9:
            years = row.get("device_age_years", row.get("pct_life_used", 0) * 10)
            age_reason = f"Device past 90% expected life ({years:.1f} years) - plan replacement"
            if len(reasons) > 0:
                reasons[0] = reasons[0] + " | " + age_reason
            else:
                reasons.append(age_reason)
        
        comm_age = row.get("comm_age_days", 0)
        if comm_age > 90:
            comm_reason = f"No communication for {comm_age:.0f} days"
            if len(reasons) > 0:
                reasons[0] = reasons[0] + " | " + comm_reason
            else:
                reasons.append(comm_reason)
        elif comm_age > 30:
            comm_reason = f"Long communication gap ({comm_age:.0f} days)"
            if len(reasons) > 0:
                reasons[0] = reasons[0] + " | " + comm_reason
            else:
                reasons.append(comm_reason)
        
        if pd.isna(row.get("LineCurrent_val")):
            reasons.append("Missing current data")
        if pd.isna(row.get("LineTemperature_val")):
            reasons.append("Missing temperature data")

    # UM3-specific
    elif "UM3" in device_type or "UM3+" in device_type:
        comm_age = row.get("comm_age_days", 0)
        if comm_age > 90:
            if reconfigure_count == 0:
                reasons.append(f"No communication for {comm_age:.0f} days - reconfigure")
            else:
                reasons.append(f"Still no communication after reconfigure - consider replacement after 90 days")
        elif comm_age > 30:
            if reconfigure_count == 0:
                reasons.append(f"No communication for {comm_age:.0f} days - reconfigure")
            else:
                reasons.append(f"Communication issue persists after reconfigure")
        elif comm_age > 7:
            reasons.append(f"Communication delay ({comm_age:.0f} days)")

        if row.get('gps_jump_flag', 0) == 1:
            reasons.append("GPS drift detected - reconfigure recommended")
        if row.get('coord_missing_flag', 0) == 1:
            reasons.append("GPS coordinates missing - reconfigure recommended")
        
        if row.get("pct_life_used", 0) > 0.9:
            reasons.append("Device past 90% expected life - plan replacement")

    if not reasons:
        return "Normal operating conditions"
    
    return ", ".join(reasons)

# ====================================================
# Main Router
# ====================================================

def build_health_features(df_devices: pd.DataFrame, install_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df_devices.copy()
    
    if "Device_Type" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Device_Type' column.")
    
    df["Device_Type"] = df["Device_Type"].astype(str).str.strip().str.upper()
    
    def standardize_device_type(dev_type):
        dev_type = str(dev_type).upper()
        if "ZM1" in dev_type:
            return "ZM1"
        elif "UM3" in dev_type:
            return "UM3"
        elif "MM3" in dev_type:
            return "MM3"
        else:
            return dev_type
    
    df["Device_Type_Standardized"] = df["Device_Type"].apply(standardize_device_type)
    
    rename_map = {
        "LineTemperatrue": "LineTemperature",
        "LineTemperture": "LineTemperature",
        "LineTemp": "LineTemperature",
        "Line_Temperature": "LineTemperature",
    }
    df = df.rename(columns=rename_map)
    
    if "Latitude" not in df.columns:
        df["Latitude"] = 0.0
    if "Longitude" not in df.columns:
        df["Longitude"] = 0.0
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    
    df = add_common_time_features(df)
    
    if install_df is not None:
        df = add_install_age_features(df, install_df)
    
    df["comm_age_days"] = df["comm_age_days"].clip(upper=365)
    if "pct_life_used" in df.columns:
        df["pct_life_used"] = df["pct_life_used"].fillna(0).clip(upper=1.0)
    
    df = add_gps_drift_features(df)
    df = add_variance_features(df)
    df = add_frequency_features(df)
    
    # Initialize shared flags & risk columns
    for col in [
        "battery_level", "battery_low_flag", "battery_warning_flag", "battery_report_age_days",
        "LineCurrent_val", "LineTemperature_val", "zero_current_flag", "low_current_flag",
        "high_current_flag", "critical_current_flag", "high_offpeak_flag", "overheat_flag",
        "device_age_months", "battery_drain_rate", "battery_drain_rate_per_day",
        "temperature_exceedence_count", "age_adjusted_battery_risk", "old_and_hot_flag",
        "maintenance_urgency_score", "risk_score_zm1", "risk_score_um3", "risk_score_mm3",
    ]:
        if col not in df.columns:
            if col == "battery_level":
                df[col] = np.nan
            elif col in ["risk_score_zm1", "risk_score_um3", "risk_score_mm3"]:
                df[col] = 0.0
            elif "flag" in col or col == "old_and_hot_flag":
                df[col] = 0
            else:
                df[col] = 0.0
    
    df["LineTemperature_val"] = df["LineTemperature_val"].clip(upper=100)
    df["LineCurrent_val"] = df["LineCurrent_val"].clip(upper=1000)
    
    zm1_mask = df["Device_Type_Standardized"] == "ZM1"
    if zm1_mask.any() and "battery_report_age_days" in df.columns:
        df.loc[zm1_mask, "battery_report_age_days"] = df.loc[zm1_mask, "battery_report_age_days"].clip(upper=180)
    
    mask_zm1 = df["Device_Type_Standardized"] == "ZM1"
    mask_um3 = df["Device_Type_Standardized"] == "UM3"
    mask_mm3 = df["Device_Type_Standardized"] == "MM3"
    
    if mask_zm1.any():
        sub = compute_zm1_features(df.loc[mask_zm1].copy())
        for col in ["battery_level", "battery_low_flag", "battery_warning_flag", "battery_report_age_days",
                    "LineCurrent_val", "LineTemperature_val", "zero_current_flag", "low_current_flag",
                    "high_current_flag", "critical_current_flag", "overheat_flag", "device_age_months",
                    "battery_drain_rate", "battery_drain_rate_per_day", "temperature_exceedence_count",
                    "age_adjusted_battery_risk", "old_and_hot_flag", "maintenance_urgency_score", "risk_score_zm1"]:
            if col in sub.columns:
                df.loc[mask_zm1, col] = sub[col].values
    
    if mask_um3.any():
        sub = compute_um3_features(df.loc[mask_um3].copy())
        for col in ["LineCurrent_val", "LineTemperature_val", "zero_current_flag", "low_current_flag",
                    "high_current_flag", "critical_current_flag", "overheat_flag", "battery_level",
                    "battery_low_flag", "battery_warning_flag", "risk_score_um3"]:
            if col in sub.columns:
                df.loc[mask_um3, col] = sub[col].values
    
    if mask_mm3.any():
        sub = compute_mm3_features(df.loc[mask_mm3].copy())
        for col in ["LineCurrent_val", "LineTemperature_val", "zero_current_flag", "low_current_flag",
                    "high_current_flag", "critical_current_flag", "high_offpeak_flag", "overheat_flag",
                    "battery_level", "battery_low_flag", "battery_warning_flag", "risk_score_mm3"]:
            if col in sub.columns:
                df.loc[mask_mm3, col] = sub[col].values
    
    df["risk_score"] = df[["risk_score_zm1", "risk_score_um3", "risk_score_mm3"]].max(axis=1)
    df["risk_score"] = df["risk_score"].round(2)
    df["comm_age_days"] = df["comm_age_days"].round(1)
    df["risk_reason"] = df.apply(explain_risk, axis=1)
    
    return df

# ====================================================
# Utility
# ====================================================

def get_top_risk_devices(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if "risk_score" not in df.columns:
        raise ValueError("risk_score not found. Did you call build_health_features()?")
    df_positive = df[df["risk_score"] > 0.0]
    return df_positive.sort_values("risk_score", ascending=False).head(n)

# ====================================================
# Suggested Output Schema
# ====================================================

FEATURE_HEALTH_OUTPUT_COLUMNS: List[str] = [
    "Serial", "Device_Type", "Last_Heard", "Last_Heard_dt", "comm_age_days",
    "LineCurrent", "LineTemperature", "LineCurrent_val", "LineTemperature_val",
    "battery_level", "battery_low_flag", "battery_warning_flag", "battery_report_age_days",
    "device_age_days", "device_age_months", "expected_lifetime_days", "pct_life_used",
    "battery_drain_rate", "battery_drain_rate_per_day", "aging_acceleration_factor",
    "battery_degradation_factor", "temperature_exceedence_count", "age_adjusted_battery_risk",
    "old_and_hot_flag", "maintenance_urgency_score", "Latitude", "Longitude",
    "distance_drift_m", "gps_jump_flag", "current_variance", "temp_variance",
    "zero_current_flag", "low_current_flag", "high_current_flag", "critical_current_flag",
    "high_offpeak_flag", "overheat_flag", "zero_current_frequency", "overheat_frequency",
    "comm_fail_frequency", "gps_jump_frequency", "risk_score_zm1", "risk_score_um3",
    "risk_score_mm3", "risk_score", "risk_reason"
]