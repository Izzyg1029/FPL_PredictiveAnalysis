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

    # Standard (value - min) / (max - min)
    return (s - min_val) / (max_val - min_val)


def add_common_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 'comm_age_hours' for EVERY device.
    This measures how long ago the device last communicated.
    """
    df = df.copy()

    # Convert Last_Heard string → datetime object
    df["Last_Heard_dt"] = pd.to_datetime(df["Last_Heard"])

    # Find the most recent timestamp in the dataset.
    # This acts as the "data export time".
    snapshot_time = df["Last_Heard_dt"].max()

    # Compute time difference in HOURS:
    # (snapshot_time - last_heard_time)
    df["comm_age_hours"] = (
        snapshot_time - df["Last_Heard_dt"]
    ).dt.total_seconds() / 3600.0

    return df

def add_install_age_features(df: pd.DataFrame, install_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aging features using FPL install-date sheet.

    Parameters:
        df          = Ample export DataFrame
        install_df  = DataFrame with install dates (must include Serial + InstallDate columns)

    Output:
        df with:
            - install_date_dt
            - device_age_days
            - expected_lifetime_days
            - pct_life_used (0–1 scale)
    """
    df = df.copy()

    # -------------------------------------------------------
    # 1. Make sure install_df's InstallDate is a datetime field
    # -------------------------------------------------------
    install_df = install_df.copy()
    install_df["InstallDate_dt"] = pd.to_datetime(
        install_df["InstallDate"], errors="coerce"
    )

    # -------------------------------------------------------
    # 2. Merge install dates into the main dataset via Serial
    # -------------------------------------------------------
    df = df.merge(
        install_df[["Serial", "InstallDate_dt"]],
        on="Serial",
        how="left"
    )

    # -------------------------------------------------------
    # 3. Compute device age in days
    # -------------------------------------------------------
    snapshot = df["Last_Heard_dt"].max()  # same logic as comm_age
    df["device_age_days"] = (
        snapshot - df["InstallDate_dt"]
    ).dt.total_seconds() / (3600 * 24)

    # -------------------------------------------------------
    # 4. Assign expected lifetimes per device type
    # -------------------------------------------------------
    # Typical FCI lifetimes (starter values):
    # - ZM1 = ~5 years (battery device)
    # - UM3 = ~10 years (underground housing)
    # - MM3 = ~8 years (line powered overhead)
    df["expected_lifetime_days"] = np.where(
        df["Device_Type"].str.contains("ZM1", case=False, na=False), 5 * 365,
        np.where(
            df["Device_Type"].str.contains("UM3", case=False, na=False), 10 * 365,
            8 * 365  # default for MM3 or unknown
        )
    )

    # -------------------------------------------------------
    # 5. Compute percent of life used (0–1)
    # -------------------------------------------------------
    df["pct_life_used"] = df["device_age_days"] / df["expected_lifetime_days"]

    # Handle negative or missing ages
    df["pct_life_used"] = df["pct_life_used"].clip(lower=0)

    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two lat/lon points in meters.
    Uses the Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_gps_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes GPS drift per device.
    Requires multiple rows per device over time.
    
    Adds:
        - gps_prev_lat
        - gps_prev_lon
        - distance_drift_m
        - gps_jump_flag
    """
    df = df.copy()

    # Ensure lat/lon numeric
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Sort by device + time
    df = df.sort_values(["Serial", "Last_Heard_dt"])

    # Previous coordinates per device
    df["gps_prev_lat"] = df.groupby("Serial")["Latitude"].shift(1)
    df["gps_prev_lon"] = df.groupby("Serial")["Longitude"].shift(1)

    # Compute drift only where previous coordinate exists
    df["distance_drift_m"] = df.apply(
        lambda row: haversine_distance(
            row["gps_prev_lat"],
            row["gps_prev_lon"],
            row["Latitude"],
            row["Longitude"]
        )
        if pd.notna(row["gps_prev_lat"]) and pd.notna(row["gps_prev_lon"])
        else 0,
        axis=1
    )

    # Flag large jumps:
    # > 30 m jump usually means GPS noise underground OR storm movement overhead
    df["gps_jump_flag"] = (df["distance_drift_m"] > 30).astype(int)

    return df

def add_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for variance features across multiple days.

    True variance requires multiple timestamped records per device.
    Since current data contains only one day, this function prepares
    the structure and will be implemented once multi-day data is available.

    Future calculations:
        - rolling variance of LineCurrent over N days
        - rolling variance of LineTemperatrue over N days
        - variance of comm_age_hours across time
    """
    df = df.copy()

    # TODO: when multi-day data is available:
    # Example:
    # df["current_variance"] = df.groupby("Serial")["LineCurrent_val"].rolling(window=7).var().reset_index(0, drop=True)
    # df["temp_variance"] = df.groupby("Serial")["LineTemperatrue_val"].rolling(window=7).var().reset_index(0, drop=True)

    # For now, create placeholder zeros so downstream logic works
    df["current_variance"] = 0.0
    df["temp_variance"] = 0.0

    return df

def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for frequency-of-failure features.

    Requires multiple days of data to compute:
        - frequency of zero_current_flag events
        - frequency of high/overheat temperature events
        - communication failure frequency
        - % of days with GPS drift above threshold

    These cannot be computed with a single-day snapshot.

    This function adds placeholder fields for now.
    """
    df = df.copy()

    # TODO: when multi-day data is available:
    # Example:
    # df["zero_current_frequency"] = df.groupby("Serial")["zero_current_flag"].rolling(window=30).mean().reset_index(0, drop=True)
    # df["comm_fail_frequency"] = df.groupby("Serial")["comm_fail_flag"].rolling(window=30).mean().reset_index(0, drop=True)

    # Placeholder values for now
    df["zero_current_frequency"] = 0.0
    df["overheat_frequency"] = 0.0
    df["comm_fail_frequency"] = 0.0
    df["gps_jump_frequency"] = 0.0

    return df


# ----------------------------------------------------
# Device-Specific Feature Engineering
# ----------------------------------------------------

def compute_zm1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features for ZM1 devices (battery-powered overhead).
    """
    df = df.copy()

    # Convert BatteryLevel column to numeric values
    df["battery_level"] = pd.to_numeric(df["BatteryLevel"], errors="coerce")

    # Flag if battery is below 20% (bad)
    df["battery_low_flag"] = (df["battery_level"] < 20).astype(int)

    # Convert BatteryLatestReport → datetime
    df["BatteryLatestReport_dt"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")

    # Compute age of the battery report (same idea as comm_age_hours)
    snapshot_time = df["Last_Heard_dt"].max()
    df["battery_report_age_hours"] = (
        snapshot_time - df["BatteryLatestReport_dt"]
    ).dt.total_seconds() / 3600.0

    # Convert current and temperature into numeric form
    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    # Flag zero current (sensor may be stuck or device may be off)
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)

    # Overheat flag if temp > 45°C (arbitrary starter threshold)
    df["overheat_flag"] = (df["LineTemperatrue_val"] > 45).astype(int)

    # --------------------------
    # Compute the ZM1 risk score
    # --------------------------

    # Weighted combination using normalized values
    risk = (
        0.4 * normalize(df["comm_age_hours"]) +
        0.3 * normalize(df["battery_report_age_hours"].fillna(0)) +
        0.2 * normalize(df["LineTemperatrue_val"].fillna(0)) +
        0.1 * normalize(df["zero_current_flag"])
    )

    # Convert from 0–1 scale → 0–100 scale
    risk = risk * 100

    # Add penalties for bad conditions
    risk += 20 * df["battery_low_flag"]
    risk += 10 * df["overheat_flag"]

    # Ensure final score stays between 0–100
    df["risk_score_zm1"] = risk.apply(clamp)

    return df


def compute_um3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features for UM3 devices (underground).
    """
    df = df.copy()

    # Convert fields to numeric
    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    # Current stuck at zero → bad underground indicator
    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)

    # Underground equipment should not exceed 40°C
    df["high_temp_flag"] = (df["LineTemperatrue_val"] > 40).astype(int)

    # Risk score weighted more heavily on communication (underground RF issues)
    risk = (
        0.5 * normalize(df["comm_age_hours"]) +
        0.3 * normalize(df["LineTemperatrue_val"].fillna(0)) +
        0.2 * normalize(df["zero_current_flag"])
    )

    risk = risk * 100
    risk += 10 * df["high_temp_flag"]

    df["risk_score_um3"] = risk.apply(clamp)

    return df


def compute_mm3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute health features for MM3 devices (line-powered overhead).
    """
    df = df.copy()

    df["LineCurrent_val"] = pd.to_numeric(df["LineCurrent"], errors="coerce")
    df["LineTemperatrue_val"] = pd.to_numeric(df["LineTemperatrue"], errors="coerce")

    df["zero_current_flag"] = (df["LineCurrent_val"] == 0).astype(int)

    # MM3 overhead temp threshold higher since it's line-powered
    df["overheat_flag"] = (df["LineTemperatrue_val"] > 50).astype(int)

    risk = (
        0.6 * normalize(df["comm_age_hours"]) +
        0.2 * normalize(df["LineCurrent_val"].fillna(0)) +
        0.2 * normalize(df["zero_current_flag"])
    )

    risk = risk * 100
    risk += 10 * df["overheat_flag"]

    df["risk_score_mm3"] = risk.apply(clamp)

    return df


# ----------------------------------------------------
# Main Routing Function (selects feature logic based on device type)
# ----------------------------------------------------

def build_health_features(df_devices: pd.DataFrame, install_df=None) -> pd.DataFrame:
    """
    Main function that:
    1. Adds common communication features
    2. Adds install-age features (if install_df provided)
    3. Adds GPS drift features
    4. Routes devices into ZM1 / UM3 / MM3 logic
    """
    df = df_devices.copy()

    # 1. Communication health for all devices
    df = add_common_time_features(df)

    # 2. Install-date aging (only if install sheet is provided)
    if install_df is not None:
        df = add_install_age_features(df, install_df)

    # 3. GPS drift features
    # (Works once you have multi-day CSVs for the same devices)
    df = add_gps_drift_features(df)
    
    # 4. Variance (placeholder)
    df = add_variance_features(df)

    # 5. Frequency (placeholder)
    df = add_frequency_features(df)

    # 6. Identify each device type
    mask_zm1 = df["Device_Type"].str.contains("ZM1", case=False, na=False)
    mask_um3 = df["Device_Type"].str.contains("UM3", case=False, na=False)
    mask_mm3 = df["Device_Type"].str.contains("MM3", case=False, na=False)

    # 7. Apply device-specific health + risk scoring
    if mask_zm1.any():
        df.loc[mask_zm1] = compute_zm1_features(df.loc[mask_zm1])

    if mask_um3.any():
        df.loc[mask_um3] = compute_um3_features(df.loc[mask_um3])

    if mask_mm3.any():
        df.loc[mask_mm3] = compute_mm3_features(df.loc[mask_mm3])

    return df
