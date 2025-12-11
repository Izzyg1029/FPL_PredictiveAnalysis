import pandas as pd
from feature_health.health_features import (
    build_health_features,
    get_top_risk_devices,
)

# ===================================================================
# Test Script for Feature Health Pipeline
# ===================================================================
# NOTE TO REVIEWERS:
# This test script is meant to demonstrate example usage of the feature
# health pipeline and how risk scores are computed. It is NOT part of
# the production pipeline. The purpose is to:
#   - show how risk scores behave with sample device data,
#   - verify the feature engineering logic,
#   - generate example outputs for code review.
# ===================================================================

DEVICE_FILE = "data/2025-09-13-FPL-device-export.csv"
INSTALL_FILE = "data/FPL_install_dates.csv"


print("\n📥 Loading device export...")
df_raw = pd.read_csv(DEVICE_FILE)

# Optional install-date sheet
try:
    install_df = pd.read_csv(INSTALL_FILE)
    print("📥 Install-date sheet loaded.\n")
except FileNotFoundError:
    install_df = None
    print("⚠️ No install-date sheet found. Aging features skipped.\n")


print("⚙️ Generating health features...")
df_features = build_health_features(df_raw, install_df=install_df)
print("✅ Features generated!\n")

# Make a copy just for pretty printing
df_print = df_features.copy()

# Round days to 2 decimals for readability
if "comm_age_days" in df_print.columns:
    df_print["comm_age_days"] = df_print["comm_age_days"].round(2)
if "device_age_days" in df_print.columns:
    df_print["device_age_days"] = df_print["device_age_days"].round(2)

# --------------------------------------------------------
# Summary Table
# --------------------------------------------------------

summary_cols = [
    "Serial",
    "Device_Type",
    "comm_age_days",
    "LineCurrent_val" if "LineCurrent_val" in df_print.columns else None,
    "LineTemperature_val" if "LineTemperature_val" in df_print.columns else None,
    "zero_current_flag" if "zero_current_flag" in df_print.columns else None,
    "overheat_flag" if "overheat_flag" in df_print.columns else None,
    "battery_low_flag" if "battery_low_flag" in df_print.columns else None,
    "risk_score",  # ⭐ include final combined risk score
]

summary_cols = [c for c in summary_cols if c in df_print.columns]

print("📊 Device Summary (first 20 rows):")
print(df_print[summary_cols].head(20).to_string(index=False))


# --------------------------------------------------------
# Top 10 Highest-Risk Devices
# --------------------------------------------------------

print("\n🔥 Top 10 Highest-Risk Devices (risk > 0):")
top10 = get_top_risk_devices(df_print, n=10)

if top10.empty:
    print("No devices have a risk_score above 0. 👍")
else:
    print(
        top10[["Serial", "Device_Type", "risk_score", "comm_age_days"]]
        .to_string(index=False)
    )


print("\n🎉 Test script completed.\n")
