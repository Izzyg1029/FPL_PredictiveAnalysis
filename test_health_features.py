import pandas as pd
from feature_health.health_features import build_health_features

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
# 
# This script also includes optional reporting sections (such as
# listing devices with no communication for > 3 days) for evaluation
# purposes only.

DEVICE_FILE = "data/2025-09-13-FPL-device-export.csv"
INSTALL_FILE = "data/FPL_install_dates.csv"

HIGH_RISK_THRESHOLD = 70


print("\n📥 Loading device export...")
df_raw = pd.read_csv(DEVICE_FILE)


try:
    install_df = pd.read_csv(INSTALL_FILE)
    print("📥 Install-date sheet loaded.\n")
except FileNotFoundError:
    install_df = None
    print("⚠️ No install-date sheet found. Aging features skipped.\n")


print("⚙️ Generating health features...")
df_features = build_health_features(df_raw, install_df=install_df)
print("✅ Features generated!\n")


# --------------------------------------------------------
# Summary Table
# --------------------------------------------------------

summary_cols = [
    "Serial",
    "Device_Type",
    "comm_age_days",
    "LineCurrent_val" if "LineCurrent_val" in df_features else None,
    "LineTemperatrue_val" if "LineTemperatrue_val" in df_features else None,
    "zero_current_flag" if "zero_current_flag" in df_features else None,
    "overheat_flag" if "overheat_flag" in df_features else None,
    "battery_low_flag" if "battery_low_flag" in df_features else None,
    "risk_score_zm1" if "risk_score_zm1" in df_features else None,
    "risk_score_um3" if "risk_score_um3" in df_features else None,
    "risk_score_mm3" if "risk_score_mm3" in df_features else None,
]

summary_cols = [c for c in summary_cols if c]

print("📊 Device Summary (first 20 rows):")
print(df_features[summary_cols].head(20).to_string(index=False))


# --------------------------------------------------------
# Combine Risk Scores Safely
# --------------------------------------------------------

risk_zm1 = df_features["risk_score_zm1"] if "risk_score_zm1" in df_features else pd.Series(0, index=df_features.index)
risk_um3 = df_features["risk_score_um3"] if "risk_score_um3" in df_features else pd.Series(0, index=df_features.index)
risk_mm3 = df_features["risk_score_mm3"] if "risk_score_mm3" in df_features else pd.Series(0, index=df_features.index)

df_features["risk_score"] = pd.concat([risk_zm1, risk_um3, risk_mm3], axis=1).max(axis=1)



# --------------------------------------------------------
# Show Top 10 Highest-Risk Devices
# --------------------------------------------------------

print("\n🔥 Top 10 Highest-Risk Devices:")

# Safely pull risk score columns if device type doesn’t exist
risk_zm1 = df_features["risk_score_zm1"] if "risk_score_zm1" in df_features else pd.Series(0, index=df_features.index)
risk_um3 = df_features["risk_score_um3"] if "risk_score_um3" in df_features else pd.Series(0, index=df_features.index)
risk_mm3 = df_features["risk_score_mm3"] if "risk_score_mm3" in df_features else pd.Series(0, index=df_features.index)

# Combine risk into one field
df_features["risk_score"] = pd.concat([risk_zm1, risk_um3, risk_mm3], axis=1).max(axis=1)

# Sort by risk descending and take top 10
top10 = df_features.nlargest(10, "risk_score")

if top10.empty:
    print("No devices have any risk score computed.")
else:
    print(top10[["Serial", "Device_Type", "risk_score", "comm_age_days"]]
          .to_string(index=False))

# --------------------------------------------------------
# Devices With No Communication in > 3 Days
# --------------------------------------------------------

NO_COMM_THRESHOLD_DAYS = 3

no_comm = df_features[df_features["comm_age_days"] > NO_COMM_THRESHOLD_DAYS]

print(f"\n🚨 Devices With No Communication for > {NO_COMM_THRESHOLD_DAYS} Days:")

if no_comm.empty:
    print("All devices communicated within the past 3 days. 👍")
else:
    print(
        no_comm[["Serial", "Device_Type", "comm_age_days", "risk_score"]]
        .sort_values("comm_age_days", ascending=False)
        .to_string(index=False)
    )


print("\n🎉 Test script completed.\n")
