import pandas as pd
from feature_health.health_features import build_health_features

# --------------------------------------------------------
# Settings
# ===================================================================
# Test Script for Your Feature Health Pipeline
#
# This script:
#   1. Loads your Ample device export (CSV with ZM1/UM3/MM3 info)
#   2. Loads the FPL install-date sheet
#   3. Calls build_health_features() to compute:
#         - communication features
#         - install-age features once install dates aer provided
#         - GPS drift features (will be 0 on single-day data)
#         - device-specific risk scores (ZM1 / UM3 / MM3)
#   4. Prints out a preview of key columns
#
# You run this script to verify that the pipeline works end-to-end.
# ===================================================================
# -------------------------------------------------------------------
# 1. Update these paths to match where your CSVs are saved
#    These can be in a /data folder in your repo.
# -------------------------------------------------------------------
# --------------------------------------------------------

DEVICE_FILE = "2025-09-13-FPL-device-export.csv"
INSTALL_FILE = "FPL_install_dates.csv"   # optional

# Choose threshold for "high risk"
HIGH_RISK_THRESHOLD = 70


# --------------------------------------------------------
# Load Data
# --------------------------------------------------------

print("\n📥 Loading device export...")
df_raw = pd.read_csv(DEVICE_FILE)

# Try to load install date sheet (optional)
try:
    install_df = pd.read_csv(INSTALL_FILE)
    print("📥 Install-date sheet loaded successfully.\n")
except:
    install_df = None
    print("⚠️ No install-date sheet found. Aging features will be skipped.\n")


# --------------------------------------------------------
# Build Features
# --------------------------------------------------------

print("⚙️ Generating health features...")
df_features = build_health_features(df_raw, install_df=install_df)

print("✅ Features generated successfully!\n")


# --------------------------------------------------------
# Clean Summary Table
# --------------------------------------------------------

# Pick columns that make sense for human inspection
summary_cols = [
    "Serial",
    "Device_Type",
    "comm_age_hours",
    "LineCurrent_val",
    "LineTemperatrue_val",
    "zero_current_flag",
    "overheat_flag" if "overheat_flag" in df_features.columns else None,
    "battery_low_flag" if "battery_low_flag" in df_features.columns else None,
    "risk_score_zm1" if "risk_score_zm1" in df_features.columns else None,
    "risk_score_um3" if "risk_score_um3" in df_features.columns else None,
    "risk_score_mm3" if "risk_score_mm3" in df_features.columns else None
]

# Remove invalid (None) entries
summary_cols = [c for c in summary_cols if c in df_features.columns]

summary_df = df_features[summary_cols]

print("📊 Device Health Summary (first 20 rows):")
print(summary_df.head(20).to_string(index=False))


# --------------------------------------------------------
# Highlight At-Risk Devices
# --------------------------------------------------------

# Combine all risk score columns into one "risk_score" for sorting/display
df_features["risk_score"] = df_features[
    ["risk_score_zm1","risk_score_um3","risk_score_mm3"]
].max(axis=1)

high_risk = df_features[df_features["risk_score"] >= HIGH_RISK_THRESHOLD]

print("\n🔥 Devices Above High-Risk Threshold (risk ≥ {}):".format(HIGH_RISK_THRESHOLD))

if high_risk.empty:
    print("No high-risk devices found today. ✅")
else:
    print(high_risk[["Serial", "Device_Type", "risk_score", "comm_age_hours"]]
          .sort_values("risk_score", ascending=False)
          .to_string(index=False))


# --------------------------------------------------------
# Done
# --------------------------------------------------------

print("\n🎉 Test script completed.\n")
