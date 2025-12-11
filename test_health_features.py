import pandas as pd
from feature_health.health_features import build_health_features

# --------------------------------------------------------
# Settings
# ===================================================================
# Test Script for Your Feature Health Pipeline
#
# This script:
#   1. Loads Ample device export CSV
#   2. Loads optional install-date sheet
#   3. Runs build_health_features()
#   4. Prints readable summaries + identifies high-risk devices
#
# Use this to verify that your pipeline works end-to-end.
# ===================================================================
# --------------------------------------------------------
# File Paths (relative to your repo)
# --------------------------------------------------------

DEVICE_FILE = "data/2025-09-13-FPL-device-export.csv"
INSTALL_FILE = "data/FPL_install_dates.csv"   # optional

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
    # Current & Temp (only appear after scoring functions)
    "LineCurrent_val",
    "LineTemperatrue_val"
    # Flags (only exist for certain device types),
    "zero_current_flag",
    "overheat_flag" if "overheat_flag" in df_features.columns else None,
    "battery_low_flag" if "battery_low_flag" in df_features.columns else None,
    # Device-specific risk scores
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
# ---------------------------------------------------------

risk_zm1 = df_features["risk_score_zm1"] if "risk_score_zm1" in df_features else pd.Series(0, index=df_features.index)
risk_um3 = df_features["risk_score_um3"] if "risk_score_um3" in df_features else pd.Series(0, index=df_features.index)
risk_mm3 = df_features["risk_score_mm3"] if "risk_score_mm3" in df_features else pd.Series(0, index=df_features.index)
# Combine into a single final risk score
df_features["risk_score"] = pd.concat([risk_zm1, risk_um3, risk_mm3], axis=1).max(axis=1)

# --------------------------------------------------------
# High-Risk Device Listing
# --------------------------------------------------------
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
