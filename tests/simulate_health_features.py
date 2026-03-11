"""TEST CASE: INT-001 - Health Features with Simulated Data
Author: Cassidy 
Date: 2026-03-10
Requirements Tested:
  - 3.1.1: Device type detection
  - 3.2.1: Risk score calculation
  - 3.3.1: Flag generation (overheat, zero current)
  - 3.4.1: Risk reason generation
  
Success Criteria:
  - All device types processed correctly
  - Risk scores > 80 for extreme cases
  - Risk scores < 40 for normal cases
  - Appropriate flags set (overheat_flag, zero_current_flag)
  - Risk reasons contain expected text
"""


import pandas as pd
from feature_health.health_features import build_health_features, get_top_risk_devices

# =====================================================
# Simulated dataset with TRUE variation for normalization
# =====================================================

df_sim = pd.DataFrame([
    # ZM1 — battery + severe overheat + long comm gap
    {"Serial": "SIM001", "Device_Type": "ZM1",
     "Last_Heard": "2025-01-01",
     "LineCurrent": 0, "LineTemperature": 75,
     "BatteryLevel": 5, "BatteryLatestReport": "2023-01-01",
     "Latitude": 27.90, "Longitude": -82.40},

    # ZM1 — medium risk
    {"Serial": "SIM002", "Device_Type": "ZM1",
     "Last_Heard": "2025-01-10",
     "LineCurrent": 3, "LineTemperature": 45,
     "BatteryLevel": 25, "BatteryLatestReport": "2024-06-01",
     "Latitude": 27.91, "Longitude": -82.41},

    # ZM1 — low risk
    {"Serial": "SIM003", "Device_Type": "ZM1",
     "Last_Heard": "2025-01-15",
     "LineCurrent": 15, "LineTemperature": 20,
     "BatteryLevel": 90, "BatteryLatestReport": "2025-01-14",
     "Latitude": 27.92, "Longitude": -82.42},

    # UM3 — underground high temp + zero current
    {"Serial": "SIM004", "Device_Type": "UM3",
     "Last_Heard": "2024-12-15",
     "LineCurrent": 0, "LineTemperature": 60,
     "Latitude": 27.93, "Longitude": -82.43},

    # UM3 — moderate temp
    {"Serial": "SIM005", "Device_Type": "UM3",
     "Last_Heard": "2025-01-08",
     "LineCurrent": 10, "LineTemperature": 42,
     "Latitude": 27.94, "Longitude": -82.44},

    # UM3 — normal operation
    {"Serial": "SIM006", "Device_Type": "UM3",
     "Last_Heard": "2025-01-15",
     "LineCurrent": 15, "LineTemperature": 25,
     "Latitude": 27.95, "Longitude": -82.45},

    # MM3 — extreme overheat + zero current
    {"Serial": "SIM007", "Device_Type": "MM3",
     "Last_Heard": "2024-12-20",
     "LineCurrent": 0, "LineTemperature": 90,
     "Latitude": 27.96, "Longitude": -82.46},

    # MM3 — moderate overheat
    {"Serial": "SIM008", "Device_Type": "MM3",
     "Last_Heard": "2025-01-05",
     "LineCurrent": 12, "LineTemperature": 55,
     "Latitude": 27.97, "Longitude": -82.47},

    # MM3 — low risk baseline
    {"Serial": "SIM009", "Device_Type": "MM3",
     "Last_Heard": "2025-01-15",
     "LineCurrent": 20, "LineTemperature": 25,
     "Latitude": 27.98, "Longitude": -82.48},
])

# =====================================================
# Run Health Feature Pipeline
# =====================================================

df_features = build_health_features(df_sim)

# =====================================================
# Debug: show all columns available
# =====================================================

print("\n🔍 DEBUG — Available Columns:")
print(df_features.columns.tolist())


# Columns we want to safely view (only if they exist)
debug_cols = [
    "Serial", "Device_Type",
    "LineCurrent", "LineTemperature",
    "zero_current_flag", "overheat_flag",
    "risk_score_zm1", "risk_score_um3", "risk_score_mm3",
    "risk_score"
]

safe_cols = [c for c in debug_cols if c in df_features.columns]

print("\n🔍 DEBUG — Feature Extract Snapshot:")
print(df_features[safe_cols].to_string(index=False))


# =====================================================
# Final Risk Score Summary
# =====================================================

print("\n📊 Simulated Device Risk Scores (with reasons):")
print(df_features[[
    "Serial", "Device_Type", "comm_age_days",
    "risk_score"
]].to_string(index=False))

# =====================================================
# Top-5 Output (Column View)
# =====================================================

import textwrap

top5 = get_top_risk_devices(df_features, n=5)

print("\n🔥 Top 5 High-Risk Devices (Readable Columns):")
print("-" * 110)
print(f"{'Serial':<10} {'Type':<6} {'Risk':<8}  Reason")
print("-" * 110)

for _, row in top5.iterrows():
    wrapped_reason = textwrap.fill(row["risk_reason"], width=80, subsequent_indent=" " * 26)
    print(f"{row['Serial']:<10} {row['Device_Type']:<6} {row['risk_score']:<8}  {wrapped_reason}")

print("-" * 110)
