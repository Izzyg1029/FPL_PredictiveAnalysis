import pandas as pd
from feature_health.health_features import build_health_features

# ===================================================================
# Test Script for Your Feature Health Pipeline
#
# This script:
#   1. Loads your Ample device export (CSV with ZM1/UM3/MM3 info)
#   2. Loads the FPL install-date sheet
#   3. Calls build_health_features() to compute:
#         - communication features
#         - install-age features
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

# Path to the Ample device export (the one you uploaded earlier)
AMPLE_CSV_PATH = "data/2025-09-13-FPL-device-export.csv"

# Path to your FPL install-date sheet (must contain Serial + InstallDate)
INSTALL_DATE_PATH = "data/FPL_install_dates.csv"


def main():
    # -------------------------------------------------------------------
    # 2. Load the Ample export file
    #    This contains columns like:
    #       Serial, Device_Type, Last_Heard, BatteryLevel, etc.
    # -------------------------------------------------------------------
    df_raw = pd.read_csv(AMPLE_CSV_PATH)

    # -------------------------------------------------------------------
    # 3. Load the install-date sheet
    #    MUST contain at least:
    #       Serial        (to match devices)
    #       InstallDate   (to compute device age)
    # -------------------------------------------------------------------
    install_df = pd.read_csv(INSTALL_DATE_PATH)

    # -------------------------------------------------------------------
    # 4. Call your feature builder
    #
    #    This will compute:
    #       comm_age_hours
    #       device_age_days
    #       pct_life_used
    #       distance_drift_m
    #       gps_jump_flag
    #       risk_score_zm1 / risk_score_um3 / risk_score_mm3
    #
    #    install_df=install_df enables the install-date aging features.
    # -------------------------------------------------------------------
    df_features = build_health_features(df_raw, install_df=install_df)

    # -------------------------------------------------------------------
    # 5. Choose which columns to preview in the output.
    #
    #    Some columns only exist for certain device types, so we check
    #    if they exist before printing.
    # -------------------------------------------------------------------
    cols_to_show = [
        "Serial",
        "Device_Type",
        "Last_Heard",
        "comm_age_hours",
        "device_age_days",
        "pct_life_used",
        "distance_drift_m",
        "gps_jump_flag",
        "risk_score_zm1",
        "risk_score_um3",
        "risk_score_mm3",
    ]

    # Only include columns that actually exist in df_features
    cols_present = [c for c in cols_to_show if c in df_features.columns]

    # -------------------------------------------------------------------
    # 6. Print the first few rows to verify that everything works.
    # -------------------------------------------------------------------
    print("\n================ FEATURE PREVIEW ================\n")
    print(df_features[cols_present].head())
    print("\n=================================================\n")


# -------------------------------------------------------------------
# Entry point: runs main() only when executing this script directly
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
