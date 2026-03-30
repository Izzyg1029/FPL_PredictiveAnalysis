# fci_complete_export.py 
# Creates ONE file: FCI_Device_Health_Export.csv 

import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import joblib

print("SCRIPT STARTING - If you see this, Python is working")
import sys
print(f"Python version: {sys.version}")

# ====================================================
# CONFIGURATION
# ====================================================


HISTORY_PATH = Path("data/processed/fci_history.parquet")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path(__file__).parent.parent / "powerbi_exports"
LABEL_TO_NAME = {
    0: "NO_ACTION",
    1: "RECONFIGURE",
    2: "REPLACE",  # Direct replacement after reconfigure fails
}
def check_if_already_ran(output_file):
    """Check if the export file already exists and when it was created"""
    if output_file.exists():
        file_time = datetime.fromtimestamp(output_file.stat().st_mtime)
        time_diff = datetime.now() - file_time
        hours = time_diff.total_seconds() / 3600
        
        print(f"\n EXPORT FILE ALREADY EXISTS")
        print(f"   File: {output_file.name}")
        print(f"   Created: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Age: {hours:.1f} hours ago")
        print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        if hours < 1:
            print(f"     File is very recent (<1 hour old)")
        elif hours < 24:
            print(f"    File is from today")
        else:
            print(f"   File is {hours/24:.1f} days old")
        
        response = input("\nDo you want to regenerate the export? (y/n): ")
        return response.lower() == 'y'
    return True

def normalize_device_type_value(x):
    """Standardize device type names"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ("", "NONE", "NAN", "NULL"):
        return np.nan
    if s == "M3":
        return "MM3"
    if s == "UM3":
        return "UM3+"
    # fallback detect inside
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").replace("/", "")
    if "ZM1" in s2:
        return "ZM1"
    if "MM3" in s2 or s2 == "M3":
        return "MM3"
    if "UM3" in s2:
        return "UM3+"
    return s

def load_model_bundle(device_type: str):
    """Load ML model for a specific device type"""
    model_path = MODELS_DIR / device_type / "action_rf.joblib"
    if not model_path.exists():
        return None, None
    try:
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj and "features" in obj:
            return obj["model"], obj["features"]
        if hasattr(obj, "predict"):
            feats = list(getattr(obj, "feature_names_in_", []))
            return obj, feats if feats else None
    except:
        return None, None
    return None, None
    print("\n Loading device history...")
    df = pd.read_parquet(HISTORY_PATH)
    print(f"DEBUG: Loaded {len(df):,} rows")

def create_battery_categories(df):
    """Add battery health categories for Power BI"""
    if 'battery_current' in df.columns:
        df['battery_status'] = np.where(
            df['battery_current'] < 10, 'CRITICAL',
            np.where(df['battery_current'] < 20, 'Warning',
            np.where(df['battery_current'] < 30, 'Low',
            np.where(df['battery_current'] < 50, 'Moderate', 'Good')))
        )
        
        df['battery_level_category'] = pd.cut(
            df['battery_current'].fillna(0),
            bins=[-1, 0, 10, 20, 30, 50, 80, 101],
            labels=['Dead (0%)', 'Critical (1-10%)', 'Very Low (11-20%)', 
                   'Low (21-30%)', 'Moderate (31-50%)', 'Good (51-80%)', 'Excellent (81-100%)']
        )
    return df

def create_timeline_categories(df):
    """Add replacement timeline categories"""
    if 'days_until_battery_critical' in df.columns:
        df['replacement_timeline'] = np.where(
            df['days_until_battery_critical'] < 7, 'EMERGENCY (<1 week)',
            np.where(df['days_until_battery_critical'] < 30, 'Critical (<1 month)',
            np.where(df['days_until_battery_critical'] < 90, 'Urgent (1-3 months)',
            np.where(df['days_until_battery_critical'] < 180, 'Soon (3-6 months)',
            np.where(df['days_until_battery_critical'] < 365, 'Near (6-12 months)',
                    'Future (>1 year)'))))
        )
        
        df['needs_replacement_30d'] = df['days_until_battery_critical'] < 30
        df['needs_replacement_90d'] = df['days_until_battery_critical'] < 90
        df['needs_replacement_1yr'] = df['days_until_battery_critical'] < 365
    return df

def create_risk_categories(df):
    """Add risk categories"""
    if 'risk_score_current' in df.columns:
        df['risk_category'] = pd.cut(
            df['risk_score_current'].fillna(0),
            bins=[-1, 20, 40, 60, 80, 101],
            labels=['Low (0-20)', 'Medium (21-40)', 'High (41-60)', 'Critical (61-80)', 'Emergency (81-100)']
        )
        
        df['is_high_risk'] = df['risk_score_current'] > 60
        df['is_critical_risk'] = df['risk_score_current'] > 80
    return df

def create_age_categories(df):
    """Add device age categories"""
    if 'total_days_observed' in df.columns:
        df['device_age_years'] = df['total_days_observed'] / 365
        df['age_category'] = pd.cut(
            df['device_age_years'].fillna(0),
            bins=[-1, 0.5, 1, 2, 3, 5, 10, 100],
            labels=['<6 months', '6-12 months', '1-2 years', '2-3 years', '3-5 years', '5-10 years', '10+ years']
        )
    return df

def create_drain_categories(df):
    """Add drain rate categories using yearly rate"""
    if 'battery_drain_rate' in df.columns:
        expected_yearly_rate = 10.0  # 10% per year nominal
        
        df['drain_rate_category'] = pd.cut(
            df['battery_drain_rate'].fillna(0),
            bins=[-1, 0.1, expected_yearly_rate*0.5, expected_yearly_rate, 
                  expected_yearly_rate*2, expected_yearly_rate*5, 100],
            labels=['No Drain', 'Very Slow', 'Normal', 'Slightly Fast', 'Fast', 'Very Fast']
        )
        
        df['is_normal_drain'] = df['battery_drain_rate'] <= expected_yearly_rate * 1.5
        df['is_fast_drain'] = df['battery_drain_rate'] > expected_yearly_rate * 1.5
    return df

def create_critical_flags(df):
    """Create combined critical device flags"""
    conditions = []
    
    if 'battery_current' in df.columns:
        conditions.append(df['battery_current'] < 20)
    if 'days_until_battery_critical' in df.columns:
        conditions.append(df['days_until_battery_critical'] < 30)
    if 'risk_score_current' in df.columns:
        conditions.append(df['risk_score_current'] > 80)
    
    if conditions:
        if len(conditions) > 1:
            df['is_critical_device'] = pd.concat(conditions, axis=1).any(axis=1)
        else:
            df['is_critical_device'] = conditions[0]
    else:
        df['is_critical_device'] = False
    
    return df

def add_global_stats(df):
    """Add global statistics as columns for easy KPI cards"""
    df['total_devices'] = len(df)
    
    if 'battery_current' in df.columns:
        df['global_avg_battery'] = df['battery_current'].mean()
        df['global_devices_below_20'] = (df['battery_current'] < 20).sum()
        df['global_devices_below_30'] = (df['battery_current'] < 30).sum()
    
    if 'days_until_battery_critical' in df.columns:
        df['global_emergency_7d'] = (df['days_until_battery_critical'] < 7).sum()
        df['global_critical_30d'] = (df['days_until_battery_critical'] < 30).sum()
        df['global_urgent_90d'] = (df['days_until_battery_critical'] < 90).sum()
    
    return df

def add_replacement_dates(df):
    """Add estimated replacement dates"""
    if 'days_until_battery_critical' in df.columns:
        today = pd.Timestamp.now()
        df['estimated_replacement_date'] = None
        df['replacement_quarter'] = None
        df['replacement_year'] = None
        df['replacement_month'] = None
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('days_until_battery_critical')) and row['days_until_battery_critical'] > 0:
                try:
                    days = min(row['days_until_battery_critical'], 3650)
                    replacement_date = today + pd.Timedelta(days=days)
                    df.at[idx, 'estimated_replacement_date'] = replacement_date.strftime('%Y-%m-%d')
                    df.at[idx, 'replacement_quarter'] = f"Q{(replacement_date.month - 1) // 3 + 1}-{replacement_date.year}"
                    df.at[idx, 'replacement_year'] = replacement_date.year
                    df.at[idx, 'replacement_month'] = replacement_date.strftime('%Y-%m')
                except:
                    pass
    return df

def add_ttl_and_age(df):
    """Add Time-to-Live and device age columns"""
    print("\n Adding TTL and device age columns...")
    
    # Check if we have battery data
    if 'BatteryLevel' in df.columns:
        print("   Found battery data - calculating TTL for ZM1 devices")
        
        # For ZM1 devices only
        zm1_mask = df['device_type'] == 'ZM1'
        
        # We need a drain rate - use a default or estimate
        # Typical ZM1 battery drain is about 0.027% per day (10-year life)
        DEFAULT_DRAIN_RATE = 0.027  # % per day
        
        # Create TTL columns
        df['ttl_days'] = np.nan
        df.loc[zm1_mask, 'ttl_days'] = df.loc[zm1_mask, 'BatteryLevel'] / DEFAULT_DRAIN_RATE
        
        df['ttl_months'] = df['ttl_days'] / 30.44
        df['ttl_years'] = df['ttl_days'] / 365
        
        # TTL categories
        df['ttl_category'] = np.where(
            df['ttl_days'] < 30, 'Critical (<1 month)',
            np.where(df['ttl_days'] < 90, 'Urgent (1-3 months)',
            np.where(df['ttl_days'] < 180, 'Soon (3-6 months)',
            np.where(df['ttl_days'] < 365, 'Near (6-12 months)',
                    'Future (>1 year)')))
        )
        print(f"   Added TTL columns for ZM1 devices using default drain rate")
    else:
        print("   WARNING: Cannot calculate TTL - missing battery data")
    
    # Device age not possible without install dates
    print("   Note: Device age columns not added (install/first_seen data missing)")
    
    return df
def ensure_risk_scores_in_export(df):
    """Add risk score columns to export if missing"""
    print("\n Adding risk scores to export...")
    
    if df is None:
        print("  ERROR: Input dataframe is None")
        return pd.DataFrame()
    # Try to load from device profiles
    profiles_path = Path(__file__).parent.parent / "data" / "processed" / "time_series" / "all_device_profiles_summary.csv"
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        if 'risk_score_current' in profiles.columns:
            # Merge risk scores
            risk_cols = ['Serial', 'risk_score_current', 'risk_reason_current']
            existing = [c for c in risk_cols if c in profiles.columns]
            if existing:
                df = df.merge(profiles[existing], on='Serial', how='left')
                df.rename(columns={'risk_score_current': 'risk_score', 
                                  'risk_reason_current': 'risk_reason'}, inplace=True)
                print(f"  Added risk scores for {df['risk_score'].notna().sum()} devices")
    
    # Create risk categories if risk_score exists
    if 'risk_score' in df.columns:
        df['risk_category'] = pd.cut(
            df['risk_score'].fillna(0),
            bins=[-1, 20, 40, 60, 80, 101],
            labels=['Low (0-20)', 'Medium (21-40)', 'High (41-60)', 
                   'Critical (61-80)', 'Emergency (81-100)']
        )
        df['is_high_risk'] = df['risk_score'] > 60
        df['is_critical_risk'] = df['risk_score'] > 80
    
    return df

print("DEBUG: Starting main function")
def main():
    """Main function to create combined export"""
    print("DEBUG: Entered main function")
    print("=" * 60)
    print(" FCI COMPLETE HEALTH EXPORT - SINGLE FILE")
    print("=" * 60)
    
    print("DEBUG: Step 1 - Checking HISTORY_PATH")
    # ====================================================
    # 1. LOAD AND PROCESS DEVICE DATA
    # ====================================================
    
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORY_PATH}. Run pipelines/update_history.py first.")
    
    print("\n Loading device history...")
    df = pd.read_parquet(HISTORY_PATH)
    print("DEBUG: Step 2 - Loading parquet")
    print(f" Loaded {len(df):,} historical records")
    
    print("DEBUG: Step 3 - Processing device types")
    # Standardize device types
    if "Device_Type" in df.columns:
        dt = df["Device_Type"]
        if "device_type" in df.columns:
            dt = dt.where(~dt.isna(), df["device_type"])
    elif "device_type" in df.columns:
        dt = df["device_type"]
    else:
        dt = pd.Series([np.nan] * len(df))
    
    df["device_type"] = dt.map(normalize_device_type_value)
    
    # Drop missing device types
    before = len(df)
    df = df[df["device_type"].notna()].copy()
    print(f" Dropped {before - len(df):,} rows with missing device_type")
    
    # Convert dates
    if "BatteryLatestReport" in df.columns:
        df["BatteryLatestReport"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")
    
    # Get latest record per device
    if "BatteryLatestReport" in df.columns:
        latest = df.sort_values("BatteryLatestReport").drop_duplicates("Serial", keep="last")
    else:
        latest = df.drop_duplicates("Serial", keep="last")
    
    print(f" Latest device records: {len(latest):,} devices")
    
    # ===== CREATE result VARIABLE =====
    result = latest.copy()
    
    
    # ====================================================
    # 2. ADD POWER BI CATEGORIES
    # ====================================================
    print("\n Adding Power BI categories...")
    
    result = create_battery_categories(result)
    result = create_timeline_categories(result)
    result = create_risk_categories(result)
    result = create_age_categories(result)
    result = create_drain_categories(result)
    result = create_critical_flags(result)

    # ====================================================
    # 3. ADD ML PREDICTIONS
    # ====================================================
    
    print("\n Adding ML predictions...")
    
    all_predictions = []
    
    # Process each device type with a model
    for device_dir in MODELS_DIR.iterdir():
        if not device_dir.is_dir():
            continue
        
        device_type = device_dir.name
        model, feature_cols = load_model_bundle(device_type)
        
        if model is None or feature_cols is None:
            print(f" Skipping {device_type}: no model found")
            continue
        
        # Get devices of this type
        subset = result[result["device_type"] == device_type].copy()
        if subset.empty:
            continue
        
        print(f"   • Predicting for {device_type}: {len(subset)} devices")
        
        # Prepare features
        for col in feature_cols:
            if col not in subset.columns:
                subset[col] = 0
        
        X = subset[feature_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Make predictions
        preds = model.predict(X)
        probs = model.predict_proba(X).max(axis=1)
        
        # Add to subset
        subset["PredictedAction"] = [LABEL_TO_NAME.get(int(p), "NO_ACTION") for p in preds]
        subset["PredictedActionCode"] = preds
        subset["PredictionConfidence"] = probs
        subset["PredictionDate"] = datetime.now().strftime('%Y-%m-%d')
        
        all_predictions.append(subset[["Serial", "PredictedAction", "PredictedActionCode", 
                                       "PredictionConfidence", "PredictionDate"]])
    
    # Merge predictions with main data
    if all_predictions:
        pred_df = pd.concat(all_predictions, ignore_index=True)
        
        # Merge on Serial
        result = result.merge(pred_df, on="Serial", how="left")
        
        # Fill missing predictions
        result["PredictedAction"] = result["PredictedAction"].fillna("NO_ACTION")
        result["PredictedActionCode"] = result["PredictedActionCode"].fillna(0)
        result["PredictionConfidence"] = result["PredictionConfidence"].fillna(0)
        result["PredictionDate"] = result["PredictionDate"].fillna(datetime.now().strftime('%Y-%m-%d'))
        
        print(f" Added predictions for {len(pred_df)} devices")
    else:
        print(" No predictions generated")
        result["PredictedAction"] = "NO_ACTION"
        result["PredictedActionCode"] = 0
        result["PredictionConfidence"] = 0
        result["PredictionDate"] = datetime.now().strftime('%Y-%m-%d')

    # ====================================================
    # 3.5 ADD FLAGS FROM PREDICTIONS FILE
    # ====================================================
    
    print("\n Adding action flags from predictions...")
    
    # Load the predictions file which contains all the flags
    predictions_file = OUTPUT_DIR / "predictions_latest.csv"
    if predictions_file.exists():
        flags_df = pd.read_csv(predictions_file)
        print(f" Loaded predictions file with {len(flags_df)} rows")
        print(f" Columns in predictions file: {flags_df.columns.tolist()}")
        
        # Select the flag columns we want to merge
        flag_cols = ['Serial', 'days_since_last_report', 'battery_low_flag', 
                     'offline_flag', 'online_flag', 'intermittent_flag', 
                     'standby_flag', 'zero_current_flag', 'overheat_flag',
                     'coord_missing_flag']
        
        # Only keep columns that exist
        available_flags = [col for col in flag_cols if col in flags_df.columns]
        print(f" Available flag columns: {available_flags}")
        
        if len(available_flags) > 1:  # At least Serial plus one flag
            flags_subset = flags_df[available_flags].copy()
            
            # Check if Serial column exists in both dataframes
            if 'Serial' in result.columns and 'Serial' in flags_subset.columns:
                # Merge flags into result
                before_cols = len(result.columns)
                result = result.merge(flags_subset, on="Serial", how="left")
                after_cols = len(result.columns)
                
                print(f" Added {after_cols - before_cols} columns from predictions file")
                
                # Verify flags have values
                for col in available_flags:
                    if col != 'Serial' and col in result.columns:
                        if col in ['PredictedActionName', 'PredictedActionLabel']:
                            non_zero = "N/A"
                        else:
                            non_zero = (result[col] > 0).sum() if pd.api.types.is_numeric_dtype(result[col]) else "N/A"
                        print(f"   {col}: {non_zero} non-zero values")
            else:
                print(" ERROR: 'Serial' column missing in one of the dataframes!")
        else:
            print(" No flag columns found in predictions file")
    else:
        print(" predictions_latest.csv not found - flags will be missing")
   
    # ====================================================
    # 4. ADD RISK SCORES
    # ====================================================
    result = ensure_risk_scores_in_export(result)
    
    # ====================================================
    # 5. ADD REPLACEMENT DATES AND GLOBAL STATS
    # ====================================================
    
    print("\n Adding replacement forecasts...")
    result = add_replacement_dates(result)
    result = add_global_stats(result)
    
    # ====================================================
    # 6. ADD TTL AND DEVICE AGE COLUMNS
    # ====================================================
    
    result = add_ttl_and_age(result)
    
    # ====================================================
    # 7. ADD METADATA COLUMNS
    # ====================================================
    
    result["ExportDate"] = datetime.now().strftime('%Y-%m-%d')
    result["ExportTimestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result["DataSource"] = "FCI Complete Export"
    
    # ====================================================
    # 8. SAVE SINGLE FILE
    # ====================================================
    
    print("\n Saving single export file...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the EXACT filename specified in the user guide
    output_file = OUTPUT_DIR / "FCI_Device_Health_Export.csv"
    

    # Check if file already exists and ask user
    if not check_if_already_ran(output_file):
        print("❌ Export cancelled by user")
        return

    # Save to CSV
    result.to_csv(output_file, index=False)
    print(f" Created: {output_file}")
    
    # Also save a timestamped version for backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = OUTPUT_DIR / f"FCI_Device_Health_Export_{timestamp}.csv"
    result.to_csv(backup_file, index=False)
    print(f" Backup: {backup_file}")
    
    # ====================================================
    # 9. PRINT SUMMARY
    # ====================================================
    
    print("\n" + "=" * 60)
    print(" EXPORT COMPLETE - READY FOR POWER BI")
    print("=" * 60)
    
    print(f"\n FILE:")
    print(f"   • Name: FCI_Device_Health_Export.csv")
    print(f"   • Location: {output_file}")
    print(f"   • Devices: {len(result):,}")
    print(f"   • Columns: {len(result.columns)}")
    print(f"   • File size: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    print(f"\n BATTERY HEALTH:")
    if 'battery_current' in result.columns:
        print(f"   • Average: {result['battery_current'].mean():.1f}%")
        print(f"   • Critical (<20%): {(result['battery_current'] < 20).sum():,}")
        print(f"   • Low (20-30%): {((result['battery_current'] >= 20) & (result['battery_current'] < 30)).sum():,}")
    
    print(f"\n REPLACEMENT TIMELINE:")
    if 'days_until_battery_critical' in result.columns:
        print(f"   • Emergency (<7 days): {(result['days_until_battery_critical'] < 7).sum():,}")
        print(f"   • Critical (<30 days): {(result['days_until_battery_critical'] < 30).sum():,}")
        print(f"   • Urgent (<90 days): {(result['days_until_battery_critical'] < 90).sum():,}")
        print(f"   • This year: {(result['days_until_battery_critical'] < 365).sum():,}")
    
    print(f"\n TIME-TO-LIVE (TTL):")
    if 'ttl_days' in result.columns:
        print(f"   • Average TTL: {result['ttl_days'].mean():.1f} days")
        print(f"   • Critical (<30 days): {(result['ttl_days'] < 30).sum():,}")
        print(f"   • Urgent (30-90 days): {((result['ttl_days'] >= 30) & (result['ttl_days'] < 90)).sum():,}")
    
    print(f"\n DEVICE AGE:")
    if 'device_age_days' in result.columns:
        print(f"   • Average age: {result['device_age_days'].mean():.1f} days")
        print(f"   • New (<1 year): {(result['device_age_days'] < 365).sum():,}")
        print(f"   • Old (>5 years): {(result['device_age_days'] > 1825).sum():,}")

    print(f"\n ML PREDICTIONS:")
    if 'PredictedAction' in result.columns:
        action_counts = result['PredictedAction'].value_counts()
        for action, count in action_counts.items():
            pct = (count / len(result)) * 100
            print(f"   • {action}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  CRITICAL DEVICES:")
    print(f"   • Total critical: {result['is_critical_device'].sum():,}")
    
    print(f"\n DEVICE TYPES:")
    if 'device_type' in result.columns:
        type_counts = result['device_type'].value_counts()
        for dtype, count in type_counts.items():
            print(f"   • {dtype}: {count:,}")
    
    print(f"\n" + "=" * 60)
    print(" POWER BI INSTRUCTIONS (from your user guide):")
    print("=" * 60)
    print("""
1. Open Power BI Desktop
2. Click Home  Get Data  Text/CSV
3. Navigate to: powerbi_exports/FCI_Device_Health_Export.csv
4. Click Open
5. Click Load

The file contains ALL columns needed for:
   • Health Summary Dashboard
   • Device Detail View  
   • Analytics & Reports
   • ML Predictions (NO_ACTION, RECONFIGURE, RELOCATE, REPLACE)
   • Battery categories
   • Replacement timelines
   • Risk scores
   • Time-to-Live (TTL) metrics
   • Device age analytics

To refresh data later:
1. Run this script again: python fci_complete_export.py
2. In Power BI, click Refresh on the Home tab
    """)
if __name__ == "__main__":
    print("DEBUG: About to call main()")
    main()
    print("DEBUG: main() completed")