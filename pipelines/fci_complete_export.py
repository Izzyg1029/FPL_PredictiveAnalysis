# fci_complete_export.py - COMBINED SINGLE EXPORT
# Creates ONE file: FCI_Device_Health_Export.csv with ALL data

import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import joblib

# ====================================================
# CONFIGURATION
# ====================================================
HISTORY_PATH = Path("data/processed/fci_history.parquet")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("powerbi_exports")
LABEL_TO_NAME = {
    0: "NO_ACTION",
    1: "RECONFIGURE",
    2: "RELOCATE",
    3: "REPLACE",
}

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
    """Add drain rate categories"""
    if 'battery_drain_rate_per_day' in df.columns:
        expected_rate = 100 / (10 * 365)  # 10-year nominal
        
        df['drain_rate_category'] = pd.cut(
            df['battery_drain_rate_per_day'].fillna(0),
            bins=[-1, 0.001, expected_rate*0.5, expected_rate, expected_rate*2, expected_rate*5, 100],
            labels=['No Drain', 'Very Slow', 'Normal', 'Slightly Fast', 'Fast', 'Very Fast']
        )
        
        df['is_normal_drain'] = df['battery_drain_rate_per_day'] <= expected_rate * 1.5
        df['is_fast_drain'] = (df['battery_drain_rate_per_day'] > expected_rate * 1.5)
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

def main():
    """Main function to create combined export"""
    
    print("=" * 60)
    print(" FCI COMPLETE HEALTH EXPORT - SINGLE FILE")
    print("=" * 60)
    
    # ====================================================
    # 1. LOAD AND PROCESS DEVICE DATA
    # ====================================================
    
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORY_PATH}. Run pipelines/update_history.py first.")
    
    print("\n📦 Loading device history...")
    df = pd.read_parquet(HISTORY_PATH)
    print(f" Loaded {len(df):,} historical records")
    
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
    print(f"🧹 Dropped {before - len(df):,} rows with missing device_type")
    
    # Convert dates
    if "BatteryLatestReport" in df.columns:
        df["BatteryLatestReport"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")
    
    # Get latest record per device
    if "BatteryLatestReport" in df.columns:
        latest = df.sort_values("BatteryLatestReport").drop_duplicates("Serial", keep="last")
    else:
        latest = df.drop_duplicates("Serial", keep="last")
    
    print(f" Latest device records: {len(latest):,} devices")
    
    # ====================================================
    # 2. ADD POWER BI CATEGORIES
    # ====================================================
    
    print("\n Adding Power BI categories...")
    
    result = latest.copy()
    result = create_battery_categories(result)
    result = create_timeline_categories(result)
    result = create_risk_categories(result)
    result = create_age_categories(result)
    result = create_drain_categories(result)
    result = create_critical_flags(result)
    
    # ====================================================
    # 3. ADD ML PREDICTIONS
    # ====================================================
    
    print("\n🤖 Adding ML predictions...")
    
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
    # 4. ADD REPLACEMENT DATES AND GLOBAL STATS
    # ====================================================
    
    print("\n Adding replacement forecasts...")
    result = add_replacement_dates(result)
    result = add_global_stats(result)
    
    # ====================================================
    # 5. ADD METADATA COLUMNS
    # ====================================================
    
    result["ExportDate"] = datetime.now().strftime('%Y-%m-%d')
    result["ExportTimestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result["DataSource"] = "FCI Complete Export"
    
    # ====================================================
    # 6. SAVE SINGLE FILE - EXACTLY AS SPECIFIED
    # ====================================================
    
    print("\n Saving single export file...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the EXACT filename specified in the user guide
    output_file = OUTPUT_DIR / "FCI_Device_Health_Export.csv"
    
    # Save to CSV
    result.to_csv(output_file, index=False)
    print(f" Created: {output_file}")
    
    # Also save a timestamped version for backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = OUTPUT_DIR / f"FCI_Device_Health_Export_{timestamp}.csv"
    result.to_csv(backup_file, index=False)
    print(f" Backup: {backup_file}")
    
    # ====================================================
    # 7. PRINT SUMMARY
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
    
    print(f"\n⏰ REPLACEMENT TIMELINE:")
    if 'days_until_battery_critical' in result.columns:
        print(f"   • Emergency (<7 days): {(result['days_until_battery_critical'] < 7).sum():,}")
        print(f"   • Critical (<30 days): {(result['days_until_battery_critical'] < 30).sum():,}")
        print(f"   • Urgent (<90 days): {(result['days_until_battery_critical'] < 90).sum():,}")
        print(f"   • This year: {(result['days_until_battery_critical'] < 365).sum():,}")
    
    print(f"\n🤖 ML PREDICTIONS:")
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
2. Click Home → Get Data → Text/CSV
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

To refresh data later:
1. Run this script again: python fci_complete_export.py
2. In Power BI, click Refresh on the Home tab
    """)

if __name__ == "__main__":
    main()