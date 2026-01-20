# pipelines/run_health_features.py
import pandas as pd
from pathlib import Path
import sys
import os
import importlib.util

# Get the absolute path to health_features.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from pipelines/
feature_health_path = os.path.join(project_root, "feature_health", "health_features.py")

print(f"🔍 Loading module from: {feature_health_path}")

if not os.path.exists(feature_health_path):
    print(f"❌ File not found: {feature_health_path}")
    sys.exit(1)

# Load the module directly
spec = importlib.util.spec_from_file_location("health_features", feature_health_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Get the functions
build_health_features = module.build_health_features
get_top_risk_devices = module.get_top_risk_devices
print("✅ Successfully loaded functions from health_features")

# Process files - paths relative to project root
CLEAN_DIR = Path(project_root) / "data" / "clean" 
CLEAN_DAILY_DIR = CLEAN_DIR / "daily"
OUT_DIR = Path(project_root) / "data" / "processed" / "daily"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all clean Excel files
excel_files = list(CLEAN_DAILY_DIR.glob("*-clean.xlsx"))
print(f"📁 Found {len(excel_files)} Excel file(s) to process")

for file in excel_files:
    print(f"\n{'='*60}")
    print(f"🧠 Processing: {file.name}")
    print(f"{'='*60}")
    
    try:
        # Load the data
        df = pd.read_excel(file)
        print(f"   📊 Loaded {len(df)} rows")
        print(f"   Columns: {', '.join(df.columns[:5])}...")  # Show first 5 columns
        
        # OPTIONAL: Load install dates if available
        install_file = CLEAN_DIR / "install_dates.csv"
        install_df = None
        if install_file.exists():
            install_df = pd.read_csv(install_file)
            print(f"   📅 Loaded {len(install_df)} install records")
        else:
            print(f"   ⚠️  No install_dates.csv found - device age features will be limited")
        
        # Build health features
        if install_df is not None:
            df_features = build_health_features(df, install_df=install_df)
            print(f"   ✅ Built features WITH device age data")
        else:
            df_features = build_health_features(df)
            print(f"   ✅ Built features WITHOUT device age data")
        
        # Show statistics
        print(f"\n   📈 Risk Score Statistics:")
        print(f"      Min: {df_features['risk_score'].min():.2f}")
        print(f"      Max: {df_features['risk_score'].max():.2f}")
        print(f"      Mean: {df_features['risk_score'].mean():.2f}")
        risk_count = (df_features['risk_score'] > 0).sum()
        print(f"      Devices with risk > 0: {risk_count} ({risk_count/len(df_features)*100:.1f}%)")
        
        # Device type breakdown
        if 'Device_Type' in df_features.columns:
            print(f"\n   📱 Device Type Distribution:")
            for dev_type in ['ZM1', 'UM3', 'MM3']:
                count = (df_features['Device_Type'] == dev_type).sum()
                if count > 0:
                    avg_risk = df_features[df_features['Device_Type'] == dev_type]['risk_score'].mean()
                    print(f"      {dev_type}: {count} devices, avg risk: {avg_risk:.2f}")
        
        # Show top risky devices
        top_risky = get_top_risk_devices(df_features, n=5)
        if len(top_risky) > 0:
            print(f"\n   🚨 Top 5 Risky Devices:")
            for idx, row in top_risky.iterrows():
                short_reason = row['risk_reason'][:60] + "..." if len(row['risk_reason']) > 60 else row['risk_reason']
                print(f"      {row['Serial']} ({row['Device_Type']}): {row['risk_score']:.1f} - {short_reason}")
        
        # Save to CSV (as shown in your structure)
        out_file = OUT_DIR / f"{file.stem.replace('-clean','')}-health.csv"
        df_features.to_csv(out_file, index=False)
        
        print(f"\n💾 Saved to: {out_file}")
        print(f"   File size: {out_file.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("✨ All files processed!")
print(f"Output directory: {OUT_DIR}")
print(f"{'='*60}")