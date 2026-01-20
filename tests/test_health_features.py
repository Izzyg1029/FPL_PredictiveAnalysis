import sys
import pandas as pd
from pathlib import Path

# ===================================================================
# FIXED: Add project root to path first
# ===================================================================

# Get the project root and add it to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("HEALTH FEATURES TEST")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"Running from: {SCRIPT_DIR}")

# Try to import
try:
    from feature_health.health_features import (
        build_health_features,
        get_top_risk_devices,
    )
    print("✅ Successfully imported health_features!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nChecking what's available...")
    
    # Check if feature_health folder exists
    feature_health_path = PROJECT_ROOT / "feature_health"
    print(f"Looking for feature_health at: {feature_health_path}")
    print(f"Exists: {feature_health_path.exists()}")
    
    if feature_health_path.exists():
        print(f"Contents: {list(feature_health_path.glob('*'))}")
    
    exit()

# ===================================================================
# Test with your clean data
# ===================================================================

# Find clean files
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
INSTALL_FILE = PROJECT_ROOT / "data" / "clean" / "install_dates.csv"

clean_files = list(CLEAN_DIR.glob("*.xlsx")) + list(CLEAN_DIR.glob("*.xls"))
clean_files = [f for f in clean_files if "rejected" not in f.name.lower()]

if not clean_files:
    print(f"\n❌ No clean files found in {CLEAN_DIR}")
    print("Run your cleaning script first!")
    exit()

# Use the first clean file
DEVICE_FILE = clean_files[0]
print(f"\n📥 Testing with: {DEVICE_FILE.name}")

# Load data
df_raw = pd.read_excel(DEVICE_FILE)
print(f"✅ Loaded {len(df_raw)} devices")

# Load install dates if available
try:
    install_df = pd.read_csv(INSTALL_FILE)
    print(f"📥 Install dates loaded from: install_dates.csv")
except FileNotFoundError:
    install_df = None
    print("⚠️ No install dates found")

# Test the functions
print("\n⚙️ Testing build_health_features...")
try:
    df_features = build_health_features(df_raw, install_df=install_df)
    print(f"✅ Success! Created features with {len(df_features.columns)} columns")

    if "risk_score" in df_features.columns:
        print(f"\n📊 Risk scores calculated:")
        print(f"   Min: {df_features['risk_score'].min():.2f}")
        print(f"   Max: {df_features['risk_score'].max():.2f}")
        print(f"   Mean: {df_features['risk_score'].mean():.2f}")
        print(f"   Devices with risk > 0: {(df_features['risk_score'] > 0).sum()}")
        
        print("\n" + "=" * 70)
        print("🔍 DEBUGGING RISK SCORES:")
        print("=" * 70)
        
        # Check a few devices with high scores
        high_risk = df_features[df_features['risk_score'] > 50].head(3)
        
        if not high_risk.empty:
            print(f"\nFound {len(high_risk)} devices with risk_score > 50:")
            for idx, row in high_risk.iterrows():
                print(f"\nDevice {row['Serial']} (Type: {row['Device_Type']}, Score: {row['risk_score']:.2f}):")
                
                # Check contributing factors
                factors = []
                if row.get('comm_age_days', 0) > 1:
                    factors.append(f"comm_age_days: {row.get('comm_age_days', 0):.1f}")
                if row.get('pct_life_used', 0) > 0:
                    factors.append(f"pct_life_used: {row.get('pct_life_used', 0)*100:.1f}%")
                if row.get('zero_current_flag', 0) == 1:
                    factors.append("zero_current_flag")
                if row.get('overheat_flag', 0) == 1:
                    factors.append("overheat_flag")
                if row.get('battery_low_flag', 0) == 1:
                    factors.append("battery_low_flag")
                if row.get('LineTemperature_val', 0) > 50:
                    factors.append(f"LineTemperature: {row.get('LineTemperature_val', 0):.1f}°C")
                if row.get('LineCurrent_val', 0) > 100:
                    factors.append(f"LineCurrent: {row.get('LineCurrent_val', 0):.1f}A")
                
                print(f"  Factors: {', '.join(factors) if factors else 'None'}")
                print(f"  Risk reason: {row.get('risk_reason', 'N/A')}")
        else:
            print("\nNo devices with risk_score > 50")
       
        print("\n📡 COMMUNICATION AGE ANALYSIS:")
        print(f"comm_age_days statistics:")
        print(f"  Min: {df_features['comm_age_days'].min():.1f} days")
        print(f"  Max: {df_features['comm_age_days'].max():.1f} days")
        print(f"  Mean: {df_features['comm_age_days'].mean():.1f} days")
        print(f"  Std: {df_features['comm_age_days'].std():.1f} days")

        # Count devices by communication age
        print("\n📊 Devices by communication age:")
        bins = [0, 1, 3, 7, 14, 30, 1000]
        labels = ["<1 day", "1-3 days", "3-7 days", "7-14 days", "14-30 days", ">30 days"]
        df_features['comm_age_category'] = pd.cut(df_features['comm_age_days'], bins=bins, labels=labels)
        print(df_features['comm_age_category'].value_counts().sort_index())
        
        # ZM1 analysis
        print("\n🔋 CHECKING ZM1 DEVICES:")
        zm1_high_risk = df_features[(df_features['Device_Type'] == 'ZM1') & (df_features['risk_score'] > 50)]
        print(f"ZM1 devices with risk > 50: {len(zm1_high_risk)}")
        if not zm1_high_risk.empty:
            for idx, row in zm1_high_risk.head(3).iterrows():
                print(f"  {row['Serial']}: Score {row['risk_score']:.2f}, Battery: {row.get('battery_level', 'N/A')}%, Reason: {row.get('risk_reason', 'N/A')}")
       
        # Normalization caps check
        print("\n" + "="*70)
        print("🔍 CHECKING NORMALIZATION CAPS")
        print("="*70)

        # Check if caps are working
        print("Checking communication age caps (should be capped at 365 days):")
        extreme_comm = df_features[df_features['comm_age_days'] > 365]
        print(f"  Devices with comm_age_days > 365: {len(extreme_comm)}")
        if not extreme_comm.empty:
            sample = extreme_comm.iloc[0]
            print(f"  Sample: {sample['Serial']}, comm_age_days: {sample['comm_age_days']:.1f}, risk_score: {sample['risk_score']:.2f}")
            
 
        near_100 = df_features[df_features['pct_life_used'] > 0.9]
        print(f"\nDevices with pct_life_used > 90%: {len(near_100)}")
        if not near_100.empty:
            print("First 3 devices >90% life:")
            for idx, row in near_100.head(3).iterrows():
                print(f"  {row['Serial']}: {row['pct_life_used']*100:.1f}%, Risk: {row['risk_score']:.2f}")
                
        # NEW: Check devices with score 100
        print("\n" + "="*70)
        print("🔍 CHECKING DEVICES WITH SCORE 100")
        print("="*70)
        hundred_scores = df_features[df_features['risk_score'] == 100]

        if not hundred_scores.empty:
            print(f"Found {len(hundred_scores)} devices with score 100:")
            print(f"By device type:")
            print(hundred_scores['Device_Type'].value_counts())
            
        # Device types and top risky devices
        print("\n" + "="*70)
        print("📊 Device types in results:")
        print("="*70)   
        print(df_features['Device_Type'].value_counts())

        print("\n" + "=" * 70)
        print("🔋 NEW ZM1 COMBINED FEATURES ANALYSIS")
        print("=" * 70)

        # Check if ZM1 devices exist
        zm1_mask = df_features['Device_Type'].str.contains('ZM1', case=False, na=False)
        if zm1_mask.any():
            zm1_devices = df_features[zm1_mask]
            print(f"ZM1 devices found: {len(zm1_devices)}")
    
        # Show new features for top ZM1 devices
            if 'maintenance_urgency_score' in zm1_devices.columns:
                top_zm1 = zm1_devices.nlargest(5, 'maintenance_urgency_score')
                print("\n🚨 Top 5 ZM1 devices by maintenance urgency:")
                for idx, row in top_zm1.iterrows():
                    print(f"\n{row['Serial']}:")
                    print(f"  Maintenance urgency: {row['maintenance_urgency_score']:.3f}")
                    print(f"  Device age: {row['device_age_months']:.1f} months ({row['pct_life_used']*100:.1f}% life used)")
                    print(f"  Battery drain rate: {row['battery_drain_rate']:.1f}%/year")
                    print(f"  Age-adjusted battery risk: {row['age_adjusted_battery_risk']:.1f}")
                    print(f"  Old & hot flag: {row['old_and_hot_flag']}")
                    print(f"  Current battery: {row['battery_level']:.1f}%")
                    print(f"  Temperature: {row['LineTemperature_val']:.1f}°C")
                    print(f"  Risk score: {row['risk_score']:.2f}")
                    print(f"  Risk reasons: {row['risk_reason']}")
            else:
                print("❌ New ZM1 features not found in output - check your code updates")
    
        # Show statistics
            print(f"\n📊 ZM1 Feature Statistics:")
            print(f"  Avg maintenance urgency: {zm1_devices['maintenance_urgency_score'].mean():.3f}")
            print(f"  Avg device age: {zm1_devices['device_age_months'].mean():.1f} months")
            print(f"  Old & hot devices: {zm1_devices['old_and_hot_flag'].sum()} devices")
            print(f"  Avg age-adjusted battery risk: {zm1_devices['age_adjusted_battery_risk'].mean():.1f}")
        else:
            print("No ZM1 devices found in dataset")
        # Save results for inspection
        # Create processed directory structure
        PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
        HEALTH_FEATURES_DIR = PROCESSED_DIR / "health_features"
        HEALTH_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_file = HEALTH_FEATURES_DIR / f"{DEVICE_FILE.stem}-health-test.xlsx"
        
        df_features.to_excel(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file.name}")
            
except Exception as e:
    print(f"❌ Error in build_health_features: {e}")
import traceback
traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)