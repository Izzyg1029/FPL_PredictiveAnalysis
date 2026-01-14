#change to excel processed by changing 
# df_features.to_excel(out_file, index=False)
import pandas as pd
from pathlib import Path
import sys
import os
import importlib.util

# Get the absolute path to health_features.py
project_root = os.path.dirname(os.path.abspath(__file__))
feature_health_path = os.path.join(project_root, "..", "feature_health", "health_features.py")
feature_health_path = os.path.abspath(feature_health_path)

print(f"🔍 Loading module from: {feature_health_path}")

if not os.path.exists(feature_health_path):
    print(f"❌ File not found: {feature_health_path}")
    sys.exit(1)

# Load the module directly
spec = importlib.util.spec_from_file_location("health_features", feature_health_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Get the function
build_health_features = module.build_health_features
print("✅ Successfully loaded build_health_features function")

# Process files
CLEAN_DIR = Path("data/clean")
OUT_DIR = Path("data/processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for file in CLEAN_DIR.glob("*-clean.xlsx"):
    print(f"\n🧠 Processing: {file.name}")
    
    try:
        df = pd.read_excel(file)
        print(f"   Loaded {len(df)} rows")
        
        df_features = build_health_features(df)
        
        out_file = OUT_DIR / f"{file.stem.replace('-clean','')}-health.csv"
        df_features.to_csv(out_file, index=False)
        
        print(f"✅ Saved to: {out_file}")
        print(f"   Risk scores: {df_features['risk_score'].min():.1f} to {df_features['risk_score'].max():.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n✨ Done!")