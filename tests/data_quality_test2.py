import pandas as pd
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "daily"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BATCH CLEANING - PROCESSING ALL EXCEL FILES")
print("=" * 60)

# Get ALL Excel files in the raw folder
excel_files = list(INPUT_DIR.glob("*.xlsx")) + list(INPUT_DIR.glob("*.xls"))

if not excel_files:
    print(f"❌ No Excel files found in {INPUT_DIR}")
    exit()

print(f"📁 Found {len(excel_files)} Excel file(s) to process:")

# Process each file
for excel_file in excel_files:
    print(f"\n{'='*50}")
    print(f"📥 Processing: {excel_file.name}")
    print(f"{'='*50}")
    
    # Load data
    df_raw = pd.read_excel(excel_file)
    print(f"   Total devices loaded: {len(df_raw)}")
    
    # Show device types
    if 'Device_Type' in df_raw.columns:
        print(f"\n   📊 Device types in raw data:")
        print(f"   {df_raw['Device_Type'].value_counts().to_string()}")
    
    # ======================================================
    # CLEANING: Only remove rows missing Device_Type or Last_Heard
    # ======================================================
    clean_mask = df_raw['Device_Type'].notna() & df_raw['Last_Heard'].notna()
    df_clean = df_raw[clean_mask].copy()
    
    # Create output filename (add -clean before extension)
    output_name = f"{excel_file.stem}-clean.csv"  # Save as CSV
    output_file = OUTPUT_DIR / output_name
    
    # Save clean data
    df_clean.to_csv(output_file, index=False)  # Save as CSV
    
    print(f"\n   ✅ Clean devices: {len(df_clean)}/{len(df_raw)}")
    print(f"   💾 Saved to: {output_file.name}")
    
    # Show clean device distribution
    if 'Device_Type' in df_clean.columns:
        print(f"\n   📊 Device types in clean data:")
        print(f"   {df_clean['Device_Type'].value_counts().to_string()}")

print(f"\n{'='*60}")
print("✅ BATCH PROCESSING COMPLETE!")
print(f"   Processed {len(excel_files)} file(s)")
print(f"   Clean files saved to: {OUTPUT_DIR}")
print("=" * 60)