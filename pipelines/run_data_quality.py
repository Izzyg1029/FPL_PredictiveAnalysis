# run_data_quality.py (FIXED FOR CSV FILES)

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import re

# --------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # pipelines folder
PROJECT_ROOT = SCRIPT_DIR.parent              # project root
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "daily"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DATA QUALITY PIPELINE - PROCESSING ALL FILES")
print("=" * 60)
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Get ALL CSV files (FIXED: look for CSV instead of Excel)
csv_files = sorted(list(INPUT_DIR.glob("*.csv")))
print(f"\nFound {len(csv_files)} CSV file(s):")
for f in csv_files:
    print(f"   - {f.name}")

if not csv_files:
    print("No CSV files found!")
    print(f"   Place CSV files in: {INPUT_DIR}")
    sys.exit(1)

# Process each file
for device_file in csv_files:
    print(f"\n{'='*60}")
    print(f"PROCESSING: {device_file.name}")
    print(f"{'='*60}")
    
    try:
        # Load data (FIXED: use read_csv instead of read_excel)
        df_raw = pd.read_csv(device_file)
        print(f"Devices loaded: {len(df_raw):,}")
        
        # ========== DIAGNOSTICS ==========
        print(f"\nCOLUMN ANALYSIS:")
        print(f"   Total columns: {len(df_raw.columns)}")
        
        # Check for device type columns
        device_cols = [col for col in df_raw.columns if 'device' in col.lower() and 'type' in col.lower()]
        if device_cols:
            device_col = device_cols[0]
            print(f"\nDEVICE TYPES IN '{device_col}':")
            counts = df_raw[device_col].value_counts()
            for dev_type, count in counts.items():
                percentage = (count / len(df_raw)) * 100
                print(f"   {dev_type}: {count:,} ({percentage:.1f}%)")
        else:
            print(f"\nNo Device_Type column found")
            print(f"   Available columns: {list(df_raw.columns)[:10]}...")
        
        # Check critical columns
        CRITICAL_COLS = []
        col_mapping = {}
        for expected in ["Device_Type", "Last_Heard", "LineCurrent", "LineTemperature", "Serial"]:
            for actual in df_raw.columns:
                if expected.lower() in actual.lower():
                    CRITICAL_COLS.append(actual)
                    col_mapping[expected] = actual
                    break
        
        print(f"\nCRITICAL COLUMNS FOUND:")
        for expected, actual in col_mapping.items():
            print(f"   {expected}: '{actual}'")
        
        # Clean data
        print(f"\nCLEANING DATA...")
        if CRITICAL_COLS:
            # Remove rows with missing critical data
            before = len(df_raw)
            clean_mask = df_raw[CRITICAL_COLS].notna().all(axis=1)
            df_clean = df_raw[clean_mask].copy()
            after = len(df_clean)
            removed = before - after
            
            print(f"   Removed {removed:,} rows with missing critical data")
            print(f"   Kept {after:,} rows ({after/before*100:.1f}%)")
            
            # Add rejection reasons for removed rows
            if removed > 0:
                df_rejected = df_raw[~clean_mask].copy()
                df_rejected["rejection_reason"] = df_rejected.apply(
                    lambda r: ", ".join([col for col in CRITICAL_COLS if pd.isna(r[col])]),
                    axis=1
                )
        else:
            df_clean = df_raw.copy()
            print(f"   No critical columns found - keeping all data")
        
        # Create output filenames
        base_name = device_file.stem
        output_clean = OUTPUT_DIR / f"{base_name}-clean.csv"
        output_rejected = OUTPUT_DIR / f"{base_name}-rejected.csv"
        
        # Save files
        df_clean.to_csv(output_clean, index=False)
        print(f"\nSAVED CLEAN DATA: {output_clean.name}")
        print(f"   Size: {output_clean.stat().st_size / 1024:.1f} KB")
        
        if removed > 0:
            df_rejected.to_csv(output_rejected, index=False)
            print(f"SAVED REJECTED DATA: {output_rejected.name}")
        
        # Final summary
        if device_cols and device_col in df_clean.columns:
            print(f"\nFINAL CLEAN DATA DISTRIBUTION:")
            final_counts = df_clean[device_col].value_counts()
            for dev_type in ['ZM1', 'UM3', 'MM3']:
                if dev_type in final_counts.index:
                    print(f"   {dev_type}: {final_counts[dev_type]:,}")
        
        print(f"\nSUCCESS: {device_file.name} processed")
        
    except Exception as e:
        print(f"\nERROR processing {device_file.name}:")
        print(f"   {str(e)[:200]}...")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("DATA QUALITY PIPELINE COMPLETE!")
print(f"   Processed {len(csv_files)} file(s)")
print(f"   Clean files saved to: {OUTPUT_DIR}")
print("=" * 60)
print("\nReady for health features pipeline!")
print("   Run: python run_health_features.py")