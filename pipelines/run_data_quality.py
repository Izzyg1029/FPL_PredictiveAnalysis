# run_data_quality.py - WITH DEVICE-SPECIFIC VALIDATION

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

# Get ALL CSV files
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
        # Load data
        df_raw = pd.read_csv(device_file)
        print(f"Devices loaded: {len(df_raw):,}")
        
        # ========== DIAGNOSTICS ==========
        print(f"\nCOLUMN ANALYSIS:")
        print(f"   Total columns: {len(df_raw.columns)}")
        
        # Check for device type columns
        device_col = None
        for col in df_raw.columns:
            if 'device' in col.lower() and 'type' in col.lower():
                device_col = col
                break
        
        if device_col:
            print(f"\nDEVICE TYPES IN '{device_col}':")
            counts = df_raw[device_col].value_counts()
            for dev_type, count in counts.items():
                percentage = (count / len(df_raw)) * 100
                print(f"   {dev_type}: {count:,} ({percentage:.1f}%)")
        else:
            print(f"\nNo Device_Type column found")
            print(f"   Available columns: {list(df_raw.columns)[:10]}...")
            # Skip processing if no device type column
            print(f"\nSKIPPING {device_file.name} - No device type column")
            continue
        
        # ===== DEVICE-SPECIFIC CRITICAL COLUMNS =====
        print(f"\nCHECKING DEVICE-SPECIFIC CRITICAL COLUMNS...")
        
        # Create masks for each device type
        df_raw['temp_device_type'] = df_raw[device_col].astype(str).str.upper().str.strip()
        is_mm3 = df_raw['temp_device_type'].str.contains('MM3', na=False)
        is_zm1 = df_raw['temp_device_type'].str.contains('ZM1', na=False)
        is_um3 = df_raw['temp_device_type'].str.contains('UM3', na=False)
        
        # Track which rows are valid
        valid_mask = pd.Series(False, index=df_raw.index)
        rejection_reasons = pd.Series('', index=df_raw.index)
        
        # Helper function to find column
        def find_column(possible_names):
            for name in possible_names:
                for actual in df_raw.columns:
                    if name.lower() in actual.lower():
                        return actual
            return None
        
        # MM3 devices need current, temperature, and Last_Heard
        if is_mm3.any():
            mm3_cols = []
            mm3_needed = {
                'LineCurrent': ['LineCurrent', 'Current', 'LINE_CURRENT'],
                'LineTemperature': ['LineTemperature', 'Temperature', 'TEMP', 'LineTemperatrue'],
                'Last_Heard': ['Last_Heard', 'LAST_HEARD', 'LastHeard']
            }
            
            col_mapping = {}
            for field, possible_names in mm3_needed.items():
                found = find_column(possible_names)
                if found:
                    mm3_cols.append(found)
                    col_mapping[field] = found
                else:
                    print(f"   WARNING: No column found for {field}")
            
            if mm3_cols:
                mm3_valid = df_raw.loc[is_mm3, mm3_cols].notna().all(axis=1)
                valid_mask[is_mm3] = mm3_valid
                
                # Record rejection reasons for invalid MM3
                invalid_mm3 = is_mm3 & ~mm3_valid
                if invalid_mm3.any():
                    missing_cols = []
                    for col in mm3_cols:
                        missing = df_raw.loc[invalid_mm3, col].isna()
                        if missing.any():
                            missing_cols.append(col)
                    rejection_reasons[invalid_mm3] = "Missing critical data: " + ", ".join(missing_cols)
                
                print(f"   MM3: {mm3_valid.sum()}/{is_mm3.sum()} valid")
        
        # ZM1 devices need battery and Last_Heard
        if is_zm1.any():
            zm1_cols = []
            zm1_needed = {
                'BatteryLevel': ['BatteryLevel', 'BATTERY', 'Battery', 'battery_level'],
                'Last_Heard': ['Last_Heard', 'LAST_HEARD', 'LastHeard']
            }
            
            col_mapping = {}
            for field, possible_names in zm1_needed.items():
                found = find_column(possible_names)
                if found:
                    zm1_cols.append(found)
                    col_mapping[field] = found
            
            if zm1_cols:
                zm1_valid = df_raw.loc[is_zm1, zm1_cols].notna().all(axis=1)
                valid_mask[is_zm1] = zm1_valid
                
                # Record rejection reasons for invalid ZM1
                invalid_zm1 = is_zm1 & ~zm1_valid
                if invalid_zm1.any():
                    missing_cols = []
                    for col in zm1_cols:
                        missing = df_raw.loc[invalid_zm1, col].isna()
                        if missing.any():
                            missing_cols.append(col)
                    rejection_reasons[invalid_zm1] = "Missing critical data: " + ", ".join(missing_cols)
                
                print(f"   ZM1: {zm1_valid.sum()}/{is_zm1.sum()} valid")
        
        # UM3 devices only need Last_Heard
        if is_um3.any():
            um3_cols = []
            um3_needed = {
                'Last_Heard': ['Last_Heard', 'LAST_HEARD', 'LastHeard']
            }
            
            for field, possible_names in um3_needed.items():
                found = find_column(possible_names)
                if found:
                    um3_cols.append(found)
            
            if um3_cols:
                um3_valid = df_raw.loc[is_um3, um3_cols].notna().all(axis=1)
                valid_mask[is_um3] = um3_valid
                
                # Record rejection reasons for invalid UM3
                invalid_um3 = is_um3 & ~um3_valid
                if invalid_um3.any():
                    rejection_reasons[invalid_um3] = "Missing Last_Heard data"
                
                print(f"   UM3: {um3_valid.sum()}/{is_um3.sum()} valid")
        
                    # Clean data based on device-specific validation
        before = len(df_raw)
        df_clean = df_raw[valid_mask].copy()
        after = len(df_clean)
        removed = before - after
        
        print(f"\n   TOTAL VALID: {after}/{before} rows ({after/before*100:.1f}%)")
        
        # Create output filenames
        base_name = device_file.stem
        output_clean = OUTPUT_DIR / f"{base_name}-clean.csv"
        output_rejected = OUTPUT_DIR / f"{base_name}-rejected.csv"
        
        # Save clean file (this is the important part!)
        if 'temp_device_type' in df_clean.columns:
            df_clean = df_clean.drop(columns=['temp_device_type'])
        
        df_clean.to_csv(output_clean, index=False)
        print(f"\n SAVED CLEAN DATA: {output_clean.name}")
        print(f"   Size: {output_clean.stat().st_size / 1024:.1f} KB")
        print(f"   Columns: {len(df_clean.columns)}")
        
        # Skip rejected files for now to avoid errors
        if removed > 0:
            print(f"    Rejected {removed} rows (not saving rejected file)")
            # Rejected file code is temporarily disabled
            # We'll fix this later
            
        # Final summary by device type
        if device_col in df_clean.columns:
            print(f"\n FINAL CLEAN DATA DISTRIBUTION:")
            final_counts = df_clean[device_col].value_counts()
            for dev_type, count in final_counts.items():
                pct = (count / after) * 100
                print(f"   {dev_type}: {count:,} ({pct:.1f}%)")
        
        print(f"\n SUCCESS: {device_file.name} processed")
        
    except Exception as e:
        print(f"\n ERROR processing {device_file.name}:")
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

