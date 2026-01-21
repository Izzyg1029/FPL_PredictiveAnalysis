# pipelines/process_daily_time_series.py (ZM1 ONLY VERSION)
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import re
import zipfile  # ADD THIS FOR ZIP SUPPORT
import tempfile
import shutil

# Add project root to path to import feature_health
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from feature_health import build_health_features
    print("✅ Successfully imported health_features module")
except ImportError as e:
    print(f"❌ Failed to import health_features: {e}")
    sys.exit(1)

def extract_zip_files(raw_dir):
    """
    Extract CSV and Excel files from ZIP archives in the raw directory.
    Returns number of files extracted.
    """
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        return 0
    
    print(f"\n📦 Found {len(zip_files)} ZIP file(s):")
    for zf in zip_files:
        print(f"   • {zf.name}")
    
    extracted_count = 0
    
    for zip_file in zip_files:
        print(f"\n   📦 Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Get list of files in ZIP
                file_list = zf.namelist()
                csv_excel_files = [f for f in file_list if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
                
                if not csv_excel_files:
                    print(f"      No CSV or Excel files found in {zip_file.name}")
                    continue
                
                print(f"      Contains {len(csv_excel_files)} CSV/Excel file(s)")
                
                # Extract each CSV/Excel file
                for file_in_zip in csv_excel_files:
                    # Skip directories
                    if file_in_zip.endswith('/'):
                        continue
                    
                    # Get the filename without path
                    filename = Path(file_in_zip).name
                    
                    # Check if file already exists
                    target_file = raw_dir / filename
                    
                    # Handle duplicates by adding _1, _2, etc.
                    counter = 1
                    original_stem = Path(filename).stem
                    original_suffix = Path(filename).suffix
                    
                    while target_file.exists():
                        filename = f"{original_stem}_{counter}{original_suffix}"
                        target_file = raw_dir / filename
                        counter += 1
                    
                    # Extract the file
                    zf.extract(file_in_zip, raw_dir)
                    
                    # If extracted to subdirectory, move it
                    extracted_path = raw_dir / file_in_zip
                    if extracted_path != target_file:
                        if extracted_path.exists():
                            extracted_path.rename(target_file)
                            # Clean up empty directories
                            try:
                                extracted_path.parent.rmdir()
                            except:
                                pass
                    
                    print(f"      ✓ Extracted: {filename}")
                    extracted_count += 1
                
                print(f"   ✅ Successfully extracted {len(csv_excel_files)} files from {zip_file.name}")
                
        except Exception as e:
            print(f"      ❌ Error extracting {zip_file.name}: {str(e)[:100]}")
    
    return extracted_count

def process_daily_time_series():
    """
    Pipeline to process daily ZM1 data into a time series dataset.
    Supports: CSV, Excel (.xlsx, .xls), and ZIP files containing CSV/Excel.
    """
    print("=" * 70)
    print("DAILY TIME SERIES PROCESSING PIPELINE - ZM1 ONLY")
    print("=" * 70)
    
    # Paths (relative to project root) - UPDATED FOR NEW STRUCTURE
    RAW_DAILY_DIR = project_root / "data" / "raw" / "daily"
    CLEAN_DAILY_DIR = project_root / "data" / "clean" / "daily"
    TIME_SERIES_DIR = project_root / "data" / "clean" / "time_series"
    PROCESSED_TS_DIR = project_root / "data" / "processed" / "time_series"
    
    # Create directories
    for directory in [CLEAN_DAILY_DIR, TIME_SERIES_DIR, PROCESSED_TS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Extract any ZIP files first
    print("\n" + "=" * 50)
    print("STEP 1: CHECKING FOR ZIP FILES")
    print("=" * 50)
    
    extracted_count = extract_zip_files(RAW_DAILY_DIR)
    if extracted_count > 0:
        print(f"\n✅ Extracted {extracted_count} files from ZIP archives")
    
    # STEP 2: Load install dates
    print("\n" + "=" * 50)
    print("STEP 2: LOADING INSTALLATION DATA")
    print("=" * 50)
    
    install_file = project_root / "data" / "clean" / "install_dates.csv"
    if install_file.exists():
        install_df = pd.read_csv(install_file)
        print(f"📅 Loaded {len(install_df)} install records")
    else:
        install_df = None
        print("⚠️  No install dates file found - device age features will be limited")
    
    # STEP 3: Find all CSV and Excel files
    print("\n" + "=" * 50)
    print("STEP 3: FINDING DATA FILES")
    print("=" * 50)
    
    print(f"📁 Looking for files in: {RAW_DAILY_DIR}")
    
    # Get all CSV and Excel files (including those extracted from ZIPs)
    excel_files = sorted(RAW_DAILY_DIR.glob("*.xlsx")) + sorted(RAW_DAILY_DIR.glob("*.xls"))
    csv_files = sorted(RAW_DAILY_DIR.glob("*.csv"))
    daily_files = list(excel_files) + list(csv_files)
    
    print(f"📁 Found {len(daily_files)} data files ready for processing")
    print(f"   Excel files: {len(excel_files)} (.xlsx/.xls)")
    print(f"   CSV files: {len(csv_files)}")
    
    # Show what files were found
    if daily_files:
        print("\n   Files to process:")
        for f in daily_files[:15]:  # Show first 15 files
            print(f"   • {f.name}")
        if len(daily_files) > 15:
            print(f"   ... and {len(daily_files) - 15} more")
    else:
        print("❌ No data files found!")
        print(f"   Check path: {RAW_DAILY_DIR}")
        print(f"   Supported formats: .csv, .xlsx, .xls, or .zip containing these formats")
        return
    
    # STEP 4: Process each file
    print("\n" + "=" * 50)
    print("STEP 4: PROCESSING FILES")
    print("=" * 50)
    
    all_daily_data = []
    processing_stats = []
    
    for i, daily_file in enumerate(daily_files):
        print(f"\n[{i+1}/{len(daily_files)}] Processing: {daily_file.name}")
        
        try:
            # Load data based on file type
            if daily_file.suffix.lower() in ['.xlsx', '.xls']:
                df_daily = pd.read_excel(daily_file)
                print(f"   📊 Loaded Excel file: {len(df_daily)} rows")
            else:  # .csv
                df_daily = pd.read_csv(daily_file)
                print(f"   📊 Loaded CSV file: {len(df_daily)} rows")
            
            # Extract date from filename using regex
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', daily_file.stem)
            if date_match:
                date_from_filename = date_match.group(1)
                print(f"   📅 Extracted date: {date_from_filename}")
            else:
                # Fallback: use entire filename stem
                date_from_filename = daily_file.stem
                print(f"   ⚠️  Could not extract date from filename, using: {date_from_filename}")
            
            # Check if output already exists, skip if it does
            output_file = CLEAN_DAILY_DIR / f"{date_from_filename}_health_zm1only.csv"
            if output_file.exists():
                print(f"   ⏭️  Skipping {daily_file.name} - output already exists")
                
                # Still record stats for skipped file
                processing_stats.append({
                    'date': date_from_filename,
                    'file': daily_file.name,
                    'total_rows': len(df_daily),
                    'zm1_rows': 0,
                    'other_rows': 0,
                    'zm1_percentage': 0,
                    'health_features': 0,
                    'processed': False,
                    'output_file': output_file.name,
                    'skip_reason': 'output_already_exists'
                })
                continue
            
            # Check if we have Device_Type column
            if 'Device_Type' not in df_daily.columns:
                print(f"   ⚠️  No 'Device_Type' column, checking for device type in other columns...")
                type_cols = [col for col in df_daily.columns if 'type' in col.lower() or 'device' in col.lower()]
                if type_cols:
                    df_daily = df_daily.rename(columns={type_cols[0]: 'Device_Type'})
                    print(f"   🔧 Renamed '{type_cols[0]}' to 'Device_Type'")
            
            # Show ALL device types before filtering
            if 'Device_Type' in df_daily.columns:
                print(f"   📊 ALL Device Types in file:")
                counts = df_daily['Device_Type'].value_counts()
                for dev_type, count in counts.items():
                    percentage = (count / len(df_daily)) * 100
                    print(f"      {dev_type}: {count} ({percentage:.1f}%)")
            
            # Filter to ZM1 devices only
            zm1_mask = df_daily['Device_Type'].astype(str).str.contains('ZM1', case=False, na=False)
            df_zm1 = df_daily[zm1_mask].copy()
            
            if len(df_zm1) == 0:
                print(f"   ⚠️  No ZM1 devices found in this file - skipping")
                processing_stats.append({
                    'date': date_from_filename,
                    'file': daily_file.name,
                    'total_rows': len(df_daily),
                    'zm1_rows': 0,
                    'other_rows': len(df_daily),
                    'zm1_percentage': 0,
                    'health_features': 0,
                    'processed': False,
                    'output_file': None,
                    'skip_reason': 'no_zm1_devices'
                })
                continue
            
            print(f"   🔋 Filtered to {len(df_zm1)} ZM1 devices ({(len(df_zm1)/len(df_daily))*100:.1f}% of total)")
            
            # Add date column if not present
            if 'date' not in df_zm1.columns:
                df_zm1['date'] = date_from_filename
                df_zm1['timestamp'] = pd.to_datetime(date_from_filename)
            
            # Build health features
            print(f"   ⚙️  Building health features...")
            df_health = build_health_features(df_zm1, install_df)
            
            # Ensure date is preserved
            if 'date' not in df_health.columns:
                df_health['date'] = date_from_filename
            
            # Save cleaned daily file
            clean_file = CLEAN_DAILY_DIR / f"{date_from_filename}_health_zm1only.csv"
            df_health.to_csv(clean_file, index=False)
            
            # Add to combined time series
            all_daily_data.append(df_health)
            
            # Record stats
            processing_stats.append({
                'date': date_from_filename,
                'file': daily_file.name,
                'total_rows': len(df_daily),
                'zm1_rows': len(df_zm1),
                'other_rows': len(df_daily) - len(df_zm1),
                'zm1_percentage': (len(df_zm1) / len(df_daily)) * 100,
                'health_features': len(df_health.columns),
                'processed': True,
                'output_file': clean_file.name
            })
            
            print(f"   ✅ Saved ZM1-only file: {clean_file.name}")
            print(f"   📈 Generated {len(df_health.columns)} health features")
            
        except Exception as e:
            print(f"   ❌ Error processing {daily_file.name}: {str(e)[:100]}...")
            import traceback
            traceback.print_exc()
            processing_stats.append({
                'date': daily_file.stem if daily_file else 'unknown',
                'file': daily_file.name,
                'total_rows': 0,
                'zm1_rows': 0,
                'other_rows': 0,
                'zm1_percentage': 0,
                'health_features': 0,
                'processed': False,
                'output_file': None,
                'error': str(e)[:200]
            })
    
    # STEP 5: Create combined time series
    print("\n" + "=" * 50)
    print("STEP 5: CREATING TIME SERIES")
    print("=" * 50)
    
    if all_daily_data:
        time_series_df = pd.concat(all_daily_data, ignore_index=True)
        
        # Sort by device and date
        time_series_df = time_series_df.sort_values(['Serial', 'date'])
        
        # Save raw combined time series
        raw_ts_file = TIME_SERIES_DIR / "zm1_daily_time_series.csv"
        time_series_df.to_csv(raw_ts_file, index=False)
        
        # Add time-based features
        print("   ⏳ Adding time-based features...")
        time_series_df = add_time_based_features(time_series_df)
        
        # Get date range for the combined file name
        min_date = time_series_df['date'].min()
        max_date = time_series_df['date'].max()
        
        # Save enhanced time series with date range in filename
        enhanced_ts_file = PROCESSED_TS_DIR / f"{min_date}_to_{max_date}_health_zm1only_timeseries.csv"
        time_series_df.to_csv(enhanced_ts_file, index=False)
        
        # Save statistics
        stats_df = pd.DataFrame(processing_stats)
        stats_file = TIME_SERIES_DIR / "daily_processing_stats.csv"
        stats_df.to_csv(stats_file, index=False)
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"✅ Raw ZM1 time series: {raw_ts_file} ({len(time_series_df):,} records)")
        print(f"✅ Enhanced ZM1 time series: {enhanced_ts_file}")
        print(f"✅ Processing stats: {stats_file}")
        
        # Summary
        print(f"\n📊 ZM1-ONLY SUMMARY:")
        print(f"   Unique ZM1 devices: {time_series_df['Serial'].nunique()}")
        print(f"   Date range: {min_date} to {max_date}")
        print(f"   Total days processed: {time_series_df['date'].nunique()}")
        
        # Show processing results
        processed_count = sum(1 for s in processing_stats if s.get('processed') == True)
        skipped_count = sum(1 for s in processing_stats if s.get('processed') == False)
        print(f"   Successfully processed: {processed_count}/{len(processing_stats)} days")
        print(f"   Skipped: {skipped_count} days")
        
        # Show output files
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   Clean daily files: {processed_count} files in {CLEAN_DAILY_DIR}")
        print(f"   Time series: {enhanced_ts_file.name}")
        
        # Show overall ZM1 statistics
        if len(processing_stats) > 0:
            total_raw = sum([s['total_rows'] for s in processing_stats if s.get('processed') == True])
            total_zm1 = sum([s['zm1_rows'] for s in processing_stats if s.get('processed') == True])
            percentages = [s['zm1_percentage'] for s in processing_stats if s.get('processed') == True]
            if percentages:
                avg_percentage = np.mean(percentages)
                print(f"\n📈 ZM1 EXTRACTION STATS:")
                print(f"   Total raw devices processed: {total_raw:,}")
                print(f"   Total ZM1 devices extracted: {total_zm1:,}")
                print(f"   Average ZM1 percentage: {avg_percentage:.1f}%")
        
    else:
        print("\n❌ No ZM1 data was processed!")
        print("   Check if your daily files contain ZM1 devices")

def add_time_based_features(df):
    """Add time-based features to the time series."""
    df = df.copy()
    
    # Ensure sorted by device and date
    df = df.sort_values(['Serial', 'date'])
    
    # Calculate daily changes
    if 'battery_level' in df.columns:
        df['battery_change_1d'] = df.groupby('Serial')['battery_level'].diff()
    
    if 'LineTemperature_val' in df.columns:
        df['temp_change_1d'] = df.groupby('Serial')['LineTemperature_val'].diff()
    
    # Rolling averages (3, 7, 14 days)
    for window in [3, 7, 14]:
        if 'battery_level' in df.columns:
            df[f'battery_avg_{window}d'] = df.groupby('Serial')['battery_level'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        if 'LineTemperature_val' in df.columns:
            df[f'temp_avg_{window}d'] = df.groupby('Serial')['LineTemperature_val'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Days since last event flags
    for flag in ['overheat_flag', 'zero_current_flag', 'battery_low_flag']:
        if flag in df.columns:
            df[f'days_since_{flag}'] = df.groupby('Serial').apply(
                lambda grp: (pd.to_datetime(grp['date']) - 
                           pd.to_datetime(grp.loc[grp[flag] == 1, 'date']).max()).dt.days
                if (grp[flag] == 1).any() else np.nan
            ).reset_index(level=0, drop=True)
    
    return df

if __name__ == "__main__":
    process_daily_time_series()