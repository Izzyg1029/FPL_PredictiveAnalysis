# pipelines/process_daily_time_series.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path to import feature_health
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from feature_health import build_health_features
    print("✅ Successfully imported health_features module")
except ImportError as e:
    print(f"❌ Failed to import health_features: {e}")
    sys.exit(1)

def process_daily_time_series():
    """
    Pipeline to process daily ZM1 data into a time series dataset.
    """
    print("=" * 70)
    print("DAILY TIME SERIES PROCESSING PIPELINE")
    print("=" * 70)
    
    # Paths (relative to project root) - UPDATED FOR NEW STRUCTURE
    RAW_DAILY_DIR = project_root / "data" / "raw" / "daily"
    CLEAN_DAILY_DIR = project_root / "data" / "clean" / "daily"
    TIME_SERIES_DIR = project_root / "data" / "clean" / "time_series"
    PROCESSED_TS_DIR = project_root / "data" / "processed" / "time_series"
    
    # Create directories
    for directory in [CLEAN_DAILY_DIR, TIME_SERIES_DIR, PROCESSED_TS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Load install dates
    install_file = project_root / "data" / "clean" / "install_dates.csv"
    if install_file.exists():
        install_df = pd.read_csv(install_file)
        print(f"📅 Loaded {len(install_df)} install records")
    else:
        install_df = None
        print("⚠️  No install dates file found - device age features will be limited")
    
    # Get all daily files
    # Get all daily files (BOTH CSV AND EXCEL)
    excel_files = sorted(RAW_DAILY_DIR.glob("*.xlsx"))
    csv_files = sorted(RAW_DAILY_DIR.glob("*.csv"))
    daily_files = list(excel_files) + list(csv_files)

    print(f"📁 Found {len(daily_files)} daily files in {RAW_DAILY_DIR}")
    print(f"   Excel files: {len(excel_files)}, CSV files: {len(csv_files)}")
    if not daily_files:
        print("❌ No daily CSV files found!")
        print(f"   Expected path: {RAW_DAILY_DIR}")
        print(f"   Files should be named like: 2024-01-01.csv, 2024-01-02.csv, etc.")
        return
    
    all_daily_data = []
    processing_stats = []
    
    # Process each daily file
    for i, daily_file in enumerate(daily_files):
        print(f"\n[{i+1}/{len(daily_files)}] Processing: {daily_file.name}")
        
        try:
         # Load daily data (support both Excel and CSV)
            if daily_file.suffix.lower() == '.xlsx':
                df_daily = pd.read_excel(daily_file)
            else:  # .csv
                df_daily = pd.read_csv(daily_file)
            # Extract date from filename (format: YYYY-MM-DD.csv)
            date_from_filename = daily_file.stem
         
            
            # Check if we have ZM1 devices
            if 'Device_Type' not in df_daily.columns:
                print(f"   ⚠️  No 'Device_Type' column, checking for device type in other columns...")
                # Look for device type in other columns
                type_cols = [col for col in df_daily.columns if 'type' in col.lower() or 'device' in col.lower()]
                if type_cols:
                    df_daily = df_daily.rename(columns={type_cols[0]: 'Device_Type'})
                    print(f"   🔧 Renamed '{type_cols[0]}' to 'Device_Type'")
            
            # Filter to ZM1 devices only
            zm1_mask = df_daily['Device_Type'].astype(str).str.contains('ZM1', case=False, na=False)
            df_zm1 = df_daily[zm1_mask].copy()
            
            if len(df_zm1) == 0:
                print(f"   ⚠️  No ZM1 devices found in this file")
                # Show what device types are present
                device_types = df_daily['Device_Type'].unique()[:5]
                print(f"   Available device types: {device_types}")
                continue
            
            print(f"   🔋 Found {len(df_zm1)} ZM1 devices")
            
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
            clean_file = CLEAN_DAILY_DIR / f"{daily_file.stem}-clean.csv"
            df_health.to_csv(clean_file, index=False)
            
            # Add to combined time series
            all_daily_data.append(df_health)
            
            # Record stats
            processing_stats.append({
                'date': date_from_filename,
                'file': daily_file.name,
                'total_rows': len(df_daily),
                'zm1_rows': len(df_zm1),
                'health_features': len(df_health.columns),
                'processed': True
            })
            
            print(f"   ✅ Saved to: {clean_file}")
            print(f"   📈 Generated {len(df_health.columns)} health features")
            
        except Exception as e:
            print(f"   ❌ Error processing {daily_file.name}: {str(e)[:100]}...")
            processing_stats.append({
                'date': daily_file.stem if daily_file else 'unknown',
                'file': daily_file.name,
                'total_rows': 0,
                'zm1_rows': 0,
                'health_features': 0,
                'processed': False,
                'error': str(e)[:200]
            })
    
    # Create combined time series
    if all_daily_data:
        print("\n" + "=" * 70)
        print("CREATING COMBINED TIME SERIES")
        print("=" * 70)
        
        time_series_df = pd.concat(all_daily_data, ignore_index=True)
        
        # Sort by device and date
        time_series_df = time_series_df.sort_values(['Serial', 'date'])
        
        # Save raw combined time series (in clean/time_series/)
        raw_ts_file = TIME_SERIES_DIR / "zmi_daily_time_series.csv"
        time_series_df.to_csv(raw_ts_file, index=False)
        
        # Add time-based features
        print("   ⏳ Adding time-based features...")
        time_series_df = add_time_based_features(time_series_df)
        
        # Save enhanced time series (in processed/time_series/)
        enhanced_ts_file = PROCESSED_TS_DIR / "zmi_daily_health_time_series.csv"
        time_series_df.to_csv(enhanced_ts_file, index=False)
        
        # Save statistics (in clean/time_series/)
        stats_df = pd.DataFrame(processing_stats)
        stats_file = TIME_SERIES_DIR / "daily_processing_stats.csv"  # Updated name
        stats_df.to_csv(stats_file, index=False)
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"✅ Raw time series: {raw_ts_file} ({len(time_series_df):,} records)")
        print(f"✅ Enhanced time series: {enhanced_ts_file}")
        print(f"✅ Processing stats: {stats_file}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Unique ZM1 devices: {time_series_df['Serial'].nunique()}")
        print(f"   Date range: {time_series_df['date'].min()} to {time_series_df['date'].max()}")
        print(f"   Total days processed: {time_series_df['date'].nunique()}")
        print(f"   Successfully processed: {sum(stats_df['processed'])}/{len(stats_df)} days")
        
    else:
        print("\n❌ No ZM1 data was processed!")
        print("   Check if your daily files contain ZM1 devices")

def add_time_based_features(df):
    """Add time-based features to the time series."""
    df = df.copy()
    
    # Ensure sorted by device and date
    df = df.sort_values(['Serial', 'date'])
    
    # Calculate daily changes
    df['battery_change_1d'] = df.groupby('Serial')['battery_level'].diff()
    df['temp_change_1d'] = df.groupby('Serial')['LineTemperature_val'].diff()
    
    # Rolling averages (3, 7, 14 days)
    for window in [3, 7, 14]:
        df[f'battery_avg_{window}d'] = df.groupby('Serial')['battery_level'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
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