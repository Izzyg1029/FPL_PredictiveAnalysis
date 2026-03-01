# pipelines/process_daily_time_series.py (ALL DEVICES VERSION) - NO EMOJIS
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import re
import zipfile
import tempfile
import shutil

# Add project root to path to import feature_health
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from feature_health import build_health_features
    print("Successfully imported health_features module")
except ImportError as e:
    print(f"Failed to import health_features: {e}")
    sys.exit(1)

def extract_zip_files(raw_dir):
    """
    Extract CSV and Excel files from ZIP archives in the raw directory.
    Returns number of files extracted.
    """
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        return 0
    
    print(f"\nFound {len(zip_files)} ZIP file(s):")
    for zf in zip_files:
        print(f"   - {zf.name}")
    
    extracted_count = 0
    
    for zip_file in zip_files:
        print(f"\n   Extracting: {zip_file.name}")
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
                    
                    print(f"      Extracted: {filename}")
                    extracted_count += 1
                
                print(f"   Successfully extracted {len(csv_excel_files)} files from {zip_file.name}")
                
        except Exception as e:
            print(f"      Error extracting {zip_file.name}: {str(e)[:100]}")
    
    return extracted_count

def calculate_daily_battery_trend(battery_df):
    """
    Calculate daily battery drain trends for devices.
    """
    import numpy as np
    import pandas as pd
    
    trends = []
    
    # Check what date column exists
    date_col = None
    for col in ['Date', 'date', 'Last_Heard', 'Last_Heard_dt', 'timestamp', 'observation_date']:
        if col in battery_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("ERROR: No date column found in battery_df")
        print(f"Available columns: {battery_df.columns.tolist()}")
        return pd.DataFrame()
    
    print(f"Using '{date_col}' as date column for trend analysis")
    
    # ===== CRITICAL FIX: CONVERT DATES TO DATETIME =====
    battery_df = battery_df.copy()
    battery_df[date_col] = pd.to_datetime(battery_df[date_col], errors='coerce')
    
    # Also check if BatteryLevel column exists
    if 'BatteryLevel' not in battery_df.columns:
        print("ERROR: No BatteryLevel column found in battery_df")
        return pd.DataFrame()
    
    processed_count = 0
    error_count = 0
    
    for serial, group in battery_df.groupby('Serial'):
        if len(group) < 2:
            continue  # Need at least 2 data points
            
        group = group.sort_values(date_col)  # Use the CORRECT column name
        
        # Extract dates and battery levels
        dates = group[date_col].values  # These should now be datetime objects
        battery_levels = group['BatteryLevel'].values
        
        # Skip if dates are invalid
        if pd.isna(dates[0]) or pd.isna(dates[-1]):
            error_count += 1
            continue
            
        # Calculate days difference safely
        try:
            # Now dates should be datetime objects
            days_diff = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
        except (TypeError, AttributeError) as e:
            error_count += 1
            print(f"Error calculating days diff for {serial}: {e}")
            print(f"Date types: {type(dates[0])}, {type(dates[-1])}")
            print(f"Date values: '{dates[0]}', '{dates[-1]}'")
            continue
            
        if days_diff <= 0:
            continue  # Invalid time range
            
        # Calculate daily drain rate
        battery_diff = battery_levels[-1] - battery_levels[0]
        daily_drain_rate = battery_diff / days_diff
        
        # Linear regression for trend
        try:
            # Convert dates to numeric (days since first date)
            days_since_start = (dates - dates[0]) / np.timedelta64(1, 'D')
            slope, intercept = np.polyfit(days_since_start, battery_levels, 1)
        except:
            slope, intercept = np.nan, np.nan
        
        trend_info = {
            'Serial': serial,
            'data_points': len(group),
            'date_range_days': days_diff,
            'battery_start': battery_levels[0],
            'battery_end': battery_levels[-1],
            'battery_change': battery_diff,
            'daily_drain_rate': daily_drain_rate,
            'trend_slope': slope,
            'trend_intercept': intercept,
            'first_date': dates[0],
            'last_date': dates[-1]
        }
        
        trends.append(trend_info)
        processed_count += 1
    
    print(f"\nTrend analysis completed: {processed_count} devices processed, {error_count} errors")
    
    return pd.DataFrame(trends)

def _calculate_drain_rate_from_trend(device_data, battery_clean, battery_series):
    """
    Calculate battery drain rate from trend data
    """
    # Need at least 2 data points
    if len(battery_clean) < 2:
        return 0.027  # Default
    
    try:
        valid_indices = battery_clean.index
        
        # Convert dates to datetime - FIXED VERSION
        battery_dates = pd.to_datetime(device_data.loc[valid_indices, 'date'], errors='coerce')
        
        # Filter out invalid dates
        valid_date_mask = battery_dates.notna()
        if valid_date_mask.sum() < 2:
            return 0.027  # Not enough valid dates
        
        battery_dates = battery_dates[valid_date_mask]
        battery_clean_filtered = battery_clean[valid_date_mask]
        
        # Sort by date
        sorted_indices = battery_dates.argsort()
        sorted_dates = battery_dates.iloc[sorted_indices]
        sorted_battery = battery_clean_filtered.iloc[sorted_indices]
        
        # Now dates are datetime objects, this should work
        days_since_first = (sorted_dates - sorted_dates.iloc[0]).dt.days
        
        if days_since_first.iloc[-1] <= 0:
            return 0.027  # No time difference
        
        # Calculate drain rate
        if len(days_since_first.unique()) > 1:
            # Linear regression for better accuracy
            slope, intercept = np.polyfit(days_since_first.values, sorted_battery.values, 1)
            drain_rate = abs(slope)
        else:
            # Simple calculation
            total_drain = sorted_battery.iloc[-1] - sorted_battery.iloc[0]
            total_days = days_since_first.iloc[-1]
            drain_rate = abs(total_drain / total_days) if total_days > 0 else 0.027
        
        # Cap at reasonable values (0.001% to 0.5% per day)
        return max(0.001, min(0.5, drain_rate))
        
    except Exception as e:
        # If any error, return default
        return 0.027
    
def create_device_profiles(time_series_df):
    """
    Create a device summary file for ALL device types (ZM1, MM3, UM3).
    One row per device with device-specific metrics.
    """
    print("\n" + "=" * 50)
    print("CREATING DEVICE PROFILES (ALL DEVICE TYPES)")
    print("=" * 50)
    
    # Ensure we have the time series data
    if len(time_series_df) == 0:
        print("No time series data available for device profiles")
        return None
    
    device_profiles = []
    
    # Device type specifications
    device_specs = {
        "ZM1": {
            "battery_type": "Non-rechargeable",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 100 / (10 * 365),  # ~0.027%/day
            "measures_current": False,
            "measures_temp": False,
            "measures_battery": True,
            "critical_battery": 20,
            "warning_battery": 30
        },
        "UM3": {
            "battery_type": "Line-powered or long-life",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 0,  # No battery or minimal drain
            "measures_current": False,
            "measures_temp": False,
            "measures_battery": False,
            "critical_battery": 0,
            "warning_battery": 0
        },
        "MM3": {
            "battery_type": "Line-powered (NOT rechargeable)",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 0,  # Line-powered, no battery drain
            "measures_current": True,
            "measures_temp": True,
            "measures_battery": False,
            "critical_battery": 0,
            "warning_battery": 0
        }
    }
    
    # Group by device
    for serial, device_data in time_series_df.groupby('Serial'):
        # Sort by date (most recent first)
        device_data = device_data.sort_values('date', ascending=False)
        
        # Get device type
        device_type = device_data['Device_Type'].iloc[0] if 'Device_Type' in device_data.columns else 'Unknown'
        if not isinstance(device_type, str):
            device_type = 'Unknown'
        
        # Standardize device type name
        device_type_upper = device_type.upper()
        if 'ZM1' in device_type_upper:
            standardized_type = 'ZM1'
            specs = device_specs['ZM1']
        elif 'UM3' in device_type_upper:
            standardized_type = 'UM3'
            specs = device_specs['UM3']
        elif 'MM3' in device_type_upper:
            standardized_type = 'MM3'
            specs = device_specs['MM3']
        else:
            standardized_type = 'Unknown'
            specs = device_specs['ZM1']  # Default to ZM1
        
        # Basic device info
        profile_dict = {
            'Serial': serial,
            'Device_Type': device_type,
            'Device_Type_Standardized': standardized_type,
            'first_seen_date': device_data['date'].min(),
            'last_seen_date': device_data['date'].max(),
            'total_days_observed': device_data['date'].nunique(),
            'total_records': len(device_data)
        }
        
        # Add device specifications
        profile_dict.update({
            'battery_type': specs['battery_type'],
            'expected_lifetime_years': specs['expected_lifetime_years'],
            'measures_current': specs['measures_current'],
            'measures_temp': specs['measures_temp'],
            'measures_battery': specs['measures_battery'],
            'critical_battery_level': specs['critical_battery'],
            'warning_battery_level': specs['warning_battery']
        })
        
        # If install date exists
        if 'install_date' in device_data.columns:
            profile_dict['install_date'] = device_data['install_date'].iloc[0]
            if 'device_age_days' in device_data.columns:
                profile_dict['device_age_days'] = device_data['device_age_days'].iloc[0]
        
        # === DEVICE-SPECIFIC ANALYSIS ===
        
        # ZM1: Battery analysis
        if standardized_type == 'ZM1':
            # Try different battery column names
            battery_col = None
            for col in device_data.columns:
                if 'battery' in col.lower():
                    battery_col = col
                    break

            if battery_col:
                # Clean battery data
                battery_series = pd.to_numeric(device_data[battery_col], errors='coerce')
                battery_clean = battery_series[(battery_series >= 0) & (battery_series <= 100)]

                if len(battery_clean) > 0:
                    profile_dict['battery_current'] = battery_clean.iloc[0]
                    profile_dict['battery_min'] = battery_clean.min()
                    profile_dict['battery_max'] = battery_clean.max()
                    profile_dict['battery_avg'] = battery_clean.mean()
                    profile_dict['battery_std'] = battery_clean.std()

                    # Data quality
                    total_readings = len(battery_series)
                    valid_readings = len(battery_clean)
                    profile_dict['battery_data_quality_pct'] = (valid_readings / total_readings * 100) if total_readings > 0 else 0

                    # ====GET REAL DRAIN RATES FROM HEALTH FEATURES ======
            
                    # OPTION 1: Get from device_data (health features already calculated)
                    # Check for existing drain rate columns from feature_health.py
                    drain_rate_found = False
            
                    # Try battery_drain_rate_per_day first (daily rate)
                    if 'battery_drain_rate_per_day' in device_data.columns:
                        drain_rate = device_data['battery_drain_rate_per_day'].iloc[0]
                        if pd.notna(drain_rate) and drain_rate > 0:
                            profile_dict['battery_drain_rate_per_day'] = drain_rate
                            drain_rate_found = True
                            print(f"DEBUG {serial}: Using battery_drain_rate_per_day = {drain_rate:.4f}%/day")
            
                    # Try battery_drain_rate (yearly rate)
                    if not drain_rate_found and 'battery_drain_rate' in device_data.columns:
                        yearly_rate = device_data['battery_drain_rate'].iloc[0]
                        if pd.notna(yearly_rate) and yearly_rate > 0:
                            profile_dict['battery_drain_rate_per_day'] = yearly_rate / 365.0
                            drain_rate_found = True
                            print(f"DEBUG {serial}: Using battery_drain_rate = {yearly_rate:.1f}%/year -> {yearly_rate/365.0:.4f}%/day")
            
                    # OPTION 2: Calculate from trend if health features not available
                    if not drain_rate_found:
                        print(f"DEBUG {serial}: Calculating drain rate from trend...")
                        profile_dict['battery_drain_rate_per_day'] = _calculate_drain_rate_from_trend(
                            device_data, battery_clean, battery_series
                            )
                        
            
                    # ====== CALCULATE DAYS UNTIL CRITICAL ======
                    current_battery = profile_dict['battery_current']
                    daily_drain = profile_dict.get('battery_drain_rate_per_day', 0.027)
            
                    if current_battery > 20 and daily_drain > 0.0001:
                        days_to_20 = (current_battery - 20) / daily_drain
                        profile_dict['days_until_battery_critical'] = max(0, days_to_20)
                    else:
                        profile_dict['days_until_battery_critical'] = 0
            
                    # Expected battery life
                    if daily_drain > 0.0001:
                        profile_dict['expected_battery_life_days'] = current_battery / daily_drain
                        profile_dict['expected_battery_life_years'] = profile_dict['expected_battery_life_days'] / 365
                    else:
                        profile_dict['expected_battery_life_days'] = float('inf')
                        profile_dict['expected_battery_life_years'] = float('inf')
                    # Count low/warning days
                    profile_dict['battery_low_days_count'] = (battery_clean < 20).sum()
                    profile_dict['battery_warning_days_count'] = (battery_clean < 30).sum()

                else:
                    # No valid battery readings
                    profile_dict['battery_current'] = np.nan
                    profile_dict['battery_drain_rate_per_day'] = 0.027  # Default
                    profile_dict['days_until_battery_critical'] = 0
                    profile_dict['expected_battery_life_days'] = float('inf')
                    profile_dict['expected_battery_life_years'] = float('inf')
            else:
                # No battery column found
                profile_dict['battery_current'] = np.nan
                profile_dict['battery_drain_rate_per_day'] = 0.027  # Default
                profile_dict['days_until_battery_critical'] = 0
                profile_dict['expected_battery_life_days'] = float('inf')
                profile_dict['expected_battery_life_years'] = float('inf')

        # MM3: Current and temperature analysis
        elif standardized_type == 'MM3':
            # Current analysis
            current_col = None
            for col in device_data.columns:
                if 'current' in col.lower() and 'line' in col.lower():
                    current_col = col
                    break
            
            if current_col:
                current_series = pd.to_numeric(device_data[current_col], errors='coerce')
                current_clean = current_series[(current_series >= 0) & (current_series <= 1000)]
                
                if len(current_clean) > 0:
                    profile_dict['current_current'] = current_clean.iloc[0]
                    profile_dict['current_avg'] = current_clean.mean()
                    profile_dict['current_max'] = current_clean.max()
                    profile_dict['current_std'] = current_clean.std()
                    
                    # Flag abnormal currents
                    profile_dict['zero_current_days'] = (current_clean == 0).sum()
                    profile_dict['high_current_days'] = (current_clean > 700).sum()
                    profile_dict['critical_current_days'] = (current_clean > 850).sum()
            
            # Temperature analysis
            temp_col = None
            for col in device_data.columns:
                if 'temp' in col.lower() and 'line' in col.lower():
                    temp_col = col
                    break
            
            if temp_col:
                temp_series = pd.to_numeric(device_data[temp_col], errors='coerce')
                temp_clean = temp_series[(temp_series >= -40) & (temp_series <= 100)]
                
                if len(temp_clean) > 0:
                    profile_dict['temp_current'] = temp_clean.iloc[0]
                    profile_dict['temp_avg'] = temp_clean.mean()
                    profile_dict['temp_max'] = temp_clean.max()
                    
                    # Overheat days
                    profile_dict['overheat_days'] = (temp_clean > 85).sum()
            
            profile_dict['battery_current'] = None

        # UM3: Communication analysis only
        elif standardized_type == 'UM3':
            # UM3 typically measures nothing, focus on communication
            profile_dict['communication_only_device'] = True
            profile_dict['battery_current'] = None
            
        # === RISK SCORE ANALYSIS ===
        if 'risk_score' in device_data.columns:
            risk_data = device_data['risk_score'].dropna()
            if len(risk_data) > 0:
                profile_dict['risk_score_current'] = risk_data.iloc[0]
                profile_dict['risk_score_avg'] = risk_data.mean()
                profile_dict['risk_score_max'] = risk_data.max()
                profile_dict['risk_score_min'] = risk_data.min()
                profile_dict['risk_score_std'] = risk_data.std()
        
        # === COMMUNICATION ANALYSIS ===
        if 'hours_since_last_heard' in device_data.columns:
            comm_data = device_data['hours_since_last_heard'].dropna()
            if len(comm_data) > 0:
                profile_dict['comms_gap_current_hours'] = comm_data.iloc[0]
                profile_dict['comms_gap_avg_hours'] = comm_data.mean()
                profile_dict['comms_gap_max_hours'] = comm_data.max()
        
        # === ADD DEVICE HEALTH STATUS ===
        def get_health_status(row):
            risk = row.get('risk_score_current', 0)
            device_type = row.get('Device_Type_Standardized', 'Unknown')
            
            if device_type == 'ZM1':
                battery = row.get('battery_current', 100)
                if risk > 80 or battery < 10:
                    return 'CRITICAL'
                elif risk > 60 or battery < 20:
                    return 'POOR'
                elif risk > 40 or battery < 40:
                    return 'FAIR'
                elif risk > 20 or battery < 60:
                    return 'GOOD'
                else:
                    return 'EXCELLENT'
            elif device_type == 'MM3':
                # MM3: focus on current and temperature
                current = row.get('current_current', 0)
                temp = row.get('temp_current', 0)
                if risk > 80 or current == 0 or temp > 85:
                    return 'CRITICAL'
                elif risk > 60 or current > 700 or temp > 70:
                    return 'POOR'
                elif risk > 40 or current > 500:
                    return 'FAIR'
                elif risk > 20:
                    return 'GOOD'
                else:
                    return 'EXCELLENT'
            else:  # UM3
                if risk > 80:
                    return 'CRITICAL'
                elif risk > 60:
                    return 'POOR'
                elif risk > 40:
                    return 'FAIR'
                elif risk > 20:
                    return 'GOOD'
                else:
                    return 'EXCELLENT'
        
        if 'risk_score_current' in profile_dict:
            profile_dict['device_health_status'] = get_health_status(profile_dict)
        
        # Add to device profiles list
        device_profiles.append(profile_dict)
    
    # Convert to DataFrame
    profiles_df = pd.DataFrame(device_profiles)
    
    # Sort by device type and risk
    if 'risk_score_current' in profiles_df.columns:
        profiles_df = profiles_df.sort_values(['Device_Type_Standardized', 'risk_score_current'], 
                                              ascending=[True, False])
    
    print(f"Created profiles for {len(profiles_df)} devices")
    
    # Show device type distribution
    print(f"\nDEVICE TYPE DISTRIBUTION:")
    type_counts = profiles_df['Device_Type_Standardized'].value_counts()
    for dev_type, count in type_counts.items():
        percentage = (count / len(profiles_df)) * 100
        print(f"   {dev_type}: {count} devices ({percentage:.1f}%)")
    
    return profiles_df

def add_time_based_features(df):
    """Add time-based features to the time series for all device types."""
    df = df.copy()
    
    # Ensure sorted by device and date
    df = df.sort_values(['Serial', 'date'])
    
    # Determine device type if not already standardized
    if 'Device_Type_Standardized' not in df.columns:
        # Create standardized device type
        def standardize_device_type(dev_type):
            dev_type = str(dev_type).upper()
            if 'ZM1' in dev_type:
                return 'ZM1'
            elif 'UM3' in dev_type:
                return 'UM3'
            elif 'MM3' in dev_type:
                return 'MM3'
            else:
                return 'Unknown'
        
        if 'Device_Type' in df.columns:
            df['Device_Type_Standardized'] = df['Device_Type'].apply(standardize_device_type)
        else:
            df['Device_Type_Standardized'] = 'Unknown'
    
    # Device-specific calculations
    for device_type in ['ZM1', 'MM3', 'UM3']:
        mask = df['Device_Type_Standardized'] == device_type
        
        if device_type == 'ZM1':
            # ZM1: Battery calculations
            if 'battery_level' in df.columns:
                df.loc[mask, 'battery_change_1d'] = df.loc[mask].groupby('Serial')['battery_level'].diff()
                
                # Rolling averages for battery
                for window in [3, 7, 14]:
                    df.loc[mask, f'battery_avg_{window}d'] = df.loc[mask].groupby('Serial')['battery_level'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
        
        elif device_type == 'MM3':
            # MM3: Current and temperature calculations
            if 'LineCurrent_val' in df.columns:
                df.loc[mask, 'current_change_1d'] = df.loc[mask].groupby('Serial')['LineCurrent_val'].diff()
                
                # Rolling averages for current
                for window in [3, 7, 14]:
                    df.loc[mask, f'current_avg_{window}d'] = df.loc[mask].groupby('Serial')['LineCurrent_val'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
            
            if 'LineTemperature_val' in df.columns:
                df.loc[mask, 'temp_change_1d'] = df.loc[mask].groupby('Serial')['LineTemperature_val'].diff()
                
                # Rolling averages for temperature
                for window in [3, 7, 14]:
                    df.loc[mask, f'temp_avg_{window}d'] = df.loc[mask].groupby('Serial')['LineTemperature_val'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
    
    # TEMPORARY FIX: Skip the problematic days_since calculation for now
    # Just initialize the columns with NaN
    for flag in ['overheat_flag', 'zero_current_flag', 'battery_low_flag']:
        if flag in df.columns:
            df[f'days_since_{flag}'] = np.nan
    
    return df

def process_daily_time_series():
    """
    Pipeline to process daily data for ALL device types (ZM1, MM3, UM3).
    """
    print("=" * 70)
    print("DAILY TIME SERIES PROCESSING PIPELINE - ALL DEVICES")
    print("=" * 70)
    
    # Paths (relative to project root)
    RAW_DAILY_DIR = project_root / "data" / "raw" / "daily"
    CLEAN_DAILY_DIR = project_root / "data" / "clean" / "daily"
    TIME_SERIES_DIR = project_root / "data" / "clean" / "time_series"
    PROCESSED_TS_DIR = project_root / "data" / "processed" / "time_series"
    
    # Create subdirectories for different device types
    DEVICE_TYPE_DIRS = {
        'all': CLEAN_DAILY_DIR / "all_devices",
        'zm1': CLEAN_DAILY_DIR / "zm1_only",
        'mm3': CLEAN_DAILY_DIR / "mm3_only",
        'um3': CLEAN_DAILY_DIR / "um3_only"
    }
    
    for dir_path in DEVICE_TYPE_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    for directory in [TIME_SERIES_DIR, PROCESSED_TS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Extract any ZIP files first
    print("\n" + "=" * 50)
    print("STEP 1: CHECKING FOR ZIP FILES")
    print("=" * 50)
    
    extracted_count = extract_zip_files(RAW_DAILY_DIR)
    if extracted_count > 0:
        print(f"\nExtracted {extracted_count} files from ZIP archives")
    
    # STEP 2: Load install dates
    print("\n" + "=" * 50)
    print("STEP 2: LOADING INSTALLATION DATA")
    print("=" * 50)
    
    install_file = project_root / "data" / "clean" / "install_dates.csv"
    if install_file.exists():
        install_df = pd.read_csv(install_file)
        print(f"Loaded {len(install_df)} install records")
    else:
        install_df = None
        print("No install dates file found - device age features will be limited")
    
    # STEP 3: Find all CSV and Excel files
    print("\n" + "=" * 50)
    print("STEP 3: FINDING DATA FILES")
    print("=" * 50)
    
    print(f"Looking for files in: {RAW_DAILY_DIR}")
    
    # Get all CSV and Excel files (including those extracted from ZIPs)
    excel_files = sorted(RAW_DAILY_DIR.glob("*.xlsx")) + sorted(RAW_DAILY_DIR.glob("*.xls"))
    csv_files = sorted(RAW_DAILY_DIR.glob("*.csv"))
    daily_files = list(excel_files) + list(csv_files)
    
    print(f"Found {len(daily_files)} data files ready for processing")
    
    # Show what files were found
    if daily_files:
        print("\n   Files to process:")
        for f in daily_files[:10]:  # Show first 10 files
            print(f"   - {f.name}")
        if len(daily_files) > 10:
            print(f"   ... and {len(daily_files) - 10} more")
    else:
        print("No data files found!")
        print(f"   Check path: {RAW_DAILY_DIR}")
        print(f"   Supported formats: .csv, .xlsx, .xls, or .zip containing these formats")
        return
    
    # STEP 4: Process each file for ALL device types
    print("\n" + "=" * 50)
    print("STEP 4: PROCESSING FILES (ALL DEVICE TYPES)")
    print("=" * 50)
    
    all_data_by_type = {
        'all': [],
        'zm1': [],
        'mm3': [],
        'um3': []
    }
    
    processing_stats = []
    
    for i, daily_file in enumerate(daily_files):
        print(f"\n[{i+1}/{len(daily_files)}] Processing: {daily_file.name}")
        
        try:
            # Load data based on file type
            if daily_file.suffix.lower() in ['.xlsx', '.xls']:
                df_daily = pd.read_excel(daily_file)
                print(f"   Loaded Excel file: {len(df_daily)} rows")
            else:  # .csv
                df_daily = pd.read_csv(daily_file)
                print(f"   Loaded CSV file: {len(df_daily)} rows")
            
            # Extract date from filename using regex
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', daily_file.stem)
            if date_match:
                date_from_filename = date_match.group(1)
                print(f"   Extracted date: {date_from_filename}")
            else:
                # Fallback: use entire filename stem
                date_from_filename = daily_file.stem
                print(f"   Could not extract date from filename, using: {date_from_filename}")
            
            # Check if we have Device_Type column
            if 'Device_Type' not in df_daily.columns:
                print(f"   No 'Device_Type' column, checking for device type in other columns...")
                type_cols = [col for col in df_daily.columns if 'type' in col.lower() or 'device' in col.lower()]
                if type_cols:
                    df_daily = df_daily.rename(columns={type_cols[0]: 'Device_Type'})
                    print(f"   Renamed '{type_cols[0]}' to 'Device_Type'")
            
            # Show ALL device types before filtering
            if 'Device_Type' in df_daily.columns:
                print(f"   ALL Device Types in file:")
                counts = df_daily['Device_Type'].value_counts()
                for dev_type, count in counts.items():
                    percentage = (count / len(df_daily)) * 100
                    print(f"      {dev_type}: {count} ({percentage:.1f}%)")
            
            # Process each device type separately
            device_types_to_process = ['ZM1', 'MM3', 'UM3']
            
            for device_type in device_types_to_process:
                # Filter by device type (case insensitive)
                mask = df_daily['Device_Type'].astype(str).str.contains(device_type, case=False, na=False)
                df_device = df_daily[mask].copy()
                
                if len(df_device) == 0:
                    continue
                
                print(f"   Processing {device_type}: {len(df_device)} devices")
                
                # Check if output already exists
                device_type_lower = device_type.lower()
                output_file = DEVICE_TYPE_DIRS[device_type_lower] / f"{date_from_filename}_health_{device_type_lower}.csv"
                
                if output_file.exists():
                    print(f"   Skipping {device_type} - output already exists")
                    continue
                
                # Add date column if not present
                if 'date' not in df_device.columns:
                    df_device['date'] = date_from_filename
                    df_device['timestamp'] = pd.to_datetime(date_from_filename)
                
                # DEDUPLICATION: Remove duplicate entries for same device/date
                before_dedup = len(df_device)
                df_device = df_device.drop_duplicates(subset=['Serial', 'date'], keep='first')
                duplicates_removed = before_dedup - len(df_device)
                
                if duplicates_removed > 0:
                    print(f"      Removed {duplicates_removed} duplicate entries")
                
                # Build health features
                print(f"      Building health features...")
                df_health = build_health_features(df_device, install_df)
                
                # Ensure date is preserved
                if 'date' not in df_health.columns:
                    df_health['date'] = date_from_filename
                
                # Save device-specific file
                df_health.to_csv(output_file, index=False)
                
                # Add to appropriate lists
                all_data_by_type[device_type_lower].append(df_health)
                all_data_by_type['all'].append(df_health)
                
                # Record stats for this device type
                processing_stats.append({
                    'date': date_from_filename,
                    'file': daily_file.name,
                    'device_type': device_type,
                    'rows_processed': len(df_health),
                    'unique_devices': df_health['Serial'].nunique(),
                    'output_file': output_file.name,
                    'processed': True
                })
                
                print(f"      Saved {device_type} file: {output_file.name}")
                print(f"      Generated {len(df_health.columns)} health features")
            
            # Save combined file for all devices from this day
            clean_all_file = DEVICE_TYPE_DIRS['all'] / f"{date_from_filename}_health_all_devices.csv"
            df_daily.to_csv(clean_all_file, index=False)
            print(f"   Saved all devices file: {clean_all_file.name}")
            
        except Exception as e:
            print(f"   Error processing {daily_file.name}: {str(e)[:100]}...")
            import traceback
            traceback.print_exc()
            processing_stats.append({
                'date': daily_file.stem if daily_file else 'unknown',
                'file': daily_file.name,
                'device_type': 'ALL',
                'rows_processed': 0,
                'error': str(e)[:200]
            })
    
    # STEP 5: Create time series for each device type
    print("\n" + "=" * 50)
    print("STEP 5: CREATING TIME SERIES DATASETS")
    print("=" * 50)
    
    time_series_datasets = {}
    
    for device_type in ['all', 'zm1', 'mm3', 'um3']:
        if all_data_by_type[device_type]:
            # Combine data
            combined_df = pd.concat(all_data_by_type[device_type], ignore_index=True)
            combined_df = combined_df.sort_values(['Serial', 'date'])
            
            # Save raw combined time series
            raw_ts_file = TIME_SERIES_DIR / f"{device_type}_daily_time_series.csv"
            combined_df.to_csv(raw_ts_file, index=False)
            
            # Add time-based features
            print(f"   Adding time-based features for {device_type.upper()}...")
            combined_df = add_time_based_features(combined_df)
            
            # Get date range for the combined file name
            min_date = combined_df['date'].min()
            max_date = combined_df['date'].max()
            
            # Save enhanced time series with date range in filename
            enhanced_ts_file = PROCESSED_TS_DIR / f"{min_date}_to_{max_date}_health_{device_type}_timeseries.csv"
            print(f"Saving {len(combined_df):,} rows to {enhanced_ts_file}...")

            # Clean problematic values
            combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

            # Convert object columns to string
            for col in combined_df.columns:
                 if combined_df[col].dtype == object:
                    combined_df[col] = combined_df[col].fillna('').astype(str)

            combined_df.to_csv(enhanced_ts_file, index=False, compression='gzip')
            print("File saved successfully!")

            time_series_datasets[device_type] = combined_df
            
            print(f"Created {device_type.upper()} time series: {enhanced_ts_file.name}")
            print(f"   Devices: {combined_df['Serial'].nunique()}")
            print(f"   Records: {len(combined_df):,}")
            print(f"   Date range: {min_date} to {max_date}")
    
    # STEP 6: Create device profiles for ALL devices
    print("\n" + "=" * 50)
    print("STEP 6: CREATING DEVICE PROFILES")
    print("=" * 50)

    # Initialize variables
    device_profiles = None
    trends_df = None
    trend_file = None
    
    # STEP 6.1: Create device profiles from all data
    if 'all' in time_series_datasets:
        device_profiles = create_device_profiles(time_series_datasets['all'])
    
    if device_profiles is not None:
        # STEP 6.2: Add battery trend analysis for ZM1 devices
        print("   Adding battery trend analysis...")
        
        # Get ZM1 data for trend analysis
        zm1_data = None
        
        # Method 1: Check if we have separate ZM1 dataset
        if 'zm1' in time_series_datasets:
            zm1_data = time_series_datasets['zm1']
            print("   Using separate ZM1 time series data")
        
        # Method 2: Filter ZM1 from 'all' dataset
        elif 'all' in time_series_datasets and 'Device_Type_Standardized' in time_series_datasets['all'].columns:
            zm1_data = time_series_datasets['all'][time_series_datasets['all']['Device_Type_Standardized'] == 'ZM1']
            print("   Filtered ZM1 devices from 'all' dataset")
        
        # Method 3: Try to filter by device type name
        elif 'all' in time_series_datasets and 'Device_Type' in time_series_datasets['all'].columns:
            zm1_mask = time_series_datasets['all']['Device_Type'].str.contains('ZM1', case=False, na=False)
            zm1_data = time_series_datasets['all'][zm1_mask]
            print("   Filtered ZM1 devices by Device_Type column")
        
        # Calculate trends if we found ZM1 data
        if zm1_data is not None and len(zm1_data) > 0:
            print(f"   Found {len(zm1_data)} ZM1 records for trend analysis")
            trends_df = calculate_daily_battery_trend(zm1_data)
            print(f"Battery trend analysis completed for {len(trends_df)} ZM1 devices")
        else:
            print("No ZM1 battery data available for trend analysis")
            trends_df = pd.DataFrame()
        
        # STEP 6.3: Merge trend data with device profiles
        if trends_df is not None and len(trends_df) > 0:
            # Check for common columns to avoid duplicates
            common_cols = set(device_profiles.columns) & set(trends_df.columns)
            if common_cols - {'Serial'}:
                print(f"   Removing duplicate columns before merge: {common_cols - {'Serial'}}")
                # Keep device_profiles version, drop from trends_df
                trends_df = trends_df.drop(columns=common_cols - {'Serial'})
            
            device_profiles = device_profiles.merge(trends_df, on='Serial', how='left')
            print(f"   Added trend data for {len(trends_df)} devices")
        else:
            print("   No trend data to add to device profiles")
        
        # STEP 6.4: Save trend data separately for Power BI
        if trends_df is not None and len(trends_df) > 0:
            # Check which trend columns exist
            trend_columns_available = []
            for col in ['daily_drain_rate', 'trend_slope', 'trend_intercept', 
                        'battery_change', 'date_range_days', 'battery_start', 
                        'battery_end', 'data_points']:
                if col in trends_df.columns:
                    trend_columns_available.append(col)
            
            if trend_columns_available:
                columns_to_save = ['Serial'] + trend_columns_available
                trend_file = PROCESSED_TS_DIR / "battery_trend_analysis.csv"
                trends_df[columns_to_save].to_csv(trend_file, index=False)
                print(f"Power BI trend file: {trend_file.name}")
                print(f"   Trend metrics saved: {', '.join(trend_columns_available)}")
            else:
                print("No trend columns found in trends DataFrame")
        else:
            print("No trend data available for Power BI export")
        
        # STEP 6.5: Save device profiles
        profiles_file = PROCESSED_TS_DIR / "all_device_profiles_summary.csv"
        device_profiles.to_csv(profiles_file, index=False)
        
        print(f"\nDevice profiles summary:")
        print(f"   File: {profiles_file}")
        print(f"   Devices: {len(device_profiles):,}")
        print(f"   Columns: {len(device_profiles.columns)}")
        if trends_df is not None:
            print(f"   Trend data added: {len(trends_df)} devices")
    else:
        print("Failed to create device profiles")

 # Summary
    print(f"\nALL-DEVICES SUMMARY:")
    
    for device_type in ['zm1', 'mm3', 'um3']:
        if device_type in time_series_datasets:
            df = time_series_datasets[device_type]
            print(f"\n   {device_type.upper()}:")
            print(f"      Unique devices: {df['Serial'].nunique()}")
            print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"      Total days processed: {df['date'].nunique()}")
    
    # Show processing results
    processed_by_type = {}
    for stat in processing_stats:
        if stat.get('processed'):
            dev_type = stat.get('device_type', 'unknown')
            processed_by_type[dev_type] = processed_by_type.get(dev_type, 0) + 1
    
    print(f"\nPROCESSING RESULTS:")
    for dev_type, count in processed_by_type.items():
        print(f"   {dev_type}: {count} days processed")
    
    total_processed = sum(processed_by_type.values()) / 3  # Avoid double counting
    print(f"   Total days processed: {total_processed:.0f}/{len(daily_files)}")
    
    print(f"\nOUTPUT FILES CREATED:")
    print(f"   Clean daily files in subdirectories of: {CLEAN_DAILY_DIR}")
    print(f"   Time series files in: {PROCESSED_TS_DIR}")
    
    # Return the time series datasets for potential further use
    return time_series_datasets, device_profiles  # Optional: return data


# This should be at the OUTERMOST level (no indentation)
if __name__ == "__main__":
    # Run the updated all-devices pipeline
    process_daily_time_series()