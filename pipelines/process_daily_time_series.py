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
                file_list = zf.namelist()
                csv_excel_files = [f for f in file_list if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
                
                if not csv_excel_files:
                    print(f"      No CSV or Excel files found in {zip_file.name}")
                    continue
                
                print(f"      Contains {len(csv_excel_files)} CSV/Excel file(s)")
                
                for file_in_zip in csv_excel_files:
                    if file_in_zip.endswith('/'):
                        continue
                    
                    filename = Path(file_in_zip).name
                    target_file = raw_dir / filename
                    
                    counter = 1
                    original_stem = Path(filename).stem
                    original_suffix = Path(filename).suffix
                    
                    while target_file.exists():
                        filename = f"{original_stem}_{counter}{original_suffix}"
                        target_file = raw_dir / filename
                        counter += 1
                    
                    zf.extract(file_in_zip, raw_dir)
                    
                    extracted_path = raw_dir / file_in_zip
                    if extracted_path != target_file:
                        if extracted_path.exists():
                            extracted_path.rename(target_file)
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
    trends = []
    
    date_col = None
    for col in ['Date', 'date', 'Last_Heard', 'Last_Heard_dt', 'timestamp', 'observation_date']:
        if col in battery_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("ERROR: No date column found in battery_df")
        return pd.DataFrame()
    
    print(f"Using '{date_col}' as date column for trend analysis")
    
    battery_df = battery_df.copy()
    battery_df[date_col] = pd.to_datetime(battery_df[date_col], errors='coerce')
    
    if 'BatteryLevel' not in battery_df.columns:
        print("ERROR: No BatteryLevel column found in battery_df")
        return pd.DataFrame()
    
    processed_count = 0
    error_count = 0
    
    for serial, group in battery_df.groupby('Serial'):
        if len(group) < 2:
            continue
            
        group = group.sort_values(date_col)
        
        dates = group[date_col].values
        battery_levels = group['BatteryLevel'].values
        
        if pd.isna(dates[0]) or pd.isna(dates[-1]):
            error_count += 1
            continue
            
        try:
            days_diff = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
        except (TypeError, AttributeError) as e:
            error_count += 1
            continue
            
        if days_diff <= 0:
            continue
            
        battery_diff = battery_levels[-1] - battery_levels[0]
        daily_drain_rate = battery_diff / days_diff
        
        try:
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
    if len(battery_clean) < 2:
        return 0.027
    
    try:
        valid_indices = battery_clean.index
        
        battery_dates = pd.to_datetime(device_data.loc[valid_indices, 'date'], errors='coerce')
        
        valid_date_mask = battery_dates.notna()
        if valid_date_mask.sum() < 2:
            return 0.027
        
        battery_dates = battery_dates[valid_date_mask]
        battery_clean_filtered = battery_clean[valid_date_mask]
        
        sorted_indices = battery_dates.argsort()
        sorted_dates = battery_dates.iloc[sorted_indices]
        sorted_battery = battery_clean_filtered.iloc[sorted_indices]
        
        days_since_first = (sorted_dates - sorted_dates.iloc[0]).dt.days
        
        if days_since_first.iloc[-1] <= 0:
            return 0.027
        
        if len(days_since_first.unique()) > 1:
            slope, intercept = np.polyfit(days_since_first.values, sorted_battery.values, 1)
            drain_rate = abs(slope)
        else:
            total_drain = sorted_battery.iloc[-1] - sorted_battery.iloc[0]
            total_days = days_since_first.iloc[-1]
            drain_rate = abs(total_drain / total_days) if total_days > 0 else 0.027
        
        return max(0.001, min(0.5, drain_rate))
        
    except Exception as e:
        return 0.027
    
def create_device_profiles(time_series_df):
    """
    Create a device summary file for ALL device types (ZM1, MM3, UM3).
    """
    print("\n" + "=" * 50)
    print("CREATING DEVICE PROFILES (ALL DEVICE TYPES)")
    print("=" * 50)
    
    if len(time_series_df) == 0:
        print("No time series data available for device profiles")
        return None
    
    device_profiles = []
    
    device_specs = {
        "ZM1": {
            "battery_type": "Non-rechargeable",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 100 / (10 * 365),
            "measures_current": False,
            "measures_temp": False,
            "measures_battery": True,
            "critical_battery": 20,
            "warning_battery": 30
        },
        "UM3": {
            "battery_type": "Line-powered or long-life",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 0,
            "measures_current": False,
            "measures_temp": False,
            "measures_battery": False,
            "critical_battery": 0,
            "warning_battery": 0
        },
        "MM3": {
            "battery_type": "Line-powered (NOT rechargeable)",
            "expected_lifetime_years": 10,
            "expected_daily_drain": 0,
            "measures_current": True,
            "measures_temp": True,
            "measures_battery": False,
            "critical_battery": 0,
            "warning_battery": 0
        }
    }
    
    for serial, device_data in time_series_df.groupby('Serial'):
        device_data = device_data.sort_values('date', ascending=False)
        
        device_type = device_data['Device_Type'].iloc[0] if 'Device_Type' in device_data.columns else 'Unknown'
        if not isinstance(device_type, str):
            device_type = 'Unknown'
        
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
            specs = device_specs['ZM1']
        
        profile_dict = {
            'Serial': serial,
            'Device_Type': device_type,
            'Device_Type_Standardized': standardized_type,
            'first_seen_date': device_data['date'].min(),
            'last_seen_date': device_data['date'].max(),
            'total_days_observed': device_data['date'].nunique(),
            'total_records': len(device_data)
        }
        
        profile_dict.update({
            'battery_type': specs['battery_type'],
            'expected_lifetime_years': specs['expected_lifetime_years'],
            'measures_current': specs['measures_current'],
            'measures_temp': specs['measures_temp'],
            'measures_battery': specs['measures_battery'],
            'critical_battery_level': specs['critical_battery'],
            'warning_battery_level': specs['warning_battery']
        })
        
        if 'install_date' in device_data.columns:
            profile_dict['install_date'] = device_data['install_date'].iloc[0]
            if 'device_age_days' in device_data.columns:
                profile_dict['device_age_days'] = device_data['device_age_days'].iloc[0]
        
        # ZM1: Battery analysis
        if standardized_type == 'ZM1':
            battery_col = None
            for col in device_data.columns:
                if 'battery' in col.lower():
                    battery_col = col
                    break

            if battery_col:
                battery_series = pd.to_numeric(device_data[battery_col], errors='coerce')
                battery_clean = battery_series[(battery_series >= 0) & (battery_series <= 100)]

                if len(battery_clean) > 0:
                    profile_dict['battery_current'] = battery_clean.iloc[0]
                    profile_dict['battery_min'] = battery_clean.min()
                    profile_dict['battery_max'] = battery_clean.max()
                    profile_dict['battery_avg'] = battery_clean.mean()
                    profile_dict['battery_std'] = battery_clean.std()

                    total_readings = len(battery_series)
                    valid_readings = len(battery_clean)
                    profile_dict['battery_data_quality_pct'] = (valid_readings / total_readings * 100) if total_readings > 0 else 0

                    # ===== NEW: Calculate battery metrics from observed data =====
                    def calculate_battery_metrics(device_data):
                        device_data = device_data.sort_values('date')
                        battery_readings = pd.to_numeric(device_data['battery_level'], errors='coerce').values
                        dates = pd.to_datetime(device_data['date'], errors='coerce').values
                        
                        if len(battery_readings) < 2:
                            return {
                                'drain_rate_per_day': np.nan,
                                'drain_status': 'Insufficient data',
                                'drain_category': 'Unknown'
                            }
                        
                        baseline_battery = battery_readings[0]
                        current_battery = battery_readings[-1]
                        time_span = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
                        
                        if time_span > 0 and baseline_battery > 0:
                            total_drain = baseline_battery - current_battery
                            drain_rate_per_day = total_drain / time_span
                            drain_rate_per_year = drain_rate_per_day * 365
                            
                            if drain_rate_per_day > 0.001:
                                status = f"Draining at {drain_rate_per_year:.1f}%/year"
                                if drain_rate_per_year < 10:
                                    category = "Excellent"
                                elif drain_rate_per_year < 20:
                                    category = "Normal"
                                elif drain_rate_per_year < 40:
                                    category = "Slightly Fast"
                                elif drain_rate_per_year < 100:
                                    category = "Fast"
                                else:
                                    category = "Critical"
                            else:
                                status = "Stable battery level"
                                category = "Normal"
                        else:
                            drain_rate_per_day = np.nan
                            status = "Insufficient data to calculate drain rate"
                            category = "Unknown"
                        
                        return {
                            'drain_rate_per_day': drain_rate_per_day,
                            'drain_status': status,
                            'drain_category': category,
                            'baseline_battery': baseline_battery if time_span > 0 else np.nan,
                            'time_span_days': time_span
                        }
                    
                    battery_metrics = calculate_battery_metrics(device_data)
                    
                    profile_dict['battery_drain_rate_per_day'] = battery_metrics['drain_rate_per_day']
                    profile_dict['battery_drain_status'] = battery_metrics['drain_status']
                    profile_dict['drain_rate_category'] = battery_metrics['drain_category']
                    profile_dict['battery_baseline'] = battery_metrics.get('baseline_battery', np.nan)
                    profile_dict['battery_observation_days'] = battery_metrics.get('time_span_days', 0)
                    
                    if pd.notna(battery_metrics['drain_rate_per_day']):
                        profile_dict['battery_drain_rate'] = battery_metrics['drain_rate_per_day'] * 365
                    else:
                        profile_dict['battery_drain_rate'] = np.nan

                    current_battery = profile_dict['battery_current']
                    daily_drain = profile_dict.get('battery_drain_rate_per_day', 0.027)
                    
                    if daily_drain > 0.0001:
                        days_to_20 = (current_battery - 20) / daily_drain
                        profile_dict['days_until_battery_critical'] = max(0, days_to_20)
                    else:
                        profile_dict['days_until_battery_critical'] = float('inf')
                    
                    if daily_drain > 0.0001:
                        profile_dict['expected_battery_life_days'] = current_battery / daily_drain
                        profile_dict['expected_battery_life_years'] = profile_dict['expected_battery_life_days'] / 365
                    else:
                        profile_dict['expected_battery_life_days'] = float('inf')
                        profile_dict['expected_battery_life_years'] = float('inf')
                    
                    profile_dict['battery_low_days_count'] = (battery_clean < 20).sum()
                    profile_dict['battery_warning_days_count'] = (battery_clean < 30).sum()

                else:
                    profile_dict['battery_current'] = np.nan
                    profile_dict['battery_drain_rate_per_day'] = 0.027
                    profile_dict['drain_rate_category'] = 'Insufficient Data'
                    profile_dict['days_until_battery_critical'] = 0
                    profile_dict['expected_battery_life_days'] = float('inf')
                    profile_dict['expected_battery_life_years'] = float('inf')
            else:
                profile_dict['battery_current'] = np.nan
                profile_dict['battery_drain_rate_per_day'] = 0.027
                profile_dict['drain_rate_category'] = 'No Battery Data'
                profile_dict['days_until_battery_critical'] = 0
                profile_dict['expected_battery_life_days'] = float('inf')
                profile_dict['expected_battery_life_years'] = float('inf')

        # MM3: Current and temperature analysis
        elif standardized_type == 'MM3':
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
                    
                    profile_dict['zero_current_days'] = (current_clean == 0).sum()
                    profile_dict['high_current_days'] = (current_clean > 700).sum()
                    profile_dict['critical_current_days'] = (current_clean > 850).sum()
            
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
                    profile_dict['overheat_days'] = (temp_clean > 85).sum()
            
            profile_dict['battery_current'] = None

        # UM3: Communication analysis only
        elif standardized_type == 'UM3':
            profile_dict['communication_only_device'] = True
            profile_dict['battery_current'] = None
            
        # Risk score analysis
        if 'risk_score' in device_data.columns:
            risk_data = device_data['risk_score'].dropna()
            if len(risk_data) > 0:
                profile_dict['risk_score_current'] = risk_data.iloc[0]
                profile_dict['risk_score_avg'] = risk_data.mean()
                profile_dict['risk_score_max'] = risk_data.max()
                profile_dict['risk_score_min'] = risk_data.min()
                profile_dict['risk_score_std'] = risk_data.std()
        
        # Communication analysis
        if 'hours_since_last_heard' in device_data.columns:
            comm_data = device_data['hours_since_last_heard'].dropna()
            if len(comm_data) > 0:
                profile_dict['comms_gap_current_hours'] = comm_data.iloc[0]
                profile_dict['comms_gap_avg_hours'] = comm_data.mean()
                profile_dict['comms_gap_max_hours'] = comm_data.max()
        
        # Device health status
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
            else:
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
        
        device_profiles.append(profile_dict)
    
    profiles_df = pd.DataFrame(device_profiles)
    
    if 'risk_score_current' in profiles_df.columns:
        profiles_df = profiles_df.sort_values(['Device_Type_Standardized', 'risk_score_current'], 
                                              ascending=[True, False])
    
    print(f"Created profiles for {len(profiles_df)} devices")
    
    print(f"\nDEVICE TYPE DISTRIBUTION:")
    type_counts = profiles_df['Device_Type_Standardized'].value_counts()
    for dev_type, count in type_counts.items():
        percentage = (count / len(profiles_df)) * 100
        print(f"   {dev_type}: {count} devices ({percentage:.1f}%)")
    
    return profiles_df

def add_time_based_features(df):
    """Add time-based features to the time series."""
    df = df.copy()
    df = df.sort_values(['Serial', 'date'])
    
    if 'Device_Type_Standardized' not in df.columns:
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
    
    for device_type in ['ZM1', 'MM3', 'UM3']:
        mask = df['Device_Type_Standardized'] == device_type
        
        if device_type == 'ZM1':
            if 'battery_level' in df.columns:
                df.loc[mask, 'battery_change_1d'] = df.loc[mask].groupby('Serial')['battery_level'].diff()
                for window in [3, 7, 14]:
                    df.loc[mask, f'battery_avg_{window}d'] = df.loc[mask].groupby('Serial')['battery_level'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
        
        elif device_type == 'MM3':
            if 'LineCurrent_val' in df.columns:
                df.loc[mask, 'current_change_1d'] = df.loc[mask].groupby('Serial')['LineCurrent_val'].diff()
                for window in [3, 7, 14]:
                    df.loc[mask, f'current_avg_{window}d'] = df.loc[mask].groupby('Serial')['LineCurrent_val'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
            
            if 'LineTemperature_val' in df.columns:
                df.loc[mask, 'temp_change_1d'] = df.loc[mask].groupby('Serial')['LineTemperature_val'].diff()
                for window in [3, 7, 14]:
                    df.loc[mask, f'temp_avg_{window}d'] = df.loc[mask].groupby('Serial')['LineTemperature_val'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
    
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
    
    RAW_DAILY_DIR = project_root / "data" / "raw" / "daily"
    CLEAN_DAILY_DIR = project_root / "data" / "clean" / "daily"
    TIME_SERIES_DIR = project_root / "data" / "clean" / "time_series"
    PROCESSED_TS_DIR = project_root / "data" / "processed" / "time_series"
    
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
    
    print("\n" + "=" * 50)
    print("STEP 1: CHECKING FOR ZIP FILES")
    print("=" * 50)
    
    extracted_count = extract_zip_files(RAW_DAILY_DIR)
    if extracted_count > 0:
        print(f"\nExtracted {extracted_count} files from ZIP archives")
    
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
    
    print("\n" + "=" * 50)
    print("STEP 3: FINDING DATA FILES")
    print("=" * 50)
    
    print(f"Looking for files in: {RAW_DAILY_DIR}")
    
    excel_files = sorted(RAW_DAILY_DIR.glob("*.xlsx")) + sorted(RAW_DAILY_DIR.glob("*.xls"))
    csv_files = sorted(RAW_DAILY_DIR.glob("*.csv"))
    daily_files = list(excel_files) + list(csv_files)
    
    print(f"Found {len(daily_files)} data files ready for processing")
    
    if daily_files:
        print("\n   Files to process:")
        for f in daily_files[:10]:
            print(f"   - {f.name}")
        if len(daily_files) > 10:
            print(f"   ... and {len(daily_files) - 10} more")
    else:
        print("No data files found!")
        return
    
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
            if daily_file.suffix.lower() in ['.xlsx', '.xls']:
                df_daily = pd.read_excel(daily_file)
                print(f"   Loaded Excel file: {len(df_daily)} rows")
            else:
                df_daily = pd.read_csv(daily_file)
                print(f"   Loaded CSV file: {len(df_daily)} rows")
            
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', daily_file.stem)
            if date_match:
                date_from_filename = date_match.group(1)
                print(f"   Extracted date: {date_from_filename}")
            else:
                date_from_filename = daily_file.stem
                print(f"   Could not extract date from filename, using: {date_from_filename}")
            
            if 'Device_Type' not in df_daily.columns:
                print(f"   No 'Device_Type' column, checking for device type in other columns...")
                type_cols = [col for col in df_daily.columns if 'type' in col.lower() or 'device' in col.lower()]
                if type_cols:
                    df_daily = df_daily.rename(columns={type_cols[0]: 'Device_Type'})
                    print(f"   Renamed '{type_cols[0]}' to 'Device_Type'")
            
            if 'Device_Type' in df_daily.columns:
                print(f"   ALL Device Types in file:")
                counts = df_daily['Device_Type'].value_counts()
                for dev_type, count in counts.items():
                    percentage = (count / len(df_daily)) * 100
                    print(f"      {dev_type}: {count} ({percentage:.1f}%)")
            
            device_types_to_process = ['ZM1', 'MM3', 'UM3']
            
            for device_type in device_types_to_process:
                mask = df_daily['Device_Type'].astype(str).str.contains(device_type, case=False, na=False)
                df_device = df_daily[mask].copy()
                
                if len(df_device) == 0:
                    continue
                
                print(f"   Processing {device_type}: {len(df_device)} devices")
                
                device_type_lower = device_type.lower()
                output_file = DEVICE_TYPE_DIRS[device_type_lower] / f"{date_from_filename}_health_{device_type_lower}.csv"
                
                if output_file.exists():
                    print(f"   Skipping {device_type} - output already exists")
                    continue
                
                if 'date' not in df_device.columns:
                    df_device['date'] = date_from_filename
                    df_device['timestamp'] = pd.to_datetime(date_from_filename)
                
                before_dedup = len(df_device)
                df_device = df_device.drop_duplicates(subset=['Serial', 'date'], keep='first')
                duplicates_removed = before_dedup - len(df_device)
                
                if duplicates_removed > 0:
                    print(f"      Removed {duplicates_removed} duplicate entries")
                
                print(f"      Building health features...")
                df_health = build_health_features(df_device, install_df)
                
                if 'date' not in df_health.columns:
                    df_health['date'] = date_from_filename
                
                df_health.to_csv(output_file, index=False)
                
                all_data_by_type[device_type_lower].append(df_health)
                all_data_by_type['all'].append(df_health)
                
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
    
    print("\n" + "=" * 50)
    print("STEP 5: CREATING TIME SERIES DATASETS")
    print("=" * 50)
    
    time_series_datasets = {}
    
    for device_type in ['all', 'zm1', 'mm3', 'um3']:
        if all_data_by_type[device_type]:
            combined_df = pd.concat(all_data_by_type[device_type], ignore_index=True)
            combined_df = combined_df.sort_values(['Serial', 'date'])
            
            raw_ts_file = TIME_SERIES_DIR / f"{device_type}_daily_time_series.csv"
            combined_df.to_csv(raw_ts_file, index=False)
            
            print(f"   Adding time-based features for {device_type.upper()}...")
            combined_df = add_time_based_features(combined_df)
            
            min_date = combined_df['date'].min()
            max_date = combined_df['date'].max()
            
            enhanced_ts_file = PROCESSED_TS_DIR / f"{min_date}_to_{max_date}_health_{device_type}_timeseries.csv"
            print(f"Saving {len(combined_df):,} rows to {enhanced_ts_file}...")

            combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

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
    
    print("\n" + "=" * 50)
    print("STEP 6: CREATING DEVICE PROFILES")
    print("=" * 50)

    device_profiles = None
    trends_df = None
    trend_file = None
    
    if 'all' in time_series_datasets:
        device_profiles = create_device_profiles(time_series_datasets['all'])
    
    if device_profiles is not None:
        print("   Adding battery trend analysis...")
        
        zm1_data = None
        
        if 'zm1' in time_series_datasets:
            zm1_data = time_series_datasets['zm1']
            print("   Using separate ZM1 time series data")
        elif 'all' in time_series_datasets and 'Device_Type_Standardized' in time_series_datasets['all'].columns:
            zm1_data = time_series_datasets['all'][time_series_datasets['all']['Device_Type_Standardized'] == 'ZM1']
            print("   Filtered ZM1 devices from 'all' dataset")
        elif 'all' in time_series_datasets and 'Device_Type' in time_series_datasets['all'].columns:
            zm1_mask = time_series_datasets['all']['Device_Type'].str.contains('ZM1', case=False, na=False)
            zm1_data = time_series_datasets['all'][zm1_mask]
            print("   Filtered ZM1 devices by Device_Type column")
        
        if zm1_data is not None and len(zm1_data) > 0:
            print(f"   Found {len(zm1_data)} ZM1 records for trend analysis")
            trends_df = calculate_daily_battery_trend(zm1_data)
            print(f"Battery trend analysis completed for {len(trends_df)} ZM1 devices")
        else:
            print("No ZM1 battery data available for trend analysis")
            trends_df = pd.DataFrame()
        
        if trends_df is not None and len(trends_df) > 0:
            common_cols = set(device_profiles.columns) & set(trends_df.columns)
            if common_cols - {'Serial'}:
                print(f"   Removing duplicate columns before merge: {common_cols - {'Serial'}}")
                trends_df = trends_df.drop(columns=common_cols - {'Serial'})
            
            device_profiles = device_profiles.merge(trends_df, on='Serial', how='left')
            print(f"   Added trend data for {len(trends_df)} devices")
        else:
            print("   No trend data to add to device profiles")
        
        if trends_df is not None and len(trends_df) > 0:
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

    print(f"\nALL-DEVICES SUMMARY:")
    
    for device_type in ['zm1', 'mm3', 'um3']:
        if device_type in time_series_datasets:
            df = time_series_datasets[device_type]
            print(f"\n   {device_type.upper()}:")
            print(f"      Unique devices: {df['Serial'].nunique()}")
            print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"      Total days processed: {df['date'].nunique()}")
    
    processed_by_type = {}
    for stat in processing_stats:
        if stat.get('processed'):
            dev_type = stat.get('device_type', 'unknown')
            processed_by_type[dev_type] = processed_by_type.get(dev_type, 0) + 1
    
    print(f"\nPROCESSING RESULTS:")
    for dev_type, count in processed_by_type.items():
        print(f"   {dev_type}: {count} days processed")
    
    total_processed = sum(processed_by_type.values()) / 3
    print(f"   Total days processed: {total_processed:.0f}/{len(daily_files)}")
    
    print(f"\nOUTPUT FILES CREATED:")
    print(f"   Clean daily files in subdirectories of: {CLEAN_DAILY_DIR}")
    print(f"   Time series files in: {PROCESSED_TS_DIR}")
    
    return time_series_datasets, device_profiles

if __name__ == "__main__":
    process_daily_time_series()
