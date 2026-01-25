# test_battery_calc_fixed.py
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import glob

script_dir = Path(__file__).parent
project_root = script_dir.parent  # Go up one level from tests/


# Change the function to start from project_root
def find_time_series_files():
    """Find all time series files in the project"""
    print(f"🔍 SEARCHING FROM: {project_root}")
    
    # Common locations - start from project_root
    search_paths = [
        project_root / "data" / "processed" / "time_series",
        project_root / "data" / "clean" / "time_series", 
        project_root / "data" / "processed",
        project_root / "data" / "clean",
        project_root / "data",
        project_root
    ]
    all_csv_files = []
    
    for search_path in search_paths:
        if search_path.exists():
            # Look for time series or profile files
            patterns = ["*timeseries*.csv", "*profile*.csv", "*health*.csv", "*daily*.csv"]
            
            for pattern in patterns:
                files = list(search_path.rglob(pattern))
                if files:
                    print(f"\n📁 Found in {search_path.relative_to(project_root)}/:")
                    for f in files:
                        print(f"   • {f.name} ({f.stat().st_size/1024:.1f} KB)")
                        all_csv_files.append(f)
    
    if not all_csv_files:
        print("\n❌ No CSV files found!")
        # List all files in data directory
        data_dir = project_root / "data"
        if data_dir.exists():
            print(f"\n📁 Contents of data/ directory:")
            for item in data_dir.rglob("*"):
                if item.is_file():
                    print(f"   {item.relative_to(project_root)}")
    
    return all_csv_files

def debug_battery_calculation():
    """Debug battery calculations"""
    print("=" * 60)
    print("BATTERY CALCULATION DEBUGGER")
    print("=" * 60)
    
    # Find files
    csv_files = find_time_series_files()
    
    if not csv_files:
        return
    
    # Try to load the largest/likely time series file
    target_file = None
    for f in csv_files:
        if any(keyword in f.name.lower() for keyword in ['timeseries', 'profile', 'health']):
            target_file = f
            break
    
    if target_file is None:
        target_file = csv_files[0]  # Use first file
    
    print(f"\n📊 LOADING: {target_file}")
    
    try:
        df = pd.read_csv(target_file)
        print(f"✅ Successfully loaded")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Check for battery columns
        battery_cols = [col for col in df.columns if 'battery' in col.lower()]
        print(f"\n🔋 BATTERY COLUMNS FOUND: {len(battery_cols)}")
        for col in battery_cols:
            non_null = df[col].notna().sum()
            print(f"   {col}: {non_null} non-null values")
            if non_null > 0:
                print(f"      Min: {df[col].min():.1f}, Max: {df[col].max():.1f}, Mean: {df[col].mean():.1f}")
        
        # Check a few devices if we have serial numbers
        if 'Serial' in df.columns:
            sample_devices = df['Serial'].dropna().unique()[:2]
            print(f"\n📱 SAMPLE DEVICES (first 2):")
            
            for serial in sample_devices:
                device_data = df[df['Serial'] == serial].sort_values('date' if 'date' in df.columns else 'Date')
                print(f"\n   Device: {serial}")
                print(f"   Records: {len(device_data)}")
                
                if 'battery_level' in df.columns or 'battery_current' in df.columns:
                    battery_col = 'battery_level' if 'battery_level' in df.columns else 'battery_current'
                    if len(device_data) >= 2:
                        battery_vals = device_data[battery_col].values
                        print(f"   Battery values: {battery_vals}")
                        
                        # Simple drain calculation
                        if 'date' in device_data.columns:
                            dates = pd.to_datetime(device_data['date'])
                            days_diff = (dates.iloc[-1] - dates.iloc[0]).days
                            if days_diff > 0:
                                battery_diff = battery_vals[-1] - battery_vals[0]
                                daily_rate = battery_diff / days_diff
                                print(f"   Drain rate: {daily_rate:.4f}%/day (over {days_diff} days)")
        
        # Check if this looks like device profiles
        if 'battery_current' in df.columns and 'battery_drain_rate_per_day' in df.columns:
            print(f"\n📈 DEVICE PROFILES ANALYSIS:")
            print(f"   Devices: {len(df)}")
            print(f"   Avg battery current: {df['battery_current'].mean():.1f}%")
            print(f"   Avg drain rate: {df['battery_drain_rate_per_day'].mean():.6f}%/day")
            print(f"   Negative drain rates: {(df['battery_drain_rate_per_day'] < 0).sum()}")
            print(f"   Zero drain rates: {(df['battery_drain_rate_per_day'] == 0).sum()}")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    debug_battery_calculation()