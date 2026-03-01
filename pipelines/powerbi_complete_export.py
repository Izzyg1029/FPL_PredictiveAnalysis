# powerbi_complete_export.py - ULTRA SIMPLE SINGLE FILE VERSION
# This exports ONLY ONE file - nothing else!

import pandas as pd
from pathlib import Path
import numpy as np

def create_complete_powerbi_export():
    """
    Export ONE single file for Power BI.
    """
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "powerbi_exports"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(" CREATING SINGLE POWER BI EXPORT FILE")
    print("=" * 60)
    
    # ====================================================
    # 1. LOAD DEVICE PROFILES
    # ====================================================
    
    profiles_file = project_root / "data" / "processed" / "time_series" / "all_device_profiles_summary.csv"
    if profiles_file.exists():
        df = pd.read_csv(profiles_file, low_memory=False)
        print(f"Loaded: {len(df):,} devices")
    else:
        print(" File not found")
        return
    
    # ====================================================
    # 2. ADD A FEW USEFUL COLUMNS
    # ====================================================
    
    print(" Adding useful columns...")
    
    # Battery level categories
    if 'battery_current' in df.columns:
        df['battery_status'] = np.where(
            df['battery_current'] < 10, 'CRITICAL',
            np.where(df['battery_current'] < 20, 'Warning',
            np.where(df['battery_current'] < 30, 'Low',
            np.where(df['battery_current'] < 50, 'Moderate', 'Good')))
        )
    
    # Time categories
    if 'days_until_battery_critical' in df.columns:
        df['replacement_timeline'] = np.where(
            df['days_until_battery_critical'] < 30, 'Critical (<30 days)',
            np.where(df['days_until_battery_critical'] < 90, 'Urgent (30-90 days)',
            np.where(df['days_until_battery_critical'] < 365, 'Soon (3-12 months)',
                    'Future (>1 year)')))
    
    # Risk categories
    if 'risk_score_current' in df.columns:
        df['risk_level'] = np.where(
            df['risk_score_current'] > 80, 'Emergency',
            np.where(df['risk_score_current'] > 60, 'Critical',
            np.where(df['risk_score_current'] > 40, 'High',
            np.where(df['risk_score_current'] > 20, 'Medium', 'Low'))))
    
    # Critical device flag
    df['critical_device'] = False
    if 'battery_current' in df.columns:
        df.loc[df['battery_current'] < 20, 'critical_device'] = True
    if 'days_until_battery_critical' in df.columns:
        df.loc[df['days_until_battery_critical'] < 30, 'critical_device'] = True
    
    # ====================================================
    # 3. SAVE - ONLY ONE FILE!
    # ====================================================
    
    print("\n Saving ONE file...")
    
    # Save with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"powerbi_export_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f" Created: {output_file}")
    
    # Also save as latest (overwrites previous)
    latest_file = output_dir / "powerbi_export_latest.csv"
    df.to_csv(latest_file, index=False)
    print(f" Created: {latest_file}")
    
    # ====================================================
    # 4. SUMMARY
    # ====================================================
    
    print("\n" + "=" * 60)
    print("DONE - ONE FILE CREATED")
    print("=" * 60)
    print(f"\nFile: {latest_file}")
    print(f"Devices: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nReady for Power BI!")

if __name__ == "__main__":
    create_complete_powerbi_export()