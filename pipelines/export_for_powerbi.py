# export_for_powerbi.py
import pandas as pd
from pathlib import Path
import numpy as np

def create_powerbi_exports():
    """Create optimized files for Power BI import."""
    project_root = Path(__file__).parent.parent  # Go up one level
    output_dir = project_root / "powerbi_exports"  # Creates in project root
    output_dir.mkdir(exist_ok=True)
    
    print("📊 CREATING POWER BI EXPORTS")
    print("=" * 50)
    
    # 1. DEVICE PROFILES (Main dashboard data)
    profiles_file = project_root / "data" / "processed" / "time_series" / "device_profiles_summary.csv"
    if profiles_file.exists():
        profiles = pd.read_csv(profiles_file)
        
        # Add derived columns for Power BI
        profiles['battery_health_category'] = pd.cut(
            profiles['battery_current'],
            bins=[0, 20, 40, 60, 80, 101],
            labels=['Critical (<20%)', 'Poor (20-40%)', 'Fair (40-60%)', 'Good (60-80%)', 'Excellent (80-100%)']
        )
        
        # Age categories
        if 'device_age_days' in profiles.columns:
            profiles['age_category'] = pd.cut(
                profiles['device_age_days'].fillna(0),
                bins=[0, 180, 365, 730, 1825, 3650],
                labels=['<6 months', '6-12 months', '1-2 years', '2-5 years', '5+ years']
            )
        
        # Risk categories
        profiles['risk_category'] = pd.cut(
            profiles['risk_score_current'].fillna(0),
            bins=[0, 20, 40, 60, 80, 101],
            labels=['Low (0-20)', 'Medium (21-40)', 'High (41-60)', 'Critical (61-80)', 'Emergency (81-100)']
        )
        
        # Save
        profiles_output = output_dir / "device_profiles_powerbi.csv"
        profiles.to_csv(profiles_output, index=False)
        print(f"✅ Device profiles: {profiles_output}")
        print(f"   Devices: {len(profiles):,}")
    
    # 2. TIME SERIES SAMPLE (for trends)
    ts_files = list((project_root / "data" / "processed" / "time_series").glob("*_health_zm1only_timeseries.csv"))
    if ts_files:
        ts_file = ts_files[-1]
        ts_data = pd.read_csv(ts_file)
        
        # Sample for Power BI (limit to last 7 days for performance)
        latest_date = pd.to_datetime(ts_data['date']).max()
        cutoff_date = latest_date - pd.Timedelta(days=7)
        ts_sample = ts_data[pd.to_datetime(ts_data['date']) >= cutoff_date]
        
        ts_output = output_dir / "time_series_sample_powerbi.csv"
        ts_sample.to_csv(ts_output, index=False)
        print(f"✅ Time series sample (last 7 days): {ts_output}")
        print(f"   Records: {len(ts_sample):,}")
    
    # 3. DAILY SUMMARY STATS
    daily_stats = []
    clean_daily_dir = project_root / "data" / "clean" / "daily"
    
    for daily_file in clean_daily_dir.glob("*.csv"):
        date = daily_file.stem.split('_')[0]  # Get date from filename
        df = pd.read_csv(daily_file)
        
        stats = {
            'date': date,
            'total_devices': len(df),
            'avg_battery': df['battery_level'].mean() if 'battery_level' in df.columns else None,
            'avg_risk': df['risk_score'].mean() if 'risk_score' in df.columns else None,
            'critical_devices': (df['risk_score'] > 80).sum() if 'risk_score' in df.columns else 0,
            'low_battery_devices': (df['battery_level'] < 30).sum() if 'battery_level' in df.columns else 0
        }
        daily_stats.append(stats)
    
    if daily_stats:
        daily_df = pd.DataFrame(daily_stats)
        daily_output = output_dir / "daily_summary_powerbi.csv"
        daily_df.to_csv(daily_output, index=False)
        print(f"✅ Daily summary: {daily_output}")
        print(f"   Days: {len(daily_df)}")
    
    # 4. HEALTH DISTRIBUTION (for pie charts)
    if profiles_file.exists():
        health_dist = profiles['device_health_status'].value_counts().reset_index()
        health_dist.columns = ['health_status', 'device_count']
        health_dist['percentage'] = (health_dist['device_count'] / health_dist['device_count'].sum() * 100).round(1)
        
        health_output = output_dir / "health_distribution_powerbi.csv"
        health_dist.to_csv(health_output, index=False)
        print(f"✅ Health distribution: {health_output}")
    
    print(f"\n🎉 All Power BI files saved to: {output_dir}")

if __name__ == "__main__":
    create_powerbi_exports()