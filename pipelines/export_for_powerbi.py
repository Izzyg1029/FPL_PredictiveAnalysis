# export_for_powerbi.py
import pandas as pd
from pathlib import Path
import numpy as np

def clean_zm1_battery_data(profiles):
    """Clean battery data for non-rechargeable ZM1 sensors"""
    df = profiles.copy()
    
    # 1. Fix impossible negative drain rates
    df.loc[df['battery_drain_rate_per_day'] < 0, 'battery_drain_rate_per_day'] = 0
    
    # 2. Add flag for data quality
    df['battery_data_issue'] = np.where(
        df['battery_drain_rate_per_day'] < 0,
        'Shows charging (impossible)',
        np.where(
            df['battery_drain_rate_per_day'] == 0,
            'Zero drain (sensor issue?)',
            'Normal'
        )
    )
    
    # 3. Calculate expected lifetime
    # First, ensure we have the expected daily drain column
    if 'expected_daily_drain' not in df.columns:
        df['expected_daily_drain'] = 100 / (10 * 365)  # ~0.027%/day for 10-year battery
    
    # Calculate with protection against division by zero
    df['expected_battery_life_days'] = np.where(
        df['battery_drain_rate_per_day'] > 0,
        df['battery_current'] / df['battery_drain_rate_per_day'],
        df['battery_current'] / df['expected_daily_drain']
    )
    
    df['years_remaining'] = df['expected_battery_life_days'] / 365
    
    # Cap unrealistic values
    df.loc[df['years_remaining'] > 20, 'years_remaining'] = 20
    
    return df

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
        
        # === ADD THIS CRITICAL LINE ===
        # Clean ZM1 battery data (fixes negative drain rates)
        if 'battery_drain_rate_per_day' in profiles.columns:
            profiles = clean_zm1_battery_data(profiles)
            print(f"   🔋 Cleaned battery data: {len(profiles):,} devices")
            print(f"   ⚠️  Battery issues found: {(profiles['battery_data_issue'] != 'Normal').sum():,}")
        
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
        
        # Add battery drain categories (new)
        if 'battery_drain_rate_per_day' in profiles.columns:
            profiles['drain_category'] = pd.cut(
                profiles['battery_drain_rate_per_day'].fillna(0),
                bins=[-1, 0.01, 0.05, 0.1, 1, 100],
                labels=['Sensor Issue', 'Excellent (<0.01%/day)', 'Good (0.01-0.05%/day)', 
                       'Moderate (0.05-0.1%/day)', 'High (>0.1%/day)']
            )
        
        # Save
        profiles_output = output_dir / "device_profiles_powerbi.csv"
        profiles.to_csv(profiles_output, index=False)
        print(f"✅ Device profiles: {profiles_output}")
        print(f"   Devices: {len(profiles):,}")
        print(f"   Columns: {len(profiles.columns)}")
    
       # 2. TIME SERIES SAMPLE (for trends) - UPDATED VERSION
    ts_files = list((project_root / "data" / "processed" / "time_series").glob("*_health_zm1only_timeseries.csv"))
    if ts_files:
        ts_file = ts_files[-1]
        ts_data = pd.read_csv(ts_file)
        
        # Convert date column
        ts_data['date'] = pd.to_datetime(ts_data['date'])
        
        # Get last 30 days for communication analysis (not just 7)
        latest_date = ts_data['date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=30)
        ts_sample = ts_data[ts_data['date'] >= cutoff_date]
        
        # === ADD CRITICAL METRICS ANALYSIS ===
        print("   ⚙️  Creating critical metrics analysis...")
        profiles = create_critical_metrics_summary(profiles, ts_sample)
        
        # Create critical devices exports
        summary = create_critical_devices_export(profiles, output_dir)
        
        # Save the updated profiles with critical metrics
        profiles_output = output_dir / "device_profiles_powerbi.csv"
        profiles.to_csv(profiles_output, index=False)
        
        ts_output = output_dir / "time_series_sample_powerbi.csv"
        ts_sample.to_csv(ts_output, index=False)
        print(f"✅ Time series sample (last 30 days): {ts_output}")
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
    
    # Show battery data summary
    if profiles_file.exists() and 'battery_drain_rate_per_day' in profiles.columns:
        print(f"\n🔋 BATTERY DATA SUMMARY:")
        print(f"   Avg drain rate: {profiles['battery_drain_rate_per_day'].mean():.4f}%/day")
        print(f"   Max drain rate: {profiles['battery_drain_rate_per_day'].max():.2f}%/day")
        print(f"   Devices showing 'charging' (fixed): {(profiles['battery_data_issue'] == 'Shows charging (impossible)').sum()}")
        print(f"   Expected 10-year rate: {100/(10*365):.4f}%/day")
# Add this RIGHT AFTER the clean_zm1_battery_data function

def create_critical_metrics_summary(profiles, ts_data):
    """Create comprehensive summary of critical device metrics"""
    
    # 1. Communication Status Analysis
    print("   📡 Analyzing communication status...")
    if 'Date' in ts_data.columns or 'date' in ts_data.columns:
        date_col = 'Date' if 'Date' in ts_data.columns else 'date'
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        latest_date = ts_data[date_col].max()
        
        # Get last communication per device
        last_comms = ts_data.groupby('Serial')[date_col].max().reset_index()
        last_comms['days_since_last_comm'] = (latest_date - last_comms[date_col]).dt.days
        
        # Merge with profiles
        profiles = profiles.merge(last_comms[['Serial', 'days_since_last_comm']], 
                                 on='Serial', how='left')
        
        # Communication status categories
        profiles['communication_status'] = np.where(
            profiles['days_since_last_comm'].isna(),
            'No Data',
            np.where(
                profiles['days_since_last_comm'] <= 1,
                'Active (<24h)',
                np.where(
                    profiles['days_since_last_comm'] <= 7,
                    'Recent (1-7d)',
                    np.where(
                        profiles['days_since_last_comm'] <= 30,
                        'Stale (8-30d)',
                        'Inactive (>30d)'
                    )
                )
            )
        )
        
        # Flag for not communicating
        profiles['not_communicating_7d'] = (profiles['days_since_last_comm'] > 7) | (profiles['days_since_last_comm'].isna())
    
    # 2. Battery Sensor Issues
    print("   ⚠️  Identifying sensor issues...")
    profiles['has_sensor_issue'] = False
    
    if 'battery_drain_rate_per_day' in profiles.columns:
        # Zero or negative drain rates (sensor stuck)
        profiles.loc[profiles['battery_drain_rate_per_day'] <= 0, 'has_sensor_issue'] = True
        
        # Unrealistically high drain rates
        profiles.loc[profiles['battery_drain_rate_per_day'] > 1.0, 'has_sensor_issue'] = True
    
    if 'battery_current' in profiles.columns:
        # Invalid battery percentages
        profiles.loc[(profiles['battery_current'] < 0) | 
                    (profiles['battery_current'] > 100), 'has_sensor_issue'] = True
    
    # 3. Battery Life Analysis - Comprehensive with <10 year flag
    print("   📅 Analyzing battery life projections...")

    profiles['projected_life_years'] = None
    profiles['years_remaining'] = None
    profiles['life_reduction_pct'] = None
    profiles['remaining_life_category'] = 'Unknown'

    # FLAGS FOR TOTAL PROJECTED LIFE:
    profiles['less_than_10yr_life'] = False      # NEW: All devices under 10 years
    profiles['is_slightly_draining'] = False     # <9 years (10%+ reduction)
    profiles['is_moderately_draining'] = False   # <8 years (20%+ reduction)
    profiles['is_fast_draining'] = False         # <7 years (30%+ reduction)  
    profiles['is_critical_draining'] = False     # <5 years (50%+ reduction)

    # FLAGS FOR REMAINING LIFE (Your requested changes):
    profiles['needs_replacement_1year'] = False  # <1 year remaining
    profiles['needs_replacement_2year'] = False  # 1-2 years remaining
    profiles['schedule_replacement_3year'] = False  # 2-3 years remaining

    if 'battery_drain_rate_per_day' in profiles.columns and 'battery_current' in profiles.columns:
    # Calculate total projected life (from 100% to 0%)
        with np.errstate(divide='ignore', invalid='ignore'):
            profiles['projected_life_years'] = 100 / (profiles['battery_drain_rate_per_day'] * 365)
    
    # Calculate remaining life (from current % to 0%)
    profiles['years_remaining'] = profiles['battery_current'] / (profiles['battery_drain_rate_per_day'] * 365)
    
    # Handle extreme values
    profiles.loc[profiles['battery_drain_rate_per_day'] <= 0.0001, 'projected_life_years'] = 30
    profiles.loc[profiles['battery_drain_rate_per_day'] <= 0.0001, 'years_remaining'] = 30
    profiles.loc[profiles['projected_life_years'] > 30, 'projected_life_years'] = 30
    profiles.loc[profiles['years_remaining'] > 30, 'years_remaining'] = 30
    
    # Life reduction from 10-year spec
    profiles['life_reduction_pct'] = ((10 - profiles['projected_life_years']) / 10) * 100
    
    # ===== TOTAL LIFE FLAGS =====
    # ALL devices with less than 10-year life
    profiles['less_than_10yr_life'] = profiles['projected_life_years'] < 10
    
    # Graduated warning levels
    profiles.loc[profiles['projected_life_years'] < 9, 'is_slightly_draining'] = True      # 10%+ reduction
    profiles.loc[profiles['projected_life_years'] < 8, 'is_moderately_draining'] = True    # 20%+ reduction
    profiles.loc[profiles['projected_life_years'] < 7, 'is_fast_draining'] = True          # 30%+ reduction
    profiles.loc[profiles['projected_life_years'] < 5, 'is_critical_draining'] = True      # 50%+ reduction
    
    # ===== REMAINING LIFE CATEGORIES =====
    profiles['remaining_life_category'] = np.where(
        profiles['years_remaining'] < 0.083,  # <1 month (30 days)
        'EMERGENCY (<1 month)',
        np.where(
            profiles['years_remaining'] < 1,  # <1 year
            'CRITICAL (<1 year)',
            np.where(
                profiles['years_remaining'] < 2,  # <2 years
                'WARNING (1-2 years)',
                'SAFE (>2 years)'
            )
        )
    )
    
    # ===== REPLACEMENT PLANNING =====
    profiles['needs_replacement_1year'] = profiles['years_remaining'] < 1
    profiles['needs_replacement_2year'] = (profiles['years_remaining'] >= 1) & (profiles['years_remaining'] < 2)
    profiles['schedule_replacement_3year'] = (profiles['years_remaining'] >= 2) & (profiles['years_remaining'] < 3)
    
    # ===== LIFE EXPECTANCY CATEGORY =====
    profiles['life_expectancy_category'] = np.where(
        profiles['projected_life_years'] < 5,
        'Short (<5 years)',
        np.where(
            profiles['projected_life_years'] < 7,
            'Below Average (5-7 years)',
            np.where(
                profiles['projected_life_years'] < 9,
                'Average (7-9 years)',
                np.where(
                    profiles['projected_life_years'] < 12,
                    'Good (9-12 years)',
                    'Excellent (>12 years)'
                )
            )
        )
    )
    
    print(f"      Average total life: {profiles['projected_life_years'].mean():.1f} years")
    print(f"      Devices with <10 year life: {profiles['less_than_10yr_life'].sum()} ({(profiles['less_than_10yr_life'].sum()/len(profiles)*100):.1f}%)")
    print(f"      Slightly draining (<9 years): {profiles['is_slightly_draining'].sum()} devices")
    print(f"      Moderately draining (<8 years): {profiles['is_moderately_draining'].sum()} devices")
    print(f"      Fast draining (<7 years): {profiles['is_fast_draining'].sum()} devices")
    print(f"      Critical draining (<5 years): {profiles['is_critical_draining'].sum()} devices")
    print(f"      Needs replacement (<1 year): {profiles['needs_replacement_1year'].sum()} devices")
   
    # 4. Batteries Closest to Death
    print("   💀 Identifying critical batteries...")
    
    # Calculate estimated days remaining
    if 'battery_current' in profiles.columns and 'battery_drain_rate_per_day' in profiles.columns:
        # Use actual drain rate if available and positive
        profiles['estimated_days_remaining'] = np.where(
            (profiles['battery_drain_rate_per_day'] > 0.0001) & 
            (profiles['battery_drain_rate_per_day'].notna()),
            profiles['battery_current'] / profiles['battery_drain_rate_per_day'],
            np.where(
                profiles['battery_current'].notna(),
                profiles['battery_current'] / (100 / (10 * 365)),  # Use 10-year rate
                np.nan
            )
        )
        
        # Flag batteries with <30 days remaining
        profiles['critical_battery_30d'] = profiles['estimated_days_remaining'] < 30
        
        # Flag batteries with <7 days remaining (emergency)
        profiles['emergency_battery_7d'] = profiles['estimated_days_remaining'] < 7
    
    # 5. Overall Risk Score
    print("   📊 Calculating overall risk score...")
    
    # Initialize risk components (0-100 scale)
    profiles['comm_risk_score'] = 0
    profiles['sensor_risk_score'] = 0
    profiles['battery_risk_score'] = 0
    profiles['drain_risk_score'] = 0
    
    # Communication risk (higher for older last communication)
    if 'days_since_last_comm' in profiles.columns:
        profiles.loc[profiles['days_since_last_comm'].isna(), 'comm_risk_score'] = 100
        profiles.loc[profiles['days_since_last_comm'] > 30, 'comm_risk_score'] = 80
        profiles.loc[profiles['days_since_last_comm'] > 7, 'comm_risk_score'] = 60
        profiles.loc[profiles['days_since_last_comm'] > 1, 'comm_risk_score'] = 20
    
    # Sensor issue risk
    profiles.loc[profiles['has_sensor_issue'], 'sensor_risk_score'] = 70
    
    # Battery level risk
    if 'battery_current' in profiles.columns:
        profiles.loc[profiles['battery_current'] < 10, 'battery_risk_score'] = 100
        profiles.loc[profiles['battery_current'] < 20, 'battery_risk_score'] = 80
        profiles.loc[profiles['battery_current'] < 30, 'battery_risk_score'] = 60
        profiles.loc[profiles['battery_current'] < 50, 'battery_risk_score'] = 40
    
    # Drain rate risk
    if 'battery_drain_rate_per_day' in profiles.columns:
        expected_rate = 100 / (10 * 365)
        profiles.loc[profiles['battery_drain_rate_per_day'] > expected_rate * 10, 'drain_risk_score'] = 100
        profiles.loc[profiles['battery_drain_rate_per_day'] > expected_rate * 5, 'drain_risk_score'] = 80
        profiles.loc[profiles['battery_drain_rate_per_day'] > expected_rate * 2, 'drain_risk_score'] = 40
    
    # Overall risk (weighted average)
    profiles['overall_risk_score'] = (
        profiles['comm_risk_score'] * 0.3 +  # 30% weight
        profiles['sensor_risk_score'] * 0.2 +  # 20% weight
        profiles['battery_risk_score'] * 0.3 +  # 30% weight
        profiles['drain_risk_score'] * 0.2  # 20% weight
    )
    
    # Risk category
    profiles['risk_category'] = pd.cut(
        profiles['overall_risk_score'],
        bins=[0, 20, 40, 60, 80, 101],
        labels=['Low', 'Medium', 'High', 'Critical', 'Emergency'],
        right=False
    )
    
    # 6. Priority for Action
    profiles['priority_level'] = 'Normal'
    
    # Emergency priority
    emergency_conditions = (
        profiles['emergency_battery_7d'] |
        (profiles['communication_status'] == 'No Data') |
        (profiles['overall_risk_score'] >= 80)
    )
    profiles.loc[emergency_conditions, 'priority_level'] = 'Emergency'
    
    # High priority
    high_conditions = (
        profiles['critical_battery_30d'] |
        profiles['is_fast_draining'] |
        profiles['has_sensor_issue'] |
        (profiles['communication_status'] == 'Inactive (>30d)') |
        (profiles['overall_risk_score'] >= 60)
    ) & (profiles['priority_level'] != 'Emergency')
    profiles.loc[high_conditions, 'priority_level'] = 'High'
    
    # Medium priority
    medium_conditions = (
        profiles['not_communicating_7d'] |
        (profiles['battery_current'] < 30) |
        (profiles['overall_risk_score'] >= 40)
    ) & (profiles['priority_level'] == 'Normal')
    profiles.loc[medium_conditions, 'priority_level'] = 'Medium'
    
    return profiles

def create_critical_devices_export(profiles, output_dir):
    """Create special exports for critical devices"""
    
    print("\n🔴 CREATING CRITICAL DEVICES EXPORTS")
    
    # 1. Devices Not Communicating (>7 days)
    not_comm_devices = profiles[profiles['not_communicating_7d'] == True]
    not_comm_file = output_dir / "critical_not_communicating.csv"
    not_comm_devices.to_csv(not_comm_file, index=False)
    print(f"✅ Devices not communicating (>7d): {len(not_comm_devices)}")
    
    # 2. Devices with Sensor Issues
    sensor_issue_devices = profiles[profiles['has_sensor_issue'] == True]
    sensor_issue_file = output_dir / "critical_sensor_issues.csv"
    sensor_issue_devices.to_csv(sensor_issue_file, index=False)
    print(f"✅ Devices with sensor issues: {len(sensor_issue_devices)}")
    
    # 3. Fast Draining Batteries
    fast_drain_devices = profiles[profiles['is_fast_draining'] == True]
    fast_drain_file = output_dir / "critical_fast_draining.csv"
    fast_drain_devices.to_csv(fast_drain_file, index=False)
    print(f"✅ Fast draining batteries: {len(fast_drain_devices)}")
    
    # 4. Batteries Closest to Death (<30 days)
    critical_battery_devices = profiles[profiles['critical_battery_30d'] == True]
    critical_battery_file = output_dir / "critical_battery_low.csv"
    critical_battery_devices.to_csv(critical_battery_file, index=False)
    print(f"✅ Batteries with <30 days remaining: {len(critical_battery_devices)}")
    
    # 5. Emergency Batteries (<7 days)
    emergency_devices = profiles[profiles['emergency_battery_7d'] == True]
    emergency_file = output_dir / "emergency_battery.csv"
    emergency_devices.to_csv(emergency_file, index=False)
    print(f"✅ EMERGENCY: Batteries with <7 days remaining: {len(emergency_devices)}")
    
    # 6. All Critical Devices (combined view)
    critical_columns = ['Serial', 'device_health_status', 'priority_level', 'risk_category', 
                       'overall_risk_score', 'battery_current', 
                       'estimated_days_remaining', 'communication_status',
                       'battery_drain_rate_per_day', 'has_sensor_issue',
                       'is_fast_draining', 'critical_battery_30d']
    
    # Filter to only devices with issues
    has_issue = profiles[
        profiles['not_communicating_7d'] | 
        profiles['has_sensor_issue'] | 
        profiles['is_fast_draining'] | 
        profiles['critical_battery_30d']
    ]
    
    if len(has_issue) > 0:
        critical_view = has_issue[critical_columns].sort_values(['priority_level', 'overall_risk_score'], ascending=[False, False])
        critical_file = output_dir / "all_critical_devices.csv"
        critical_view.to_csv(critical_file, index=False)
        print(f"✅ All critical devices view: {len(critical_view)} devices")
    
    # 7. Summary Dashboard Data
    summary_data = {
        'metric': [
            'Total Devices',
            'Not Communicating (>7d)',
            'Sensor Issues',
            'Fast Draining',
            'Critical Battery (<30d)',
            'Emergency Battery (<7d)',
            'High Priority',
            'Medium Priority',
            'Low Priority'
        ],
        'count': [
            len(profiles),
            len(not_comm_devices),
            len(sensor_issue_devices),
            len(fast_drain_devices),
            len(critical_battery_devices),
            len(emergency_devices),
            len(profiles[profiles['priority_level'] == 'High']),
            len(profiles[profiles['priority_level'] == 'Medium']),
            len(profiles[profiles['priority_level'] == 'Normal'])
        ],
        'percentage': [
            100,
            (len(not_comm_devices) / len(profiles)) * 100,
            (len(sensor_issue_devices) / len(profiles)) * 100,
            (len(fast_drain_devices) / len(profiles)) * 100,
            (len(critical_battery_devices) / len(profiles)) * 100,
            (len(emergency_devices) / len(profiles)) * 100,
            (len(profiles[profiles['priority_level'] == 'High']) / len(profiles)) * 100,
            (len(profiles[profiles['priority_level'] == 'Medium']) / len(profiles)) * 100,
            (len(profiles[profiles['priority_level'] == 'Normal']) / len(profiles)) * 100
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "critical_metrics_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Critical metrics summary created")
    
    return summary_df

if __name__ == "__main__":
    create_powerbi_exports()