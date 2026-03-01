# export_for_powerbi.py - COMPLETE FIXED VERSION
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
    
    # 3. Calculate expected lifetime using NON-LINEAR model
    if 'battery_drain_rate_per_day' in df.columns and 'battery_current' in df.columns:
        
        # Use the non-linear function from earlier in the file
        def calculate_non_linear_days(current_battery, base_daily_drain, target_level, age_factor=0.5):
            """Calculate days to reach target level with non-linear drain"""
            if pd.isna(base_daily_drain) or base_daily_drain <= 0 or pd.isna(current_battery):
                return np.nan
            
            if current_battery <= target_level:
                return 0
            
            days = 0
            battery = float(current_battery)
            max_days = 3650
            
            while battery > target_level and days < max_days:
                battery_stress = 1.0 + ((100.0 - battery) / 100.0) * 0.3
                current_age = age_factor + ((100.0 - battery) / 100.0) * 0.15
                aging_accel = 1.0 + (current_age ** 1.5) * 1.2
                
                today_drain = base_daily_drain * battery_stress * aging_accel
                battery -= today_drain
                days += 1
                
                if today_drain < 0.0001:
                    days = max_days
                    break
            
            return min(days, max_days)
        
        # Get age factor if available
        if 'pct_life_used' in df.columns:
            age_factors = df['pct_life_used'].fillna(0.5)
        else:
            age_factors = pd.Series(0.5, index=df.index)
        
        # Calculate days to zero using non-linear model
        days_to_zero_list = []
        for idx, row in df.iterrows():
            days_to_zero = calculate_non_linear_days(
                row['battery_current'], 
                row['battery_drain_rate_per_day'], 
                0,  # Target level 0%
                age_factors.iloc[idx]
            )
            days_to_zero_list.append(days_to_zero)
        
        df['expected_battery_life_days'] = days_to_zero_list
    
    df['years_remaining'] = df['expected_battery_life_days'] / 365
    df.loc[df['years_remaining'] > 20, 'years_remaining'] = 20
    
    return df

def create_critical_metrics_summary(profiles, ts_data):
    """Create comprehensive summary of critical device metrics with NON-LINEAR battery modeling"""
    
    # 1. Communication Status Analysis
    print("    Analyzing communication status...")
    if 'Date' in ts_data.columns or 'date' in ts_data.columns:
        date_col = 'Date' if 'Date' in ts_data.columns else 'date'
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        latest_date = ts_data[date_col].max()
        
        last_comms = ts_data.groupby('Serial')[date_col].max().reset_index()
        last_comms['days_since_last_comm'] = (latest_date - last_comms[date_col]).dt.days
        
        profiles = profiles.merge(last_comms[['Serial', 'days_since_last_comm']], 
                                 on='Serial', how='left')
        
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
        
        profiles['not_communicating_7d'] = (profiles['days_since_last_comm'] > 7) | (profiles['days_since_last_comm'].isna())
    
    # 2. Battery Sensor Issues (Age-adjusted for non-linear model)
    print("     Identifying sensor issues...")
    profiles['has_sensor_issue'] = False
    profiles['sensor_issue_reason'] = ''

    if 'battery_drain_rate_per_day' in profiles.columns:
        # Always flag negative/zero drain
        negative_mask = profiles['battery_drain_rate_per_day'] <= 0
        profiles.loc[negative_mask, 'has_sensor_issue'] = True
        profiles.loc[negative_mask, 'sensor_issue_reason'] = 'Negative/zero drain rate'
    
        # Age-adjusted high drain check for non-linear model
        if 'pct_life_used' in profiles.columns:
            # Allow higher drain rates for older batteries
            # New battery (0% life used): max 0.5%/day
            # Old battery (100% life used): max 2.5%/day
            max_allowed_drain = 0.5 + (profiles['pct_life_used'] * 2.0)
            
            # Consider battery stress for low batteries
            if 'battery_current' in profiles.columns:
                battery_stress = np.where(profiles['battery_current'] < 30, 0.2, 0.0)
                max_allowed_drain = max_allowed_drain + battery_stress
            
            high_drain_mask = profiles['battery_drain_rate_per_day'] > max_allowed_drain
            profiles.loc[high_drain_mask, 'sensor_issue_reason'] = 'Age-adjusted high drain rate'
        else:
            # Fallback: simple threshold
            high_drain_mask = profiles['battery_drain_rate_per_day'] > 1.0
            profiles.loc[high_drain_mask, 'sensor_issue_reason'] = 'High drain (>1.0%/day)'
        
        profiles.loc[high_drain_mask, 'has_sensor_issue'] = True

    if 'battery_current' in profiles.columns:
        # Flag impossible battery levels
        impossible_mask = (profiles['battery_current'] < 0) | (profiles['battery_current'] > 100)
        profiles.loc[impossible_mask, 'has_sensor_issue'] = True
        profiles.loc[impossible_mask, 'sensor_issue_reason'] = 'Impossible battery level'
    
    # Clean up reasons
    profiles['sensor_issue_reason'] = profiles['sensor_issue_reason'].str.strip()
    
    # 3. NON-LINEAR Battery Life Analysis
    print("    Analyzing NON-LINEAR battery life projections...")

    profiles['projected_life_years'] = None
    profiles['years_remaining'] = None
    profiles['days_to_critical'] = None
    profiles['days_to_warning'] = None
    profiles['life_reduction_pct'] = None
    profiles['remaining_life_category'] = 'Unknown'

    profiles['less_than_10yr_life'] = False
    profiles['is_slightly_draining'] = False
    profiles['is_moderately_draining'] = False
    profiles['is_fast_draining'] = False
    profiles['is_critical_draining'] = False

    profiles['needs_replacement_1year'] = False
    profiles['needs_replacement_2year'] = False
    profiles['schedule_replacement_3year'] = False

    if 'battery_drain_rate_per_day' in profiles.columns and 'battery_current' in profiles.columns:
        
        # ===== NON-LINEAR TIME-TO-FAILURE CALCULATION =====
        def calculate_non_linear_days(current_battery, base_daily_drain, target_level, age_factor=0.5):
            """Calculate days to reach target battery level with non-linear drain acceleration"""
            
            if pd.isna(base_daily_drain) or base_daily_drain <= 0 or pd.isna(current_battery):
                return np.nan
            
            if current_battery <= target_level:
                return 0
            
            days = 0
            battery = float(current_battery)
            
            # Maximum simulation to prevent infinite loops
            max_days = 3650  # 10 years
            
            while battery > target_level and days < max_days:
                # Calculate today's drain rate with NON-LINEAR acceleration
                # Factor 1: Battery stress (lower battery = faster drain)
                battery_stress = 1.0 + ((100.0 - battery) / 100.0) * 0.3
                
                # Factor 2: Aging acceleration (older batteries drain faster)
                # Current age increases as battery depletes
                current_age = age_factor + ((100.0 - battery) / 100.0) * 0.15
                aging_accel = 1.0 + (current_age ** 1.5) * 1.2  # Sub-exponential growth
                
                # Today's accelerated drain rate
                today_drain = base_daily_drain * battery_stress * aging_accel
                
                # Apply today's drain
                battery -= today_drain
                days += 1
                
                # Early exit if drain is extremely slow
                if today_drain < 0.0001:
                    days = max_days
                    break
            
            return min(days, max_days)
        
        # Get age factor for non-linear calculations
        if 'pct_life_used' in profiles.columns:
            age_factors = profiles['pct_life_used'].fillna(0.5)
        else:
            # Estimate age factor from device age
            if 'device_age_days' in profiles.columns:
                # Assume 10-year nominal life
                age_factors = (profiles['device_age_days'] / 3650).fillna(0.5).clip(upper=1.0)
            else:
                age_factors = pd.Series(0.5, index=profiles.index)
        
        # Calculate days to warning (30%) and critical (20%)
        days_to_warning_list = []
        days_to_critical_list = []
        years_remaining_list = []
        
        for idx, row in profiles.iterrows():
            current_battery = row.get('battery_current', 100)
            base_drain = row.get('battery_drain_rate_per_day', 0.0274)
            age_factor = age_factors.iloc[idx]
            
            # Calculate days to warning (30%)
            days_warning = calculate_non_linear_days(
                current_battery, base_drain, 30, age_factor
            )
            days_to_warning_list.append(days_warning)
            
            # Calculate days to critical (20%)
            days_critical = calculate_non_linear_days(
                current_battery, base_drain, 20, age_factor
            )
            days_to_critical_list.append(days_critical)
            
            # Calculate years remaining (to 0%)
            if pd.notna(days_critical):
                # Extrapolate from critical to 0% (accelerated)
                battery_at_critical = 20
                remaining_to_zero = calculate_non_linear_days(
                    battery_at_critical, base_drain * 1.5, 0, age_factor + 0.2
                )
                total_days_to_zero = days_critical + remaining_to_zero
                years_remaining = total_days_to_zero / 365.0
            else:
                years_remaining = np.nan
            
            years_remaining_list.append(years_remaining)
        
        # Assign calculated values
        profiles['days_to_warning'] = days_to_warning_list
        profiles['days_to_critical'] = days_to_critical_list
        profiles['years_remaining'] = years_remaining_list
        
        # Calculate projected total life
        profiles['projected_life_years'] = profiles['years_remaining'] / (profiles['battery_current'] / 100.0)
        
        # Handle NaN/infinite values
        profiles.loc[profiles['projected_life_years'] > 30, 'projected_life_years'] = 30
        profiles.loc[profiles['years_remaining'] > 30, 'years_remaining'] = 30
        profiles.loc[profiles['projected_life_years'] < 0, 'projected_life_years'] = 0
        profiles.loc[profiles['years_remaining'] < 0, 'years_remaining'] = 0
        
        # Life reduction percentage (compared to 10-year nominal)
        profiles['life_reduction_pct'] = ((10 - profiles['projected_life_years']) / 10) * 100
        profiles['life_reduction_pct'] = profiles['life_reduction_pct'].clip(lower=0, upper=100)
        
        # Drain speed categories based on projected life
        profiles['less_than_10yr_life'] = profiles['projected_life_years'] < 10
        profiles.loc[profiles['projected_life_years'] < 9, 'is_slightly_draining'] = True
        profiles.loc[profiles['projected_life_years'] < 8, 'is_moderately_draining'] = True
        profiles.loc[profiles['projected_life_years'] < 7, 'is_fast_draining'] = True
        profiles.loc[profiles['projected_life_years'] < 5, 'is_critical_draining'] = True
        
        # Remaining life urgency categories
        profiles['remaining_life_category'] = np.where(
            profiles['days_to_critical'] < 30,
            'EMERGENCY (<1 month)',
            np.where(
                profiles['days_to_critical'] < 90,
                'CRITICAL (1-3 months)',
                np.where(
                    profiles['days_to_critical'] < 180,
                    'HIGH (3-6 months)',
                    np.where(
                        profiles['days_to_critical'] < 365,
                        'MEDIUM (6-12 months)',
                        'SAFE (>12 months)'
                    )
                )
            )
        )
        
        # Replacement scheduling flags
        profiles['needs_replacement_1year'] = profiles['days_to_critical'] < 365
        profiles['needs_replacement_2year'] = (profiles['days_to_critical'] >= 365) & (profiles['days_to_critical'] < 730)
        profiles['schedule_replacement_3year'] = (profiles['days_to_critical'] >= 730) & (profiles['days_to_critical'] < 1095)
        
        # Life expectancy categories
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
    
    # 4. Batteries Closest to Failure
    print("   Identifying critical batteries...")
    
    if 'days_to_critical' in profiles.columns:
        profiles['estimated_days_remaining'] = profiles['days_to_critical']
        profiles['critical_battery_30d'] = profiles['days_to_critical'] < 30
        profiles['emergency_battery_7d'] = profiles['days_to_critical'] < 7
    elif 'battery_current' in profiles.columns and 'battery_drain_rate_per_day' in profiles.columns:
        # Fallback linear calculation
        profiles['estimated_days_remaining'] = np.where(
            (profiles['battery_drain_rate_per_day'] > 0.0001) & 
            (profiles['battery_drain_rate_per_day'].notna()),
            (profiles['battery_current'] - 20) / profiles['battery_drain_rate_per_day'],
            np.where(
                profiles['battery_current'].notna(),
                (profiles['battery_current'] - 20) / (80 / (10 * 365)),  # 80% over 10 years
                np.nan
            )
        )
        
        profiles['critical_battery_30d'] = profiles['estimated_days_remaining'] < 30
        profiles['emergency_battery_7d'] = profiles['estimated_days_remaining'] < 7
    
    # 5. Overall Risk Score (updated with non-linear factors)
    print("   Calculating overall risk score...")
    
    profiles['comm_risk_score'] = 0
    profiles['sensor_risk_score'] = 0
    profiles['battery_risk_score'] = 0
    profiles['drain_risk_score'] = 0
    profiles['age_accel_risk_score'] = 0  # NEW: Age acceleration risk
    
    # Communication risk
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
    
    # Drain rate risk (non-linear aware)
    if 'battery_drain_rate_per_day' in profiles.columns:
        expected_rate = 100 / (10 * 365)  # 10-year nominal
        # Higher penalty for faster-than-expected drains
        drain_ratio = profiles['battery_drain_rate_per_day'] / expected_rate
        profiles.loc[drain_ratio > 10, 'drain_risk_score'] = 100
        profiles.loc[drain_ratio > 5, 'drain_risk_score'] = 80
        profiles.loc[drain_ratio > 2, 'drain_risk_score'] = 40
        profiles.loc[drain_ratio > 1.5, 'drain_risk_score'] = 20
    
    # Age acceleration risk (NEW: penalize older devices more)
    if 'pct_life_used' in profiles.columns:
        profiles.loc[profiles['pct_life_used'] > 0.9, 'age_accel_risk_score'] = 50
        profiles.loc[profiles['pct_life_used'] > 0.7, 'age_accel_risk_score'] = 30
        profiles.loc[profiles['pct_life_used'] > 0.5, 'age_accel_risk_score'] = 15
    
    # Overall risk score (weighted)
    profiles['overall_risk_score'] = (
        profiles['comm_risk_score'] * 0.25 +      # Reduced weight
        profiles['sensor_risk_score'] * 0.15 +    # Reduced weight
        profiles['battery_risk_score'] * 0.25 +   # Same weight
        profiles['drain_risk_score'] * 0.20 +     # Increased weight
        profiles['age_accel_risk_score'] * 0.15   # NEW: Age acceleration risk
    )
    
    profiles['risk_category'] = pd.cut(
        profiles['overall_risk_score'],
        bins=[0, 20, 40, 60, 80, 101],
        labels=['Low', 'Medium', 'High', 'Critical', 'Emergency'],
        right=False
    )
    
    # 6. Priority for Action (non-linear aware)
    profiles['priority_level'] = 'Normal'
    
    # EMERGENCY: Critical batteries or no communication
    emergency_conditions = (
        profiles['emergency_battery_7d'] |
        (profiles['communication_status'] == 'No Data') |
        (profiles['overall_risk_score'] >= 80) |
        (profiles['is_critical_draining'] & (profiles['days_to_critical'] < 90))
    )
    profiles.loc[emergency_conditions, 'priority_level'] = 'Emergency'
    
    # HIGH: Fast draining or soon-to-fail
    high_conditions = (
        profiles['critical_battery_30d'] |
        profiles['is_fast_draining'] |
        (profiles['is_moderately_draining'] & (profiles['days_to_critical'] < 180)) |
        profiles['has_sensor_issue'] |
        (profiles['communication_status'] == 'Inactive (>30d)') |
        (profiles['overall_risk_score'] >= 60)
    ) & (profiles['priority_level'] != 'Emergency')
    profiles.loc[high_conditions, 'priority_level'] = 'High'
    
    # MEDIUM: Warning signs
    medium_conditions = (
        profiles['not_communicating_7d'] |
        (profiles['battery_current'] < 30) |
        profiles['is_moderately_draining'] |
        (profiles['overall_risk_score'] >= 40)
    ) & (profiles['priority_level'] == 'Normal')
    profiles.loc[medium_conditions, 'priority_level'] = 'Medium'
    
    # Add non-linear acceleration warning
    if 'battery_degradation_factor' in profiles.columns:
        profiles['high_acceleration_warning'] = profiles['battery_degradation_factor'] > 1.3
        profiles.loc[profiles['high_acceleration_warning'], 'priority_level'] = np.where(
            profiles.loc[profiles['high_acceleration_warning'], 'priority_level'] == 'Normal',
            'Medium',
            profiles.loc[profiles['high_acceleration_warning'], 'priority_level']
        )
    
    # Add drain acceleration factor if available
    if 'aging_acceleration_factor' in profiles.columns:
        profiles['drain_acceleration_ratio'] = profiles['aging_acceleration_factor']
        profiles['is_accelerating_drain'] = profiles['drain_acceleration_ratio'] > 1.2
    
    return profiles

def create_critical_devices_export(profiles, output_dir):
    """Create special exports for critical devices"""
    
    print("\n CREATING CRITICAL DEVICES EXPORTS")
    
    # 1. Devices Not Communicating (>7 days)
    not_comm_devices = profiles[profiles['not_communicating_7d'] == True]
    not_comm_file = output_dir / "critical_not_communicating.csv"
    not_comm_devices.to_csv(not_comm_file, index=False)
    print(f"Devices not communicating (>7d): {len(not_comm_devices)}")
    
    # 2. Devices with Sensor Issues
    sensor_issue_devices = profiles[profiles['has_sensor_issue'] == True]
    sensor_issue_file = output_dir / "critical_sensor_issues.csv"
    sensor_issue_devices.to_csv(sensor_issue_file, index=False)
    print(f" Devices with sensor issues: {len(sensor_issue_devices)}")
    
    # 3. Fast Draining Batteries
    fast_drain_devices = profiles[profiles['is_fast_draining'] == True]
    fast_drain_file = output_dir / "critical_fast_draining.csv"
    fast_drain_devices.to_csv(fast_drain_file, index=False)
    print(f"Fast draining batteries: {len(fast_drain_devices)}")
    
    # 4. Batteries Closest to Death (<30 days)
    critical_battery_devices = profiles[profiles['critical_battery_30d'] == True]
    critical_battery_file = output_dir / "critical_battery_low.csv"
    critical_battery_devices.to_csv(critical_battery_file, index=False)
    print(f" Batteries with <30 days remaining: {len(critical_battery_devices)}")
    
    # 5. Emergency Batteries (<7 days)
    emergency_devices = profiles[profiles['emergency_battery_7d'] == True]
    emergency_file = output_dir / "emergency_battery.csv"
    emergency_devices.to_csv(emergency_file, index=False)
    print(f" EMERGENCY: Batteries with <7 days remaining: {len(emergency_devices)}")
    
    # 6. All Critical Devices (combined view)
    critical_columns = ['Serial', 'device_health_status', 'priority_level', 'risk_category', 
                       'overall_risk_score', 'battery_current', 
                       'estimated_days_remaining', 'communication_status',
                       'battery_drain_rate_per_day', 'has_sensor_issue',
                       'is_fast_draining', 'critical_battery_30d']
    
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
        print(f" All critical devices view: {len(critical_view)} devices")
    
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
    print(f" Critical metrics summary created")
    
    return summary_df

# ====================================================
# Time-to-Failure Functions
# ====================================================

def create_time_to_failure_estimation(profiles, output_dir):
    """
    Calculate NON-LINEAR time-to-failure based on battery drain rates
    """
    
    print("\n CREATING NON-LINEAR TIME-TO-FAILURE ESTIMATES")
    print("=" * 50)
    
    # Filter for ZM1 devices only
    zm1_devices = profiles[profiles['Device_Type_Standardized'] == 'ZM1'].copy()
    
    if len(zm1_devices) == 0:
        print("  No ZM1 devices found")
        return None
    
    print(f"   ZM1 devices: {len(zm1_devices)}")
    
    # Expected drain rates for different scenarios (for fallback only)
    EXPECTED_DRAIN_RATES = {
        'optimistic': 0.0274,      # 10-year battery life
        'typical': 0.0548,         # 5-year actual life
        'pessimistic': 0.1096,     # 2.5-year life
        'fast_draining': 0.2740    # 1-year life
    }
    
    failure_data = []
    devices_with_issues = 0
    
    for _, device in zm1_devices.iterrows():
        serial = device['Serial']
        current_battery = device.get('battery_current', 100)
        
        # Get non-linear factors
        base_drain = device.get('battery_drain_rate_per_day', 0.0274)
        age_factor = device.get('pct_life_used', 0.5)
        
        # Determine data source
        if ('battery_drain_rate_per_day' in device and 
            not pd.isna(device['battery_drain_rate_per_day']) and
            device['battery_drain_rate_per_day'] > 0.01):
            drain_rate = device['battery_drain_rate_per_day']
            data_source = 'Actual'
        else:
            # Use expected rates based on device condition
            if device.get('is_fast_draining', False):
                drain_rate = EXPECTED_DRAIN_RATES['fast_draining']
                data_source = 'Expected (Fast)'
            elif device.get('is_moderately_draining', False):
                drain_rate = EXPECTED_DRAIN_RATES['pessimistic']
                data_source = 'Expected (Moderate)'
            elif device.get('is_slightly_draining', False):
                drain_rate = EXPECTED_DRAIN_RATES['typical']
                data_source = 'Expected (Typical)'
            else:
                drain_rate = EXPECTED_DRAIN_RATES['optimistic']
                data_source = 'Expected (Optimistic)'
        
        # Ensure drain rate is positive
        if drain_rate <= 0:
            drain_rate = EXPECTED_DRAIN_RATES['optimistic']
            data_source = 'Expected (Optimistic - zero drain)'
        
        # ===== NON-LINEAR CALCULATION =====
        def non_linear_days_to_level(target_level, current_battery, base_drain, age_factor):
            """Calculate days to reach target battery level with non-linear drain"""
            
            if base_drain <= 0 or current_battery <= target_level:
                return 0
            
            days = 0
            battery = float(current_battery)
            
            # Maximum simulation to prevent infinite loops
            max_days = 3650  # 10 years
            
            while battery > target_level and days < max_days:
                # Calculate today's drain rate with NON-LINEAR acceleration
                # Factor 1: Battery stress (lower battery = faster drain)
                battery_stress = 1.0 + ((100.0 - battery) / 100.0) * 0.3
                
                # Factor 2: Aging acceleration (older batteries drain faster)
                # Current age increases as battery depletes
                current_age = age_factor + ((100.0 - battery) / 100.0) * 0.15
                aging_accel = 1.0 + (current_age ** 1.5) * 1.2
                
                # Today's accelerated drain rate
                today_drain = base_drain * battery_stress * aging_accel
                
                # Apply today's drain
                battery -= today_drain
                days += 1
                
                # Early exit if drain is extremely slow
                if today_drain < 0.0001:
                    days = max_days
                    break
            
            return min(days, max_days)
        
        # Calculate days to warning (30%) and critical (20%)
        days_to_warning = non_linear_days_to_level(30, current_battery, drain_rate, age_factor)
        days_to_critical = non_linear_days_to_level(20, current_battery, drain_rate, age_factor)
        
        # Calculate days to zero (0%) - use accelerated rate
        days_to_zero = non_linear_days_to_level(0, current_battery, drain_rate * 1.2, age_factor)
        
        # Handle edge cases
        if pd.isna(days_to_critical):
            days_to_critical = 3650
        
        if days_to_critical < 0:
            days_to_critical = 0
            devices_with_issues += 1
        
        # Convert to more readable formats
        months_to_critical = days_to_critical / 30.44
        years_to_critical = days_to_critical / 365
        
        # Determine urgency category
        if days_to_critical <= 30:
            urgency = 'EMERGENCY (<1 month)'
            priority = 1
        elif days_to_critical <= 90:
            urgency = 'CRITICAL (1-3 months)'
            priority = 2
        elif days_to_critical <= 180:
            urgency = 'HIGH (3-6 months)'
            priority = 3
        elif days_to_critical <= 365:
            urgency = 'MEDIUM (6-12 months)'
            priority = 4
        elif days_to_critical <= 730:
            urgency = 'LOW (1-2 years)'
            priority = 5
        elif days_to_critical <= 1825:
            urgency = 'PLAN (2-5 years)'
            priority = 6
        else:
            urgency = 'FUTURE (>5 years)'
            priority = 7
        
        # Calculate replacement date
        today = pd.Timestamp.now()
        
        try:
            replacement_date = today + pd.Timedelta(days=days_to_critical)
            replacement_date_str = replacement_date.date()
        except:
            replacement_date = today + pd.Timedelta(days=3650)
            replacement_date_str = replacement_date.date()
        
        failure_data.append({
            'Serial': serial,
            'Current_Battery_%': current_battery,
            'Drain_Rate_%_per_Day': drain_rate,
            'Drain_Rate_Source': data_source,
            'Base_Drain_Rate': base_drain,  # Keep original for reference
            'Age_Factor': age_factor,
            'Days_To_Warning': round(days_to_warning),
            'Days_To_Critical': round(days_to_critical),
            'Months_To_Critical': round(months_to_critical, 1),
            'Years_To_Critical': round(years_to_critical, 1),
            'Days_To_Zero': round(days_to_zero),
            'Urgency_Category': urgency,
            'Priority_Level': priority,
            'Expected_Replacement_Date': replacement_date_str,
            'Quarter_Replacement': f"Q{(replacement_date.month - 1) // 3 + 1}-{replacement_date.year}",
            'Year_Replacement': replacement_date.year,
            'Latitude': device.get('Latitude'),
            'Longitude': device.get('Longitude'),
            'Device_Age_Days': device.get('device_age_days'),
            'pct_life_used': device.get('pct_life_used', 0.5),
            'Original_Expected_Life_Days': 3650,
            'Remaining_Life_%': (days_to_zero / 3650 * 100) if days_to_zero < float('inf') else 100,
            'Is_Non_Linear_Model': True  # Flag to indicate this is non-linear
        })
    
    # Create DataFrame
    failure_df = pd.DataFrame(failure_data)
    
    if len(failure_df) == 0:
        print(" No failure data created")
        return None
    
    # Sort by urgency (soonest first)
    failure_df = failure_df.sort_values('Priority_Level')
    
    # Save detailed data
    failure_file = output_dir / "time_to_failure_detailed.csv"
    failure_df.to_csv(failure_file, index=False)
    
    # Create summary statistics
    summary_data = {
        'Metric': [
            'Total ZM1 Devices',
            'Emergency (<1 month)',
            'Critical (1-3 months)',
            'High (3-6 months)',
            'Medium (6-12 months)',
            'Low (1-2 years)',
            'Plan (2-5 years)',
            'Future (>5 years)',
            'Avg Days to Critical',
            'Avg Years to Critical',
            'Devices Using Actual Data',
            'Devices Using Expected Data',
            'Devices with Battery < 20%',
            'Devices Already Critical',
            'Non-Linear Model Used'
        ],
        'Value': [
            len(failure_df),
            len(failure_df[failure_df['Priority_Level'] == 1]),
            len(failure_df[failure_df['Priority_Level'] == 2]),
            len(failure_df[failure_df['Priority_Level'] == 3]),
            len(failure_df[failure_df['Priority_Level'] == 4]),
            len(failure_df[failure_df['Priority_Level'] == 5]),
            len(failure_df[failure_df['Priority_Level'] == 6]),
            len(failure_df[failure_df['Priority_Level'] == 7]),
            round(failure_df['Days_To_Critical'].mean()),
            round(failure_df['Years_To_Critical'].mean(), 1),
            len(failure_df[failure_df['Drain_Rate_Source'] == 'Actual']),
            len(failure_df[failure_df['Drain_Rate_Source'].str.contains('Expected')]),
            len(failure_df[failure_df['Current_Battery_%'] < 20]),
            devices_with_issues,
            'Yes'  # Always yes now
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "time_to_failure_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Create calendar view for replacement scheduling
    calendar_view = failure_df[['Serial', 'Expected_Replacement_Date', 'Quarter_Replacement', 
                               'Year_Replacement', 'Urgency_Category', 'Current_Battery_%',
                               'Days_To_Critical']].copy()
    calendar_file = output_dir / "replacement_calendar.csv"
    calendar_view.to_csv(calendar_file, index=False)
    
    print(f"\n NON-LINEAR time-to-failure estimates created:")
    print(f"   Detailed data: {failure_file}")
    print(f"   Summary: {summary_file}")
    print(f"   Replacement calendar: {calendar_file}")
    
    # Print key insights
    print(f"\n KEY INSIGHTS (NON-LINEAR MODEL):")
    print(f"   Average time to critical: {failure_df['Years_To_Critical'].mean():.1f} years")
    print(f"   Minimum time to critical: {failure_df['Days_To_Critical'].min():.0f} days")
    
    urgent_count = len(failure_df[failure_df['Priority_Level'] <= 3])
    if urgent_count > 0:
        print(f"     {urgent_count} devices need attention within 6 months")
    
    actual_data_pct = (len(failure_df[failure_df['Drain_Rate_Source'] == 'Actual']) / len(failure_df) * 100)
    print(f"   {actual_data_pct:.1f}% using actual drain rates, rest using expected rates")
    
    if devices_with_issues > 0:
        print(f"     {devices_with_issues} devices already at or below 20% battery")
    
    # Show acceleration factor
    if 'Age_Factor' in failure_df.columns:
        avg_acceleration = failure_df['Age_Factor'].mean()
        print(f"   Average age acceleration factor: {avg_acceleration:.2f}")
    
    return failure_df

def create_failure_timeline_visualization(failure_df, output_dir):
    """
    Create visualization data showing failure timeline
    """
    
    print("\n CREATING FAILURE TIMELINE VISUALIZATION")
    print("=" * 50)
    
    # Group by replacement quarter
    timeline_data = failure_df.groupby('Quarter_Replacement').agg({
        'Serial': 'count',
        'Current_Battery_%': 'mean',
        'Days_To_Critical': 'mean'
    }).reset_index()
    
    timeline_data = timeline_data.rename(columns={
        'Serial': 'Device_Count',
        'Current_Battery_%': 'Avg_Battery_%',
        'Days_To_Critical': 'Avg_Days_To_Critical'
    })
    
    # Sort by quarter
    timeline_data['Year'] = timeline_data['Quarter_Replacement'].str.split('-').str[1].astype(int)
    timeline_data['Quarter'] = timeline_data['Quarter_Replacement'].str.split('-').str[0].str[1].astype(int)
    timeline_data = timeline_data.sort_values(['Year', 'Quarter'])
    
    # Save
    timeline_file = output_dir / "failure_timeline_by_quarter.csv"
    timeline_data.to_csv(timeline_file, index=False)
    
    print(f" Failure timeline by quarter: {timeline_file}")
    
    # Create monthly view for next 12 months
    today = pd.Timestamp.now()
    monthly_data = []
    
    for months_ahead in range(1, 13):
        target_date = today + pd.Timedelta(days=30 * months_ahead)
        target_quarter = f"Q{(target_date.month - 1) // 3 + 1}-{target_date.year}"
        
        # Count devices that will reach critical by this date
        devices_due = failure_df[
            (failure_df['Days_To_Critical'] <= months_ahead * 30) & 
            (failure_df['Days_To_Critical'] > (months_ahead - 1) * 30)
        ]
        
        monthly_data.append({
            'Month': target_date.strftime('%b %Y'),
            'Month_Number': months_ahead,
            'Devices_Due': len(devices_due),
            'Cumulative_Due': len(failure_df[failure_df['Days_To_Critical'] <= months_ahead * 30])
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    monthly_file = output_dir / "monthly_failure_forecast.csv"
    monthly_df.to_csv(monthly_file, index=False)
    
    print(f" Monthly failure forecast: {monthly_file}")
    
    return timeline_data, monthly_df

def create_battery_drain_simulation(failure_df, output_dir):
    """Create NON-LINEAR battery drain simulation for urgent devices"""
    
    print("\n CREATING NON-LINEAR BATTERY DRAIN SIMULATION")
    print("=" * 50)
    
    # Get urgent devices (fast-draining AND soon-to-fail)
    fast_devices = failure_df[failure_df['Drain_Rate_%_per_Day'] > 0.1].copy()
    soon_devices = failure_df[failure_df['Days_To_Critical'] < 365].copy()
    
    # Combine and remove duplicates
    urgent_devices = pd.concat([fast_devices, soon_devices]).drop_duplicates(subset=['Serial'])
    
    # If we have too few, include borderline cases
    if len(urgent_devices) < 10:
        moderate = failure_df[
            (failure_df['Drain_Rate_%_per_Day'] > 0.05) & 
            (failure_df['Drain_Rate_%_per_Day'] <= 0.1)
        ].copy()
        urgent_devices = pd.concat([urgent_devices, moderate]).drop_duplicates(subset=['Serial'])
    
    # Limit to top 30 for visualization
    if len(urgent_devices) > 30:
        urgent_devices = urgent_devices.head(30)
    
    print(f"   Simulating {len(urgent_devices)} urgent devices")
    
    simulation_data = []
    
    for _, device in urgent_devices.iterrows():
        serial = device['Serial']
        current_battery = device['Current_Battery_%']
        base_drain = device['Drain_Rate_%_per_Day']
        age_factor = device.get('Age_Factor', 0.5)
        
        # Simulate for appropriate timeframe
        simulation_days = min(180, int(device['Days_To_Critical']) + 30)
        
        battery = float(current_battery)
        
        for day in range(0, simulation_days + 1, 2):  # Every 2 days for performance
            # Calculate NON-LINEAR drain for this day
            battery_stress = 1.0 + ((100.0 - battery) / 100.0) * 0.3
            current_age = age_factor + ((100.0 - battery) / 100.0) * 0.15
            aging_accel = 1.0 + (current_age ** 1.5) * 1.2
            
            today_drain = base_drain * battery_stress * aging_accel
            
            # Apply drain (but only simulate forward from day 0)
            simulated_battery = max(0, current_battery - (today_drain * day))
            
            # Determine status
            if simulated_battery < 20:
                status = 'CRITICAL'
            elif simulated_battery < 30:
                status = 'WARNING'
            else:
                status = 'NORMAL'
            
            simulation_data.append({
                'Serial': serial,
                'Date': pd.Timestamp.now() + pd.Timedelta(days=day),
                'Battery_Level': round(simulated_battery, 2),
                'Day': day,
                'Status': status,
                'Drain_Rate': base_drain,
                'Age_Factor': age_factor,
                'Days_To_Critical': max(0, device['Days_To_Critical'] - day),
                'Current_Battery_Start': current_battery,
                'Is_Fast_Draining': base_drain > 0.1,
                'Is_Soon_Failing': device['Days_To_Critical'] < 365,
                'Estimated_Failure_Date': device['Expected_Replacement_Date']
            })
    
    simulation_df = pd.DataFrame(simulation_data)
    
    # Add reference lines
    reference_data = []
    max_date = simulation_df['Date'].max()
    date_range = pd.date_range(start=pd.Timestamp.now(), end=max_date, periods=30)
    
    for date in date_range:
        reference_data.append({
            'Serial': 'REF_20%_CRITICAL',
            'Date': date,
            'Battery_Level': 20,
            'Day': (date - pd.Timestamp.now()).days,
            'Status': 'REFERENCE',
            'Is_Reference_Line': True
        })
        
        reference_data.append({
            'Serial': 'REF_30%_WARNING',
            'Date': date,
            'Battery_Level': 30,
            'Day': (date - pd.Timestamp.now()).days,
            'Status': 'REFERENCE',
            'Is_Reference_Line': True
        })
    
    reference_df = pd.DataFrame(reference_data)
    simulation_df = pd.concat([simulation_df, reference_df], ignore_index=True)
    
    # Save simulation data
    simulation_file = output_dir / "battery_drain_simulation.csv"
    simulation_df.to_csv(simulation_file, index=False)

    print(f" Battery drain simulation created: {simulation_file}")
    print(f"   Total records: {len(simulation_df):,}")
    print(f"   Unique devices: {simulation_df[simulation_df['Is_Reference_Line'] != True]['Serial'].nunique()}")
    
    # Create summary
    urgent_summary = urgent_devices[[
        'Serial', 'Current_Battery_%', 'Drain_Rate_%_per_Day', 
        'Days_To_Critical', 'Urgency_Category', 'Age_Factor'
    ]].copy()
    
    urgent_summary_file = output_dir / "urgent_devices_summary.csv"
    urgent_summary.to_csv(urgent_summary_file, index=False)
    
    print(f" Urgent devices summary: {urgent_summary_file}")
    
    return simulation_df

# ====================================================
# Main Function
# ====================================================

def create_powerbi_exports():
    """Create optimized files for Power BI import."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "powerbi_exports"
    output_dir.mkdir(exist_ok=True)
    
    print(" CREATING POWER BI EXPORTS")
    print("=" * 50)
    
    # 1. DEVICE PROFILES (Main dashboard data)
    profiles_file = project_root / "data" / "processed" / "time_series" / "all_device_profiles_summary.csv"
    if profiles_file.exists():
        try:
            profiles = pd.read_csv(profiles_file, low_memory=False)
            print(f"    Loaded profiles: {len(profiles):,} devices")
        except Exception as e:
            print(f"    Error loading profiles: {e}")
            profiles = pd.DataFrame()
    else:
        print(" No device profiles file found")
        profiles = pd.DataFrame()
    
    if len(profiles) > 0:
        # Clean ZM1 battery data
        if 'battery_drain_rate_per_day' in profiles.columns:
            profiles = clean_zm1_battery_data(profiles)
            print(f"   🔋 Cleaned battery data")
        
        # Add derived columns for Power BI
        if 'battery_current' in profiles.columns:
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
        if 'risk_score_current' in profiles.columns:
            profiles['risk_category'] = pd.cut(
                profiles['risk_score_current'].fillna(0),
                bins=[0, 20, 40, 60, 80, 101],
                labels=['Low (0-20)', 'Medium (21-40)', 'High (41-60)', 'Critical (61-80)', 'Emergency (81-100)']
            )
        
        # Add battery drain categories
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
        print(f" Device profiles: {profiles_output}")
        print(f"   Devices: {len(profiles):,}")
        print(f"   Columns: {len(profiles.columns)}")
    
    # 2. TIME SERIES SAMPLE (for trends)
    ts_files = list((project_root / "data" / "processed" / "time_series").glob("*_health_all_timeseries.csv"))
    if ts_files:
        ts_file = ts_files[-1]
        try:
            ts_data = pd.read_csv(ts_file, low_memory=False)
            print(f"    Loaded time series: {len(ts_data):,} records")
        except Exception as e:
            print(f"    Error loading time series: {e}")
            ts_data = pd.DataFrame()
    else:
        print(" No time series files found")
        ts_data = pd.DataFrame()
    
    if len(ts_data) > 0:
        # Convert date column
        if 'date' in ts_data.columns:
            ts_data['date'] = pd.to_datetime(ts_data['date'])
        elif 'Date' in ts_data.columns:
            ts_data['Date'] = pd.to_datetime(ts_data['Date'])
        
        # Get last 30 days for communication analysis
        date_col = 'date' if 'date' in ts_data.columns else 'Date'
        if date_col in ts_data.columns:
            latest_date = ts_data[date_col].max()
            cutoff_date = latest_date - pd.Timedelta(days=30)
            ts_sample = ts_data[ts_data[date_col] >= cutoff_date]
            
            # Create critical metrics analysis
            if len(profiles) > 0:
                print("    Creating critical metrics analysis...")
                profiles = create_critical_metrics_summary(profiles, ts_sample)
            
            # Create critical devices exports
            if len(profiles) > 0:
                summary_df = create_critical_devices_export(profiles, output_dir)
            
            # Save the updated profiles with critical metrics
            if len(profiles) > 0:
                profiles_output = output_dir / "device_profiles_powerbi.csv"
                profiles.to_csv(profiles_output, index=False)
            
            # Save time series sample
            ts_output = output_dir / "time_series_sample_powerbi.csv"
            ts_sample.to_csv(ts_output, index=False)
            print(f" Time series sample (last 30 days): {ts_output}")
            print(f"   Records: {len(ts_sample):,}")
    
    # 3. TIME-TO-FAILURE AND SIMULATION
    if len(profiles) > 0:
        print("\n CREATING TIME-TO-FAILURE ESTIMATES")
        failure_df = create_time_to_failure_estimation(profiles, output_dir)
        
        if failure_df is not None:
            # Create timeline visualization
            create_failure_timeline_visualization(failure_df, output_dir)
            
            # Create battery drain simulation
            simulation_df = create_battery_drain_simulation(failure_df, output_dir)
    
    # 4. DAILY SUMMARY STATS
    daily_stats = []
    clean_daily_dir = project_root / "data" / "clean" / "daily"
    
    if clean_daily_dir.exists():
        daily_files = list(clean_daily_dir.glob("*.csv"))
        print(f"   Found {len(daily_files)} daily files")
        
        for daily_file in daily_files[:10]:  # Process first 10
            try:
                date = daily_file.stem.split('_')[0]
                df = pd.read_csv(daily_file, low_memory=False)
                
                stats = {
                    'date': date,
                    'total_devices': len(df),
                    'avg_battery': df['battery_level'].mean() if 'battery_level' in df.columns else None,
                    'avg_risk': df['risk_score'].mean() if 'risk_score' in df.columns else None,
                    'critical_devices': (df['risk_score'] > 80).sum() if 'risk_score' in df.columns else 0,
                    'low_battery_devices': (df['battery_level'] < 30).sum() if 'battery_level' in df.columns else 0
                }
                daily_stats.append(stats)
            except Exception as e:
                continue
    
    if daily_stats:
        daily_df = pd.DataFrame(daily_stats)
        daily_output = output_dir / "daily_summary_powerbi.csv"
        daily_df.to_csv(daily_output, index=False)
        print(f" Daily summary: {daily_output}")
        print(f"   Days: {len(daily_df)}")
    
    # 5. HEALTH DISTRIBUTION
    if len(profiles) > 0 and 'device_health_status' in profiles.columns:
        health_dist = profiles['device_health_status'].value_counts().reset_index()
        health_dist.columns = ['health_status', 'device_count']
        health_dist['percentage'] = (health_dist['device_count'] / health_dist['device_count'].sum() * 100).round(1)
        
        health_output = output_dir / "health_distribution_powerbi.csv"
        health_dist.to_csv(health_output, index=False)
        print(f" Health distribution: {health_output}")
    
    print(f"\n All Power BI files saved to: {output_dir}")
    
    # Show battery data summary
    if len(profiles) > 0 and 'battery_drain_rate_per_day' in profiles.columns:
        print(f"\n BATTERY DATA SUMMARY:")
        print(f"   Avg drain rate: {profiles['battery_drain_rate_per_day'].mean():.4f}%/day")
        print(f"   Max drain rate: {profiles['battery_drain_rate_per_day'].max():.2f}%/day")
        if 'battery_data_issue' in profiles.columns:
            charging_count = (profiles['battery_data_issue'] == 'Shows charging (impossible)').sum()
            print(f"   Devices showing 'charging' (fixed): {charging_count}")
        print(f"   Expected 10-year rate: {100/(10*365):.4f}%/day")

if __name__ == "__main__":
    create_powerbi_exports()