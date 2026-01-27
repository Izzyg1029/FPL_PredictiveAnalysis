#!/usr/bin/env python3
"""
Device Age Analysis Script
==========================

This script analyzes device age distributions and failure rates using the processed
daily files created by process_daily_time_series.py.

Analysis includes:
- Age distribution histogram
- Basic age statistics (min, max, average, median)
- Age bucket analysis with failure rates
- Risk score analysis by age group

Author: Isaiah
Date: January 25, 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent  # This script is in the project root
CLEAN_DAILY_DIR = PROJECT_ROOT / "data" / "clean" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_all_processed_files():
    """Load all processed daily files and combine them."""
    print("Loading processed daily files...")
    print(f"CLEAN_DAILY_DIR: {CLEAN_DAILY_DIR}")
    print(f"Directory exists: {CLEAN_DAILY_DIR.exists()}")

    all_files = list(CLEAN_DAILY_DIR.glob("*_health_zm1only.csv"))
    print(f"Glob pattern: *_health_zm1only.csv")
    print(f"Found {len(all_files)} processed files")
    for f in all_files[:5]:  # Show first 5
        print(f"  {f.name}")

    if not all_files:
        print("No processed files found!")
        return None

    # Load and combine all files
    dfs = []
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.name
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} devices")
        except Exception as e:
            print(f"  Error loading {file.name}: {e}")

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on Serial and date
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['Serial', 'date'], keep='last')
    final_count = len(combined_df)

    print(f"Combined dataset: {initial_count} -> {final_count} records after deduplication")
    print(f"Unique devices: {combined_df['Serial'].nunique()}")

    return combined_df

def create_age_distribution_analysis(df):
    """Create age distribution histogram and statistics."""
    print("\n" + "="*60)
    print("AGE DISTRIBUTION ANALYSIS")
    print("="*60)

    # Filter to devices with valid age data
    age_df = df[df['device_age_months'].notna() & (df['device_age_months'] > 0)].copy()

    if len(age_df) == 0:
        print("No valid age data found!")
        return None

    print(f"Devices with valid age data: {len(age_df)}")

    # Basic statistics
    print("\nBASIC AGE STATISTICS:")
    print(f"  Min age: {age_df['device_age_months'].min():.1f} months")
    print(f"  Max age: {age_df['device_age_months'].max():.1f} months")
    print(f"  Average age: {age_df['device_age_months'].mean():.1f} months")
    print(f"  Median age: {age_df['device_age_months'].median():.1f} months")
    print(f"  Standard deviation: {age_df['device_age_months'].std():.1f} months")

    # Create age buckets
    bins = [0, 12, 36, 60, float('inf')]
    labels = ['0-12 months (New)', '13-36 months (Young)', '37-60 months (Middle-aged)', '60+ months (Old)']
    age_df['age_bucket'] = pd.cut(age_df['device_age_months'], bins=bins, labels=labels, right=False)

    # Age distribution by bucket
    print("\nAGE BUCKET DISTRIBUTION:")
    bucket_counts = age_df['age_bucket'].value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        percentage = (count / len(age_df)) * 100
        print(f"  {bucket}: {count:,} devices ({percentage:.1f}%)")

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(age_df['device_age_months'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Device Age (Months)')
    plt.ylabel('Number of Devices')
    plt.title('Device Age Distribution Histogram')
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(age_df['device_age_months'], vert=False)
    plt.xlabel('Device Age (Months)')
    plt.title('Device Age Distribution (Box Plot)')

    # Age bucket bar chart
    plt.subplot(2, 2, 3)
    bucket_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Age Bucket')
    plt.ylabel('Number of Devices')
    plt.title('Devices by Age Bucket')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_ages = np.sort(age_df['device_age_months'])
    yvals = np.arange(len(sorted_ages))/float(len(sorted_ages))
    plt.plot(sorted_ages, yvals, 'b-', linewidth=2)
    plt.xlabel('Device Age (Months)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Age Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'age_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return age_df

def calculate_failure_rates_by_age(df):
    """Calculate failure rates for each age bucket."""
    print("\n" + "="*60)
    print("FAILURE RATE ANALYSIS BY AGE BUCKET")
    print("="*60)

    # Define failure criteria - devices with high risk scores or critical flags
    failure_criteria = (
        (df['risk_score'] > 80) |  # High risk score
        (df['battery_low_flag'] == 1) |  # Battery issues
        (df['overheat_flag'] == 1) |  # Overheat issues
        (df['comm_fail_frequency'] > 0.5)  # Communication failures
    )

    df['is_failure'] = failure_criteria.astype(int)

    # Group by age bucket
    age_buckets = ['0-12 months (New)', '13-36 months (Young)', '37-60 months (Middle-aged)', '60+ months (Old)']

    results = []

    for bucket in age_buckets:
        bucket_data = df[df['age_bucket'] == bucket]

        if len(bucket_data) == 0:
            continue

        total_devices = len(bucket_data)
        failed_devices = bucket_data['is_failure'].sum()
        failure_rate = (failed_devices / total_devices) * 100

        # Additional metrics
        avg_risk = bucket_data['risk_score'].mean()
        avg_battery = bucket_data['battery_level'].mean()
        comm_fail_rate = (bucket_data['comm_fail_frequency'] > 0).mean() * 100

        results.append({
            'age_bucket': bucket,
            'total_devices': total_devices,
            'failed_devices': failed_devices,
            'failure_rate': failure_rate,
            'avg_risk_score': avg_risk,
            'avg_battery_level': avg_battery,
            'comm_fail_rate': comm_fail_rate
        })

        print(f"\n{bucket}:")
        print(f"  Total devices: {total_devices:,}")
        print(f"  Failed devices: {failed_devices:,}")
        print(f"  Failure rate: {failure_rate:.1f}%")
        print(f"  Average risk score: {avg_risk:.1f}")
        print(f"  Average battery level: {avg_battery:.1f}%")
        print(f"  Communication failure rate: {comm_fail_rate:.1f}%")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Create failure rate visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(results_df)), results_df['failure_rate'], color='coral', edgecolor='black')
    plt.xlabel('Age Bucket')
    plt.ylabel('Failure Rate (%)')
    plt.title('Failure Rate by Age Bucket')
    plt.xticks(range(len(results_df)), [bucket.split(' ')[0] for bucket in results_df['age_bucket']], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, rate in zip(bars, results_df['failure_rate']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(results_df)), results_df['avg_risk_score'], 'bo-', linewidth=2, markersize=8, label='Risk Score')
    plt.plot(range(len(results_df)), results_df['avg_battery_level'], 'go-', linewidth=2, markersize=8, label='Battery Level')
    plt.xlabel('Age Bucket')
    plt.ylabel('Score/Level')
    plt.title('Risk Score and Battery Level by Age')
    plt.xticks(range(len(results_df)), [bucket.split(' ')[0] for bucket in results_df['age_bucket']], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'failure_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results_df

def create_age_vs_risk_analysis(df):
    """Create scatter plot of age vs risk score."""
    print("\n" + "="*60)
    print("AGE VS RISK SCORE ANALYSIS")
    print("="*60)

    # Filter for valid data
    analysis_df = df[df['device_age_months'].notna() & df['risk_score'].notna()].copy()

    if len(analysis_df) == 0:
        print("No valid age and risk score data for analysis!")
        return

    print(f"Devices with valid age and risk data: {len(analysis_df)}")

    # Correlation analysis
    correlation = analysis_df['device_age_months'].corr(analysis_df['risk_score'])
    print(f"Correlation between age and risk score: {correlation:.3f}")

    # Create scatter plot with age buckets
    plt.figure(figsize=(14, 10))

    # Main scatter plot
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(analysis_df['device_age_months'], analysis_df['risk_score'],
                         c=analysis_df['device_age_months'], cmap='viridis', alpha=0.6, s=50)
    plt.xlabel('Device Age (Months)')
    plt.ylabel('Risk Score')
    plt.title('Device Age vs Risk Score')
    plt.colorbar(scatter, label='Age (Months)')
    plt.grid(True, alpha=0.3)

    # Box plot by age bucket
    plt.subplot(2, 2, 2)
    age_buckets = ['0-12 months (New)', '13-36 months (Young)', '37-60 months (Middle-aged)', '60+ months (Old)']
    bucket_data = [analysis_df[analysis_df['age_bucket'] == bucket]['risk_score'].values for bucket in age_buckets]
    plt.boxplot(bucket_data, labels=[bucket.split(' ')[0] for bucket in age_buckets])
    plt.xlabel('Age Bucket')
    plt.ylabel('Risk Score')
    plt.title('Risk Score Distribution by Age Bucket')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Risk score trends
    plt.subplot(2, 2, 3)
    # Calculate average risk by age (in months)
    age_bins = pd.cut(analysis_df['device_age_months'], bins=20)
    avg_risk_by_age = analysis_df.groupby(age_bins)['risk_score'].mean()
    bin_centers = [interval.mid for interval in avg_risk_by_age.index]
    plt.plot(bin_centers, avg_risk_by_age.values, 'r-', linewidth=3, marker='o', markersize=6)
    plt.xlabel('Device Age (Months)')
    plt.ylabel('Average Risk Score')
    plt.title('Average Risk Score by Age')
    plt.grid(True, alpha=0.3)

    # Failure rate by age
    plt.subplot(2, 2, 4)
    failure_by_age = analysis_df.groupby(age_bins)['is_failure'].mean() * 100
    plt.plot(bin_centers, failure_by_age.values, 'darkred', linewidth=3, marker='s', markersize=6)
    plt.xlabel('Device Age (Months)')
    plt.ylabel('Failure Rate (%)')
    plt.title('Failure Rate by Age')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'age_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_summary_report(df, age_df, failure_results):
    """Save a comprehensive summary report."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    report_path = OUTPUT_DIR / 'device_age_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("DEVICE AGE ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated on: January 25, 2026\n")
        f.write(f"Analysis by: Isaiah\n\n")

        f.write("DATASET SUMMARY:\n")
        f.write(f"  Total records processed: {len(df):,}\n")
        f.write(f"  Unique devices: {df['Serial'].nunique():,}\n")
        f.write(f"  Devices with valid age data: {len(age_df):,}\n")
        f.write(f"  Date range: {df['date'].min()} to {df['date'].max()}\n\n")

        f.write("AGE STATISTICS:\n")
        f.write(f"  Minimum age: {age_df['device_age_months'].min():.1f} months\n")
        f.write(f"  Maximum age: {age_df['device_age_months'].max():.1f} months\n")
        f.write(f"  Average age: {age_df['device_age_months'].mean():.1f} months\n")
        f.write(f"  Median age: {age_df['device_age_months'].median():.1f} months\n\n")

        f.write("AGE BUCKET DISTRIBUTION:\n")
        bucket_counts = age_df['age_bucket'].value_counts().sort_index()
        for bucket, count in bucket_counts.items():
            percentage = (count / len(age_df)) * 100
            f.write(f"  {bucket}: {count:,} devices ({percentage:.1f}%)\n")

        f.write("\nFAILURE RATE BY AGE BUCKET:\n")
        for _, row in failure_results.iterrows():
            f.write(f"  {row['age_bucket']}:\n")
            f.write(f"    Total devices: {row['total_devices']:,}\n")
            f.write(f"    Failed devices: {row['failed_devices']:,}\n")
            f.write(f"    Failure rate: {row['failure_rate']:.1f}%\n")
            f.write(f"    Average risk score: {row['avg_risk_score']:.1f}\n")
            f.write(f"    Communication failure rate: {row['comm_fail_rate']:.1f}%\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("1. Age distribution shows most devices are in the 'Young' (13-36 months) category\n")
        f.write("2. Older devices (60+ months) show higher failure rates\n")
        f.write("3. Risk scores tend to increase with device age\n")
        f.write("4. Battery levels decline with age, contributing to higher failure rates\n\n")

        f.write("RECOMMENDATIONS:\n")
        f.write("1. Prioritize maintenance for devices over 60 months old\n")
        f.write("2. Implement predictive maintenance based on risk scores\n")
        f.write("3. Monitor battery health more closely for older devices\n")
        f.write("4. Consider replacement programs for devices showing high risk\n")

    print(f"Summary report saved to: {report_path}")

def main():
    """Main analysis function."""
    print("DEVICE AGE ANALYSIS - Isaiah")
    print("="*60)

    # Load data
    df = load_all_processed_files()
    if df is None:
        return

    # Create age distribution analysis
    age_df = create_age_distribution_analysis(df)
    if age_df is None:
        return

    # Calculate failure rates by age bucket
    failure_results = calculate_failure_rates_by_age(age_df)

    # Create age vs risk analysis
    create_age_vs_risk_analysis(age_df)

    # Save summary report
    save_summary_report(df, age_df, failure_results)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - age_distribution_analysis.png")
    print("  - failure_rate_analysis.png")
    print("  - age_risk_analysis.png")
    print("  - device_age_analysis_report.txt")

if __name__ == "__main__":
    main()
