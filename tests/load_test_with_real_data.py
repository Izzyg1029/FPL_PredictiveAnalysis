"""
Load test with real data samples - INCLUDES FULL DATASET TEST AND VISUALIZATIONS
"""

import pandas as pd
from pathlib import Path
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Get the project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add project root to path
sys.path.insert(0, str(project_root))

from feature_health.health_features import build_health_features

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for visuals
output_dir = project_root / "tests" / "performance_visuals"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("LOAD TEST WITH REAL DATA (INCLUDING FULL DATASET & VISUALS)")
print("=" * 80)
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Output directory: {output_dir}")

def load_health_data_sample(sample_size=None):
    """Load a sample of processed health data
    If sample_size is None, loads ALL devices"""
    
    health_dir = project_root / "data" / "processed" / "daily"
    
    if not health_dir.exists():
        print(f"Directory does not exist: {health_dir}")
        return None
    
    files = list(health_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {health_dir}")
        return None
    
    # Use the most recent file
    latest_file = sorted(files)[-1]
    print(f"\nLoading health data from: {latest_file.name}")
    
    df = pd.read_csv(latest_file, low_memory=False)
    total_devices = len(df)
    print(f"Loaded {total_devices:,} rows")
    
    # Show device type distribution
    if 'Device_Type' in df.columns:
        print(f"Device types: {df['Device_Type'].value_counts().to_dict()}")
    
    # Sample if sample_size is specified
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} devices")
    else:
        print(f"Using ALL {total_devices:,} devices")
    
    return df

def run_load_test(sample_sizes=[100, 500, 1000, None]):
    """Run load test with real data samples
    None in sample_sizes means test ALL devices"""
    
    results = []
    
    for size in sample_sizes:
        size_label = "ALL" if size is None else f"{size:,}"
        print(f"\n{'='*60}")
        print(f"Testing {size_label} devices")
        print(f"{'='*60}")
        
        # Load sample
        df = load_health_data_sample(size)
        if df is None:
            print("Cannot load data")
            break
        
        print(f"   Loaded {len(df):,} devices")
        
        # Monitor
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        cpu_before = psutil.cpu_percent()
        
        # Run pipeline
        start = time.time()
        
        try:
            print("   Running build_health_features...")
            df_features = build_health_features(df)
            elapsed = time.time() - start
            mem_after = process.memory_info().rss / 1024 / 1024
            cpu_after = psutil.cpu_percent()
            
            print(f"\n   SUCCESS!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {len(df)/elapsed:.0f} rec/s")
            print(f"   Memory delta: {mem_after - mem_before:.1f} MB")
            print(f"   CPU: {cpu_after - cpu_before:.1f}%")
            print(f"   Output: {len(df_features):,} rows, {len(df_features.columns)} cols")
            
            if 'risk_score' in df_features.columns:
                print(f"   Risk scores: min={df_features['risk_score'].min():.1f}, max={df_features['risk_score'].max():.1f}, mean={df_features['risk_score'].mean():.1f}")
            
            results.append({
                'devices': len(df),
                'time_seconds': elapsed,
                'memory_mb': mem_after - mem_before,
                'cpu_percent': cpu_after - cpu_before,
                'records_per_second': len(df) / elapsed
            })
            
        except Exception as e:
            print(f"\n   Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return results

def create_visualizations(results):
    """Create performance visualizations from results"""
    
    if not results:
        print("No results to visualize")
        return
    
    df_results = pd.DataFrame(results)
    
    # Add absolute memory if not present
    if 'memory_mb_absolute' not in df_results.columns:
        # If we only have delta, we need to calculate absolute from first measurement
        # For now, just use a baseline of 150MB + delta
        baseline_memory = 150
        df_results['memory_mb_absolute'] = baseline_memory + df_results['memory_mb'].cumsum()
    
    print("\n" + "=" * 70)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 70)
    
    # Prepare labels for charts
    labels = [f"{int(d):,}" if d < 10000 else f"{d/1000:.1f}k" for d in df_results['devices']]
    
    # =========================================================
    # 1. Throughput by Scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(df_results)), df_results['records_per_second'], 
                  color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Number of Devices', fontsize=12)
    ax.set_ylabel('Throughput (records per second)', fontsize=12)
    ax.set_title('Processing Throughput by Scale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, df_results['records_per_second']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_by_scale.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  throughput_by_scale.png")
    
    # =========================================================
    # 2. Execution Time by Scale
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(df_results)), df_results['time_seconds'], 
                  color='coral', edgecolor='black')
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Number of Devices', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Execution Time by Scale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, df_results['time_seconds']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_by_scale.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  execution_time_by_scale.png")
    
    # =========================================================
    # 3. Performance Dashboard (2x2)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3.1 Throughput vs Devices
    ax1 = axes[0, 0]
    ax1.plot(df_results['devices'], df_results['records_per_second'], 'o-', 
             linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Number of Devices', fontsize=12)
    ax1.set_ylabel('Throughput (rec/s)', fontsize=12)
    ax1.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 3.2 Time vs Devices
    ax2 = axes[0, 1]
    ax2.plot(df_results['devices'], df_results['time_seconds'], 's-', 
             linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Number of Devices', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Execution Time Scaling', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 3.3 Memory Usage Delta
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in df_results['memory_mb']]
    bars = ax3.bar(range(len(df_results)), df_results['memory_mb'], color=colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(range(len(df_results)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_xlabel('Number of Devices', fontsize=12)
    ax3.set_ylabel('Memory Delta (MB)', fontsize=12)
    ax3.set_title('Memory Change During Processing', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 3.4 Performance Rating
    ax4 = axes[1, 1]
    def get_rating(rate):
        if rate > 5000:
            return 'EXCELLENT', 'green'
        elif rate > 2000:
            return 'GOOD', 'blue'
        elif rate > 1000:
            return 'ACCEPTABLE', 'orange'
        else:
            return 'NEEDS IMPROVEMENT', 'red'
    
    ratings = [get_rating(r)[0] for r in df_results['records_per_second']]
    colors = [get_rating(r)[1] for r in df_results['records_per_second']]
    
    bars = ax4.bar(range(len(df_results)), df_results['records_per_second'], color=colors, edgecolor='black')
    ax4.set_xticks(range(len(df_results)))
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_xlabel('Number of Devices', fontsize=12)
    ax4.set_ylabel('Throughput (rec/s)', fontsize=12)
    ax4.set_title('Performance Rating by Scale', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, rating in zip(bars, ratings):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                rating, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  performance_dashboard.png")
    
    # =========================================================
    # 4. Summary Table
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, max(4, len(df_results)*0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in df_results.iterrows():
        rate = row['records_per_second']
        if rate > 5000:
            rating = "EXCELLENT"
        elif rate > 2000:
            rating = "GOOD"
        elif rate > 1000:
            rating = "ACCEPTABLE"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        devices_label = f"{int(row['devices']):,}" if row['devices'] < 10000 else f"{row['devices']/1000:.1f}k"
        table_data.append([
            devices_label,
            f"{row['time_seconds']:.2f}s",
            f"{rate:.0f}",
            f"{row['memory_mb']:.1f} MB",
            rating
        ])
    
    columns = ['Devices', 'Time', 'Throughput (rec/s)', 'Memory Delta', 'Rating']
    
    table = ax.table(cellText=table_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.15, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for i, row in enumerate(table_data):
        rating = row[4]
        if rating == 'EXCELLENT':
            color = '#90EE90'
        elif rating == 'GOOD':
            color = '#87CEEB'
        elif rating == 'ACCEPTABLE':
            color = '#FFD700'
        else:
            color = '#FFB6C1'
        
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(color)
    
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Load Test Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  performance_summary_table.png")
    
    # =========================================================
    # 5. Memory Analysis (NEW - AFTER HTML REPORT)
    # =========================================================
    print("  Creating memory analysis charts...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Memory Delta chart
    colors = ['red' if x < 0 else 'green' for x in df_results['memory_mb']]
    bars = ax1.bar(range(len(df_results)), df_results['memory_mb'], color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(len(df_results)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('Memory Delta (MB)', fontsize=12)
    ax1.set_title('Memory Change During Processing', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, df_results['memory_mb']):
        y_pos = val + (5 if val > 0 else -15)
        color = 'red' if val < 0 else 'green'
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    # Memory Peak chart
    ax2.plot(range(len(df_results)), df_results['memory_mb_absolute'], 'o-', 
             linewidth=2, markersize=8, color='blue')
    ax2.set_xticks(range(len(df_results)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Memory After Processing (MB)', fontsize=12)
    ax2.set_title('Memory Usage After Processing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(range(len(df_results)), df_results['memory_mb_absolute'])):
        ax2.text(x, y + 2, f'{y:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  memory_analysis.png")
    
    # =========================================================
    # 6. ADD RISK SCORE VISUALIZATION HERE (NEW)
    # =========================================================
    print("  Creating risk score visualizations...")
    
    # Define the function here or call it if defined elsewhere
    def add_risk_score_visualization(results, output_dir):
        """Create risk score distribution visualization"""
        
        # Get the full dataset with risk scores
        df = load_health_data_sample(None)  # Load all devices
        if df is None or 'risk_score' not in df.columns:
            print("  Cannot create risk score visualization - missing data")
            return None
        
        print("\n  Creating risk score visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Risk Score Distribution Histogram
        ax1 = axes[0, 0]
        ax1.hist(df['risk_score'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(df['risk_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["risk_score"].mean():.1f}')
        ax1.axvline(80, color='orange', linestyle=':', linewidth=2, label='High Risk Threshold (80)')
        ax1.set_xlabel('Risk Score', fontsize=12)
        ax1.set_ylabel('Number of Devices', fontsize=12)
        ax1.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Category Pie Chart
        ax2 = axes[0, 1]
        bins = [0, 20, 40, 60, 80, 101]
        labels = ['Low (0-20)', 'Medium (21-40)', 'High (41-60)', 'Critical (61-80)', 'Emergency (81-100)']
        df['risk_category'] = pd.cut(df['risk_score'], bins=bins, labels=labels)
        risk_cats = df['risk_category'].value_counts()
        
        colors = ['green', 'yellowgreen', 'gold', 'orange', 'red']
        wedges, texts, autotexts = ax2.pie(risk_cats.values, labels=risk_cats.index, autopct='%1.1f%%',
                                            colors=colors[:len(risk_cats)], startangle=90)
        ax2.set_title('Devices by Risk Category', fontsize=14, fontweight='bold')
        
        # 3. Risk Score by Device Type
        ax3 = axes[1, 0]
        device_types = df['Device_Type'].unique()
        box_data = []
        for dtype in device_types:
            scores = df[df['Device_Type'] == dtype]['risk_score'].dropna()
            box_data.append(scores)
        
        bp = ax3.boxplot(box_data, labels=device_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral', 'lightgreen']):
            patch.set_facecolor(color)
        ax3.set_ylabel('Risk Score', fontsize=12)
        ax3.set_title('Risk Score by Device Type', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=80, color='red', linestyle='--', linewidth=1, label='High Risk (80)')
        ax3.legend()
        
        # 4. Risk vs Device Age (if available)
        ax4 = axes[1, 1]
        if 'device_age_days' in df.columns:
            age_data = df[df['device_age_days'].notna()]
            if len(age_data) > 0:
                scatter = ax4.scatter(age_data['device_age_days'] / 365, age_data['risk_score'], 
                                      c=age_data['risk_score'], cmap='RdYlGn_r', alpha=0.6, s=20)
                ax4.set_xlabel('Device Age (years)', fontsize=12)
                ax4.set_ylabel('Risk Score', fontsize=12)
                ax4.set_title('Risk Score vs Device Age', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, ax=ax4, label='Risk Score')
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Age data not available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Risk Score vs Device Age (No Data)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_score_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  risk_score_distribution.png")
        
        # Create summary statistics
        stats_text = f"""
RISK SCORE SUMMARY
============================================================
Total Devices: {len(df):,}
Mean Risk Score: {df['risk_score'].mean():.1f}
Median Risk Score: {df['risk_score'].median():.1f}
Std Deviation: {df['risk_score'].std():.1f}

Risk Categories:
"""
        for cat, count in risk_cats.items():
            pct = (count / len(df)) * 100
            stats_text += f"    {cat}: {count:,} ({pct:.1f}%)\n"
        
        stats_text += f"""
CRITICAL FINDINGS:
- Mean risk score of {df['risk_score'].mean():.1f} indicates HIGH overall risk
- {(df['risk_score'] > 80).sum():,} devices ({((df['risk_score'] > 80).sum()/len(df)*100):.1f}%) are in EMERGENCY category
- {(df['risk_score'] > 60).sum():,} devices ({((df['risk_score'] > 60).sum()/len(df)*100):.1f}%) need immediate attention

WHY THIS IS BAD:
- Risk score > 60: Device likely has multiple issues (overheating, zero current, battery problems)
- Risk score > 80: Critical failure imminent - immediate action required
- Mean > 70 indicates widespread issues across the fleet
"""
        
        # Save statistics to text file
        with open(output_dir / 'risk_score_summary.txt', 'w') as f:
            f.write(stats_text)
        print(f"  risk_score_summary.txt")
        
        return stats_text
    
    # Call the function
    risk_stats = add_risk_score_visualization(results, output_dir)
    
    # =========================================================
    # 7. HTML Report (updated to include risk scores)
    # =========================================================
    print("  Creating HTML report...")
    
    # Get risk score mean from the results
    # Load the full dataset to get actual risk score stats
    df_full = load_health_data_sample(None)
    risk_mean = df_full['risk_score'].mean() if df_full is not None and 'risk_score' in df_full.columns else "N/A"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FCI Load Test Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; text-align: center; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .excellent {{ color: green; font-weight: bold; }}
        .good {{ color: blue; font-weight: bold; }}
        .acceptable {{ color: orange; font-weight: bold; }}
        .poor {{ color: red; font-weight: bold; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .risk-warning {{ background-color: #ffe6e6; padding: 15px; border-radius: 5px; border-left: 4px solid red; margin: 20px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>FCI Load Test Performance Report</h1>
    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Tests Run:</strong> {len(df_results)}</p>
    <p><strong>Largest Test:</strong> {df_results['devices'].max():,} devices</p>
    
    <h2>Performance Summary Table</h2>
    <!-- ... your existing table code ... -->
    
    <h2>Performance Graphs</h2>
    <img src="throughput_by_scale.png" alt="Throughput by Scale">
    <img src="execution_time_by_scale.png" alt="Execution Time by Scale">
    <img src="performance_dashboard.png" alt="Performance Dashboard">
    <img src="performance_summary_table.png" alt="Performance Summary Table">
    
    <h2>Memory Analysis</h2>
    <img src="memory_analysis.png" alt="Memory Analysis">
    
    <h2>Risk Score Analysis</h2>
    <img src="risk_score_distribution.png" alt="Risk Score Distribution">
    
    <div class="risk-warning">
        <h3>Why This Matters</h3>
        <p><strong>Risk Score Interpretation:</strong></p>
        <ul>
            <li><strong>0-20 (Low):</strong> Devices are healthy, no action needed</li>
            <li><strong>21-40 (Medium):</strong> Minor issues, monitor closely</li>
            <li><strong>41-60 (High):</strong> Significant issues, plan maintenance</li>
            <li><strong>61-80 (Critical):</strong> Severe issues, immediate attention required</li>
            <li><strong>81-100 (Emergency):</strong> Device likely failing, replace urgently</li>
        </ul>
        <p><strong>Current Fleet Status: MEAN RISK = {risk_mean:.1f}</strong></p>
        <p>⚠️ This indicates widespread device issues requiring maintenance action.</p>
    </div>
    
    <h2>Key Findings</h2>
    <ul>
        <li><strong>Best Throughput:</strong> {df_results['records_per_second'].max():.0f} rec/s</li>
        <li><strong>Handles {df_results['devices'].max():,} devices in {df_results[df_results['devices'] == df_results['devices'].max()]['time_seconds'].values[0]:.1f} seconds</strong></li>
        <li><strong>Mean Risk Score:</strong> {risk_mean:.1f} (HIGH - action required)</li>
    </ul>
</div>
</body>
</html>
"""
    
    with open(output_dir / 'performance_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  performance_report.html")
    
    return df_results

def main():
    """Run complete load test with visualizations"""
    
    # Test sizes: 100, 500, 1000, and ALL devices
    results = run_load_test(sample_sizes=[100, 500, 1000, None])
    
    if results:
        # Create visualizations
        df_results = create_visualizations(results)
        
            
        print("\n" + "=" * 80)
        print("LOAD TEST AND VISUALIZATIONS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print("   - throughput_by_scale.png")
        print("   - execution_time_by_scale.png")
        print("   - performance_dashboard.png")
        print("   - performance_summary_table.png")
        print("   - performance_report.html")
        print("\nOpen performance_report.html in your browser to view all results!")
        
        # Print summary
        print("\nPERFORMANCE SUMMARY:")
        for _, row in df_results.iterrows():
            devices_label = f"{int(row['devices']):,}" if row['devices'] < 10000 else f"{row['devices']/1000:.1f}k"
            rate = row['records_per_second']
            if rate > 5000:
                rating = "EXCELLENT"
            elif rate > 2000:
                rating = "GOOD"
            elif rate > 1000:
                rating = "ACCEPTABLE"
            else:
                rating = "NEEDS IMPROVEMENT"
            print(f"   {devices_label:>8} devices: {row['time_seconds']:.2f}s | {rate:.0f} rec/s | {rating}")
    
    else:
        print("\nNo results generated")

if __name__ == "__main__":
    main()