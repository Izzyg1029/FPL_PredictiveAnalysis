# run_full_pipeline.py
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_name, step_number, total_steps):
    """Run a Python script and print status"""
    print(f"\n{'='*60}")
    print(f"STEP {step_number}/{total_steps}: Running {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                                capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            try:
                print(result.stdout.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except:
                print(result.stdout)
        
        if result.returncode != 0:
            print(f"ERROR in {script_name}:")
            if result.stderr:
                try:
                    print(result.stderr.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                except:
                    print(result.stderr)
            return False
        else:
            print(f"Completed: {script_name}")
            return True
            
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("FCI COMPLETE PIPELINE - STARTING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scripts = [
        "scripts/download_from_drive.py",
        "scripts/update_history.py",
        "scripts/run_data_quality.py",
        "scripts/run_health_features.py",
        "scripts/process_daily_time_series.py",
        "scripts/label_actions.py",
        "scripts/train_action_models_rf.py",
        "scripts/predict_daily_actions.py",
        "scripts/fci_complete_export.py",
        "scripts/upload_to_drive.py",
    ]
    
    total_steps = len(scripts)
    success_count = 0
    
    for i, script in enumerate(scripts, 1):
        if run_script(script, i, total_steps):
            success_count += 1
        else:
            print(f"\nPipeline stopped at step {i}")
            break
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps succeeded: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("\nALL STEPS SUCCESSFUL!")
        print("Your Power BI file is ready:")
        print("   powerbi_exports/FCI_Device_Health_Export.csv")
    else:
        print("\nPipeline incomplete. Check errors above.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
