# pipelines/reconfigure_tracker.py - MINIMAL VERSION for counting only

import pandas as pd
from pathlib import Path
from datetime import datetime

RECONFIGURE_TRACKER_PATH = Path("state/reconfigure_attempts.csv")

def load_reconfigure_tracker():
    """Load history of reconfigure attempts"""
    if RECONFIGURE_TRACKER_PATH.exists():
        return pd.read_csv(RECONFIGURE_TRACKER_PATH)
    return pd.DataFrame(columns=['Serial', 'attempt_date'])

def mark_reconfigure_attempted(serial):
    """Record a reconfigure attempt (minimal version)"""
    tracker = load_reconfigure_tracker()
    
    new_record = pd.DataFrame([{
        'Serial': serial,
        'attempt_date': datetime.now().strftime('%Y-%m-%d')
    }])
    
    tracker = pd.concat([tracker, new_record], ignore_index=True)
    RECONFIGURE_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracker.to_csv(RECONFIGURE_TRACKER_PATH, index=False)

def get_reconfigure_attempts(serial, days_back=90):
    """Get reconfigure attempts for a device"""
    tracker = load_reconfigure_tracker()
    if tracker.empty:
        return 0
    
    device_attempts = tracker[tracker['Serial'] == serial]
    return len(device_attempts)

def get_all_devices_reconfigure_counts(serials_list):
    """Get reconfigure counts for multiple devices"""
    tracker = load_reconfigure_tracker()
    if tracker.empty:
        return {serial: 0 for serial in serials_list}
    
    counts = tracker.groupby('Serial').size().to_dict()
    return {serial: counts.get(serial, 0) for serial in serials_list}