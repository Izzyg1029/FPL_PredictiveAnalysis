# pipelines/upload_to_drive.py
# Commits FCI_Device_Health_Export.csv back to the GitHub repo

import subprocess
from pathlib import Path
from datetime import datetime

EXPORT_FILE = Path(__file__).resolve().parent.parent / "powerbi_exports" / "FCI_Device_Health_Export.csv"

def commit_to_github():
    """Commit the export file back to the repo"""
    print("=" * 60)
    print(" COMMITTING EXPORT TO GITHUB")
    print("=" * 60)

    if not EXPORT_FILE.exists():
        raise FileNotFoundError(f"Export file not found: {EXPORT_FILE}")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    subprocess.run(["git", "config", "user.name", "github-actions"], check=True)
    subprocess.run(["git", "config", "user.email", "actions@github.com"], check=True)
    subprocess.run(["git", "add", str(EXPORT_FILE)], check=True)

    result = subprocess.run(
        ["git", "commit", "-m", f"Auto update FCI export - {timestamp}"],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        subprocess.run(["git", "push"], check=True)
        print(f" Committed and pushed: {EXPORT_FILE.name}")
        print(f" Timestamp: {timestamp}")
    else:
        print(" No changes to commit - file unchanged since last run")

    print("=" * 60)

if __name__ == "__main__":
    commit_to_github()
