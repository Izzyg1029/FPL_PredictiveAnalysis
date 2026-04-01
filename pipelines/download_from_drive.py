"""
download_from_drive.py

Downloads new files from a shared Google Drive folder, unzips them,
and saves raw CSVs to data/raw/daily/ for the quality pipeline.

Requirements:
    pip install gdown
"""

import json
import logging
import zipfile
import subprocess
import sys
import os
from pathlib import Path
import gdown

# =====================================================
# CONFIGURATION
# =====================================================

FOLDER_ID = "1A9Fv4_u--CgtDg06vuGptqqgLByBiFdL"

# =====================================================
# PATH SETUP
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "daily"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = RAW_DIR / ".downloaded_manifest.json"

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RAW_DIR / "download_log.txt"),
    ],
)
log = logging.getLogger()

# =====================================================
# HELPERS
# =====================================================

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

def unzip_and_clean(zip_path: Path) -> list:
    """Unzip into raw/daily and delete the zip. Returns list of extracted CSV filenames."""
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if member.lower().endswith('.csv'):
                z.extract(member, RAW_DIR)
                extracted.append(member)
                log.info(f"  Extracted: {member}")
    zip_path.unlink()
    log.info(f"  Deleted zip: {zip_path.name}")
    return extracted

# =====================================================
# MAIN
# =====================================================

def download_new_files() -> int:
    """Returns number of new files downloaded."""
    log.info(f"Raw data directory: {RAW_DIR}")
    log.info("Fetching file list from Google Drive folder...")

    folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

    files = gdown.download_folder(
        url=folder_url,
        output=str(RAW_DIR),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )

    if not files:
        log.info("No files found (check the folder is shared as 'Anyone with the link').")
        return 0

    manifest = load_manifest()
    new_files = []

    print(f"\nFiles found in Drive folder: {len(files)}")
    for f in files:
        print(f"  - {f.path}")
    print()

    for f in files:
        filename = f.path
        file_id = f.id

        if file_id in manifest:
            log.info(f"Already downloaded, skipping: {filename}")
            continue

        dest = RAW_DIR / filename
        log.info(f"Downloading: {filename}")

        gdown.download(id=file_id, output=str(dest), quiet=False)

        # Unzip if needed
        if dest.suffix.lower() == ".zip":
            log.info(f"Unzipping: {filename}")
            extracted = unzip_and_clean(dest)
            new_files.extend(extracted)
        else:
            new_files.append(filename)

        manifest[file_id] = filename

    save_manifest(manifest)

    if new_files:
        log.info(f"\nDownloaded & extracted {len(new_files)} new CSV file(s):")
        for name in new_files:
            log.info(f"  - {name}")
    else:
        log.info("No new files to download.")

    return len(new_files)


def run_quality_pipeline():
    """Trigger the data quality pipeline."""
    quality_script = SCRIPT_DIR / "run_data_quality.py"
    if not quality_script.exists():
        log.warning(f"Quality pipeline not found at: {quality_script}")
        log.warning("Skipping quality pipeline — download complete.")
        return

    log.info("\n" + "=" * 60)
    log.info("STARTING DATA QUALITY PIPELINE...")
    log.info("=" * 60)
    subprocess.run([sys.executable, str(quality_script)], check=True)


if __name__ == "__main__":
    new_count = download_new_files()

    if new_count > 0:
        run_quality_pipeline()
    else:
        log.info("No new files — skipping quality pipeline.")

    # Only pause when running locally, not in GitHub Actions
    if not os.environ.get("CI"):
        input("\nPress Enter to exit...")
