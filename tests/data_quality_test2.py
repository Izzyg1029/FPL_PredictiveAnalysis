import pandas as pd
from pathlib import Path
import os

# --------------------------------------------------------
# Base directory = folder where this script lives
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "clean"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_FILE = INPUT_DIR / "9-13.xlsx"
OUTPUT_CLEAN = OUTPUT_DIR / "9-13-new.xlsx"
OUTPUT_REJECTED = OUTPUT_DIR / "9-13-rejected.xlsx"

print("🧭 Current working directory:", os.getcwd())
print("📌 Script folder (BASE_DIR):", BASE_DIR)
print("🔎 Looking for file at:", DEVICE_FILE)
#looking for the file
if not DEVICE_FILE.exists():
    print("\n❌ File still not found.")
    print("📂 Files currently in data/raw are:")
    for p in INPUT_DIR.glob("*"):
        print(" -", p.name)
    raise FileNotFoundError(f"Missing: {DEVICE_FILE}")

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
df_raw = pd.read_excel(DEVICE_FILE)
print(f"\n📥 Devices loaded: {len(df_raw)}")

CRITICAL_COLS = ["Device_Type", "Last_Heard", "LineCurrent", "LineTemperatrue"]

missing = [c for c in CRITICAL_COLS if c not in df_raw.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

broken_mask = df_raw[CRITICAL_COLS].isnull().any(axis=1)
df_clean = df_raw[~broken_mask].copy()
df_rejected = df_raw[broken_mask].copy()

df_rejected["rejection_reason"] = df_rejected.apply(
    lambda r: ", ".join([c for c in CRITICAL_COLS if pd.isna(r[c])]),
    axis=1
)

df_clean.to_excel(OUTPUT_CLEAN, index=False)
df_rejected.to_excel(OUTPUT_REJECTED, index=False)

print(f"✅ Clean devices: {len(df_clean)} → {OUTPUT_CLEAN}")
print(f"❌ Rejected devices: {len(df_rejected)} → {OUTPUT_REJECTED}")
