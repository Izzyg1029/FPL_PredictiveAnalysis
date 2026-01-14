import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CRITICAL_COLS = ["Device_Type", "Last_Heard", "LineCurrent", "LineTemperatrue"]

for file in RAW_DIR.glob("*.xlsx"):
    print(f"\n📥 Processing {file.name}")
    df = pd.read_excel(file)

    broken_mask = df[CRITICAL_COLS].isnull().any(axis=1)
    df_clean = df[~broken_mask].copy()

    out_file = CLEAN_DIR / f"{file.stem}-clean.xlsx"
    df_clean.to_excel(out_file, index=False)

    print(f"✅ Cleaned → {out_file}")
