import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

for file in RAW_DIR.glob("*.xlsx"):
    print(f"\n📥 Processing {file.name}")
    df = pd.read_excel(file)
    
    # Only remove rows where Device_Type or Last_Heard is missing
    df_clean = df.dropna(subset=["Device_Type", "Last_Heard"])
    
    out_file = CLEAN_DIR / f"{file.stem}-clean.xlsx"
    df_clean.to_excel(out_file, index=False)
    
    print(f"✅ Cleaned → {out_file}")
    print(f"   Kept {len(df_clean)}/{len(df)} devices")
    
    # Show device types
    if "Device_Type" in df_clean.columns:
        device_counts = df_clean['Device_Type'].value_counts()
        print("   Device type counts:")
        for device, count in device_counts.items():
            print(f"     {device}: {count}")