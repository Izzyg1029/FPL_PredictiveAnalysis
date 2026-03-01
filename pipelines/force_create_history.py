# force_create_history.py
import pandas as pd
from pathlib import Path

raw_dir = Path("C:/Users/cassi/Capstone/FPL_PredictiveAnalysis/data/raw/daily")
all_csvs = sorted(raw_dir.glob("*.csv"))

if not all_csvs:
    print("No CSV files found!")
    exit(1)

print(f"Found {len(all_csvs)} CSV files")
frames = []

for f in all_csvs:
    print(f"Reading {f.name}...")
    df = pd.read_csv(f)
    print(f"  - {len(df)} rows")
    df["_source_file"] = f.name
    frames.append(df)

print("Combining all files...")
combined = pd.concat(frames, ignore_index=True)
print(f"Combined data: {len(combined):,} rows")

# Save as parquet
output_path = Path("data/processed/fci_history.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output_path, index=False)

print(f"\nSuccessfully created: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Total rows: {len(combined):,}")

# Also create a small sample to verify
print("\nFirst 5 rows of data:")
print(combined.head())