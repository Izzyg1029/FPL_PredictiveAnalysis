import pandas as pd
from pathlib import Path
from feature_health.health_features import build_health_features

CLEAN_DIR = Path("data/clean")
OUT_DIR = Path("data/processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for file in CLEAN_DIR.glob("*-clean.xlsx"):
    print(f"\n🧠 Generating health features for {file.name}")
    df = pd.read_excel(file)

    df_features = build_health_features(df)

    out_file = OUT_DIR / f"{file.stem.replace('-clean','')}-health.csv"
    df_features.to_csv(out_file, index=False)

    print(f"✅ Output → {out_file}")
