# update_history.py - FIXED VERSION (creates history if missing)

from pathlib import Path
import json
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/daily")
STATE_PATH = Path("state/ingest_state.json")
HISTORY_PATH = Path("data/processed/fci_history.parquet")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"processed_files": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix columns that should be numeric but sometimes arrive as strings like '60' or 'NULL'.
    This prevents pyarrow/parquet write failures.
    """
    numeric_cols = ["LineCurrent", "LineTemp", "BatteryLevel", "BatteryLev"]

    for col in numeric_cols:
        if col in df.columns:
            s = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .replace({"NULL": np.nan, "null": np.nan, "None": np.nan, "nan": np.nan, "": np.nan, " ": np.nan})
            )
            df[col] = pd.to_numeric(s, errors="coerce")

    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = load_state()
    processed = set(state.get("processed_files", []))

    all_csvs = sorted(RAW_DIR.glob("*.csv"))
    new_csvs = [f for f in all_csvs if f.name not in processed]

    # SPECIAL CASE: If history doesn't exist at all, process ALL files
    if not HISTORY_PATH.exists():
        print("No history file found. Creating new history from ALL files...")
        new_csvs = all_csvs  # Process all files
        hist = pd.DataFrame()  # Start with empty history
    else:
        # Load existing history
        hist = pd.read_parquet(HISTORY_PATH)

    if not new_csvs:
        print("No new daily files. History is up to date.")
        return

    new_frames = []
    for f in new_csvs:
        print(f"Processing: {f.name}")
        df = pd.read_csv(f)
        df["_source_file"] = f.name
        new_frames.append(df)

    new_data = pd.concat(new_frames, ignore_index=True)

    if "BatteryLatestReport" in new_data.columns:
        new_data["BatteryLatestReport"] = pd.to_datetime(
            new_data["BatteryLatestReport"],
            errors="coerce"
        )

    combined = pd.concat([hist, new_data], ignore_index=True)

    # Fix numeric columns BEFORE parquet save
    combined = coerce_numeric_columns(combined)

    # Dedupe if keys exist
    if "Serial" in combined.columns and "BatteryLatestReport" in combined.columns:
        combined = combined.sort_values(["Serial", "BatteryLatestReport"])
        combined = combined.drop_duplicates(["Serial", "BatteryLatestReport"], keep="last")

    combined.to_parquet(HISTORY_PATH, index=False)

    # Update state with processed files
    if not HISTORY_PATH.exists():  # If we just created it, mark ALL as processed
        state["processed_files"] = sorted([f.name for f in all_csvs])
    else:
        state["processed_files"] = sorted(list(processed | {f.name for f in new_csvs}))
    
    save_state(state)

    print(f"Added {len(new_csvs)} new file(s). Total rows now: {len(combined):,}")
    print(f"History saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()