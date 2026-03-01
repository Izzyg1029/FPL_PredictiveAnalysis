from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

HISTORY_PATH = Path("data/processed/fci_history.parquet")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("powerbi_exports")

LABEL_TO_NAME = {
    0: "NO_ACTION",
    1: "RECONFIGURE",
    2: "RELOCATE",
    3: "REPLACE",
}


def normalize_device_type_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ("", "NONE", "NAN", "NULL"):
        return np.nan
    if s == "M3":
        return "MM3"
    if s == "UM3":
        return "UM3+"
    # fallback detect inside
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").replace("/", "")
    if "ZM1" in s2:
        return "ZM1"
    if "MM3" in s2 or s2 == "M3":
        return "MM3"
    if "UM3" in s2:
        return "UM3+"
    return s


def load_model_bundle(device_type: str):
    model_path = MODELS_DIR / device_type / "action_rf.joblib"
    if not model_path.exists():
        return None, None

    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj and "features" in obj:
        return obj["model"], obj["features"]

    # fallback
    if hasattr(obj, "predict"):
        feats = list(getattr(obj, "feature_names_in_", []))
        return obj, feats if feats else None

    return None, None


def safe_write_csv(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        df.to_csv(alt, index=False)
        return alt


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORY_PATH}. Run pipelines/update_history.py first.")

    print("📦 Loading history...")
    df = pd.read_parquet(HISTORY_PATH)

    if "Serial" not in df.columns:
        raise ValueError("Missing required column: Serial")

    # canonical device types + drop missing
    if "Device_Type" in df.columns:
        dt = df["Device_Type"]
        if "device_type" in df.columns:
            dt = dt.where(~dt.isna(), df["device_type"])
    elif "device_type" in df.columns:
        dt = df["device_type"]
    else:
        dt = pd.Series([np.nan] * len(df))

    df["device_type"] = dt.map(normalize_device_type_value)

    before = len(df)
    df = df[df["device_type"].notna()].copy()
    if len(df) != before:
        print(f"🧹 Dropped {before-len(df):,} rows with missing device_type")

    if "BatteryLatestReport" in df.columns:
        df["BatteryLatestReport"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")

    # latest per Serial
    if "BatteryLatestReport" in df.columns:
        latest = df.sort_values("BatteryLatestReport").drop_duplicates("Serial", keep="last")
    else:
        latest = df.drop_duplicates("Serial", keep="last")

    print(f"✅ Latest records: {len(latest):,}")
    print("device_type latest counts:")
    print(latest["device_type"].value_counts(dropna=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = []

    # use only folders that exist as models
    for device_dir in MODELS_DIR.iterdir():
        if not device_dir.is_dir():
            continue
        device_type = device_dir.name

        model, feature_cols = load_model_bundle(device_type)
        if model is None or feature_cols is None:
            print(f"⚠️ Skipping {device_type}: missing model/features")
            continue

        subset = latest[latest["device_type"] == device_type].copy()
        if subset.empty:
            print(f"⚠️ No rows for {device_type}, skipping")
            continue

        # Build X with exact training features
        for c in feature_cols:
            if c not in subset.columns:
                subset[c] = 0

        X = subset[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        preds = model.predict(X)
        probs = model.predict_proba(X).max(axis=1)

        subset["PredictedActionLabel"] = preds
        subset["PredictedActionName"] = [LABEL_TO_NAME.get(int(p), "NO_ACTION") for p in preds]
        subset["Confidence"] = probs

        predictions.append(subset)
        print(f"✅ Predicted {device_type}: {len(subset):,}")

    if not predictions:
        print("⚠️ No predictions generated.")
        return

    result = pd.concat(predictions, ignore_index=True)

    out_path = OUTPUT_DIR / "predictions_latest.csv"
    written = safe_write_csv(result, out_path)
    print(f"📊 Predictions saved → {written}")


if __name__ == "__main__":
    main()