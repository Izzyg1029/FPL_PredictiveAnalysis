from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

HISTORY_PATH = Path("data/processed/fci_history.parquet")
MODELS_DIR = Path("models")  # primary per-device models (trained by train_action_models_rf.py)
PIPELINE_MODELS_DIR = Path("../pipelines/models")  # richer 54-feature pipeline models
OUTPUT_DIR = Path("powerbi_exports")

# Confidence threshold: predictions below this are downgraded to NO_ACTION
CONFIDENCE_THRESHOLD = 0.95

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
    """Try pipeline models first (richer features), fall back to base models."""
    for models_dir in [PIPELINE_MODELS_DIR, MODELS_DIR]:
        model_path = models_dir / device_type / "action_rf.joblib"
        if not model_path.exists():
            continue

        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj and "features" in obj:
            print(f"Loaded model for {device_type} from {models_dir}")
            return obj["model"], obj["features"]

        # fallback for raw estimator
        if hasattr(obj, "predict"):
            feats = list(getattr(obj, "feature_names_in_", []))
            print(f"Loaded raw estimator for {device_type} from {models_dir}")
            return obj, feats if feats else None

    return None, None


def recompute_features(subset: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute key diagnostic features from raw columns at prediction time.

    This is necessary because the stored `days_since_last_report` field is
    pre-computed relative to an old reference date and arrives as 0 for all
    rows, making every device look healthy.  We recompute from the raw
    timestamp columns using today's actual date.
    """
    today = pd.Timestamp.now().normalize()

    # -- days_since_last_report -----------------------------------------------
    ts_col = next(
        (c for c in ["BatteryLatestReport", "Last_Heard", "LastHeard", "last_heard"] if c in subset.columns),
        None,
    )
    if ts_col:
        ts = pd.to_datetime(subset[ts_col], errors="coerce")
        # Also check secondary timestamp column and take the later of the two
        ts2_col = next(
            (c for c in ["Last_Heard", "BatteryLatestReport"] if c in subset.columns and c != ts_col),
            None,
        )
        if ts2_col:
            ts2 = pd.to_datetime(subset[ts2_col], errors="coerce")
            ts = ts.combine_first(ts2)
        computed_days = (today - ts).dt.days.clip(lower=0)
        # Always override - the stored value is stale
        subset["days_since_last_report"] = computed_days.fillna(0)
        print(f"days_since_last_report recomputed: mean={computed_days.mean():.0f}d, "
              f">14d: {(computed_days>14).sum()}, >90d: {(computed_days>90).sum()}")
    else:
        print("WARNING: No timestamp column found -- days_since_last_report left as-is")

    # -- coord flags ---------------------------------------------------------
    lat_col = next((c for c in ["Latitude", "lat", "LATITUDE"] if c in subset.columns), None)
    lon_col = next((c for c in ["Longitude", "lon", "LONGITUDE"] if c in subset.columns), None)

    if lat_col and lon_col:
        subset[lat_col] = pd.to_numeric(subset[lat_col], errors="coerce")
        subset[lon_col] = pd.to_numeric(subset[lon_col], errors="coerce")
        missing_lat = subset[lat_col].isna() | (subset[lat_col] == 0)
        missing_lon = subset[lon_col].isna() | (subset[lon_col] == 0)
        subset["coord_missing_flag"] = (missing_lat | missing_lon).astype(int)
        print(f"coord_missing_flag recomputed: {subset['coord_missing_flag'].sum()} devices")

        # Detect coordinate drift vs prior reading (per-serial, sorted by timestamp)
        if ts_col:
            sorted_idx = subset.sort_values(ts_col).index
            prev_lat = subset.loc[sorted_idx].groupby("Serial")[lat_col].shift(1)
            prev_lon = subset.loc[sorted_idx].groupby("Serial")[lon_col].shift(1)
            subset["prev_lat"] = prev_lat
            subset["prev_lon"] = prev_lon
        else:
            subset["prev_lat"] = np.nan
            subset["prev_lon"] = np.nan

        lat_changed = (
            subset["prev_lat"].notna()
            & subset[lat_col].notna()
            & ((subset[lat_col] - subset["prev_lat"]).abs() > 0.001)
        )
        lon_changed = (
            subset["prev_lon"].notna()
            & subset[lon_col].notna()
            & ((subset[lon_col] - subset["prev_lon"]).abs() > 0.001)
        )
        subset["coord_changed_flag"] = (lat_changed | lon_changed).astype(int)
        print(f"coord_changed_flag: {subset['coord_changed_flag'].sum()} devices with location drift")
    else:
        subset["coord_missing_flag"] = 0
        subset["coord_changed_flag"] = 0
        subset["prev_lat"] = np.nan
        subset["prev_lon"] = np.nan

    # -- battery_low_flag ----------------------------------------------------
    bat_col = next(
        (c for c in ["BatteryLevel", "Battery_Level", "battery_level", "Battery"] if c in subset.columns),
        None,
    )
    if bat_col:
        subset[bat_col] = pd.to_numeric(subset[bat_col], errors="coerce")
        subset["battery_low_flag"] = (subset[bat_col] < 20).astype(int)
        subset["battery_low_flag"] = subset["battery_low_flag"].fillna(0).astype(int)
        print(f"battery_low_flag recomputed: {subset['battery_low_flag'].sum()} devices")

    # -- status flags (offline/online/intermittent/standby) ------------------
    status_col = next(
        (c for c in ["Status", "status", "DeviceStatus", "Profile_Status"] if c in subset.columns),
        None,
    )
    if status_col:
        s = subset[status_col].astype(str).str.upper().str.strip()
        subset["offline_flag"] = (s == "OFFLINE").astype(int)
        subset["online_flag"] = (s == "ONLINE").astype(int)
        subset["intermittent_flag"] = (s == "INTERMITTENT").astype(int)
        subset["standby_flag"] = (s == "STANDBY").astype(int)

    return subset


def _rule_based_actions(subset: pd.DataFrame, device_type: str, mask: pd.Series):
    """
    Deterministic 4-class rule assignment for rows selected by `mask`.
    Priority: RELOCATE > REPLACE > RECONFIGURE > NO_ACTION

    For devices where reconfigure_count == 0 but comm_age > 90 days we treat
    the situation as an implicit reconfigure failure -- the device has been
    unreachable long enough that a reconfigure attempt is futile.
    """
    ca         = subset["days_since_last_report"]
    coord_miss = subset.get("coord_missing_flag",    pd.Series(0, index=subset.index))
    coord_chg  = subset.get("coord_changed_flag",    pd.Series(0, index=subset.index))
    bat_low    = subset.get("battery_low_flag",      pd.Series(0, index=subset.index))
    zero_curr  = subset.get("zero_current_flag",     pd.Series(0, index=subset.index))
    overheat   = subset.get("overheat_flag",          pd.Series(0, index=subset.index))
    intermt    = subset.get("intermittent_flag",     pd.Series(0, index=subset.index))
    offline    = subset.get("offline_flag",          pd.Series(0, index=subset.index))
    crit_curr  = subset.get("critical_current_flag", pd.Series(0, index=subset.index))
    rc         = subset.get("reconfigure_count",     pd.Series(0, index=subset.index)).fillna(0)
    hrs        = subset.get("hours_since_reconfigure", pd.Series(999999, index=subset.index)).fillna(999999)

    after_reconf  = mask & (rc >= 1) & (hrs > 48)
    # Treat 90+ days without comms as implicit reconfigure failure
    implicit_fail = mask & (ca > 90)
    actionable    = after_reconf | implicit_fail

    if device_type == "MM3":
        relocate = mask & ((actionable & (coord_miss == 1)) | (coord_chg == 1))
        replace  = mask & ~relocate & actionable & (
            (crit_curr == 1) | (overheat == 1) | (zero_curr == 1) | (ca > 90)
        )
        reconf   = mask & ~relocate & ~replace & (
            (zero_curr == 1) | (overheat == 1) | (crit_curr == 1)
            | (intermt == 1) | (coord_miss == 1)
            | ((ca > 7) & (ca <= 90))
        )

    elif device_type == "ZM1":
        relocate = mask & ((coord_miss == 1) & (ca > 14) | (coord_chg == 1))
        replace  = mask & ~relocate & actionable & ((bat_low == 1) | (ca > 90))
        reconf   = mask & ~relocate & ~replace & (
            (coord_miss == 1) | ((ca > 14) & (ca <= 90))
        )

    elif device_type == "UM3+":
        relocate = mask & ((actionable & (coord_miss == 1)) | (coord_chg == 1))
        replace  = mask & ~relocate & actionable & ((offline == 1) | (ca > 90))
        reconf   = mask & ~relocate & ~replace & (
            (offline == 1) | (intermt == 1) | (coord_miss == 1)
            | ((ca > 7) & (ca <= 90))
        )

    else:
        relocate = replace = reconf = pd.Series(False, index=subset.index)

    actions = pd.Series("NO_ACTION", index=subset.index)
    labels  = pd.Series(0,           index=subset.index)
    actions[reconf]   = "RECONFIGURE"; labels[reconf]   = 1
    actions[relocate] = "RELOCATE";    labels[relocate] = 2
    actions[replace]  = "REPLACE";     labels[replace]  = 3
    return actions, labels


def apply_confidence_threshold(subset: pd.DataFrame, device_type: str) -> pd.DataFrame:
    """
    Where model confidence < CONFIDENCE_THRESHOLD, replace the model prediction
    with a deterministic rule-based action so actionable devices still surface.
    """
    low_conf = subset["Confidence"] < CONFIDENCE_THRESHOLD

    if low_conf.sum() == 0:
        return subset

    print(f"{low_conf.sum():,} rows below {CONFIDENCE_THRESHOLD} confidence -- applying rule-based fallback")

    actions, labels = _rule_based_actions(subset, device_type, low_conf)
    subset.loc[low_conf, "PredictedActionName"]  = actions[low_conf]
    subset.loc[low_conf, "PredictedActionLabel"] = labels[low_conf]
    subset.loc[low_conf, "UsedRuleFallback"]     = True

    print(f"Fallback actions: {actions[low_conf].value_counts().to_dict()}")

    return subset


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

    print("Loading history...")
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
        print(f"Dropped {before-len(df):,} rows with missing device_type")

    if "BatteryLatestReport" in df.columns:
        df["BatteryLatestReport"] = pd.to_datetime(df["BatteryLatestReport"], errors="coerce")

    # latest per Serial
    if "BatteryLatestReport" in df.columns:
        latest = df.sort_values("BatteryLatestReport").drop_duplicates("Serial", keep="last")
    else:
        latest = df.drop_duplicates("Serial", keep="last")

    print(f"Latest records: {len(latest):,}")
    print("device_type latest counts:")
    print(latest["device_type"].value_counts(dropna=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = []

    # Only predict for active device types
    ACTIVE_DEVICE_TYPES = ["MM3", "ZM1"]  # add "UM3+" here when ready
    all_data_types = latest["device_type"].dropna().unique().tolist()
    data_device_types = [d for d in all_data_types if d in ACTIVE_DEVICE_TYPES]
    skipped = [d for d in all_data_types if d not in ACTIVE_DEVICE_TYPES]
    if skipped:
        print(f" Skipping device types (not active): {skipped}")

    for device_type in data_device_types:
        model, feature_cols = load_model_bundle(device_type)
        if model is None or feature_cols is None:
            print(f"Skipping {device_type}: no model found in {PIPELINE_MODELS_DIR} or {MODELS_DIR}")
            continue

        subset = latest[latest["device_type"] == device_type].copy()
        if subset.empty:
            print(f"No rows for {device_type}, skipping")
            continue

        # Recompute all diagnostic features from raw columns before alignment
        print(f"Recomputing features for {device_type}...")
        subset = recompute_features(subset)

        # Build X with exact training features (fill missing with 0)
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
        subset["UsedRuleFallback"] = False

        # Apply confidence threshold -- low-confidence rows get rule-based fallback
        subset = apply_confidence_threshold(subset, device_type)

        # ===== ADD FLAG COLUMNS =====
        flag_cols = [
            "days_since_last_report", "battery_low_flag", "offline_flag",
            "online_flag", "intermittent_flag", "standby_flag",
            "zero_current_flag", "coord_missing_flag", "coord_changed_flag",
            "prev_lat", "prev_lon",
        ]
        for col in flag_cols:
            if col not in subset.columns:
                subset[col] = 0

        predictions.append(subset)

        # Summary
        action_counts = subset["PredictedActionName"].value_counts().to_dict()
        high_conf = (subset["Confidence"] >= CONFIDENCE_THRESHOLD).sum()
        print(f"Predicted {device_type}: {len(subset):,} | actions={action_counts} | high-conf={high_conf}")

    if not predictions:
        print("No predictions generated.")
        return

    result = pd.concat(predictions, ignore_index=True)

    print("\n===== FINAL SUMMARY =====")
    print("Actions by device type:")
    print(result.groupby(["device_type", "PredictedActionName"]).size().unstack(fill_value=0))
    print(f"\nHigh-confidence (>= {CONFIDENCE_THRESHOLD}): {(result['Confidence'] >= CONFIDENCE_THRESHOLD).sum():,} of {len(result):,}")
    print(f"Used rule fallback: {result['UsedRuleFallback'].sum():,}")

    out_path = OUTPUT_DIR / "predictions_latest.csv"
    written = safe_write_csv(result, out_path)
    print(f"\nPredictions saved to: {written}")


if __name__ == "__main__":
    main()
