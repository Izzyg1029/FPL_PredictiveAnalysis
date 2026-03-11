"""
Train one RandomForest per device_type to predict action_label:
  0 = NO_ACTION
  1 = RECONFIGURE
  2 = RELOCATE
  3 = REPLACE
"""
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


def pick_feature_cols(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    drop = set(drop_cols + ["action_label", "action_name", "Serial", "_source_file", "BatteryLatestReport"])
    feats = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/fci_labeled.parquet")
    p.add_argument("--device_type_col", default="Device_Type")  # Changed from "device_type"
    p.add_argument("--models_dir", default="models")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--drop_cols", nargs="*", default=[
        "Region","Substation","Feeder","Site","Phase","IP_Address","NetworkGroup","Profile_Name","SensorGateway"
    ])
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run pipelines/label_actions.py first.")

    df = pd.read_parquet(in_path)

    if args.device_type_col not in df.columns:
        raise KeyError(f"Missing device type column '{args.device_type_col}'")
    if "action_label" not in df.columns:
        raise KeyError("Missing action_label. Run pipelines/label_actions.py first.")

    #  drop missing device types
    before = len(df)
    df = df[df[args.device_type_col].notna()].copy()
    if len(df) != before:
        print(f"🧹 Dropped {before-len(df):,} rows with missing device_type")

    # normalize types
    df[args.device_type_col] = (
        df[args.device_type_col].astype(str).str.strip().str.upper().replace({"M3": "MM3", "UM3": "UM3+"})
    )

    feature_cols = pick_feature_cols(df, args.drop_cols)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found after dropping columns.")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    device_types = sorted(df[args.device_type_col].dropna().unique().tolist())
    print("device types:", device_types)

    for dt in device_types:
        sub = df[df[args.device_type_col] == dt].copy()

        # need at least 2 classes
        if sub["action_label"].nunique() < 2:
            print(f" Skipping {dt}: only one action class present.")
            print(sub["action_label"].value_counts())
            continue

        X = sub[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = sub["action_label"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred).tolist()

        outdir = models_dir / str(dt)
        outdir.mkdir(parents=True, exist_ok=True)

        #  save bundle with exact training features
        joblib.dump({"model": rf, "features": feature_cols}, outdir / "action_rf.joblib")

        metrics = {
            "device_type": str(dt),
            "rows": int(len(sub)),
            "n_features": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "classes": rf.classes_.tolist(),
            "y_test_counts": pd.Series(y_test).value_counts().to_dict(),
            "pred_counts": pd.Series(pred).value_counts().to_dict(),
            "confusion_matrix": cm,
            "classification_report": report,
        }
        (outdir / "action_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"Trained RF for {dt} -> {outdir}")

    print("Done.")


if __name__ == "__main__":
    main()