"""
pipelines/train_trust_stack.py


OUTPUTS (default):
   outputs/trust_stack/model_report.txt
   outputs/trust_stack/decision_tree_rules.txt
   outputs/trust_stack/rf_feature_importance.csv
   outputs/trust_stack/test_predictions_with_explanations.csv

RECOMMENDED RUN (PowerShell, from project root):
python pipelines\train_trust_stack.py `
  --input data\processed\time_series\2025-09-13_to_2025-10-13_health_zm1only_timeseries.csv `
  --date_col Last_Heard `
  --battery_col BatteryLevel `
  --current_col LineCurrent `
  --temp_col LineTemperatrue `
  --auto_label_from_rules `
  --risk_threshold 35 `
  --out_dir outputs\trust_stack
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Helpers: cleaning/sanitization
# -----------------------------
def sanitize_numeric(df: pd.DataFrame, cols):
    """
    Converts mixed-type numeric columns safely:
      - "85%" -> 85
      - "N/A", "--", "" -> NaN
      - keeps floats/ints
    """
    for c in cols:
        if c in df.columns:
            s = df[c]

            # Fast path if already numeric
            if pd.api.types.is_numeric_dtype(s):
                df[c] = pd.to_numeric(s, errors="coerce")
                continue

            # Otherwise strip non-numeric characters
            df[c] = (
                s.astype(str)
                .str.strip()
                .replace({"None": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
                .str.replace(r"[^\d\.\-]", "", regex=True)  # keep digits, dot, minus
                .replace("", np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_datetime_column(df: pd.DataFrame, date_col: str):
    if date_col not in df.columns:
        raise KeyError(f"Missing date_col='{date_col}'. Available: {list(df.columns)[:40]} ...")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def add_days_since_last_heard(df: pd.DataFrame, last_heard_col: str, out_col: str):
    """
    days_since_last_heard = (max_date_in_file - Last_Heard) in days
    This creates a usable 'days_since_last_heard' even if you don't have it.
    """
    if last_heard_col not in df.columns:
        return df
    max_dt = df[last_heard_col].max()
    df[out_col] = (max_dt - df[last_heard_col]).dt.total_seconds() / 86400.0
    return df


def chrono_split(df: pd.DataFrame, date_col: str, train_frac: float = 0.8):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def safe_auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        # AUC undefined if only one class present
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_positive_proba(model, X):
    """
    Returns P(class=1) safely even if model was trained with a single class.
    - If both classes exist: use [:,1]
    - If only one class exists: return all-ones or all-zeros accordingly
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    only_class = int(model.classes_[0])
    return np.ones(len(X)) if only_class == 1 else np.zeros(len(X))


# -----------------------------
# Rule-based model (transparent)
# -----------------------------
def rule_based_predict(
    df: pd.DataFrame,
    battery_col="BatteryLevel",
    last_heard_days_col="days_since_last_heard",
    current_col="LineCurrent",
    temp_col="LineTemperatrue",
    device_status_col=None,
    risk_threshold=50.0,
):
    """
    Produces:
      - rule_risk_score (0..100)
      - rule_label (0/1)
      - rule_reasons (string)

    NOTE: Includes a rule that triggers even without last_heard_days,
          so you actually get some positives.
    """
    n = len(df)
    score = np.zeros(n, dtype=float)
    reasons = [[] for _ in range(n)]

    def add(mask, points, reason):
        idx = np.where(mask.values)[0]
        score[idx] += points
        for i in idx:
            reasons[i].append(reason)

    battery = df[battery_col] if battery_col in df.columns else pd.Series(np.nan, index=df.index)
    last_heard_days = df[last_heard_days_col] if last_heard_days_col in df.columns else pd.Series(np.nan, index=df.index)
    current = df[current_col] if current_col in df.columns else pd.Series(np.nan, index=df.index)
    temp = df[temp_col] if temp_col in df.columns else pd.Series(np.nan, index=df.index)

    # ---------- Engineering rules (edit as needed) ----------
    # Battery
    add((battery < 20), 35, "Battery <20%")  # <-- ensures some positives exist
    add((battery < 10), 30, "Battery <10%")

    # Comms / last heard
    add((battery < 20) & (last_heard_days > 3), 25, "Battery <20% AND not heard >3d")
    add((last_heard_days > 7), 25, "Not heard >7d")

    # Sensor sanity
    add((current == 0) & (battery < 30), 25, "LineCurrent=0 with low battery")
    add((current == 0) & (last_heard_days > 3), 20, "LineCurrent=0 AND not heard >3d")

    # Temperature stress
    add((temp > 85), 20, "Temperature >85C")
    add((temp < -20), 20, "Temperature <-20C")

    # Optional installed/active status
    if device_status_col and device_status_col in df.columns:
        status = df[device_status_col].astype(str).str.lower()
        add(status.isin(["installed", "active"]) & (current == 0), 10, "Installed/Active but LineCurrent=0")

    # Clamp and label
    score = np.clip(score, 0, 100)
    rule_label = (score >= float(risk_threshold)).astype(int)

    out = pd.DataFrame(
        {
            "rule_risk_score": score,
            "rule_label": rule_label,
            "rule_reasons": ["; ".join(r) if r else "No rule triggered" for r in reasons],
        },
        index=df.index,
    )
    return out


def top_rf_explanations(row: pd.Series, feature_names, importances, k=3):
    """
    Simple explanation for a row:
    - Select top-k globally important features
    - Report their values (human-readable)
    """
    idx = np.argsort(importances)[::-1][:k]
    parts = []
    for j in idx:
        f = feature_names[j]
        v = row.get(f, np.nan)
        parts.append(f"{f}={v}")
    return "Top signals: " + ", ".join(parts)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV (or parquet) with features")
    ap.add_argument("--date_col", default="Last_Heard", help="Date column for chronological split")
    ap.add_argument("--label_col", default="failure_label", help="Binary target label (0/1)")

    ap.add_argument("--out_dir", default="outputs/trust_stack", help="Output directory")
    ap.add_argument("--train_frac", type=float, default=0.8, help="Train fraction for chrono split")
    ap.add_argument("--risk_threshold", type=float, default=50.0, help="Rule label threshold on 0..100 risk score")
    ap.add_argument(
        "--auto_label_from_rules",
        action="store_true",
        help="If set, creates/overwrites label_col using rule_label (for baseline training)",
    )

    # Feature column names (match your CSV)
    ap.add_argument("--battery_col", default="BatteryLevel")
    ap.add_argument("--days_since_last_heard_col", default="days_since_last_heard")
    ap.add_argument("--current_col", default="LineCurrent")
    ap.add_argument("--temp_col", default="LineTemperatrue")
    ap.add_argument("--device_status_col", default=None)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Load (use low_memory=False to reduce mixed-type issues)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, low_memory=False)

    # Parse dates
    df = parse_datetime_column(df, args.date_col)

    # If days_since_last_heard missing OR mostly empty, compute it
    if (args.days_since_last_heard_col not in df.columns) or (df[args.days_since_last_heard_col].isna().mean() > 0.95):
        df = add_days_since_last_heard(df, args.date_col, args.days_since_last_heard_col)

    # Numeric sanitization (prevents float vs str comparisons)
    numeric_cols = [args.battery_col, args.days_since_last_heard_col, args.current_col, args.temp_col]
    df = sanitize_numeric(df, numeric_cols)

    # Clamp battery to physical range if present
    if args.battery_col in df.columns:
        df[args.battery_col] = df[args.battery_col].clip(lower=0, upper=100)

    # Drop rows missing date (needed for chrono split)
    df = df.dropna(subset=[args.date_col]).copy()

    # 1) Rule-based outputs
    rules_out = rule_based_predict(
        df,
        battery_col=args.battery_col,
        last_heard_days_col=args.days_since_last_heard_col,
        current_col=args.current_col,
        temp_col=args.temp_col,
        device_status_col=args.device_status_col,
        risk_threshold=args.risk_threshold,
    )
    df = pd.concat([df, rules_out], axis=1)

    # 1b) Labels
    if args.auto_label_from_rules:
        df[args.label_col] = df["rule_label"].astype(int)
    else:
        if args.label_col not in df.columns:
            raise KeyError(
                f"label_col='{args.label_col}' not found and --auto_label_from_rules not set.\n"
                f"Either pass --auto_label_from_rules OR provide a real label column."
            )
        df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=[args.label_col]).copy()
        df[args.label_col] = df[args.label_col].astype(int)

    # 2) Chronological split
    train_df, test_df = chrono_split(df, args.date_col, train_frac=args.train_frac)

    # Debug: show class balance
    print("\n Label counts:")
    print("Train:\n", train_df[args.label_col].value_counts(dropna=False))
    print("Test:\n", test_df[args.label_col].value_counts(dropna=False))

    # 3) Train Decision Tree + Random Forest
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        raise ImportError(
            "scikit-learn is required.\n"
            "Install:  python -m pip install scikit-learn\n"
            f"Original error: {e}"
        )

    # Features (include rule_risk_score to align ML with engineering baseline)
    features = [
        args.battery_col,
        args.days_since_last_heard_col,
        args.current_col,
        args.temp_col,
        "rule_risk_score",
    ]
    features = [c for c in features if c in df.columns]

    if len(features) == 0:
        raise ValueError("No ML features found. Check your column names passed to the script.")

    X_train = train_df[features].copy()
    y_train = train_df[args.label_col].astype(int).copy()
    X_test = test_df[features].copy()
    y_test = test_df[args.label_col].astype(int).copy()

    # Fill NA using train medians
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    # Decision Tree (interpretable)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = safe_positive_proba(dt, X_test)

    dt_rules_txt = export_text(dt, feature_names=list(features))
    (out_dir / "decision_tree_rules.txt").write_text(dt_rules_txt, encoding="utf-8")

    # Random Forest (stronger)
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = safe_positive_proba(rf, X_test)

    # Feature importances
    importances = rf.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feat_imp.to_csv(out_dir / "rf_feature_importance.csv", index=False)

    # Reports (force confusion matrix shape with labels=[0,1])
    report_lines = []
    report_lines.append("=== INPUT ===")
    report_lines.append(f"File: {in_path}")
    report_lines.append(f"Rows: {len(df)}  Cols: {df.shape[1]}")
    report_lines.append("")
    report_lines.append("=== SPLIT (CHRONO) ===")
    report_lines.append(f"Train rows: {len(train_df)}")
    report_lines.append(f"Test rows : {len(test_df)}")
    report_lines.append("")
    report_lines.append("=== LABEL BALANCE ===")
    report_lines.append("Train counts:")
    report_lines.append(str(train_df[args.label_col].value_counts(dropna=False).to_dict()))
    report_lines.append("Test counts:")
    report_lines.append(str(test_df[args.label_col].value_counts(dropna=False).to_dict()))
    report_lines.append("")
    report_lines.append("=== RULE-BASED BASELINE (Test) ===")
    report_lines.append(str(confusion_matrix(y_test, test_df["rule_label"].astype(int), labels=[0, 1])))
    report_lines.append(classification_report(y_test, test_df["rule_label"].astype(int), labels=[0, 1], digits=4, zero_division=0))
    report_lines.append("")
    report_lines.append("=== DECISION TREE max_depth=3 (Test) ===")
    report_lines.append(str(confusion_matrix(y_test, dt_pred, labels=[0, 1])))
    report_lines.append(classification_report(y_test, dt_pred, labels=[0, 1], digits=4, zero_division=0))
    report_lines.append(f"AUC: {safe_auc(y_test, dt_prob)}")
    report_lines.append("")
    report_lines.append("=== RANDOM FOREST (Test) ===")
    report_lines.append(str(confusion_matrix(y_test, rf_pred, labels=[0, 1])))
    report_lines.append(classification_report(y_test, rf_pred, labels=[0, 1], digits=4, zero_division=0))
    report_lines.append(f"AUC: {safe_auc(y_test, rf_prob)}")

    (out_dir / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # Per-row predictions + explanations
    test_out = test_df.copy()
    test_out["dt_pred"] = dt_pred
    test_out["dt_prob"] = dt_prob
    test_out["rf_pred"] = rf_pred
    test_out["rf_prob"] = rf_prob
    test_out["agree_rule_dt"] = (test_out["rule_label"].astype(int) == test_out["dt_pred"].astype(int)).astype(int)
    test_out["agree_rule_rf"] = (test_out["rule_label"].astype(int) == test_out["rf_pred"].astype(int)).astype(int)

    tmp_feat = test_out[features].copy().fillna(med)
    test_out["rf_explain"] = [
        top_rf_explanations(row, features, importances, k=3)
        for _, row in tmp_feat.iterrows()
    ]

    test_out.to_csv(out_dir / "test_predictions_with_explanations.csv", index=False)

    print(f"\n Saved outputs to: {out_dir.resolve()}")
    print("   - model_report.txt")
    print("   - decision_tree_rules.txt")
    print("   - rf_feature_importance.csv")
    print("   - test_predictions_with_explanations.csv")
    print("\n If labels are still all 0s, LOWER --risk_threshold (try 25 or 30).")
    print("    If too many 1s, raise it (try 50 or 70).\n")


if __name__ == "__main__":
    main()
