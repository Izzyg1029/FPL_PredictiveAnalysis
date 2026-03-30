This repository contains Cass’s Feature Health Tree, which produces a dataset of engineered features, risk scores, and reasons for risk.

The purpose of this document is to explain how the other team members (Mansib, Isaiah, Sora) should use Cass’s pipeline when building:

ML Prediction Tree
Data Quality Tree
Alert Tree

✅ 1. What Cass’s Feature Health Pipeline Provides
When teammates import:

from feature_health.health_features import build_health_features

and run:

df_features = build_health_features(df_devices)

they receive a DataFrame that already contains the most useful engineered information, including:

Core engineered features
comm_age_days
LineCurrent_val
LineTemperature_val
battery_level, battery_low_flag
device_age_days (if install file included)
distance_drift_m, gps_jump_flag
zero_current_flag, overheat_flag

Device-specific risk scores
risk_score_zm1
risk_score_um3
risk_score_mm3
Final combined health score
risk_score (0–100)

Human-readable explanation
risk_reason

These features are consistent, cleaned, normalized, and ready for downstream use.

🚀 2. How the ML Prediction Tree Should Use This
The ML team does NOT need to redo feature engineering.

They should directly use Cass’s output dataset as ML inputs.

Recommended ML input features:
comm_age_days
LineCurrent_val
LineTemperature_val
battery_level
battery_report_age_days
zero_current_flag
overheat_flag
distance_drift_m
gps_jump_flag
pct_life_used
Cass’s risk_score can also be used as a feature

ML Team To-Do List
Gather historical device failure logs (if available).

Join Cass’s engineered features by Serial and timestamp.

Choose a model:
Logistic Regression (baseline)

Decision Tree / RandomForest

Gradient Boosting (XGBoost / LightGBM)

Train to predict:
failure probability

time-to-failure (TTF)

Output a new column:
ml_predicted_risk

gives features, so the ML team can focus ONLY on the model

🔍 3. How the Data Quality Tree Should Use This

Mansib’s job is to detect “bad” or unreliable data BEFORE it enters the ML model or risk scoring.
Cass’s pipeline already contains several quality indicators that help with this:

Columns useful for Data Quality
comm_age_days → missing communication suggests stale device
battery_report_age_days → outdated reports
LineCurrent_val → corrupted if extremely high/negative
LineTemperature_val → impossible values
gps_jump_flag → sudden impossible device movement
distance_drift_m → suspicious location changes
zero_current_flag + overheat_flag → contradictory values?

Data Quality Tree outputs might be:
is_data_valid
data_quality_reason
should_exclude_from_ml (True/False)

Examples of rules:
If temperature < -40 or > 150 → corrupt
If current < 0 → corrupt
If GPS jumps > 300m → corrupted GPS
If comm_age_days > 30 → stale device

🔔 4. How the Alert Tree Should Use This

Sora creates Power BI dashboards, email/text alerts, and thresholds.
Cass’s output provides what is needed for alerts:

Alert inputs:
risk_score (0–100)
risk_reason (human readable explanation)
overheat_flag
zero_current_flag
gps_jump_flag
battery_low_flag

Suggested alert levels:
Risk Score	Alert Level	Description
80–100	🔥 Critical	Immediate field inspection
50–79	⚠ High	Investigate soon
20–49	⏳ Medium	Monitoring needed
1–19	🟢 Low	Normal
0	OK	Healthy
Suggested Alert Tree Outputs:
alert_level
alert_message
needs_inspection
powerbi_severity_color

Example alert message:
Device SIM008: High Risk (59.3) — Over temperature condition, Line temperature exceeds MM3 threshold.

✔ Cass’s risk_reason column feeds into alert descriptions.
🧭 5. How to Use Cass's Pipeline in Any Tree

Any teammate can start their code with:

from feature_health.health_features import build_health_features

df_features = build_health_features(df_raw_devices)


From here:

ML team → feed into a model

Data Quality team → apply rules

Alert team → create thresholds


No one needs to touch:

Device-level risk formulas

Temperature thresholds

Feature naming conventions

Normalization

Cass’s code handles all of that.