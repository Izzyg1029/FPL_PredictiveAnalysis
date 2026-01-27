# ml_preventative_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_ml_dataset(time_series_path=None):
    """
    Prepare dataset for ML training from time series data.
    Creates features and labels for predicting FCI failure/offline status.
    """
    print("\n" + "=" * 60)
    print("STEP 1: PREPARING ML DATASET")
    print("=" * 60)
    
    # Load time series data
    if time_series_path is None:
        time_series_path = project_root / "data" / "processed" / "time_series" / "2025-09-13_to_2025-10-13_health_zm1only_timeseries.csv"
    
    if not time_series_path.exists():
        print(f"❌ Time series file not found: {time_series_path}")
        # Try to find latest time series file
        ts_dir = project_root / "data" / "processed" / "time_series"
        ts_files = list(ts_dir.glob("*health_zm1only_timeseries.csv"))
        if ts_files:
            time_series_path = ts_files[-1]  # Use most recent
            print(f"📁 Using latest time series: {time_series_path.name}")
        else:
            print("❌ No time series files found. Run process_daily_time_series.py first.")
            return None
    
    print(f"📊 Loading time series data from: {time_series_path}")
    df = pd.read_csv(time_series_path)
    print(f"   Loaded {len(df):,} records, {df['Serial'].nunique():,} unique devices")
    
    # Sort by device and date
    df = df.sort_values(['Serial', 'date'])
    
    # ====================================================
    # FEATURE ENGINEERING
    # ====================================================
    print("\n🔧 ENGINEERING FEATURES...")
    
    # Ensure we have the 5 original features
    original_features = ["battery_level", "LineTemperature_val", "LineCurrent_val", 
                        "device_age_months", "comm_age_days"]
    
    # Check and create missing original features
    if 'device_age_months' not in df.columns and 'device_age_days' in df.columns:
        df['device_age_months'] = df['device_age_days'] / 30.44
    
    if 'comm_age_days' not in df.columns and 'hours_since_last_heard' in df.columns:
        df['comm_age_days'] = df['hours_since_last_heard'] / 24
    
    # 1. Age-adjusted battery risk (engineered feature)
    # Higher risk for older devices with low battery
    if 'battery_level' in df.columns and 'device_age_months' in df.columns:
        # Normalize battery level (0-1, where 1 is full, 0 is empty)
        battery_normalized = df['battery_level'] / 100.0
        
        # Normalize device age (0-1, where 1 is oldest)
        if df['device_age_months'].max() > 0:
            age_normalized = df['device_age_months'] / df['device_age_months'].max()
        else:
            age_normalized = df['device_age_months']
        
        # Calculate risk: higher when battery low AND device old
        df['age_adjusted_battery_risk'] = (1 - battery_normalized) * (0.3 + 0.7 * age_normalized)
        print(f"   ✓ Created: age_adjusted_battery_risk")
    
    # 2. Maintenance urgency score (engineered feature)
    # Combines multiple risk factors
    df['maintenance_urgency_score'] = 0.0
    
    # Battery component (0-40 points)
    if 'battery_level' in df.columns:
        battery_component = np.where(
            df['battery_level'] < 20, 40,
            np.where(df['battery_level'] < 30, 30,
                    np.where(df['battery_level'] < 50, 20,
                            np.where(df['battery_level'] < 70, 10, 0)))
        )
        df['maintenance_urgency_score'] += battery_component
    
    # Temperature component (0-30 points)
    if 'LineTemperature_val' in df.columns:
        temp_component = np.where(
            df['LineTemperature_val'] > 80, 30,
            np.where(df['LineTemperature_val'] > 70, 20,
                    np.where(df['LineTemperature_val'] > 60, 10, 0))
        )
        df['maintenance_urgency_score'] += temp_component
    
    # Age component (0-20 points)
    if 'device_age_months' in df.columns:
        age_component = np.where(
            df['device_age_months'] > 60, 20,  # >5 years
            np.where(df['device_age_months'] > 36, 15,  # >3 years
                    np.where(df['device_age_months'] > 24, 10,  # >2 years
                            np.where(df['device_age_months'] > 12, 5, 0)))  # >1 year
        )
        df['maintenance_urgency_score'] += age_component
    
    # Communication gap component (0-10 points)
    if 'comm_age_days' in df.columns:
        comm_component = np.where(
            df['comm_age_days'] > 7, 10,  # >1 week
            np.where(df['comm_age_days'] > 3, 7,  # >3 days
                    np.where(df['comm_age_days'] > 1, 3, 0))  # >1 day
        )
        df['maintenance_urgency_score'] += comm_component
    
    print(f"   ✓ Created: maintenance_urgency_score")
    
    # 3. Old and hot flag (engineered feature)
    # Flags devices that are both old and running hot
    if 'device_age_months' in df.columns and 'LineTemperature_val' in df.columns:
        old_threshold = df['device_age_months'].quantile(0.75)  # Top 25% oldest
        hot_threshold = df['LineTemperature_val'].quantile(0.75)  # Top 25% hottest
        
        df['old_and_hot_flag'] = np.where(
            (df['device_age_months'] > old_threshold) & 
            (df['LineTemperature_val'] > hot_threshold), 1, 0
        )
        print(f"   ✓ Created: old_and_hot_flag (thresholds: age>{old_threshold:.1f}mo, temp>{hot_threshold:.1f}°C)")
    
    # ====================================================
    # CREATE TARGET VARIABLE (LABEL) USING STATUS COLUMN
    # ====================================================
    print("\n🎯 CREATING TARGET VARIABLES FROM STATUS COLUMN...")
    
    # Check if Status column exists
    if 'Status' not in df.columns:
        print("❌ 'Status' column not found in the dataset!")
        print("   Available columns:", df.columns.tolist())
        
        # Fall back to the original method if Status column doesn't exist
        print("   Falling back to original target creation method...")
        
        # Define failure/offline events based on your criteria
        df['failure_next_7d'] = 0
        df['failure_cause'] = 'none'
        
        # Group by device for look-ahead analysis
        for serial, device_data in df.groupby('Serial'):
            device_data = device_data.sort_values('date')
            indices = device_data.index
            
            # Check for future events in the next 7 days
            for i in range(len(device_data) - 1):
                current_idx = indices[i]
                
                # Look ahead up to 7 days
                lookahead_window = min(7, len(device_data) - i - 1)
                
                for j in range(1, lookahead_window + 1):
                    future_idx = indices[i + j]
                    future_date = pd.to_datetime(device_data.loc[future_idx, 'date'])
                    current_date = pd.to_datetime(device_data.loc[current_idx, 'date'])
                    
                    days_diff = (future_date - current_date).days
                    
                    if days_diff <= 7:
                        # Check for offline/failure indicators in future
                        
                        # 1. Communication failure
                        if ('hours_since_last_heard' in device_data.columns and 
                            device_data.loc[future_idx, 'hours_since_last_heard'] > 24 * 7):  # >7 days offline
                            df.loc[current_idx, 'failure_next_7d'] = 1
                            df.loc[current_idx, 'failure_cause'] = 'communication'
                            break
                        
                        # 2. Battery critical
                        elif ('battery_level' in device_data.columns and 
                              device_data.loc[future_idx, 'battery_level'] < 10):  # <10% battery
                            df.loc[current_idx, 'failure_next_7d'] = 1
                            df.loc[current_idx, 'failure_cause'] = 'battery'
                            break
                        
                        # 3. Overheating
                        elif ('LineTemperature_val' in device_data.columns and 
                              device_data.loc[future_idx, 'LineTemperature_val'] > 85):  # >85°C
                            df.loc[current_idx, 'failure_next_7d'] = 1
                            df.loc[current_idx, 'failure_cause'] = 'overheating'
                            break
    else:
        print(f"   Found 'Status' column with values: {df['Status'].unique().tolist()}")
        
        # Clean and standardize status values
        df['Status'] = df['Status'].astype(str).str.strip().str.upper()
        
        # Map status to failure flag
        # "OFFLINE" = failure (1), "STAND BY" = no failure (0)
        # Handle other status values if they exist
        status_mapping = {
            'OFFLINE': 1,
            'OFF LINE': 1,
            'OFF-LINE': 1,
            'STAND BY': 0,
            'STANDBY': 0,
            'ACTIVE': 0,
            'ONLINE': 0,
            'ON LINE': 0,
            'ON-LINE': 0
        }
        
        # Create failure flag based on status
        df['failure_next_7d'] = df['Status'].map(status_mapping)
        
        # For any unmapped status, assume not failure
        df['failure_next_7d'] = df['failure_next_7d'].fillna(0).astype(int)
        
        # Create failure cause based on status
        df['failure_cause'] = np.where(
            df['failure_next_7d'] == 1, 'offline_status', 'active'
        )
        
        # For forward-looking prediction (if needed), we can check if device becomes offline in next 7 days
        # This is optional - depends on whether you want current status or predictive status
        if True:  # Set to True if you want to predict future failures
            print("   Creating predictive target: Will device be OFFLINE in next 7 days?")
            
            # Group by device for look-ahead analysis
            df['failure_next_7d'] = 0  # Reset to 0
            df['failure_cause'] = 'none'
            
            for serial, device_data in df.groupby('Serial'):
                device_data = device_data.sort_values('date')
                indices = device_data.index
                
                # Check for future OFFLINE status in the next 7 days
                for i in range(len(device_data) - 1):
                    current_idx = indices[i]
                    
                    # Look ahead up to 7 days
                    lookahead_window = min(7, len(device_data) - i - 1)
                    
                    for j in range(1, lookahead_window + 1):
                        future_idx = indices[i + j]
                        future_date = pd.to_datetime(device_data.loc[future_idx, 'date'])
                        current_date = pd.to_datetime(device_data.loc[current_idx, 'date'])
                        
                        days_diff = (future_date - current_date).days
                        
                        if days_diff <= 7:
                            future_status = str(device_data.loc[future_idx, 'Status']).strip().upper()
                            
                            # If device will be OFFLINE in the next 7 days, mark current as at-risk
                            if future_status in ['OFFLINE', 'OFF LINE', 'OFF-LINE']:
                                df.loc[current_idx, 'failure_next_7d'] = 1
                                df.loc[current_idx, 'failure_cause'] = 'future_offline'
                                break
    
    failure_percentage = df['failure_next_7d'].mean() * 100
    print(f"   Failure rate in dataset: {failure_percentage:.1f}% ({df['failure_next_7d'].sum():,} events)")
    
    # ====================================================
    # SELECT FINAL FEATURES FOR ML
    # ====================================================
    ml_features = [
        "battery_level",
        "LineTemperature_val", 
        "LineCurrent_val",
        "device_age_months",
        "comm_age_days",
        "age_adjusted_battery_risk",
        "maintenance_urgency_score",
        "old_and_hot_flag"
    ]
    
    # Check which features are available
    available_features = [f for f in ml_features if f in df.columns]
    missing_features = [f for f in ml_features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
        print("   Creating placeholder values...")
        
        for feature in missing_features:
            if feature == "old_and_hot_flag":
                df[feature] = 0  # Binary flag
            else:
                df[feature] = df[available_features[0]] if available_features else 0
    
    print(f"\n✅ FINAL FEATURE SET ({len(ml_features)} features):")
    for i, feat in enumerate(ml_features, 1):
        status = "✓" if feat in df.columns else "✗"
        print(f"   {i:2d}. {status} {feat}")
    
    # Create final ML dataset
    ml_df = df[['Serial', 'date', 'failure_next_7d', 'failure_cause'] + ml_features].copy()
    
    # Remove rows with missing values
    initial_rows = len(ml_df)
    ml_df = ml_df.dropna(subset=ml_features)
    rows_removed = initial_rows - len(ml_df)
    
    if rows_removed > 0:
        print(f"   Removed {rows_removed} rows with missing values")
    
    print(f"\n📈 ML DATASET READY:")
    print(f"   Total samples: {len(ml_df):,}")
    print(f"   Features: {len(ml_features)}")
    print(f"   Failure rate: {ml_df['failure_next_7d'].mean()*100:.1f}%")
    
    return ml_df, ml_features

def train_random_forest_model(ml_df, features, test_size=0.2, random_state=42):
    """
    Train Random Forest model to predict FCI failures.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    # Prepare X and y
    X = ml_df[features]
    y = ml_df['failure_next_7d']
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n🔀 Train-Test Split:")
    print(f"   Training set: {X_train.shape[0]:,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test set: {X_test.shape[0]:,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # ====================================================
    # TRAIN RANDOM FOREST
    # ====================================================
    print("\n🌲 TRAINING RANDOM FOREST...")
    
    # Define pipeline with scaling and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"   Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Train final model
    pipeline.fit(X_train, y_train)
    
    # ====================================================
    # EVALUATION
    # ====================================================
    print("\n📊 MODEL EVALUATION:")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred):.3f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Classification report
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"📊 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0, 0]}")
    print(f"   False Positives: {cm[0, 1]}")
    print(f"   False Negatives: {cm[1, 0]}")
    print(f"   True Positives:  {cm[1, 1]}")
    
    # ====================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ====================================================
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Get feature importances from trained model
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 FEATURE IMPORTANCE RANKING:")
    for i, row in feature_importance_df.iterrows():
        print(f"   {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    # Identify strongest predictor
    strongest_feature = feature_importance_df.iloc[0]
    print(f"\n🏆 STRONGEST PREDICTOR:")
    print(f"   Feature: {strongest_feature['feature']}")
    print(f"   Importance: {strongest_feature['importance']:.4f}")
    print(f"   Percentage of total: {strongest_feature['importance']/importances.sum()*100:.1f}%")
    
    # ====================================================
    # VISUALIZATION
    # ====================================================
    print("\n📈 GENERATING VISUALIZATIONS...")
    
    # Create output directory
    output_dir = project_root / "reports" / "ml_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(feature_importance_df)), 
                    feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    
    # Color the strongest predictor differently
    bars[0].set_color('red')
    
    plt.tight_layout()
    importance_plot = output_dir / "feature_importance.png"
    plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {importance_plot}")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    correlation_plot = output_dir / "feature_correlation.png"
    plt.savefig(correlation_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {correlation_plot}")
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_plot = output_dir / "roc_curve.png"
    plt.savefig(roc_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {roc_plot}")
    
    # ====================================================
    # SAVE MODEL AND RESULTS
    # ====================================================
    print("\n💾 SAVING MODEL AND RESULTS...")
    
    # Save the trained model
    model_path = output_dir / "random_forest_fci_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"   ✓ Model saved: {model_path}")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"   ✓ Feature importance saved: {importance_path}")
    
    # Save model performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'cv_mean_auc': cv_scores.mean(),
        'cv_std_auc': cv_scores.std(),
        'strongest_feature': strongest_feature['feature'],
        'strongest_importance': float(strongest_feature['importance']),
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'failure_rate_training': y_train.mean(),
        'failure_rate_test': y_test.mean()
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ✓ Model metrics saved: {metrics_path}")
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'probability': y_pred_proba
    }, index=y_test.index)
    
    # Add back original data for context
    test_indices = y_test.index
    predictions_df = pd.concat([
        predictions_df,
        ml_df.loc[test_indices, ['Serial', 'date', 'failure_cause'] + features]
    ], axis=1)
    
    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   ✓ Test predictions saved: {predictions_path}")
    
    # ====================================================
    # GENERATE INSIGHTS AND RECOMMENDATIONS
    # ====================================================
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n🔍 TOP INSIGHTS:")
    print(f"   1. Strongest predictor: {strongest_feature['feature']}")
    print(f"   2. Model can predict failures with {accuracy_score(y_test, y_pred)*100:.1f}% accuracy")
    print(f"   3. Top 3 features account for {feature_importance_df.head(3)['importance'].sum()/importances.sum()*100:.1f}% of predictive power")
    
    print("\n🎯 POTENTIAL FAILURE CAUSES & NEXT STEPS:")
    
    # Analyze feature patterns for failures
    failure_samples = ml_df[ml_df['failure_next_7d'] == 1]
    if len(failure_samples) > 0:
        print(f"\n📊 ANALYSIS OF {len(failure_samples)} FAILURE SAMPLES:")
        
        # Battery-related failures
        battery_failures = failure_samples[failure_samples['battery_level'] < 30]
        if len(battery_failures) > 0:
            avg_battery = battery_failures['battery_level'].mean()
            print(f"   🔋 Battery-related failures: {len(battery_failures)}")
            print(f"      • Average battery level: {avg_battery:.1f}%")
            print(f"      • Recommendation: Proactive battery replacement at 40% threshold")
        
        # Temperature-related failures
        temp_failures = failure_samples[failure_samples['LineTemperature_val'] > 70]
        if len(temp_failures) > 0:
            avg_temp = temp_failures['LineTemperature_val'].mean()
            print(f"   🔥 Temperature-related failures: {len(temp_failures)}")
            print(f"      • Average temperature: {avg_temp:.1f}°C")
            print(f"      • Recommendation: Thermal inspection for devices >65°C")
        
        # Age-related failures
        age_failures = failure_samples[failure_samples['device_age_months'] > 36]
        if len(age_failures) > 0:
            avg_age = age_failures['device_age_months'].mean()
            print(f"   📅 Age-related failures: {len(age_failures)}")
            print(f"      • Average device age: {avg_age:.1f} months")
            print(f"      • Recommendation: Scheduled replacement at 3-year mark")
    
    print("\n✅ ML PIPELINE COMPLETE!")
    print(f"📁 Results saved in: {output_dir}")
    
    return pipeline, feature_importance_df, predictions_df

def run_ml_pipeline(time_series_path=None):
    """
    Complete ML pipeline for FCI failure prediction.
    """
    print("=" * 70)
    print("FCI FAILURE PREDICTION ML PIPELINE")
    print("=" * 70)
    
    # Step 1: Prepare dataset
    ml_df, features = prepare_ml_dataset(time_series_path)
    
    if ml_df is None or len(ml_df) < 100:
        print("\n❌ Insufficient data for ML training. Need at least 100 samples.")
        print("   Please ensure you have processed daily time series data first.")
        return
    
    # Step 2: Train model and analyze
    model, importance_df, predictions = train_random_forest_model(ml_df, features)
    
    # Step 3: Generate deployment recommendations
    print("\n" + "=" * 60)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 60)
    
    strongest_feature = importance_df.iloc[0]['feature']
    
    recommendations = {
        "maintenance_urgency_score": "Prioritize devices with score >50 for immediate inspection",
        "battery_level": "Schedule battery replacement at 30% threshold (not 20%)",
        "LineTemperature_val": "Implement temperature alerts at 70°C (current likely 75°C)",
        "device_age_months": "Plan replacements at 36-month intervals",
        "comm_age_days": "Investigate devices offline >3 days immediately"
    }
    
    print("\n📋 RECOMMENDED ACTIONS BASED ON ML INSIGHTS:")
    for feature, recommendation in recommendations.items():
        if feature in importance_df['feature'].values:
            importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
            rank = importance_df[importance_df['feature'] == feature].index[0] + 1
            print(f"   #{rank:2d} {feature:25s} (importance: {importance:.3f})")
            print(f"      → {recommendation}")
    
    print(f"\n🎯 KEY TAKEAWAY: Focus on monitoring {strongest_feature} as primary indicator")
    
    return model, importance_df, predictions

if __name__ == "__main__":
    run_ml_pipeline()
