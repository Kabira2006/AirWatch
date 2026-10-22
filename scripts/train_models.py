import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)

def train_and_evaluate_ablation():
    print("Loading train_model_dataset.csv...")
    df = pd.read_csv("train_model_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Chronological Split (Train: 2015-2019, Test: 2020)
    train_df = df[df["Date"] < "2020-01-01"].copy()
    test_df = df[df["Date"] >= "2020-01-01"].copy()

    print(f"Train samples (2015-2019): {len(train_df)}")
    print(f"Test samples (2020): {len(test_df)}")

    target_col = "Target_PoorOrWorse_t1"
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Feature Groups for Ablation Study
    cpcb_features = [
        "AQI", "AQI_lag1", "AQI_lag2", "AQI_lag3", "AQI_roll_mean_7", "AQI_roll_std_7",
        "PM2.5", "PM2.5_lag1", "PM2.5_roll_mean_7",
        "PM10", "PM10_lag1", "PM10_roll_mean_7",
        "NO2", "NO2_lag1", "SO2", "SO2_lag1", "CO", "CO_lag1", "O3", "O3_lag1"
    ]
    # Filter available features
    cpcb_features = [f for f in cpcb_features if f in df.columns]

    weather_features = [
        "Temp_C", "Temp_C_lag1", "Temp_change_3d",
        "Humidity_pct", "Humidity_pct_lag1",
        "WindSpeed_ms", "WindSpeed_ms_lag1", "WindSpeed_roll_3d",
        "Rain_mm", "Rain_mm_lag1", "Is_Raining_t",
        "Pressure_kPa", "Pressure_kPa_lag1"
    ]
    weather_features = [f for f in weather_features if f in df.columns]

    context_features = [
        "PopDensity_per_km2", "Sat_NO2_Proxy", "Sat_Gap_Flag",
        "Month", "DayOfWeek", "DayOfYear", "Is_Winter", "Is_Monsoon"
    ]
    context_features = [f for f in context_features if f in df.columns]

    feature_sets = {
        "1. CPCB Ground Only": cpcb_features,
        "2. CPCB + Weather": cpcb_features + weather_features,
        "3. Full Model (CPCB + Weather + Pop/Sat Context)": cpcb_features + weather_features + context_features
    }

    # Evaluate Baseline Persistence Model ("Tomorrow = Today")
    y_pred_baseline = (test_df["AQI"] >= 201).astype(int).values
    baseline_metrics = {
        "Model": "Baseline (Persistence)",
        "Feature Set": "Today AQI >= 201",
        "Accuracy": accuracy_score(y_test, y_pred_baseline),
        "Precision": precision_score(y_test, y_pred_baseline, zero_division=0),
        "Recall": recall_score(y_test, y_pred_baseline, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_baseline, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_pred_baseline),
        "PR-AUC": average_precision_score(y_test, y_pred_baseline)
    }

    results = [baseline_metrics]

    # Evaluate ML Classifiers across Feature Sets
    for set_name, f_list in feature_sets.items():
        X_train = train_df[f_list].values
        X_test = test_df[f_list].values

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1)
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            res = {
                "Model": model_name,
                "Feature Set": set_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "PR-AUC": average_precision_score(y_test, y_proba)
            }
            results.append(res)

    results_df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    results_df.to_csv("data/ablation_results.csv", index=False)
    
    print("\n--- ABLATION STUDY RESULTS MATRIX ---")
    print(results_df.to_string(index=False))

    # Feature Importance for Full XGBoost Model
    xgb_full = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1)
    full_f_list = feature_sets["3. Full Model (CPCB + Weather + Pop/Sat Context)"]
    xgb_full.fit(train_df[full_f_list], y_train)

    fi_df = pd.DataFrame({
        "Feature": full_f_list,
        "Importance": xgb_full.feature_importances_
    }).sort_values("Importance", ascending=False)

    fi_df.to_csv("data/feature_importance.csv", index=False)
    print("\nTop 10 Most Important Features:")
    print(fi_df.head(10).to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate_ablation()
