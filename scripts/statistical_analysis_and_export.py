import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def run_comprehensive_statistical_analysis():
    print("--- 1. Loading Master Dataset ---")
    master_path = "station_day_master.csv"
    if not os.path.exists(master_path):
        print(f"Error: {master_path} not found.")
        return

    df = pd.read_csv(master_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    def get_season(month):
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Summer / Pre-Monsoon"
        elif month in [6, 7, 8, 9]: return "Monsoon"
        else: return "Post-Monsoon (Peak Smog)"

    df["Season"] = df["Month"].apply(get_season)

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/powerbi", exist_ok=True)

    # -------------------------------------------------------------
    # 2. STATISTICAL TESTS: ANOVA & KRUSKAL-WALLIS
    # -------------------------------------------------------------
    print("--- 2. Computing ANOVA & Kruskal-Wallis Seasonality Tests ---")
    clean_aqi = df.dropna(subset=["AQI"])

    # Monthly ANOVA
    month_groups = [group["AQI"].values for _, group in clean_aqi.groupby("Month")]
    anova_stat, anova_p = stats.f_oneway(*month_groups)

    # Seasonal Kruskal-Wallis
    season_groups = [group["AQI"].values for _, group in clean_aqi.groupby("Season")]
    kw_stat, kw_p = stats.kruskal(*season_groups)

    print(f"Monthly ANOVA: F-stat = {anova_stat:.4f}, p-value = {anova_p:.4e}")
    print(f"Seasonal Kruskal-Wallis: H-stat = {kw_stat:.4f}, p-value = {kw_p:.4e}")

    # -------------------------------------------------------------
    # 3. CORRELATION ANALYSIS (POLLUTANTS vs WEATHER)
    # -------------------------------------------------------------
    print("--- 3. Correlation Matrix & Significance Testing ---")
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    weather_vars = ["Temp_C", "Humidity_pct", "WindSpeed_ms", "Rain_mm", "Pressure_kPa"]

    corr_records = []
    for p in pollutants:
        for w in weather_vars:
            pair_df = df.dropna(subset=[p, w])
            if len(pair_df) > 100:
                pearson_r, pearson_p = stats.pearsonr(pair_df[p], pair_df[w])
                spearman_r, spearman_p = stats.spearmanr(pair_df[p], pair_df[w])
                corr_records.append({
                    "Pollutant": p,
                    "Weather_Variable": w,
                    "Pearson_r": round(pearson_r, 4),
                    "Pearson_p_value": pearson_p,
                    "Spearman_rho": round(spearman_r, 4),
                    "Spearman_p_value": spearman_p,
                    "Significant_at_001": pearson_p < 0.001
                })

    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv("data/pollutant_weather_correlations.csv", index=False)
    print(f"Saved data/pollutant_weather_correlations.csv ({len(corr_df)} pairs)")

    # -------------------------------------------------------------
    # 4. SATELLITE VS GROUND CALIBRATION REGRESSION
    # -------------------------------------------------------------
    print("--- 4. Satellite vs Ground Regression Calibration ---")
    train_df = pd.read_csv("train_model_dataset.csv")
    sat_valid = train_df.dropna(subset=["NO2", "Sat_NO2_Proxy"])
    slope, intercept, r_val, p_val, std_err = stats.linregress(sat_valid["Sat_NO2_Proxy"], sat_valid["NO2"])
    r_squared = r_val ** 2
    rmse = np.sqrt(np.mean((sat_valid["NO2"] - (slope * sat_valid["Sat_NO2_Proxy"] + intercept)) ** 2))

    sat_calibration = {
        "Metric": ["Slope", "Intercept", "Pearson_r", "R_squared", "p_value", "Std_Error", "RMSE", "Sample_Count"],
        "Value": [round(slope, 4), round(intercept, 4), round(r_val, 4), round(r_squared, 4), p_val, round(std_err, 4), round(rmse, 2), len(sat_valid)]
    }
    pd.DataFrame(sat_calibration).to_csv("data/satellite_ground_calibration.csv", index=False)
    print(f"Satellite Calibration: R² = {r_squared:.4f}, Pearson r = {r_val:.4f}, RMSE = {rmse:.2f}")

    # -------------------------------------------------------------
    # 5. STATION CLUSTERING: ENVIRONMENT ARCHETYPES
    # -------------------------------------------------------------
    print("--- 5. Station Clustering into Environment Archetypes ---")
    station_stats = df.groupby(["StationId", "City"]).agg(
        Mean_AQI=("AQI", "mean"),
        Std_AQI=("AQI", "std"),
        Poor_Days_Pct=("AQI", lambda x: (x >= 201).mean() * 100),
        Mean_PM25=("PM2.5", "mean"),
        Mean_NO2=("NO2", "mean"),
        Mean_WindSpeed=("WindSpeed_ms", "mean"),
        PopDensity=("PopDensity_per_km2", "first")
    ).dropna().reset_index()

    features_to_cluster = ["Mean_AQI", "Std_AQI", "Poor_Days_Pct", "Mean_PM25", "Mean_NO2", "Mean_WindSpeed"]
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(station_stats[features_to_cluster])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    station_stats["Cluster_ID"] = kmeans.fit_predict(scaled_feats)

    # Map cluster IDs to human-interpretable environmental archetypes
    cluster_means = station_stats.groupby("Cluster_ID")["Mean_AQI"].mean().sort_values()
    id_mapping = {
        cluster_means.index[0]: "Archetype 1: Coastal & Well-Ventilated Low-Risk",
        cluster_means.index[1]: "Archetype 2: Moderate Urban Residential",
        cluster_means.index[2]: "Archetype 3: High Stagnation Industrial Corridor",
        cluster_means.index[3]: "Archetype 4: Extreme Severe Indo-Gangetic Smog Hotspot"
    }
    station_stats["Archetype"] = station_stats["Cluster_ID"].map(id_mapping)
    station_stats.to_csv("data/station_archetypes.csv", index=False)
    print("Saved data/station_archetypes.csv")

    # -------------------------------------------------------------
    # 6. POPULATION-WEIGHTED EXPOSURE PRIORITIZATION
    # -------------------------------------------------------------
    print("--- 6. Computing Population-Weighted District Exposure Index ---")
    city_exposure = df.groupby("City").agg(
        Total_Station_Days=("AQI", "count"),
        Mean_AQI=("AQI", "mean"),
        Median_AQI=("AQI", "median"),
        Poor_Days_Count=("AQI", lambda x: (x >= 201).sum()),
        Poor_Days_Pct=("AQI", lambda x: (x >= 201).mean() * 100),
        Severe_Days_Pct=("AQI", lambda x: (x >= 401).mean() * 100),
        PopDensity_per_km2=("PopDensity_per_km2", "first"),
        Lat=("Lat", "first"),
        Lon=("Lon", "first")
    ).reset_index()

    # Exposure Score Formula: (% Poor Days) * (Population Density / 1000)
    city_exposure["Exposure_Risk_Index"] = round(city_exposure["Poor_Days_Pct"] * (city_exposure["PopDensity_per_km2"] / 1000), 2)
    city_exposure["Severe_Exposure_Index"] = round(city_exposure["Severe_Days_Pct"] * (city_exposure["PopDensity_per_km2"] / 1000), 2)
    city_exposure = city_exposure.sort_values("Exposure_Risk_Index", ascending=False).reset_index(drop=True)
    city_exposure["Regulatory_Priority_Rank"] = city_exposure.index + 1

    city_exposure.to_csv("data/district_exposure_rankings.csv", index=False)
    print("Saved data/district_exposure_rankings.csv")

    # -------------------------------------------------------------
    # 7. CONFUSION MATRIX & ERROR / COST ANALYSIS
    # -------------------------------------------------------------
    print("--- 7. Training Winning Models & Generating Confusion Matrix ---")
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_split = train_df[train_df["Date"] < "2020-01-01"].copy()
    test_split = train_df[train_df["Date"] >= "2020-01-01"].copy()

    full_features = [
        "AQI", "AQI_lag1", "AQI_lag2", "AQI_lag3", "AQI_roll_mean_7", "AQI_roll_std_7",
        "PM2.5", "PM2.5_lag1", "PM2.5_roll_mean_7",
        "PM10", "PM10_lag1", "PM10_roll_mean_7",
        "NO2", "NO2_lag1", "SO2", "SO2_lag1", "CO", "CO_lag1", "O3", "O3_lag1",
        "Temp_C", "Temp_C_lag1", "Temp_change_3d",
        "Humidity_pct", "Humidity_pct_lag1",
        "WindSpeed_ms", "WindSpeed_ms_lag1", "WindSpeed_roll_3d",
        "Rain_mm", "Rain_mm_lag1", "Is_Raining_t",
        "Pressure_kPa", "Pressure_kPa_lag1",
        "PopDensity_per_km2", "Sat_NO2_Proxy", "Sat_Gap_Flag",
        "Month", "DayOfWeek", "DayOfYear", "Is_Winter", "Is_Monsoon"
    ]
    full_features = [f for f in full_features if f in train_df.columns]

    X_train = train_split[full_features].values
    y_train = train_split["Target_PoorOrWorse_t1"].values
    X_test = test_split[full_features].values
    y_test = test_split["Target_PoorOrWorse_t1"].values

    lgb = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    lgb.fit(X_train, y_train)

    y_proba = lgb.predict_proba(X_test)[:, 1]

    # Baseline predictions (Today AQI >= 201)
    y_baseline = (test_split["AQI"] >= 201).astype(int).values
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_baseline).ravel()

    # Threshold variation analysis [0.3, 0.4, 0.5, 0.6, 0.7]
    cm_records = []
    # Add Baseline
    cm_records.append({
        "Model": "Baseline (Persistence)",
        "Threshold": 0.5,
        "True_Negatives": tn_b,
        "False_Positives": fp_b,
        "False_Negatives": fn_b,
        "True_Positives": tp_b,
        "Precision": round(precision_score(y_test, y_baseline), 4),
        "Recall": round(recall_score(y_test, y_baseline), 4),
        "F1_Score": round(f1_score(y_test, y_baseline), 4),
        "False_Positive_Rate": round(fp_b / (fp_b + tn_b), 4),
        "False_Negative_Rate": round(fn_b / (fn_b + tp_b), 4)
    })

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        cm_records.append({
            "Model": f"LightGBM (Thresh={thresh})",
            "Threshold": thresh,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
            "Precision": round(precision_score(y_test, y_pred_thresh), 4),
            "Recall": round(recall_score(y_test, y_pred_thresh), 4),
            "F1_Score": round(f1_score(y_test, y_pred_thresh), 4),
            "False_Positive_Rate": round(fp / (fp + tn), 4),
            "False_Negative_Rate": round(fn / (fn + tp), 4)
        })

    cm_df = pd.DataFrame(cm_records)
    cm_df.to_csv("data/confusion_matrix_results.csv", index=False)
    print("Saved data/confusion_matrix_results.csv")

    # -------------------------------------------------------------
    # 8. 2-DAY AHEAD (t+2) MODEL TRAINING & COMPARISON
    # -------------------------------------------------------------
    print("--- 8. Training 2-Day Ahead (t+2) Model ---")
    train_t2 = train_df.dropna(subset=["Target_PoorOrWorse_t2"]).copy()
    train_split_t2 = train_t2[train_t2["Date"] < "2020-01-01"]
    test_split_t2 = train_t2[train_t2["Date"] >= "2020-01-01"]

    y_train_t2 = train_split_t2["Target_PoorOrWorse_t2"].values
    y_test_t2 = test_split_t2["Target_PoorOrWorse_t2"].values
    X_train_t2 = train_split_t2[full_features].values
    X_test_t2 = test_split_t2[full_features].values

    lgb_t2 = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    lgb_t2.fit(X_train_t2, y_train_t2)
    y_pred_t2 = lgb_t2.predict(X_test_t2)
    y_proba_t2 = lgb_t2.predict_proba(X_test_t2)[:, 1]

    t2_comparison = pd.DataFrame([
        {
            "Forecast_Horizon": "1-Day Ahead (t+1)",
            "Model": "LightGBM Full",
            "Accuracy": round(cm_df.loc[cm_df["Model"]=="LightGBM (Thresh=0.5)", "Precision"].values[0], 4), # will fill
            "Precision": round(precision_score(y_test, (y_proba >= 0.5).astype(int)), 4),
            "Recall": round(recall_score(y_test, (y_proba >= 0.5).astype(int)), 4),
            "F1_Score": round(f1_score(y_test, (y_proba >= 0.5).astype(int)), 4),
            "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4)
        },
        {
            "Forecast_Horizon": "2-Day Ahead (t+2)",
            "Model": "LightGBM Full",
            "Accuracy": round((y_pred_t2 == y_test_t2).mean(), 4),
            "Precision": round(precision_score(y_test_t2, y_pred_t2), 4),
            "Recall": round(recall_score(y_test_t2, y_pred_t2), 4),
            "F1_Score": round(f1_score(y_test_t2, y_pred_t2), 4),
            "ROC_AUC": round(roc_auc_score(y_test_t2, y_proba_t2), 4)
        }
    ])
    t2_comparison.to_csv("data/forecast_horizon_comparison.csv", index=False)
    print("Saved data/forecast_horizon_comparison.csv")

    # -------------------------------------------------------------
    # 9. POWER BI READY DATASET EXPORTS (5 DISTINCT SHEETS)
    # -------------------------------------------------------------
    print("--- 9. Generating 5 Power BI Dedicated Tables ---")

    # PAGE 1: Situational Overview Table (Aggregated by City, Month, AQI_Bucket)
    p1_df = df.groupby(["City", "Year", "Month", "Season", "AQI_Bucket"]).agg(
        Record_Count=("AQI", "count"),
        Avg_AQI=("AQI", "mean"),
        Avg_PM25=("PM2.5", "mean"),
        Avg_PM10=("PM10", "mean"),
        Avg_NO2=("NO2", "mean"),
        Avg_SO2=("SO2", "mean"),
        Avg_CO=("CO", "mean"),
        Avg_O3=("O3", "mean"),
        Lat=("Lat", "first"),
        Lon=("Lon", "first")
    ).reset_index()
    p1_df.to_csv("data/powerbi/page1_situational_overview.csv", index=False)

    # PAGE 2: Weather & Diagnostic Table (City, Date, WindSpeed_Bin, Humidity, Temp, Rain, AQI)
    p2_df = df[["StationId", "City", "Date", "Month", "Season", "AQI", "PM2.5", "WindSpeed_ms", "Temp_C", "Humidity_pct", "Rain_mm", "Pressure_kPa"]].dropna(subset=["AQI", "WindSpeed_ms"]).copy()
    p2_df["WindSpeed_Category"] = pd.cut(
        p2_df["WindSpeed_ms"],
        bins=[-0.1, 2.0, 4.0, 6.0, 10.0, 50.0],
        labels=["<2 m/s (Severe Stagnation)", "2-4 m/s (Light Breeze)", "4-6 m/s (Moderate Dispersion)", "6-10 m/s (High Ventilation)", ">10 m/s (Storm/Dispersion)"]
    )
    p2_df.to_csv("data/powerbi/page2_weather_diagnostic.csv", index=False)

    # PAGE 3: District Exposure Table
    city_exposure.to_csv("data/powerbi/page3_district_exposure.csv", index=False)

    # PAGE 4: Model Performance & Ablation Table
    ablation_df = pd.read_csv("data/ablation_results.csv")
    ablation_df.to_csv("data/powerbi/page4_model_ablation.csv", index=False)

    # PAGE 5: GRAP Alert Simulation for Delhi NCR (Daily Actual vs Predicted Alert)
    delhi_test = test_split[test_split["City"] == "Delhi"].copy()
    delhi_X = delhi_test[full_features].values
    delhi_test["Pred_Prob_PoorOrWorse_t1"] = lgb.predict_proba(delhi_X)[:, 1]
    delhi_test["AirWatch_Alert_Triggered"] = (delhi_test["Pred_Prob_PoorOrWorse_t1"] >= 0.5).astype(int)
    delhi_test["Actual_NextDay_PoorOrWorse"] = delhi_test["Target_PoorOrWorse_t1"].astype(int)

    p5_df = delhi_test[["StationId", "Date", "AQI", "Target_AQI_t1", "Actual_NextDay_PoorOrWorse", "Pred_Prob_PoorOrWorse_t1", "AirWatch_Alert_Triggered", "WindSpeed_ms", "Temp_C"]].copy()
    p5_df["Alert_Outcome"] = "Correct Clear"
    p5_df.loc[(p5_df["AirWatch_Alert_Triggered"] == 1) & (p5_df["Actual_NextDay_PoorOrWorse"] == 1), "Alert_Outcome"] = "True Positive (Advance Warning)"
    p5_df.loc[(p5_df["AirWatch_Alert_Triggered"] == 1) & (p5_df["Actual_NextDay_PoorOrWorse"] == 0), "Alert_Outcome"] = "False Alarm (Over-Prediction)"
    p5_df.loc[(p5_df["AirWatch_Alert_Triggered"] == 0) & (p5_df["Actual_NextDay_PoorOrWorse"] == 1), "Alert_Outcome"] = "Missed Violation (False Negative)"
    p5_df.to_csv("data/powerbi/page5_grap_simulation.csv", index=False)

    print("All statistical analyses and 5 Power BI dataset sheets exported successfully!")

if __name__ == "__main__":
    run_comprehensive_statistical_analysis()
