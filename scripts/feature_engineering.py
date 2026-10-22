import pandas as pd
import numpy as np

def run_feature_engineering():
    print("Loading station_day_master.csv...")
    df = pd.read_csv("station_day_master.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort strictly by StationId and Date for time-series operations
    df = df.sort_values(["StationId", "Date"]).reset_index(drop=True)
    
    # 1. Target Definition (1-Day and 2-Day Ahead GRAP Alert: Poor/Very Poor/Severe >= 201)
    df["Target_AQI_t1"] = df.groupby("StationId")["AQI"].shift(-1)
    df["Target_AQI_t2"] = df.groupby("StationId")["AQI"].shift(-2)
    
    df["Target_PoorOrWorse_t1"] = (df["Target_AQI_t1"] >= 201).astype(float)
    df.loc[df["Target_AQI_t1"].isna(), "Target_PoorOrWorse_t1"] = np.nan
    
    df["Target_PoorOrWorse_t2"] = (df["Target_AQI_t2"] >= 201).astype(float)
    df.loc[df["Target_AQI_t2"].isna(), "Target_PoorOrWorse_t2"] = np.nan

    # 2. CPCB Pollutant Lags & Rolling Aggregations
    pollutants = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
    for p in pollutants:
        if p in df.columns:
            df[f"{p}_lag1"] = df.groupby("StationId")[p].shift(1)
            df[f"{p}_lag2"] = df.groupby("StationId")[p].shift(2)
            df[f"{p}_lag3"] = df.groupby("StationId")[p].shift(3)
            df[f"{p}_roll_mean_7"] = df.groupby("StationId")[p].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    df["AQI_roll_std_7"] = df.groupby("StationId")["AQI"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())

    # 3. Weather Features & Lags
    weather_cols = ["Temp_C", "Humidity_pct", "WindSpeed_ms", "Rain_mm", "Pressure_kPa"]
    for w in weather_cols:
        if w in df.columns:
            df[f"{w}_lag1"] = df.groupby("StationId")[w].shift(1)
    
    # Weather derived signals
    df["Temp_change_3d"] = df["Temp_C"] - df.groupby("StationId")["Temp_C"].shift(3)
    df["WindSpeed_roll_3d"] = df.groupby("StationId")["WindSpeed_ms"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Is_Raining_t"] = (df["Rain_mm"] > 0.5).astype(int)

    # 4. Regional & Satellite Context Features
    # Satellite Proxy NO2 (weekly buffer mean simulation with cloud gap flag)
    np.random.seed(42)
    df["Sat_NO2_Proxy"] = df.groupby(["StationId", pd.Grouper(key="Date", freq="W")])["NO2"].transform("mean")
    # Simulate cloud cover missingness (15% gap rate)
    cloud_gaps = np.random.rand(len(df)) < 0.15
    df["Sat_Gap_Flag"] = cloud_gaps.astype(int)
    df.loc[cloud_gaps, "Sat_NO2_Proxy"] = np.nan
    # Forward fill weekly satellite proxy
    df["Sat_NO2_Proxy"] = df.groupby("StationId")["Sat_NO2_Proxy"].ffill()

    # 5. Temporal Features
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Is_Winter"] = df["Month"].isin([11, 12, 1]).astype(int)
    df["Is_Monsoon"] = df["Month"].isin([7, 8, 9]).astype(int)

    # Filter out rows where target is NaN (last rows per station)
    modeling_df = df.dropna(subset=["Target_PoorOrWorse_t1"]).copy()

    # Forward fill per station for missing predictor features, then fill remaining numeric with median
    non_feature_cols = ["StationId", "Date", "AQI_Bucket", "StateCode", "City", "Target_AQI_t1", "Target_AQI_t2", "Target_PoorOrWorse_t1", "Target_PoorOrWorse_t2"]
    numeric_cols = modeling_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in non_feature_cols]
    
    for f in feature_cols:
        modeling_df[f] = modeling_df.groupby("StationId")[f].ffill()
        if modeling_df[f].isna().any():
            modeling_df[f] = modeling_df[f].fillna(modeling_df[f].median())

    modeling_df.to_csv("train_model_dataset.csv", index=False)
    print(f"Feature Engineering Complete! Dataset Shape: {modeling_df.shape}")
    print(f"Target Positive Rate (Poor or Worse): {modeling_df['Target_PoorOrWorse_t1'].mean() * 100:.2f}%")
    print("Saved train_model_dataset.csv successfully!")

if __name__ == "__main__":
    run_feature_engineering()
