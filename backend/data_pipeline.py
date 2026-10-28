"""
AirWatch Data Pipeline & Extended Feature Engineering
Implements clean data ingestion, rolling features, rate-of-change,
and synthetic business loss & health impact modeling (HVI, PPL, RCR).
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_health_vulnerability_index(
    pm25: pd.Series,
    pm10: pd.Series,
    no2: pd.Series,
    pop_density: Optional[pd.Series] = None
) -> pd.Series:
    """
    Computes Health Vulnerability Index (HVI) scaled from 0 to 100.
    Combines fine particulate exposure, coarse dust, nitrogen dioxide, and spatial density.
    """
    norm_pm25 = (pm25 / 250.0).clip(0, 2.5)
    norm_pm10 = (pm10 / 400.0).clip(0, 2.5)
    norm_no2 = (no2 / 120.0).clip(0, 2.5)
    
    if pop_density is not None:
        norm_density = (pop_density / 20000.0).clip(0, 2.0)
    else:
        norm_density = pd.Series(1.0, index=pm25.index)
        
    hvi_raw = (0.40 * norm_pm25 + 0.30 * norm_pm10 + 0.15 * norm_no2 + 0.15 * norm_density) * 100.0
    return hvi_raw.clip(0, 100).round(2)


def calculate_projected_productivity_loss(
    aqi: pd.Series,
    base_daily_payroll_per_1000: float = 45000.0
) -> pd.Series:
    """
    Computes Projected Productivity Loss (PPL) in USD per 1,000 corporate campus employees.
    Applies empirical environmental economics impairment thresholds:
    - AQI <= 100: 0% loss (Good/Satisfactory baseline)
    - 101 <= AQI <= 200: 4.5% cognitive & energy impairment
    - 201 <= AQI <= 300: 12.8% loss (moderate symptoms, absenteeism, ventilation throttling)
    - AQI > 300: 24.5% loss (severe fatigue, respiratory acute symptoms, high sick leave)
    """
    loss_fraction = np.select(
        [
            aqi <= 100,
            (aqi > 100) & (aqi <= 200),
            (aqi > 200) & (aqi <= 300),
            aqi > 300
        ],
        [
            0.0,
            0.045,
            0.128,
            0.245
        ],
        default=0.045
    )
    ppl = loss_fraction * base_daily_payroll_per_1000
    return pd.Series(ppl, index=aqi.index).round(2)


def calculate_respiratory_complaint_risk(
    pm25: pd.Series,
    pm10: pd.Series,
    no2: pd.Series
) -> pd.Series:
    """
    Computes Respiratory Complaint Risk (RCR) percentage (0 - 100%)
    using a sigmoid epidemiological dose-response curve.
    """
    z = -3.2 + 0.016 * pm25.fillna(pm25.median()) + 0.007 * pm10.fillna(pm10.median()) + 0.012 * no2.fillna(no2.median())
    prob = 1.0 / (1.0 + np.exp(-z))
    return (prob * 100.0).clip(2.0, 95.0).round(2)


class SensorDataPipeline:
    """
    Production-grade ETL & feature-enrichment pipeline for ambient air sensor streams.
    """
    def __init__(self, raw_filepath: str = "station_day_master.csv"):
        self.raw_filepath = raw_filepath
        self.processed_filepath = os.path.join("data", "processed_sensor_business_data.csv")

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_filepath):
            raise FileNotFoundError(f"Source file {self.raw_filepath} not found.")
        print(f"Loading raw sensor data from {self.raw_filepath}...")
        df = pd.read_csv(self.raw_filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans sensor noise and applies structured time-series imputation.
        """
        print("Executing data cleaning, bounds validation, and time-based imputation...")
        df = df.sort_values(["StationId", "Date"]).reset_index(drop=True)

        pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI"]
        weather = ["Temp_C", "Humidity_pct", "WindSpeed_ms", "Rain_mm", "Pressure_kPa"]

        # Ensure physical non-negativity
        for col in pollutants + weather:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        # Impute short gaps via linear interpolation per station
        for col in pollutants + weather:
            if col in df.columns:
                df[col] = df.groupby("StationId")[col].transform(
                    lambda s: s.interpolate(method="linear", limit=3).ffill().bfill()
                )
                # Fill remaining global nulls with column median
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates rolling moving averages (MA_3h, MA_24h), rate-of-change metrics,
        lag features, and peak-hour flags.
        """
        print("Engineering statistical rolling features, rate-of-change, and lag metrics...")
        
        # 1. Moving Averages (MA_3, MA_7, MA_24)
        df["MA_3d"] = df.groupby("StationId")["AQI"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
        df["MA_7d"] = df.groupby("StationId")["AQI"].transform(lambda x: x.rolling(7, min_periods=1).mean()).round(2)
        df["MA_24d"] = df.groupby("StationId")["AQI"].transform(lambda x: x.rolling(24, min_periods=1).mean()).round(2)
        
        df["PM25_MA_3d"] = df.groupby("StationId")["PM2.5"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
        df["PM25_MA_7d"] = df.groupby("StationId")["PM2.5"].transform(lambda x: x.rolling(7, min_periods=1).mean()).round(2)

        # 2. Rate-of-Change Metrics (delta AQI / delta t)
        df["AQI_velocity_1d"] = (df["AQI"] - df.groupby("StationId")["AQI"].shift(1)).fillna(0).round(2)
        df["AQI_velocity_3d"] = (df["AQI"] - df.groupby("StationId")["AQI"].shift(3)).fillna(0).round(2)
        df["AQI_accel_1d"] = (df["AQI_velocity_1d"] - df.groupby("StationId")["AQI_velocity_1d"].shift(1)).fillna(0).round(2)
        df["PM25_velocity_1d"] = (df["PM2.5"] - df.groupby("StationId")["PM2.5"].shift(1)).fillna(0).round(2)

        # 3. Lag Features (t-1, t-2, t-3, t-7)
        for lag in [1, 2, 3, 7]:
            df[f"AQI_lag{lag}"] = df.groupby("StationId")["AQI"].shift(lag)
            df[f"PM25_lag{lag}"] = df.groupby("StationId")["PM2.5"].shift(lag)
            df[f"NO2_lag{lag}"] = df.groupby("StationId")["NO2"].shift(lag)
            df[f"Temp_lag{lag}"] = df.groupby("StationId")["Temp_C"].shift(lag)
            df[f"Wind_lag{lag}"] = df.groupby("StationId")["WindSpeed_ms"].shift(lag)

        # Fill lag nulls per station
        lag_cols = [c for c in df.columns if "_lag" in c]
        for col in lag_cols:
            base_col = col.split("_lag")[0]
            if base_col == "PM25": base_col = "PM2.5"
            if base_col == "Temp": base_col = "Temp_C"
            if base_col == "Wind": base_col = "WindSpeed_ms"
            df[col] = df[col].fillna(df[base_col])

        # 4. Inversion Index & Peak Flags
        # Inversion condition: High humidity + low temperature + low wind speed
        df["Atmospheric_Inversion_Index"] = (
            df["Humidity_pct"] / (df["Temp_C"] + 1.0).clip(lower=1.0) * (1.0 / (df["WindSpeed_ms"] + 0.5))
        ).round(3)

        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Is_Winter"] = df["Month"].isin([11, 12, 1, 2]).astype(int)
        df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        df["Is_Peak_Season"] = ((df["Is_Winter"] == 1) & (df["Atmospheric_Inversion_Index"] > 2.0)).astype(int)

        # 5. Business Loss & Health Impact Variables
        print("Calculating synthetic business and health impact variables (HVI, PPL, RCR)...")
        pop_density = df["PopDensity_per_km2"] if "PopDensity_per_km2" in df.columns else None
        df["Health_Vulnerability_Index"] = calculate_health_vulnerability_index(
            df["PM2.5"], df["PM10"], df["NO2"], pop_density
        )
        df["Projected_Productivity_Loss_USD"] = calculate_projected_productivity_loss(df["AQI"])
        df["Respiratory_Complaint_Risk_pct"] = calculate_respiratory_complaint_risk(
            df["PM2.5"], df["PM10"], df["NO2"]
        )

        # 6. Forecasting Targets (t+1 and t+2 continuous AQI)
        df["Target_AQI_t1"] = df.groupby("StationId")["AQI"].shift(-1)
        df["Target_AQI_t2"] = df.groupby("StationId")["AQI"].shift(-2)
        df["Target_HighRisk_t1"] = (df["Target_AQI_t1"] >= 201).astype(float)
        df["Target_HighRisk_t2"] = (df["Target_AQI_t2"] >= 201).astype(float)

        return df

    def run_pipeline(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Runs complete ETL, engineering, and saves processed file.
        """
        df = self.load_data()
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} records for rapid pipeline processing...")
            df = df.sample(sample_size, random_state=42).sort_values(["StationId", "Date"]).reset_index(drop=True)
            
        df = self.clean_and_impute(df)
        df = self.engineer_features(df)
        
        os.makedirs(os.path.dirname(self.processed_filepath), exist_ok=True)
        df.to_csv(self.processed_filepath, index=False)
        print(f"Pipeline successfully completed! Processed file saved at: {self.processed_filepath}")
        print(f"Columns: {len(df.columns)}, Rows: {len(df)}")
        return df


if __name__ == "__main__":
    pipeline = SensorDataPipeline()
    pipeline.run_pipeline()
