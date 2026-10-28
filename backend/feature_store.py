"""
AirWatch Feature Store
Provides standardized access to cleaned sensor readings, rolling statistical features,
and business loss variables.
"""

import os
import pandas as pd
from typing import List, Tuple, Optional


class FeatureStore:
    """
    Centralized repository of engineered features for modeling and decision engines.
    """
    DEFAULT_DATA_PATH = os.path.join("data", "processed_sensor_business_data.csv")
    FALLBACK_RAW_PATH = "station_day_master.csv"

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self._df: Optional[pd.DataFrame] = None

    def get_data(self) -> pd.DataFrame:
        """
        Loads the processed dataset with lazy loading.
        """
        if self._df is None:
            if not os.path.exists(self.data_path):
                # If processed does not exist yet, import and trigger pipeline
                from backend.data_pipeline import SensorDataPipeline
                print(f"Feature store file {self.data_path} not found. Running pipeline...")
                pipeline = SensorDataPipeline(self.FALLBACK_RAW_PATH)
                self._df = pipeline.run_pipeline()
            else:
                self._df = pd.read_csv(self.data_path)
                self._df["Date"] = pd.to_datetime(self._df["Date"])
        return self._df

    @property
    def core_pollutants(self) -> List[str]:
        return ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

    @property
    def weather_features(self) -> List[str]:
        return ["Temp_C", "Humidity_pct", "WindSpeed_ms", "Rain_mm", "Pressure_kPa", "Atmospheric_Inversion_Index"]

    @property
    def lag_and_rolling_features(self) -> List[str]:
        return [
            "MA_3d", "MA_7d", "MA_24d", "PM25_MA_3d", "PM25_MA_7d",
            "AQI_velocity_1d", "AQI_velocity_3d", "AQI_accel_1d",
            "AQI_lag1", "AQI_lag2", "AQI_lag3",
            "PM25_lag1", "NO2_lag1", "Temp_lag1", "Wind_lag1"
        ]

    @property
    def business_impact_features(self) -> List[str]:
        return [
            "Health_Vulnerability_Index",
            "Projected_Productivity_Loss_USD",
            "Respiratory_Complaint_Risk_pct"
        ]

    def get_modeling_dataset(
        self,
        target_horizon: str = "t1"
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Extracts features and target for 24h (t1) or 48h (t2) forecast horizons.
        """
        df = self.get_data().copy()
        target_col = f"Target_AQI_{target_horizon}"
        valid_df = df.dropna(subset=[target_col]).copy()

        feature_cols = self.core_pollutants + self.weather_features + self.lag_and_rolling_features
        # Keep only available columns
        available_features = [c for c in feature_cols if c in valid_df.columns]

        X = valid_df[available_features].fillna(valid_df[available_features].median())
        y = valid_df[target_col]
        return X, y, available_features

    def get_ward_summary(self) -> pd.DataFrame:
        """
        Summarizes sensor and business metrics grouped by City/Station.
        """
        df = self.get_data()
        group_cols = ["City", "StationId"] if "StationId" in df.columns else ["City"]
        
        agg_dict = {
            "AQI": ["mean", "max", lambda x: (x > 200).mean() * 100],
            "PM2.5": "mean",
            "Health_Vulnerability_Index": "mean",
            "Projected_Productivity_Loss_USD": ["mean", "sum"],
            "Respiratory_Complaint_Risk_pct": "mean"
        }
        if "PopDensity_per_km2" in df.columns:
            agg_dict["PopDensity_per_km2"] = "first"

        summary = df.groupby(group_cols).agg(agg_dict)
        summary.columns = [
            "Avg_AQI", "Max_AQI", "Pct_Days_PoorOrWorse", "Avg_PM25",
            "Avg_HVI", "Avg_Daily_PPL_USD", "Total_PPL_USD",
            "Avg_Respiratory_Risk_pct"
        ] + (["PopDensity"] if "PopDensity_per_km2" in df.columns else [])
        
        return summary.reset_index()


if __name__ == "__main__":
    store = FeatureStore()
    df = store.get_data()
    print("Feature store loaded! Shape:", df.shape)
    X, y, feats = store.get_modeling_dataset("t1")
    print(f"X shape: {X.shape}, y shape: {y.shape}, features: {len(feats)}")
