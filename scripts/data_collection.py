import os
import time
import pandas as pd
import numpy as np
import requests

# Coordinate & Population Density Lookup for State Codes / Major City Clusters
CITY_METADATA = {
    "DL": {"City": "Delhi", "Lat": 28.6139, "Lon": 77.2090, "PopDensity": 11320},
    "KA": {"City": "Bengaluru", "Lat": 12.9716, "Lon": 77.5946, "PopDensity": 4381},
    "TG": {"City": "Hyderabad", "Lat": 17.3850, "Lon": 78.4867, "PopDensity": 8060},
    "TN": {"City": "Chennai", "Lat": 13.0827, "Lon": 80.2707, "PopDensity": 26903},
    "UP": {"City": "Lucknow", "Lat": 26.8467, "Lon": 80.9462, "PopDensity": 6940},
    "MH": {"City": "Mumbai", "Lat": 19.0760, "Lon": 72.8777, "PopDensity": 20482},
    "WB": {"City": "Kolkata", "Lat": 22.5726, "Lon": 88.3639, "PopDensity": 24252},
    "GJ": {"City": "Ahmedabad", "Lat": 23.0225, "Lon": 72.5714, "PopDensity": 12530},
    "BR": {"City": "Patna", "Lat": 25.5941, "Lon": 85.1376, "PopDensity": 18200},
    "HR": {"City": "Gurugram", "Lat": 28.4595, "Lon": 77.0266, "PopDensity": 3100},
    "AP": {"City": "Amaravati", "Lat": 16.5131, "Lon": 80.5165, "PopDensity": 2800},
    "RJ": {"City": "Jaipur", "Lat": 26.9124, "Lon": 75.7873, "PopDensity": 6500},
    "PB": {"City": "Amritsar", "Lat": 31.6340, "Lon": 74.8723, "PopDensity": 3200},
    "OD": {"City": "Brajrajnagar", "Lat": 21.8236, "Lon": 83.9213, "PopDensity": 1500},
    "JH": {"City": "Talcher", "Lat": 20.9500, "Lon": 85.2300, "PopDensity": 1200},
    "AS": {"City": "Guwahati", "Lat": 26.1445, "Lon": 91.7362, "PopDensity": 4400},
    "MP": {"City": "Bhopal", "Lat": 23.2599, "Lon": 77.4126, "PopDensity": 8550},
    "KL": {"City": "Thiruvananthapuram", "Lat": 8.5241, "Lon": 76.9366, "PopDensity": 4450},
    "ML": {"City": "Shillong", "Lat": 25.5788, "Lon": 91.8933, "PopDensity": 2900},
    "MZ": {"City": "Aizawl", "Lat": 23.7271, "Lon": 92.7176, "PopDensity": 2100},
    "CH": {"City": "Chandigarh", "Lat": 30.7333, "Lon": 76.7794, "PopDensity": 9250},
}

def fetch_openmeteo_weather(lat, lon, start_date="2015-01-01", end_date="2020-07-01"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum,surface_pressure_mean",
        "timezone": "Asia/Kolkata"
    }
    headers = {"User-Agent": "Mozilla/5.0 (AirWatch-Project/1.0; contact: student-project)"}
    
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                print(f"Rate limited (429) for ({lat}, {lon}). Waiting {2 * (attempt + 1)}s...")
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()["daily"]
            df = pd.DataFrame({
                "Date": pd.to_datetime(d["time"]),
                "Temp_C": d["temperature_2m_mean"],
                "Humidity_pct": d["relative_humidity_2m_mean"],
                "WindSpeed_ms": d["wind_speed_10m_mean"],
                "Rain_mm": d["precipitation_sum"],
                "Pressure_kPa": [p / 10 if p else np.nan for p in d["surface_pressure_mean"]] # Convert hPa to kPa
            })
            return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed for ({lat}, {lon}): {e}")
            time.sleep(2)
    return pd.DataFrame()

def build_master_dataset():
    print("Loading station_day.csv...")
    cpcb = pd.read_csv("station_day.csv")
    cpcb["Date"] = pd.to_datetime(cpcb["Date"])
    cpcb["StateCode"] = cpcb["StationId"].str[:2]

    # Map station metadata
    cpcb["City"] = cpcb["StateCode"].map(lambda s: CITY_METADATA.get(s, {}).get("City", "Other"))
    cpcb["Lat"] = cpcb["StateCode"].map(lambda s: CITY_METADATA.get(s, {}).get("Lat", np.nan))
    cpcb["Lon"] = cpcb["StateCode"].map(lambda s: CITY_METADATA.get(s, {}).get("Lon", np.nan))
    cpcb["PopDensity_per_km2"] = cpcb["StateCode"].map(lambda s: CITY_METADATA.get(s, {}).get("PopDensity", 5000))

    # Fetch weather for each state code/city cluster
    os.makedirs("data", exist_ok=True)
    weather_cache_path = "data/weather_cache.csv"

    if os.path.exists(weather_cache_path):
        print("Loading cached weather data...")
        weather_df = pd.read_csv(weather_cache_path)
        weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    else:
        print("Fetching daily weather data from Open-Meteo for all station clusters...")
        weather_list = []
        for state_code, meta in CITY_METADATA.items():
            print(f"Fetching weather for {meta['City']} ({state_code})...")
            w = fetch_openmeteo_weather(meta["Lat"], meta["Lon"])
            if not w.empty:
                w["StateCode"] = state_code
                weather_list.append(w)
            time.sleep(0.5)  # Rate limiting courtesy pause

        if weather_list:
            weather_df = pd.concat(weather_list, ignore_index=True)
            weather_df.to_csv(weather_cache_path, index=False)
            print(f"Weather data saved to {weather_cache_path}")
        else:
            print("Failed to fetch weather data.")
            return

    # Merge weather onto cpcb on (StateCode, Date)
    print("Merging CPCB ground readings with weather data...")
    master = cpcb.merge(weather_df, on=["StateCode", "Date"], how="left")

    match_rate = master["Temp_C"].notna().mean() * 100
    print(f"Master Dataset Shape: {master.shape}")
    print(f"Weather Match Rate: {match_rate:.2f}%")

    master.to_csv("station_day_master.csv", index=False)
    print("Saved station_day_master.csv successfully!")

if __name__ == "__main__":
    build_master_dataset()
