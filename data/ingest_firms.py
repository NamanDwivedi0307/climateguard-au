import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

NASA_KEY = os.getenv("NASA_FIRMS_KEY")
AUSTRALIA_BBOX = "113.338953078,-43.6345972634,153.569469029,-10.6681857235"

def fetch_firms_data(day_range=5):
    """
    Fetch recent fire detections over Australia from NASA FIRMS.
    VIIRS S-NPP sensor, max 5 days for NRT data.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{AUSTRALIA_BBOX}/{day_range}"
    
    print(f"Fetching FIRMS data for last {day_range} days over Australia...")
    response = requests.get(url, timeout=30)
    
    if response.status_code != 200:
        print(f"ERROR: Status {response.status_code} — {response.text[:200]}")
        return None
    
    if "latitude" not in response.text[:100]:
        print(f"Unexpected response: {response.text[:200]}")
        return None
    
    df = pd.read_csv(StringIO(response.text))
    
    # Keep only relevant columns
    df = df[["latitude", "longitude", "bright_ti4", "frp", "confidence", "acq_date", "acq_time", "daynight"]]
    df = df.rename(columns={"bright_ti4": "brightness", "frp": "fire_radiative_power"})
    
    print(f"Fetched {len(df)} fire detections")
    print(df.head())
    return df

if __name__ == "__main__":
    df = fetch_firms_data(day_range=5)
    if df is not None and len(df) > 0:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/firms_raw.csv", index=False)
        print(f"\nSaved {len(df)} records to data/raw/firms_raw.csv")
    else:
        print("No fire data returned")
