import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Australian postcodes sample — major regions with known fire risk profiles
POSTCODE_REGIONS = [
    # (postcode, lat, lon, base_risk, region_name)
    ("2000", -33.87, 151.21, 0.1, "Sydney CBD"),
    ("2650", -35.12, 147.37, 0.6, "Wagga Wagga"),
    ("2480", -28.82, 153.29, 0.7, "Lismore"),
    ("2630", -36.37, 148.47, 0.8, "Cooma"),
    ("3000", -37.81, 144.96, 0.1, "Melbourne CBD"),
    ("3820", -38.12, 146.00, 0.6, "Traralgon"),
    ("3444", -37.04, 144.57, 0.7, "Kyneton"),
    ("4000", -27.47, 153.02, 0.2, "Brisbane CBD"),
    ("4570", -26.19, 152.66, 0.6, "Gympie"),
    ("4380", -28.65, 151.94, 0.7, "Stanthorpe"),
    ("5000", -34.93, 138.60, 0.2, "Adelaide CBD"),
    ("5153", -35.06, 138.74, 0.7, "Cherry Gardens"),
    ("6000", -31.95, 115.86, 0.2, "Perth CBD"),
    ("6076", -31.90, 116.10, 0.8, "Mundaring"),
    ("7000", -42.88, 147.33, 0.3, "Hobart"),
    ("7140", -42.68, 146.49, 0.6, "Hamilton TAS"),
    ("0800", -12.46, 130.84, 0.5, "Darwin"),
    ("2580", -34.75, 149.72, 0.7, "Goulburn"),
    ("2630", -36.45, 148.26, 0.9, "Jindabyne"),
    ("3737", -36.73, 146.93, 0.8, "Bright VIC"),
]

def compute_ffdi(temp, humidity, wind_speed, drought_factor):
    """
    Forest Fire Danger Index (FFDI) — Mark 5 equation.
    Standard Australian fire danger rating formula.
    FFDI = 2 * exp(-0.45 + 0.987*ln(drought) - 0.0345*humidity + 0.0338*temp + 0.0234*wind)
    """
    drought_factor = max(drought_factor, 0.1)
    ffdi = 2 * np.exp(
        -0.45
        + 0.987 * np.log(drought_factor)
        - 0.0345 * humidity
        + 0.0338 * temp
        + 0.0234 * wind_speed
    )
    return np.clip(ffdi, 0, 200)

def generate_samples(n_samples=5000):
    records = []

    for _ in range(n_samples):
        # Pick a random postcode region
        pc = POSTCODE_REGIONS[np.random.randint(len(POSTCODE_REGIONS))]
        postcode, lat, lon, base_risk, region = pc

        # Simulate seasonal variation (Australian summer = Dec-Feb = high risk)
        month = np.random.randint(1, 13)
        season_factor = 1.5 if month in [12, 1, 2] else (1.2 if month in [3, 11] else 0.7)

        # Weather features with realistic Australian ranges
        temp = np.random.normal(25 + base_risk * 15, 5) * season_factor * 0.6
        temp = np.clip(temp, 5, 48)

        humidity = np.random.normal(60 - base_risk * 30, 10) / season_factor
        humidity = np.clip(humidity, 5, 100)

        wind_speed = np.random.normal(20 + base_risk * 20, 8)
        wind_speed = np.clip(wind_speed, 0, 80)

        drought_factor = np.random.normal(5 + base_risk * 5, 2) * season_factor
        drought_factor = np.clip(drought_factor, 1, 10)

        days_since_rain = np.random.exponential(10 + base_risk * 20)
        days_since_rain = np.clip(days_since_rain, 0, 120)

        rainfall_mm = np.random.exponential(5) if days_since_rain < 7 else 0

        # Vegetation density (0-1)
        vegetation = np.random.beta(2, 2) * (0.5 + base_risk * 0.5)

        # Elevation (metres)
        elevation = np.random.normal(300 + base_risk * 400, 100)
        elevation = np.clip(elevation, 0, 2000)

        # Compute FFDI
        ffdi = compute_ffdi(temp, humidity, wind_speed, drought_factor)

        # Fire radiative power from FIRMS (0 if no fire)
        has_fire_nearby = np.random.random() < base_risk * 0.3
        frp = np.random.exponential(20) if has_fire_nearby else 0

        # Risk score 0-100 (target variable)
        risk_score = (
            0.35 * min(ffdi / 100, 1.0)
            + 0.20 * (days_since_rain / 120)
            + 0.15 * base_risk
            + 0.10 * (temp / 48)
            + 0.10 * (1 - humidity / 100)
            + 0.05 * (wind_speed / 80)
            + 0.05 * vegetation
        ) * 100 * season_factor

        risk_score = np.clip(risk_score, 0, 100)

        # Risk category
        if risk_score < 25:
            risk_label = "Low"
        elif risk_score < 50:
            risk_label = "Moderate"
        elif risk_score < 75:
            risk_label = "High"
        else:
            risk_label = "Extreme"

        records.append({
            "postcode": postcode,
            "latitude": lat + np.random.normal(0, 0.05),
            "longitude": lon + np.random.normal(0, 0.05),
            "region": region,
            "month": month,
            "temperature_c": round(temp, 2),
            "humidity_pct": round(humidity, 2),
            "wind_speed_kmh": round(wind_speed, 2),
            "drought_factor": round(drought_factor, 2),
            "days_since_rain": round(days_since_rain, 1),
            "rainfall_mm": round(rainfall_mm, 2),
            "vegetation_density": round(vegetation, 3),
            "elevation_m": round(elevation, 1),
            "ffdi": round(ffdi, 2),
            "frp_nearby": round(frp, 2),
            "risk_score": round(risk_score, 2),
            "risk_label": risk_label,
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    print("Generating synthetic training data...")
    df = generate_samples(n_samples=5000)

    os.makedirs("data/synthetic", exist_ok=True)
    df.to_csv("data/synthetic/training_data.csv", index=False)

    print(f"Generated {len(df)} samples")
    print(f"\nRisk label distribution:\n{df['risk_label'].value_counts()}")
    print(f"\nFeature stats:\n{df[['temperature_c','humidity_pct','wind_speed_kmh','ffdi','risk_score']].describe().round(2)}")
