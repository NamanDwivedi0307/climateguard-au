import pandas as pd
import numpy as np
import os

def compute_ffdi(temp, humidity, wind_speed, drought_factor):
    drought_factor = np.maximum(drought_factor, 0.1)
    ffdi = 2 * np.exp(
        -0.45
        + 0.987 * np.log(drought_factor)
        - 0.0345 * humidity
        + 0.0338 * temp
        + 0.0234 * wind_speed
    )
    return np.clip(ffdi, 0, 200)

def engineer_features(df):
    # FFDI recomputed to ensure consistency
    df["ffdi"] = compute_ffdi(
        df["temperature_c"], df["humidity_pct"],
        df["wind_speed_kmh"], df["drought_factor"]
    )

    # Fire weather index categories
    df["ffdi_category"] = pd.cut(
        df["ffdi"],
        bins=[0, 12, 25, 50, 75, 200],
        labels=["Low", "High", "Very High", "Severe", "Extreme"]
    )

    # Interaction features
    df["temp_wind_interaction"] = df["temperature_c"] * df["wind_speed_kmh"]
    df["humidity_drought_interaction"] = df["humidity_pct"] * df["drought_factor"]
    df["fire_weather_composite"] = (
        df["temperature_c"] * 0.4 +
        df["wind_speed_kmh"] * 0.3 +
        (100 - df["humidity_pct"]) * 0.3
    )

    # Season encoding (Australian seasons)
    df["is_summer"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_autumn"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_winter"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_spring"] = df["month"].isin([9, 10, 11]).astype(int)

    # Drought severity bins
    df["drought_severity"] = pd.cut(
        df["drought_factor"],
        bins=[0, 3, 6, 8, 10],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Rain recency feature
    df["rain_recency"] = np.where(df["days_since_rain"] < 7, 2,
                         np.where(df["days_since_rain"] < 30, 1, 0))

    return df

def main():
    print("Loading synthetic training data...")
    df = pd.read_csv("data/synthetic/training_data.csv")
    print(f"Loaded {len(df)} records")

    print("Engineering features...")
    df = engineer_features(df)

    # Select final feature columns for model training
    feature_cols = [
        "temperature_c", "humidity_pct", "wind_speed_kmh",
        "drought_factor", "days_since_rain", "rainfall_mm",
        "vegetation_density", "elevation_m", "ffdi",
        "frp_nearby", "temp_wind_interaction",
        "humidity_drought_interaction", "fire_weather_composite",
        "is_summer", "is_autumn", "is_winter", "is_spring",
        "drought_severity", "rain_recency", "month"
    ]

    target_col = "risk_score"

    df_model = df[feature_cols + [target_col, "risk_label", "postcode", "latitude", "longitude", "region"]]

    os.makedirs("data/processed", exist_ok=True)
    df_model.to_csv("data/processed/features.csv", index=False)

    print(f"\nFinal feature set: {len(feature_cols)} features")
    print(f"Saved to data/processed/features.csv")
    print(f"\nSample:\n{df_model.head(3)}")

if __name__ == "__main__":
    main()
