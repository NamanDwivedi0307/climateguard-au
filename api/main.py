import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import shap

# ── Load models ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xgb_reg = joblib.load(os.path.join(BASE, "models/xgb_regressor.pkl"))
xgb_clf = joblib.load(os.path.join(BASE, "models/xgb_classifier.pkl"))
le      = joblib.load(os.path.join(BASE, "models/label_encoder.pkl"))

with open(os.path.join(BASE, "models/metrics.json")) as f:
    metrics = json.load(f)

explainer = shap.TreeExplainer(xgb_reg)

df_ref = pd.read_csv(os.path.join(BASE, "data/processed/features.csv"))
df_ref["postcode"] = df_ref["postcode"].astype(str)

FEATURE_COLS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "drought_factor", "days_since_rain", "rainfall_mm",
    "vegetation_density", "elevation_m", "ffdi",
    "frp_nearby", "temp_wind_interaction",
    "humidity_drought_interaction", "fire_weather_composite",
    "is_summer", "is_autumn", "is_winter", "is_spring",
    "drought_severity", "rain_recency", "month"
]

app = FastAPI(
    title="ClimateGuard-AU API",
    description="Real-time bushfire risk prediction for Australian postcodes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictRequest(BaseModel):
    postcode: str
    temperature_c: float = Field(..., ge=-10, le=55)
    humidity_pct: float = Field(..., ge=0, le=100)
    wind_speed_kmh: float = Field(..., ge=0, le=150)
    drought_factor: float = Field(5.0, ge=1, le=10)
    days_since_rain: float = Field(10.0, ge=0, le=120)
    rainfall_mm: float = Field(0.0, ge=0)
    vegetation_density: float = Field(0.5, ge=0, le=1)
    elevation_m: float = Field(200.0, ge=0, le=2500)
    frp_nearby: float = Field(0.0, ge=0)
    month: int = Field(..., ge=1, le=12)

def build_features(req: PredictRequest) -> pd.DataFrame:
    import math
    ffdi = 2 * math.exp(
        -0.45
        + 0.987 * math.log(max(req.drought_factor, 0.1))
        - 0.0345 * req.humidity_pct
        + 0.0338 * req.temperature_c
        + 0.0234 * req.wind_speed_kmh
    )
    ffdi = min(ffdi, 200)
    month = req.month
    row = {
        "temperature_c": req.temperature_c,
        "humidity_pct": req.humidity_pct,
        "wind_speed_kmh": req.wind_speed_kmh,
        "drought_factor": req.drought_factor,
        "days_since_rain": req.days_since_rain,
        "rainfall_mm": req.rainfall_mm,
        "vegetation_density": req.vegetation_density,
        "elevation_m": req.elevation_m,
        "ffdi": ffdi,
        "frp_nearby": req.frp_nearby,
        "temp_wind_interaction": req.temperature_c * req.wind_speed_kmh,
        "humidity_drought_interaction": req.humidity_pct * req.drought_factor,
        "fire_weather_composite": req.temperature_c * 0.4 + req.wind_speed_kmh * 0.3 + (100 - req.humidity_pct) * 0.3,
        "is_summer": int(month in [12, 1, 2]),
        "is_autumn": int(month in [3, 4, 5]),
        "is_winter": int(month in [6, 7, 8]),
        "is_spring": int(month in [9, 10, 11]),
        "drought_severity": 0 if req.drought_factor < 3 else (1 if req.drought_factor < 6 else (2 if req.drought_factor < 8 else 3)),
        "rain_recency": 2 if req.days_since_rain < 7 else (1 if req.days_since_rain < 30 else 0),
        "month": month
    }
    return pd.DataFrame([row])[FEATURE_COLS]

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost + LSTM ensemble", "metrics": metrics}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = build_features(req)
        risk_score = float(np.clip(xgb_reg.predict(X)[0], 0, 100))
        risk_label = str(le.inverse_transform(xgb_clf.predict(X))[0])
        shap_vals = explainer.shap_values(X)[0]
        shap_dict = {feat: round(float(val), 4) for feat, val in zip(FEATURE_COLS, shap_vals)}
        top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        pc_row = df_ref[df_ref["postcode"] == req.postcode]
        lat = float(pc_row["latitude"].mean()) if len(pc_row) else -25.0
        lon = float(pc_row["longitude"].mean()) if len(pc_row) else 133.0
        return {
            "postcode": req.postcode,
            "risk_score": round(risk_score, 2),
            "risk_label": risk_label,
            "latitude": lat,
            "longitude": lon,
            "ffdi": round(float(X["ffdi"].values[0]), 2),
            "top_risk_factors": [{"feature": k, "shap_value": v} for k, v in top_factors]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alert")
def alert(threshold: float = 60.0):
    results = []
    for postcode in df_ref["postcode"].unique():
        sub = df_ref[df_ref["postcode"] == postcode]
        avg_risk = float(sub["risk_score"].mean())
        if avg_risk >= threshold:
            results.append({
                "postcode": str(postcode),
                "region": str(sub["region"].iloc[0]),
                "avg_risk_score": round(avg_risk, 2),
                "risk_label": str(sub["risk_label"].mode().iloc[0]),
                "latitude": float(sub["latitude"].mean()),
                "longitude": float(sub["longitude"].mean())
            })
    results.sort(key=lambda x: x["avg_risk_score"], reverse=True)
    return {"threshold": threshold, "alerts": results, "count": len(results)}

@app.get("/map-data")
def map_data():
    features = []
    for postcode in df_ref["postcode"].unique():
        sub = df_ref[df_ref["postcode"] == postcode]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(sub["longitude"].mean()), float(sub["latitude"].mean())]
            },
            "properties": {
                "postcode": str(postcode),
                "region": str(sub["region"].iloc[0]),
                "risk_score": round(float(sub["risk_score"].mean()), 2),
                "risk_label": str(sub["risk_label"].mode().iloc[0])
            }
        })
    return {"type": "FeatureCollection", "features": features}
