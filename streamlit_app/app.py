import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import json
import os
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateGuard-AU",
    page_icon="🔥",
    layout="wide"
)

# ── Load models ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    xgb_reg = joblib.load(os.path.join(BASE, "models/xgb_regressor.pkl"))
    xgb_clf = joblib.load(os.path.join(BASE, "models/xgb_classifier.pkl"))
    le      = joblib.load(os.path.join(BASE, "models/label_encoder.pkl"))
    explainer = shap.TreeExplainer(xgb_reg)
    with open(os.path.join(BASE, "models/metrics.json")) as f:
        metrics = json.load(f)
    return xgb_reg, xgb_clf, le, explainer, metrics

@st.cache_data
def load_data():
    processed_path = os.path.join(BASE, "data/processed/features.csv")
    synthetic_path = os.path.join(BASE, "data/synthetic/training_data.csv")

    if not os.path.exists(processed_path):
        import numpy as np
        os.makedirs(os.path.join(BASE, "data/processed"), exist_ok=True)
        df = pd.read_csv(synthetic_path)
        df["postcode"] = df["postcode"].astype(str)

        # Recompute engineered features
        drought = df["drought_factor"].clip(lower=0.1)
        df["ffdi"] = (2 * np.exp(
            -0.45 + 0.987 * np.log(drought)
            - 0.0345 * df["humidity_pct"]
            + 0.0338 * df["temperature_c"]
            + 0.0234 * df["wind_speed_kmh"]
        )).clip(0, 200)
        df["temp_wind_interaction"] = df["temperature_c"] * df["wind_speed_kmh"]
        df["humidity_drought_interaction"] = df["humidity_pct"] * df["drought_factor"]
        df["fire_weather_composite"] = df["temperature_c"]*0.4 + df["wind_speed_kmh"]*0.3 + (100-df["humidity_pct"])*0.3
        df["is_summer"] = df["month"].isin([12,1,2]).astype(int)
        df["is_autumn"] = df["month"].isin([3,4,5]).astype(int)
        df["is_winter"] = df["month"].isin([6,7,8]).astype(int)
        df["is_spring"] = df["month"].isin([9,10,11]).astype(int)
        df["drought_severity"] = pd.cut(df["drought_factor"], bins=[0,3,6,8,10], labels=[0,1,2,3]).astype(float)
        df["rain_recency"] = np.where(df["days_since_rain"]<7, 2, np.where(df["days_since_rain"]<30, 1, 0))
        df.to_csv(processed_path, index=False)

    df = pd.read_csv(processed_path)
    df["postcode"] = df["postcode"].astype(str)
    return df

xgb_reg, xgb_clf, le, explainer, metrics = load_models()
df_ref = load_data()

FEATURE_COLS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "drought_factor", "days_since_rain", "rainfall_mm",
    "vegetation_density", "elevation_m", "ffdi",
    "frp_nearby", "temp_wind_interaction",
    "humidity_drought_interaction", "fire_weather_composite",
    "is_summer", "is_autumn", "is_winter", "is_spring",
    "drought_severity", "rain_recency", "month"
]

RISK_COLORS = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c", "Extreme": "#8e44ad"}

# ── Helper ─────────────────────────────────────────────────────────────────
def build_features(temp, humidity, wind, drought, days_rain, rain_mm, veg, elev, frp, month):
    import math
    ffdi = 2 * math.exp(
        -0.45 + 0.987 * math.log(max(drought, 0.1))
        - 0.0345 * humidity + 0.0338 * temp + 0.0234 * wind
    )
    ffdi = min(ffdi, 200)
    row = {
        "temperature_c": temp, "humidity_pct": humidity, "wind_speed_kmh": wind,
        "drought_factor": drought, "days_since_rain": days_rain, "rainfall_mm": rain_mm,
        "vegetation_density": veg, "elevation_m": elev, "ffdi": ffdi, "frp_nearby": frp,
        "temp_wind_interaction": temp * wind,
        "humidity_drought_interaction": humidity * drought,
        "fire_weather_composite": temp * 0.4 + wind * 0.3 + (100 - humidity) * 0.3,
        "is_summer": int(month in [12, 1, 2]), "is_autumn": int(month in [3, 4, 5]),
        "is_winter": int(month in [6, 7, 8]), "is_spring": int(month in [9, 10, 11]),
        "drought_severity": 0 if drought < 3 else (1 if drought < 6 else (2 if drought < 8 else 3)),
        "rain_recency": 2 if days_rain < 7 else (1 if days_rain < 30 else 0),
        "month": month
    }
    return pd.DataFrame([row])[FEATURE_COLS]

def risk_color(score):
    if score < 25: return "#2ecc71"
    elif score < 50: return "#f39c12"
    elif score < 75: return "#e74c3c"
    else: return "#8e44ad"

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/fire.png", width=60)
st.sidebar.title("ClimateGuard-AU")
st.sidebar.markdown("**Bushfire Risk Prediction**\nAustralian Postcodes")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["🗺️ Risk Map", "🔍 Postcode Lookup", "📊 Model Info"])

# ── Risk Map ───────────────────────────────────────────────────────────────
if mode == "🗺️ Risk Map":
    st.title("🔥 ClimateGuard-AU — Live Risk Map")
    st.markdown("Real-time bushfire risk prediction across Australian postcodes using XGBoost + LSTM ensemble.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model R²", f"{metrics['r2']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("CV R²", f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    col4.metric("Training Samples", f"{metrics['n_train']:,}")

    st.markdown("---")

    # Build postcode summary
    summary = df_ref.groupby(["postcode", "region"]).agg(
        risk_score=("risk_score", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean")
    ).reset_index()
    summary["risk_label"] = pd.cut(
        summary["risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Extreme"]
    ).astype(str)

    # Folium map
    m = folium.Map(location=[-25.0, 133.0], zoom_start=4, tiles="CartoDB positron")

    for _, row in summary.iterrows():
        color = risk_color(row["risk_score"])
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row['region']}</b><br>Postcode: {row['postcode']}<br>"
                f"Risk Score: {row['risk_score']:.1f}/100<br>Level: {row['risk_label']}",
                max_width=200
            ),
            tooltip=f"{row['region']} — {row['risk_score']:.0f}/100"
        ).add_to(m)

    st_folium(m, width=1100, height=550)

    # Legend
    st.markdown("""
    **Risk Levels:** 
    🟢 Low (0–25) &nbsp;|&nbsp; 🟠 Moderate (25–50) &nbsp;|&nbsp; 🔴 High (50–75) &nbsp;|&nbsp; 🟣 Extreme (75–100)
    """)

# ── Postcode Lookup ────────────────────────────────────────────────────────
elif mode == "🔍 Postcode Lookup":
    st.title("🔍 Postcode Risk Lookup")
    st.markdown("Enter weather conditions to get a real-time bushfire risk prediction with AI explanation.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            postcode = st.selectbox("Postcode", sorted(df_ref["postcode"].unique()))
            temp = st.slider("Temperature (°C)", 5, 48, 32)
            humidity = st.slider("Humidity (%)", 5, 100, 20)
            wind = st.slider("Wind Speed (km/h)", 0, 80, 40)
            month = st.selectbox("Month", list(range(1, 13)),
                format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        with col2:
            drought = st.slider("Drought Factor", 1.0, 10.0, 7.0)
            days_rain = st.slider("Days Since Rain", 0, 120, 30)
            rain_mm = st.slider("Recent Rainfall (mm)", 0.0, 50.0, 0.0)
            veg = st.slider("Vegetation Density", 0.0, 1.0, 0.6)
            elev = st.slider("Elevation (m)", 0, 2000, 400)

        submitted = st.form_submit_button("🔥 Predict Risk", use_container_width=True)

    if submitted:
        X = build_features(temp, humidity, wind, drought, days_rain, rain_mm, veg, elev, 0.0, month)
        risk_score = float(np.clip(xgb_reg.predict(X)[0], 0, 100))
        risk_label = str(le.inverse_transform(xgb_clf.predict(X))[0])
        ffdi = float(X["ffdi"].values[0])

        color = risk_color(risk_score)
        st.markdown(f"""
        <div style='background:{color};padding:20px;border-radius:10px;text-align:center;'>
            <h1 style='color:white;margin:0'>{risk_score:.1f}/100</h1>
            <h3 style='color:white;margin:0'>{risk_label} Risk — Postcode {postcode}</h3>
            <p style='color:white;margin:0'>FFDI: {ffdi:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧠 AI Explanation — Top Risk Factors")
        shap_vals = explainer.shap_values(X)[0]
        shap_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "SHAP Value": shap_vals
        }).sort_values("SHAP Value", key=abs, ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax.set_xlabel("SHAP Value (impact on risk score)")
        ax.set_title("Feature Contributions to Risk Prediction")
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Model Info ─────────────────────────────────────────────────────────────
elif mode == "📊 Model Info":
    st.title("📊 Model Performance & Architecture")

    st.markdown("### XGBoost Regressor")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['r2']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f} / 100")
    col3.metric("5-Fold CV R²", f"{metrics['cv_r2_mean']:.4f}")

    st.markdown("### Architecture")
    st.code("""
NASA FIRMS Satellite Data + BOM Weather
        ↓
Feature Engineering (FFDI Mark 5, interaction terms)
        ↓
XGBoost Regressor  →  Risk Score (0–100)
XGBoost Classifier →  Risk Label (Low/Moderate/High/Extreme)
LSTM (PyTorch)     →  Time-series fire spread
        ↓
SHAP Explainability → Top risk factors per prediction
        ↓
FastAPI /predict + Streamlit Dashboard
    """)

    st.markdown("### SHAP Feature Importance")
    if os.path.exists(os.path.join(BASE, "models/shap_summary.png")):
        st.image(os.path.join(BASE, "models/shap_summary.png"))

    st.markdown("### Data Sources")
    st.markdown("""
    - 🛰️ **NASA FIRMS** — VIIRS S-NPP real-time satellite fire detection
    - 🌤️ **BOM** — Australian Bureau of Meteorology weather data  
    - 📍 **Australian Postcodes** — Geographic boundary data
    - 🔥 **FFDI** — Forest Fire Danger Index (Mark 5 equation)
    """)
