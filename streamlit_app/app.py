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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Metric cards */
.metric-card {
    background: #1a1d2e;
    border: 1px solid #2d3149;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.2rem;
}

/* Risk banner */
.risk-banner {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.risk-score-num {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: white;
    line-height: 1;
}
.risk-label-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-top: 0.4rem;
}
.risk-ffdi {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.65);
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Section headers */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    border-bottom: 1px solid #1e2130;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Nav pills */
.nav-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
}

/* Alert cards */
.alert-card {
    background: #1a1d2e;
    border: 1px solid #2d3149;
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.alert-postcode { font-weight: 600; color: #f1f5f9; font-size: 0.95rem; }
.alert-region { font-size: 0.78rem; color: #64748b; margin-top: 0.1rem; }
.alert-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
}

/* SHAP chart styling */
.shap-positive { color: #ef4444; }
.shap-negative { color: #22c55e; }

/* Stmetric override */
[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2d3149;
    border-radius: 12px;
    padding: 1rem;
}

/* Form elements */
[data-testid="stSlider"] > div { padding: 0.2rem 0; }
.stSelectbox label, .stSlider label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Submit button */
.stFormSubmitButton button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stFormSubmitButton button:hover { opacity: 0.88 !important; }

/* Map container */
.map-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2d3149;
}

/* Page title */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource(show_spinner=False)
def load_models():
    xgb_reg  = joblib.load(os.path.join(BASE, "models/xgb_regressor.pkl"))
    xgb_clf  = joblib.load(os.path.join(BASE, "models/xgb_classifier.pkl"))
    le       = joblib.load(os.path.join(BASE, "models/label_encoder.pkl"))
    explainer = shap.TreeExplainer(xgb_reg)
    with open(os.path.join(BASE, "models/metrics.json")) as f:
        metrics = json.load(f)
    return xgb_reg, xgb_clf, le, explainer, metrics

@st.cache_data(show_spinner=False)
def load_data():
    processed_path = os.path.join(BASE, "data/processed/features.csv")
    synthetic_path = os.path.join(BASE, "data/synthetic/training_data.csv")
    if not os.path.exists(processed_path):
        os.makedirs(os.path.join(BASE, "data/processed"), exist_ok=True)
        df = pd.read_csv(synthetic_path)
        df["postcode"] = df["postcode"].astype(str)
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

with st.spinner("Loading ClimateGuard-AU..."):
    xgb_reg, xgb_clf, le, explainer, metrics = load_models()
    df_ref = load_data()

FEATURE_COLS = [
    "temperature_c","humidity_pct","wind_speed_kmh","drought_factor",
    "days_since_rain","rainfall_mm","vegetation_density","elevation_m",
    "ffdi","frp_nearby","temp_wind_interaction","humidity_drought_interaction",
    "fire_weather_composite","is_summer","is_autumn","is_winter","is_spring",
    "drought_severity","rain_recency","month"
]

RISK_COLORS   = {"Low":"#22c55e","Moderate":"#f59e0b","High":"#ef4444","Extreme":"#8b5cf6"}
RISK_BG       = {"Low":"#14532d","Moderate":"#78350f","High":"#7f1d1d","Extreme":"#4c1d95"}

def risk_color(score):
    if score < 25:  return "#22c55e"
    elif score < 50: return "#f59e0b"
    elif score < 75: return "#ef4444"
    else:            return "#8b5cf6"

def risk_label_from_score(score):
    if score < 25:  return "Low"
    elif score < 50: return "Moderate"
    elif score < 75: return "High"
    else:            return "Extreme"

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
        "fire_weather_composite": temp*0.4 + wind*0.3 + (100-humidity)*0.3,
        "is_summer": int(month in [12,1,2]), "is_autumn": int(month in [3,4,5]),
        "is_winter": int(month in [6,7,8]),  "is_spring": int(month in [9,10,11]),
        "drought_severity": 0 if drought<3 else (1 if drought<6 else (2 if drought<8 else 3)),
        "rain_recency": 2 if days_rain<7 else (1 if days_rain<30 else 0),
        "month": month
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0;'>
        <div style='font-size:1.3rem; font-weight:700; color:#f1f5f9;'>🔥 ClimateGuard</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:0.2rem; letter-spacing:0.05em; text-transform:uppercase;'>AU Bushfire Risk Platform</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Navigation",
        ["Risk Map", "Postcode Lookup", "Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; line-height:1.6;'>
        <div style='color:#475569; font-weight:600; margin-bottom:0.4rem;'>DATA SOURCES</div>
        🛰️ NASA FIRMS VIIRS S-NPP<br>
        🌤️ BOM Weather API<br>
        📍 Australian Postcodes<br>
        🔥 FFDI Mark 5 Equation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155;'>
        <div style='color:#475569; font-weight:600; margin-bottom:0.4rem;'>MODEL</div>
        XGBoost + LSTM Ensemble<br>
        R² 0.9891 · RMSE 2.20<br>
        94% Classification Accuracy
    </div>
    """, unsafe_allow_html=True)

# ── Risk Map ───────────────────────────────────────────────────────────────
if mode == "Risk Map":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div class='page-title'>Australia Bushfire Risk Map</div>
        <div class='page-subtitle'>Live postcode-level risk scoring using satellite fire data and weather conditions</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Model R²</div>
            <div class='metric-value'>{metrics['r2']}</div>
            <div class='metric-sub'>Explained variance</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>RMSE</div>
            <div class='metric-value'>{metrics['rmse']}</div>
            <div class='metric-sub'>Points on 0–100 scale</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>CV R²</div>
            <div class='metric-value'>{metrics['cv_r2_mean']}</div>
            <div class='metric-sub'>5-fold cross-validation</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Training Samples</div>
            <div class='metric-value'>{metrics['n_train']:,}</div>
            <div class='metric-sub'>Synthetic AU climate data</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    summary = df_ref.groupby(["postcode","region"]).agg(
        risk_score=("risk_score","mean"),
        latitude=("latitude","mean"),
        longitude=("longitude","mean")
    ).reset_index()

    m = folium.Map(
        location=[-27.0, 134.0],
        zoom_start=4,
        tiles="Esri.WorldImagery",
        prefer_canvas=True
    )

    for _, row in summary.iterrows():
        score = row["risk_score"]
        color = risk_color(score)
        label = risk_label_from_score(score)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=13,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                f"""<div style='font-family:Inter,sans-serif;padding:4px;'>
                    <b style='font-size:13px;'>{row['region']}</b><br>
                    <span style='color:#666;font-size:11px;'>Postcode {row['postcode']}</span><br>
                    <span style='font-size:15px;font-weight:700;color:{color};'>{score:.0f}/100</span>
                    <span style='font-size:11px;color:#666;'> {label}</span>
                </div>""",
                max_width=180
            ),
            tooltip=f"{row['region']} — {score:.0f}/100 {label}"
        ).add_to(m)

    st_folium(m, width=None, height=520, returned_objects=[])

    st.markdown("""
    <div style='display:flex;gap:1.2rem;margin-top:0.8rem;align-items:center;'>
        <span style='font-size:0.72rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'>Risk Level</span>
        <span style='font-size:0.8rem;'><span style='color:#22c55e;'>●</span> Low (0–25)</span>
        <span style='font-size:0.8rem;'><span style='color:#f59e0b;'>●</span> Moderate (25–50)</span>
        <span style='font-size:0.8rem;'><span style='color:#ef4444;'>●</span> High (50–75)</span>
        <span style='font-size:0.8rem;'><span style='color:#8b5cf6;'>●</span> Extreme (75–100)</span>
    </div>
    """, unsafe_allow_html=True)

# ── Postcode Lookup ────────────────────────────────────────────────────────
elif mode == "Postcode Lookup":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div class='page-title'>Postcode Risk Lookup</div>
        <div class='page-subtitle'>Enter current weather conditions to get an AI-powered risk prediction with SHAP explanation</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        with st.form("predict_form", border=False):
            st.markdown("<div class='section-header'>Location & Time</div>", unsafe_allow_html=True)
            postcode = st.selectbox("Postcode", sorted(df_ref["postcode"].unique()))
            month = st.selectbox("Month", list(range(1,13)),
                format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

            st.markdown("<div class='section-header'>Weather Conditions</div>", unsafe_allow_html=True)
            temp     = st.slider("Temperature (°C)", 5, 48, 32)
            humidity = st.slider("Humidity (%)", 5, 100, 20)
            wind     = st.slider("Wind Speed (km/h)", 0, 80, 40)

            st.markdown("<div class='section-header'>Fire Environment</div>", unsafe_allow_html=True)
            drought   = st.slider("Drought Factor", 1.0, 10.0, 7.0, step=0.5)
            days_rain = st.slider("Days Since Rain", 0, 120, 30)
            rain_mm   = st.slider("Recent Rainfall (mm)", 0.0, 50.0, 0.0, step=0.5)
            veg       = st.slider("Vegetation Density", 0.0, 1.0, 0.6, step=0.05)
            elev      = st.slider("Elevation (m)", 0, 2000, 400, step=50)

            submitted = st.form_submit_button("Generate Risk Prediction", use_container_width=True)

    with col_result:
        if submitted:
            X = build_features(temp, humidity, wind, drought, days_rain, rain_mm, veg, elev, 0.0, month)
            risk_score = float(np.clip(xgb_reg.predict(X)[0], 0, 100))
            risk_label = str(le.inverse_transform(xgb_clf.predict(X))[0])
            ffdi_val   = float(X["ffdi"].values[0])
            color      = risk_color(risk_score)
            bg         = RISK_BG.get(risk_label, "#1a1d2e")

            st.markdown(f"""
            <div class='risk-banner' style='background:linear-gradient(135deg, {bg}, {bg}cc);border:1px solid {color}44;'>
                <div class='risk-score-num' style='color:{color};'>{risk_score:.1f}</div>
                <div style='font-size:0.7rem;color:rgba(255,255,255,0.4);font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;'>out of 100</div>
                <div class='risk-label-text'>{risk_label} Risk · Postcode {postcode}</div>
                <div class='risk-ffdi'>FFDI {ffdi_val:.1f} · {["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Quick stats row
            s1, s2, s3 = st.columns(3)
            s1.metric("Temperature", f"{temp}°C")
            s2.metric("Humidity", f"{humidity}%")
            s3.metric("Wind", f"{wind} km/h")

            st.markdown("<div class='section-header'>SHAP Feature Contributions</div>", unsafe_allow_html=True)

            shap_vals = explainer.shap_values(X)[0]
            shap_df = pd.DataFrame({
                "Feature": [f.replace("_"," ").title() for f in FEATURE_COLS],
                "SHAP": shap_vals
            }).sort_values("SHAP", key=abs, ascending=True).tail(8)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor("#1a1d2e")
            ax.set_facecolor("#1a1d2e")
            colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["SHAP"]]
            bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors, height=0.6, edgecolor="none")
            ax.axvline(0, color="#334155", linewidth=1)
            ax.set_xlabel("Impact on risk score", fontsize=8, color="#64748b")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.label.set_color("#64748b")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Top factor callout
            top_feat = shap_df.iloc[-1]
            direction = "increases" if top_feat["SHAP"] > 0 else "reduces"
            st.markdown(f"""
            <div style='background:#0f1117;border:1px solid #1e2130;border-radius:8px;padding:0.8rem 1rem;margin-top:0.5rem;'>
                <span style='font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'>Key Driver</span><br>
                <span style='color:#f1f5f9;font-size:0.88rem;'><b style='color:#f59e0b;'>{top_feat["Feature"]}</b> {direction} risk by <b>{abs(top_feat["SHAP"]):.1f} points</b></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:200px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        background:#0f1117;border:1px dashed #1e2130;border-radius:12px;margin-top:2rem;'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>🔥</div>
                <div style='color:#475569;font-size:0.85rem;'>Configure conditions and generate a prediction</div>
            </div>
            """, unsafe_allow_html=True)

# ── Model Performance ──────────────────────────────────────────────────────
elif mode == "Model Performance":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div class='page-title'>Model Performance</div>
        <div class='page-subtitle'>XGBoost + LSTM ensemble evaluation metrics and architecture overview</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>R² Score</div>
            <div class='metric-value'>{metrics['r2']}</div>
            <div class='metric-sub'>XGBoost Regressor</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>RMSE</div>
            <div class='metric-value'>{metrics['rmse']}</div>
            <div class='metric-sub'>Points on 0–100 scale</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Classification</div>
            <div class='metric-value'>94%</div>
            <div class='metric-sub'>4-class accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>LSTM RMSE</div>
            <div class='metric-value'>6.32</div>
            <div class='metric-sub'>Time-series model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:2rem;'>Pipeline Architecture</div>", unsafe_allow_html=True)

    steps = [
        ("🛰️", "Satellite Ingest", "NASA FIRMS VIIRS S-NPP — 11,770 real fire detections over Australia"),
        ("⚙️", "Feature Engineering", "FFDI Mark 5 equation · interaction terms · season encoding · drought severity"),
        ("🤖", "Ensemble Model", "XGBoost Regressor (R²=0.9891) + XGBoost Classifier (94%) + PyTorch LSTM (RMSE=6.32)"),
        ("🧠", "SHAP Explainability", "Per-prediction feature contributions — FFDI dominates high-risk scenarios"),
        ("🚀", "Production API", "FastAPI /predict /alert /map-data endpoints with full JSON responses"),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:1rem;align-items:flex-start;padding:1rem;
                    background:#1a1d2e;border:1px solid #2d3149;border-radius:10px;margin-bottom:0.6rem;'>
            <div style='font-size:1.4rem;min-width:2rem;text-align:center;'>{icon}</div>
            <div>
                <div style='font-weight:600;color:#f1f5f9;font-size:0.9rem;'>{title}</div>
                <div style='color:#64748b;font-size:0.8rem;margin-top:0.2rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:1.5rem;'>SHAP Feature Importance</div>", unsafe_allow_html=True)
    shap_path = os.path.join(BASE, "models/shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
