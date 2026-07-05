import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features.csv")
print(f"Loaded {len(df)} samples with {df.shape[1]} columns")

FEATURE_COLS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "drought_factor", "days_since_rain", "rainfall_mm",
    "vegetation_density", "elevation_m", "ffdi",
    "frp_nearby", "temp_wind_interaction",
    "humidity_drought_interaction", "fire_weather_composite",
    "is_summer", "is_autumn", "is_winter", "is_spring",
    "drought_severity", "rain_recency", "month"
]

X = df[FEATURE_COLS]
y_regression = df["risk_score"]          # continuous 0-100
y_classification = df["risk_label"]       # Low/Moderate/High/Extreme

# ── Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
_, _, yc_train, yc_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── XGBoost Regressor (primary model) ─────────────────────────────────────
print("\nTraining XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(xgb_reg, X, y_regression, cv=5, scoring="r2")
print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── XGBoost Classifier (for risk category) ────────────────────────────────
print("\nTraining XGBoost Classifier...")
le = LabelEncoder()
yc_train_enc = le.fit_transform(yc_train)
yc_test_enc = le.transform(yc_test)

xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train, yc_train_enc)
yc_pred = xgb_clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(yc_test_enc, yc_pred, target_names=le.classes_))

# ── SHAP Explainability ────────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_reg)
shap_values = explainer.shap_values(X_test[:200])

# SHAP summary plot
shap.summary_plot(shap_values, X_test[:200], show=False)
os.makedirs("models", exist_ok=True)
plt.savefig("models/shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()
print("SHAP summary plot saved to models/shap_summary.png")

# ── Save models + metadata ─────────────────────────────────────────────────
joblib.dump(xgb_reg, "models/xgb_regressor.pkl")
joblib.dump(xgb_clf, "models/xgb_classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")

metrics = {
    "rmse": round(rmse, 4),
    "r2": round(r2, 4),
    "cv_r2_mean": round(cv_scores.mean(), 4),
    "cv_r2_std": round(cv_scores.std(), 4),
    "n_features": len(FEATURE_COLS),
    "n_train": len(X_train),
    "n_test": len(X_test)
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModels saved to models/")
print(f"Metrics: {metrics}")
