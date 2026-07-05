# ClimateGuard-AU 🔥🌏

**Real-time bushfire and climate risk prediction platform for Australian postcodes**

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/naman0307/climateguard-au)
[![GitHub](https://img.shields.io/badge/GitHub-climateguard--au-blue)](https://github.com/NamanDwivedi0307/climateguard-au)

## What it does
- Ingests NASA FIRMS satellite fire data + BOM weather data in real time
- Predicts bushfire risk score (0–100) per Australian postcode
- XGBoost + LSTM ensemble with SHAP explainability
- Interactive Folium map dashboard
- FastAPI alert endpoint for programmatic access

## Architecture
## Tech Stack
| Layer | Tools |
|---|---|
| Data | NASA FIRMS API, BOM, geopandas |
| ML | XGBoost, PyTorch LSTM, scikit-learn |
| Explainability | SHAP |
| API | FastAPI + uvicorn |
| Frontend | Streamlit + Folium |
| Deployment | HuggingFace Spaces |

## Local Setup
```bash
git clone https://github.com/NamanDwivedi0307/climateguard-au
cd climateguard-au
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Built by
Naman Dwivedi — Master of IT (AI Major), University of Melbourne
