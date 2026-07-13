# Sales Forecasting & Demand Intelligence

An end-to-end sales forecasting and demand intelligence system: time-series 
decomposition, SARIMA/Prophet/XGBoost forecasting, anomaly detection, and 
product demand segmentation — deployed as an interactive Streamlit dashboard.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit dashboard (4 pages: overview, forecasts, anomalies, segments)
- `train.csv` — sales dataset
- `analysis.ipynb` — full exploratory notebook with all modeling steps
- `summary.docx` — executive business report
