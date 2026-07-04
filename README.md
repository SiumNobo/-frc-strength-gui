# FRC Compressive Strength Prediction GUI

Publicly accessible GUI for predicting the compressive strength of fire-exposed
recycled aggregate concrete using XGBoost, with SHAP interpretability.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit GUI
- `xgb_model.pkl` — trained XGBoost model (Test R² ≈ 0.987)
- `label_encoder.pkl` — encoder for the recycled aggregate type
- `x_train_background.csv` — SHAP background data (training split)
