# ============================================================
# Fire-Resistance Recycled-Aggregate Concrete
# Compressive Strength Prediction GUI  (XGBoost + SHAP)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FRC Compressive Strength Predictor", layout="wide")

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    background = pd.read_csv("x_train_background.csv")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(background)
    return model, le, background, explainer, shap_values

model, le, background, explainer, shap_values = load_artifacts()

FEATURES = list(background.columns)
DISPLAY_NAMES = {
    "Temperature (°C )": "Temperature (°C )",
    "W/C": "W/C",
    "RA_Type_enc": "Type of RA",
    "  Binder  (kg/m3)": "Binder (kg/m³)",
    "FA (kg/m3)": "FA (kg/m³)",
    " CA (kg/m3)": "CA (kg/m³)",
    "RA (kg/m3)": "RA (kg/m³)",
}

# ---------- Header ----------
st.markdown("## Compressive Strength ($f'_c$) Prediction — Fire-Exposed Recycled Aggregate Concrete")
st.markdown("---")

col_in, col_out = st.columns([1.1, 1])

# ---------- Input Parameters ----------
with col_in:
    st.markdown("#### Input Parameters")

    temperature = st.number_input("Temperature, $T$ (°C)", min_value=20.0, max_value=900.0, value=30.0, step=10.0)
    ra_type = st.selectbox("Type of Recycled Aggregate", options=list(le.classes_), index=list(le.classes_).index("Glass") if "Glass" in le.classes_ else 0)
    wc = st.number_input("Water–Cement ratio, $W/C$", min_value=0.30, max_value=0.70, value=0.42, step=0.01, format="%.2f")
    binder = st.number_input("Binder content (kg/m³)", min_value=200.0, max_value=600.0, value=435.0, step=5.0)
    fa = st.number_input("Fine Aggregate, $FA$ (kg/m³)", min_value=200.0, max_value=900.0, value=539.55, step=5.0)
    ca = st.number_input("Coarse Aggregate, $CA$ (kg/m³)", min_value=0.0, max_value=1800.0, value=1569.0, step=5.0)
    ra = st.number_input("Recycled Aggregate, $RA$ (kg/m³)", min_value=0.0, max_value=600.0, value=0.0, step=5.0)

    b1, b2 = st.columns([1, 1])
    predict_clicked = b1.button("Predict", type="primary", use_container_width=True)
    b2.button("Reset", use_container_width=True)  # rerun resets nothing persistent

# ---------- Output ----------
with col_out:
    st.markdown("#### Output")
    if predict_clicked:
        row = pd.DataFrame([{
            "Temperature (°C )": temperature,
            "W/C": wc,
            "RA_Type_enc": int(le.transform([ra_type])[0]),
            "  Binder  (kg/m3)": binder,
            "FA (kg/m3)": fa,
            " CA (kg/m3)": ca,
            "RA (kg/m3)": ra,
        }])[FEATURES]

        pred = float(model.predict(row)[0])
        st.metric(label="Compressive Strength, $f'_c$", value=f"{pred:.4f} MPa")

        # Local SHAP explanation for this prediction
        sv_local = explainer.shap_values(row)
        st.markdown("**Contribution of each input to this prediction (SHAP)**")
        fig_local, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        order = np.argsort(np.abs(sv_local[0]))
        names = [DISPLAY_NAMES.get(FEATURES[i], FEATURES[i]) for i in order]
        vals = sv_local[0][order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value (MPa impact on prediction)")
        plt.tight_layout()
        st.pyplot(fig_local)
    else:
        st.info("Set the input parameters and click **Predict**.")

# ---------- Global SHAP section ----------
st.markdown("---")
st.markdown("### Model Interpretability (Global SHAP — XGBoost)")

g1, g2 = st.columns(2)

@st.cache_resource
def global_shap_figs():
    disp = [DISPLAY_NAMES.get(f, f) for f in FEATURES]

    fig1 = plt.figure(figsize=(7, 5), dpi=150)
    shap.summary_plot(shap_values, background, feature_names=disp, show=False)
    plt.title("SHAP Summary Plot — XGB")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)
    ax2.barh([disp[i] for i in idx], mean_abs[idx], color="purple")
    ax2.set_xlabel("mean(|SHAP value|)")
    ax2.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    return fig1, fig2

fig1, fig2 = global_shap_figs()
g1.pyplot(fig1)
g2.pyplot(fig2)

st.markdown("---")
st.caption("XGBoost model (n_estimators=100, lr=0.1, max_depth=6) · Test R² ≈ 0.987, RMSE ≈ 1.66 MPa · 616 samples, 80/20 split (random_state=0)")
