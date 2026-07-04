# ============================================================
# Fire-Resistance Recycled-Aggregate Concrete
# Compressive Strength Prediction GUI  (XGBoost + SHAP)
# SHAP plots reproduce the paper exactly:
#   shap.Explainer(model.predict, x_test) evaluated on x_test
#   (values precomputed and shipped with the app)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="FRC Compressive Strength Predictor", layout="wide")

# Paper font (Liberation Serif ~ Times New Roman); falls back to any serif
matplotlib.rcParams['font.family'] = ['Liberation Serif', 'DejaVu Serif', 'serif']

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    x_test = pd.read_csv("x_test.csv")
    shap_vals = np.load("shap_values.npy")
    shap_base = np.load("shap_base.npy")
    tree_explainer = shap.TreeExplainer(model)   # fast, for per-prediction explanation only
    return model, le, x_test, shap_vals, shap_base, tree_explainer

model, le, x_test, shap_vals, shap_base, tree_explainer = load_artifacts()

FEATURES = list(x_test.columns)  # ['Temperature (°C )','W/C','RFA Type','Binder (kg/m3)','FA (kg/m3)','CA (kg/m3)','RA (kg/m3)']
DISPLAY_NAMES = [c.replace('kg/m3', r'kg/m$^3$') if 'kg/m3' in c else c for c in FEATURES]

# ---------- Header ----------
st.markdown("## Compressive Strength ($f'_c$) Prediction — Fire-Exposed Recycled Aggregate Concrete")
st.markdown("---")

col_in, col_out = st.columns([1.1, 1])

# ---------- Input Parameters ----------
with col_in:
    st.markdown("#### Input Parameters")

    temperature = st.number_input("Temperature, $T$ (°C)", min_value=20.0, max_value=900.0, value=30.0, step=10.0)
    ra_type = st.selectbox("RFA Type", options=list(le.classes_),
                           index=list(le.classes_).index("Glass") if "Glass" in le.classes_ else 0)
    wc = st.number_input("Water–Cement ratio, $W/C$", min_value=0.30, max_value=0.70, value=0.42, step=0.01, format="%.2f")
    binder = st.number_input("Binder content (kg/m³)", min_value=200.0, max_value=600.0, value=435.0, step=5.0)
    fa = st.number_input("Fine Aggregate, $FA$ (kg/m³)", min_value=200.0, max_value=900.0, value=539.55, step=5.0)
    ca = st.number_input("Coarse Aggregate, $CA$ (kg/m³)", min_value=0.0, max_value=1800.0, value=1569.0, step=5.0)
    ra = st.number_input("Recycled Aggregate, $RA$ (kg/m³)", min_value=0.0, max_value=600.0, value=0.0, step=5.0)

    b1, b2 = st.columns([1, 1])
    predict_clicked = b1.button("Predict", type="primary", use_container_width=True)
    b2.button("Reset", use_container_width=True)

# ---------- Output ----------
with col_out:
    st.markdown("#### Output")
    if predict_clicked:
        row = pd.DataFrame([{
            "Temperature (°C )": temperature,
            "W/C": wc,
            "RFA Type": int(le.transform([ra_type])[0]),
            "Binder (kg/m3)": binder,
            "FA (kg/m3)": fa,
            "CA (kg/m3)": ca,
            "RA (kg/m3)": ra,
        }])[FEATURES]

        pred = float(model.predict(row)[0])
        st.metric(label="Compressive Strength, $f'_c$", value=f"{pred:.4f} MPa")

        # Local SHAP explanation for this single prediction
        sv_local = tree_explainer.shap_values(row)
        st.markdown("**Contribution of each input to this prediction (SHAP)**")
        fig_local, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        order = np.argsort(np.abs(sv_local[0]))
        names = [DISPLAY_NAMES[i] for i in order]
        vals = sv_local[0][order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value (MPa impact on prediction)")
        plt.tight_layout()
        st.pyplot(fig_local)
    else:
        st.info("Set the input parameters and click **Predict**.")

# ---------- Global SHAP section (exact paper figures) ----------
st.markdown("---")
st.markdown("### Model Interpretability (SHAP — XGBoost)")

g1, g2 = st.columns(2)

@st.cache_resource
def global_shap_figs():
    sv = shap.Explanation(values=shap_vals, base_values=shap_base,
                          data=x_test.values, feature_names=FEATURES)

    # --- Beeswarm: identical to paper (no title) ---
    fig1 = plt.figure(dpi=150)
    shap.summary_plot(sv, x_test, feature_names=DISPLAY_NAMES, show=False)
    ax = plt.gca()
    ax.tick_params(colors='black', which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    fig1 = plt.gcf()
    plt.tight_layout()

    # --- Bar: deeppink, descending, identical to paper (no title) ---
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    fig2 = plt.figure(figsize=(8, 5), dpi=150)
    plt.barh([DISPLAY_NAMES[i] for i in idx], mean_abs[idx], color='deeppink')
    plt.xlabel("mean(|SHAP value|)")
    plt.gca().invert_yaxis()
    ax2 = plt.gca()
    ax2.tick_params(colors='black', which='both')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    plt.tight_layout()
    return fig1, fig2

fig1, fig2 = global_shap_figs()
g1.pyplot(fig1)
g2.pyplot(fig2)


