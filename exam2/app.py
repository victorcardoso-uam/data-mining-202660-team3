"""Exam 2 Part 2 - Team 3 - EV Battery Thermal Model (Streamlit app)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exam2_team3_ev_battery.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
TARGET = "battery_cell_temp_c"
FEATURES = ["charge_rate_c", "discharge_rate_c", "ambient_temp_c",
            "coolant_flow_l_min", "internal_resistance_mOhm"]

st.set_page_config(page_title="Team 3 - EV Battery", layout="wide")
st.title("EV Battery Cell Temperature - Team 3")
st.caption("Exam 2 Part 2 | target: battery_cell_temp_c")


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def load_artifacts():
    needed = ["best_model.joblib", "scaler.joblib", "metadata.json"]
    missing = [f for f in needed if not (MODELS / f).exists()]
    if missing:
        return None, missing
    art = {
        "model": joblib.load(MODELS / "best_model.joblib"),
        "scaler": joblib.load(MODELS / "scaler.joblib"),
        "meta": json.loads((MODELS / "metadata.json").read_text()),
    }
    mlp_path = MODELS / "mlp_model.joblib"
    if mlp_path.exists():
        art["mlp"] = joblib.load(mlp_path)
    elif art["meta"]["winning_model"].startswith("MLP"):
        art["mlp"] = art["model"]
    else:
        art["mlp"] = None
    return art, []


def predict(art, row):
    meta = art["meta"]
    X = row[meta["features"]]
    if meta["requires_scaler"]:
        X = art["scaler"].transform(X)
    pred = float(art["model"].predict(X)[0])
    return float(np.exp(pred)) if meta["log_target"] else pred


if not DATA.exists():
    st.error(f"Dataset not found: {DATA}")
    st.stop()
df = load_data()
art, missing = load_artifacts()
ARTIFACTS = ["best_model.joblib", "scaler.joblib", "mlp_model.joblib", "metadata.json"]
missing_all = [f for f in ARTIFACTS if not (MODELS / f).exists()]
if art and art["meta"]["winning_model"].startswith("MLP") and missing_all == ["mlp_model.joblib"]:
    missing_all = []  # MLP is the winner, so it lives in best_model.joblib
if missing_all:
    st.error("Missing artifacts in models/: " + ", ".join(missing_all)
             + ". Run scripts/task1_regression.py and scripts/task2_ann.py first.")
else:
    st.success("Artifacts loaded: " + ", ".join(ARTIFACTS))

P75, P90 = df[TARGET].quantile(0.75), df[TARGET].quantile(0.90)
COOL_T, CRIT_T = round(P75), round(P90)


def q(c, p):
    return round(float(df[c].quantile(p)), 2)


PRESETS = {
    "normal": {"charge_rate_c": q("charge_rate_c", .5), "discharge_rate_c": q("discharge_rate_c", .1),
               "ambient_temp_c": q("ambient_temp_c", .1), "coolant_flow_l_min": q("coolant_flow_l_min", .9),
               "internal_resistance_mOhm": q("internal_resistance_mOhm", .5)},
    "cooling": {"charge_rate_c": q("charge_rate_c", .5), "discharge_rate_c": q("discharge_rate_c", .75),
                "ambient_temp_c": q("ambient_temp_c", .5), "coolant_flow_l_min": q("coolant_flow_l_min", .25),
                "internal_resistance_mOhm": q("internal_resistance_mOhm", .75)},
    "high": {"charge_rate_c": q("charge_rate_c", .5), "discharge_rate_c": q("discharge_rate_c", .95),
             "ambient_temp_c": q("ambient_temp_c", .95), "coolant_flow_l_min": q("coolant_flow_l_min", .05),
             "internal_resistance_mOhm": q("internal_resistance_mOhm", .95)},
}
for f in FEATURES:
    st.session_state.setdefault(f, float(df[f].median()))


def apply_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state[f] = v


t1, t2, t3, t4 = st.tabs(["1. Data Overview", "2. Model Comparison", "3. ANN Training",
                          "4. Real-Time What-If Scenario Inference"])

with t1:
    st.subheader("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df.head(20), width="stretch")
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe().T.round(3), width="stretch")
    st.subheader("Correlation matrix")
    st.dataframe(df.corr().round(3), width="stretch")
    st.caption("Correlation of each feature with battery_cell_temp_c")
    st.bar_chart(df.corr()[TARGET].drop(TARGET))

with t2:
    st.subheader("Test-set metrics (80/20 split, random_state=42)")
    comp = REPORTS / "model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp, index_col=0).round(4), width="stretch")
        if art:
            st.success(f"Winning model (lowest test RMSE): {art['meta']['winning_model']}")
    else:
        st.warning("reports/model_comparison.csv not found - run task2_ann.py.")
    img = FIGS / "task1_residual_dashboard.png"
    if img.exists():
        st.image(str(img), caption="OLS residual diagnostics (base linear model)")
    else:
        st.warning("Residual dashboard not found - run task1_regression.py.")

with t3:
    st.subheader("MLPRegressor architecture")
    if art and art["mlp"] is not None:
        mlp = art["mlp"]
        arch = [len(FEATURES)] + list(mlp.hidden_layer_sizes) + [1]
        n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        st.write(f"Layers: {' -> '.join(map(str, arch))} (input, hidden ReLU 64, hidden ReLU 32, linear output)")
        st.write(f"Epochs run: {mlp.n_iter_} | early_stopping=True | max_iter=1000 | "
                 f"best validation R2: {mlp.best_validation_score_:.4f} | trainable parameters: {n_params}")
        st.line_chart(pd.DataFrame({"training loss": mlp.loss_curve_}))
    else:
        st.warning("MLP model file not found (models/mlp_model.joblib).")
    img = FIGS / "task2_loss_curve.png"
    if img.exists():
        st.image(str(img), caption="Loss curve per epoch")

with t4:
    st.subheader("What-if scenario")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button("Normal scenario", on_click=apply_preset, args=("normal",))
    b2.button("Active cooling scenario", on_click=apply_preset, args=("cooling",))
    b3.button("High-risk scenario", on_click=apply_preset, args=("high",))
    labels = {"discharge_rate_c": "Discharge rate (C) - main slider",
              "charge_rate_c": "Charge rate (C)", "ambient_temp_c": "Ambient temperature (C)",
              "coolant_flow_l_min": "Coolant flow (L/min)",
              "internal_resistance_mOhm": "Internal resistance (mOhm)"}
    st.slider(labels["discharge_rate_c"], float(df["discharge_rate_c"].min()),
              float(df["discharge_rate_c"].max()), key="discharge_rate_c", step=0.01)
    cols = st.columns(2)
    for i, f in enumerate([f for f in FEATURES if f != "discharge_rate_c"]):
        cols[i % 2].slider(labels[f], float(df[f].min()), float(df[f].max()), key=f, step=0.01)

    if art:
        row = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])
        temp = predict(art, row)
        st.metric("Predicted battery cell temperature", f"{temp:.2f} C")
        if temp >= CRIT_T:
            st.error(f"CRITICAL THERMAL ALERT: {temp:.2f} C >= {CRIT_T} C. Reduce load and stop charging.")
        elif temp >= COOL_T:
            st.warning(f"ACTIVE COOLING TRIGGERED: {temp:.2f} C >= {COOL_T} C.")
        else:
            st.success(f"Normal operation: {temp:.2f} C < {COOL_T} C.")
        st.info(f"Alert policy: active cooling at {COOL_T} C = 75th percentile of battery_cell_temp_c "
                f"({P75:.2f} C, rounded), so only the hottest 25% of observed operating points need cooling. "
                f"Critical alert at {CRIT_T} C = 90th percentile ({P90:.2f} C, rounded), the hottest 10%.")
        st.caption(f"Model used: {art['meta']['winning_model']} | scaler applied: "
                   f"{art['meta']['requires_scaler']} | log target: {art['meta']['log_target']}")
