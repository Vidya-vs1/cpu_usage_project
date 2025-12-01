import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# MUST COME FIRST BEFORE ANY OTHER STREAMLIT COMMAND
st.set_page_config(
    page_title="CPU Usage Prediction",
    page_icon="🧠",
    layout="centered",
)


MODEL_PATH = Path("models/model_RandomForest_small.joblib")

CONTROLLER_KINDS = [
    "ReplicaSet",
    "DaemonSet",
    "StatefulSet",
    "ReplicationController",
    "Job",
    "Unknown",
]

# ---------- LOAD MODEL (CACHED) ----------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ---------- UI LAYOUT ----------


st.title("🧠 CPU Usage Prediction for Container Workloads")

st.write(
    """
This application predicts **CPU usage** of workloads in Kubernetes environments  
based on resource requests, limits and runtime characteristics.

Model used: **Random Forest Regressor** ✔  
Tracked using **DVC + MLflow** ✔  
Deployed using **Streamlit Cloud** ✔

"""
)

st.markdown("---")

# ---------- INPUT FORM ----------

with st.form("prediction_form"):
    st.subheader("🔧 Workload Configuration")

    cpu_request = st.number_input(
        "CPU Request (milli-cores)",
        min_value=0.0,
        value=200.0,
        step=10.0,
    )
    mem_request = st.number_input(
        "Memory Request (MiB)",
        min_value=0.0,
        value=512.0,
        step=32.0,
    )
    cpu_limit = st.number_input(
        "CPU Limit (milli-cores)",
        min_value=0.0,
        value=400.0,
        step=10.0,
    )
    mem_limit = st.number_input(
        "Memory Limit (MiB)",
        min_value=0.0,
        value=1024.0,
        step=32.0,
    )
    runtime_minutes = st.number_input(
        "Runtime (minutes)",
        min_value=0.0,
        value=60.0,
        step=5.0,
    )
    controller_kind = st.selectbox(
        "Controller Kind",
        CONTROLLER_KINDS,
        index=0,
    )

    submitted = st.form_submit_button("🔮 Predict CPU Usage")

# ---------- PREDICTION LOGIC ----------

if submitted:
    controller_val = None if controller_kind == "Unknown" else controller_kind

    input_data = pd.DataFrame([{
        "cpu_request": cpu_request,
        "mem_request": mem_request,
        "cpu_limit": cpu_limit,
        "mem_limit": mem_limit,
        "runtime_minutes": runtime_minutes,
        "controller_kind": controller_val
    }])

    try:
        pred = model.predict(input_data)[0]
        st.markdown("---")
        st.success(f"📈 Predicted CPU Usage: **{pred:.4f}**")

    except Exception as e:
        st.error("❌ Prediction failed!")
        st.exception(e)

st.markdown("---")
st.caption("Deployed on Streamlit — Developed by Vidya VS 🧠")
