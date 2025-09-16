import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dummy model list
available_models = ["AIFS", "GraphCast", "FuXi", "IFS-HRES"]
available_metrics = ["RMSE", "ACC", "CRPS", "SEEPS"]
available_variables = ["Z500", "T2M", "TP"]
available_regions = ["Global", "Europe", "North America"]
ground_truth_map = {
    "Z500": ["ERA5"],
    "T2M": ["ERA5"],
    "TP": ["ERA5", "IMERG", "CHIRPS"]
}

# ---------------------
# Load data or create dummy
# ---------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("model_scores.csv")
    except:
        lead_times = np.arange(0, 11)
        rows = []
        for model in available_models:
            for metric in available_metrics:
                for var in available_variables:
                    for region in available_regions:
                        for lt in lead_times:
                            value = np.random.rand() if metric != "ACC" else np.random.uniform(0.5, 0.95)
                            rows.append([model, metric, var, region, lt, value])
        df = pd.DataFrame(rows, columns=["model", "metric", "variable", "region", "lead_time", "value"])
    return df

data = load_data()

# ---------------------
# UI
# ---------------------
st.title("AI Weather Model Scorecard")

# Metric selection
metric = st.selectbox("Select a Metric:", available_metrics)

# Context helper
with st.expander("ℹ️ Learn About This Metric"):
    if metric == "RMSE":
        st.markdown("**Root Mean Square Error (RMSE)**: Measures the average magnitude of the forecast error. Lower is better.")
    elif metric == "ACC":
        st.markdown("**Anomaly Correlation Coefficient (ACC)**: Measures how well the forecast matches spatial patterns. Higher is better.")
    elif metric == "CRPS":
        st.markdown("**Continuous Ranked Probability Score (CRPS)**: Used for probabilistic forecasts. Lower is better.")
    elif metric == "SEEPS":
        st.markdown("**Stable Equitable Error in Probability Space (SEEPS)**: Event-based precipitation metric. Lower is better.")

# Model/var/region selectors
selected_models = st.multiselect("Select Models to Compare:", available_models, default=available_models[:2])
variable = st.selectbox("Select Variable:", available_variables)
region = st.selectbox("Select Region:", available_regions)

# Ground truth
ground_truth_options = ground_truth_map.get(variable, ["ERA5"])
ground_truth = st.selectbox("Select Ground Truth Dataset:", ground_truth_options)

# ---------------------
# Filter & Plot
# ---------------------
filtered = data[
    (data["model"].isin(selected_models)) &
    (data["metric"] == metric) &
    (data["variable"] == variable) &
    (data["region"] == region)
]

if filtered.empty:
    st.warning("No data available for selected options.")
else:
    st.subheader("📊 Metric vs Lead Time")
    fig, ax = plt.subplots()
    for model in selected_models:
        model_data = filtered[filtered["model"] == model]
        ax.plot(model_data["lead_time"], model_data["value"], label=model)
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} for {variable} over {region}")
    ax.legend()
    st.pyplot(fig)

    # Table
    st.subheader("📋 Metric Table")
    table = filtered.pivot(index="lead_time", columns="model", values="value")
    st.dataframe(table.round(3))

    # Guided Qs
    #st.subheader("🧠 Guided Questions")
    #if metric == "ACC":
     #   st.markdown(f"**Q:** Which model maintains the highest {metric} for {variable} beyond Day 7?")
    #elif metric == "CRPS":
    #    st.markdown(f"**Q:** How do models compare on probabilistic forecasting of {variable}? Why might CRPS differ from RMSE?")
    #elif metric == "RMSE":
    #    st.markdown(f"**Q:** Which model shows the lowest {metric} for {variable} at short (0-2 day) lead times?")
    #elif metric == "SEEPS":
     #   st.markdown(f"**Q:** Does any model excel at predicting rare events for {variable} based on {metric}?")
