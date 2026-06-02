import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Forecast Performance Analysis",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Forecast Performance Analysis")

st.markdown("""
Evaluate forecast performance across different market conditions and compare
machine learning forecasts against benchmark predictions.
""")

# -----------------------------
# Dummy Data
# -----------------------------
dates_case_1 = pd.date_range("2024-10-01", periods=24 * 7, freq="h")
dates_case_2 = pd.date_range("2024-12-01", periods=24 * 7, freq="h")

case_1 = pd.DataFrame({
    "datetime": dates_case_1,
    "true_price": 120 + 20 * np.sin(np.arange(len(dates_case_1)) / 8),
    "xgboost_prediction": 118 + 18 * np.sin(np.arange(len(dates_case_1)) / 8),
    "lstm_prediction": 121 + 22 * np.sin(np.arange(len(dates_case_1)) / 8),
    "naive_prediction": 115 + 15 * np.sin(np.arange(len(dates_case_1)) / 8),
})

case_2 = pd.DataFrame({
    "datetime": dates_case_2,
    "true_price": 150 + 80 * np.sin(np.arange(len(dates_case_2)) / 7),
    "xgboost_prediction": 135 + 45 * np.sin(np.arange(len(dates_case_2)) / 7),
    "lstm_prediction": 140 + 55 * np.sin(np.arange(len(dates_case_2)) / 7),
    "naive_prediction": 120 + 25 * np.sin(np.arange(len(dates_case_2)) / 7),
})

# Add artificial price spike in Case Study 2
case_2.loc[80:95, "true_price"] += 250

# -----------------------------
# Controls
# -----------------------------
scenario = st.radio(
    "Select Scenario",
    ["Case Study 1", "Case Study 2"],
    horizontal=True,
)

models = st.multiselect(
    "Select forecasts to show",
    ["xgboost_prediction", "lstm_prediction", "naive_prediction"],
    default=["xgboost_prediction", "naive_prediction"],
)

if scenario == "Case Study 1":
    plot_df = case_1.copy()
else:
    plot_df = case_2.copy()

# -----------------------------
# KPI Metrics
# -----------------------------
model_col = "xgboost_prediction"
baseline_col = "naive_prediction"

model_mae = np.mean(np.abs(plot_df["true_price"] - plot_df[model_col]))
baseline_mae = np.mean(np.abs(plot_df["true_price"] - plot_df[baseline_col]))

model_rmse = np.sqrt(np.mean((plot_df["true_price"] - plot_df[model_col]) ** 2))
baseline_rmse = np.sqrt(np.mean((plot_df["true_price"] - plot_df[baseline_col]) ** 2))

improvement = ((baseline_mae - model_mae) / baseline_mae) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Model MAE", f"{model_mae:.2f}")
c2.metric("Baseline MAE", f"{baseline_mae:.2f}")
c3.metric("Model RMSE", f"{model_rmse:.2f}")
c4.metric("Improvement", f"{improvement:.1f}%")

# -----------------------------
# Plot
# -----------------------------
cols = ["true_price"] + models

fig = px.line(
    plot_df,
    x="datetime",
    y=cols,
    labels={
        "value": "Electricity Price ($/MWh)",
        "variable": "",
        "datetime": "Datetime",
    },
)

fig.for_each_trace(
    lambda t: t.update(
        name={
            "true_price": "Actual Price",
            "xgboost_prediction": "XGBoost Forecast",
            "lstm_prediction": "LSTM Forecast",
            "naive_prediction": "Naive Baseline",
        }.get(t.name, t.name),
        line=dict(width=3),
    )
)

fig.update_layout(
    template="plotly_white",
    height=620,
    hovermode="x unified",
    title=None,
    legend_title_text="",
    font=dict(size=15, color="#111827"),
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Explanation
# -----------------------------
with st.expander("How to explain this chart"):
    st.markdown("""
    - **Actual Price** shows the real electricity market price.
    - **XGBoost Forecast** shows the machine learning model prediction.
    - **LSTM Forecast** shows the deep learning model prediction.
    - **Naive Baseline** is a simple benchmark forecast.
    - If the model line stays closer to the actual price than the baseline, the model adds value.
    """)

st.caption(
    "Note: This page currently uses dummy values. Replace with final XGBoost and LSTM outputs once final models are ready."
)
