from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Historical Prediction Replay",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬🦆 Historical Prediction Replay Center")

st.markdown(
    """
    Use this page to replay historical electricity price predictions.

    Select any timestamp and compare:

    - Actual electricity price
    - XGBoost predicted price
    - Prediction error
    - Duck market mood
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_prediction_data():

    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent

    possible_paths = [
        project_dir / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
        project_dir / "xgboost_v3_test_predictions_with_dates.csv",
        project_dir.parent / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    matches = list(project_dir.rglob("xgboost_v3_test_predictions_with_dates.csv"))

    if matches:
        return pd.read_csv(matches[0])

    st.error("Could not find xgboost_v3_test_predictions_with_dates.csv")
    st.stop()


df = load_prediction_data()


# ============================================================
# AUTO-DETECT COLUMNS
# ============================================================

datetime_col = None

for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        datetime_col = col
        break

if datetime_col is None:
    st.error("No datetime/date column found.")
    st.write(df.columns.tolist())
    st.stop()

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns for actual and predicted values.")
    st.write(df.columns.tolist())
    st.stop()


# ============================================================
# SIDEBAR COLUMN SELECTION
# ============================================================

st.sidebar.header("Replay Settings")

actual_col = st.sidebar.selectbox(
    "Actual Price Column",
    numeric_cols,
    index=0,
)

pred_col = st.sidebar.selectbox(
    "Predicted Price Column",
    numeric_cols,
    index=min(1, len(numeric_cols) - 1),
)


# ============================================================
# CREATE ERROR COLUMNS
# ============================================================

df["error"] = df[actual_col] - df[pred_col]
df["abs_error"] = df["error"].abs()


# ============================================================
# DATE RANGE FILTER
# ============================================================

st.sidebar.markdown("---")

min_date = df[datetime_col].min().date()
max_date = df[datetime_col].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:

    start_date, end_date = date_range

    df_filtered = df[
        (df[datetime_col].dt.date >= start_date)
        & (df[datetime_col].dt.date <= end_date)
    ].copy()

else:
    df_filtered = df.copy()

if df_filtered.empty:
    st.warning("No data found for selected date range.")
    st.stop()


# ============================================================
# REPLAY SLIDER
# ============================================================

st.subheader("🕰️ Select a Historical Time Point")

row_index = st.slider(
    "Move through historical predictions",
    min_value=0,
    max_value=len(df_filtered) - 1,
    value=0,
)

selected = df_filtered.iloc[row_index]

selected_time = selected[datetime_col]
actual_price = selected[actual_col]
pred_price = selected[pred_col]
error = selected["error"]
abs_error = selected["abs_error"]


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Timestamp",
    str(selected_time),
)

col2.metric(
    "Actual Price",
    f"${actual_price:,.2f}/MWh",
)

col3.metric(
    "Predicted Price",
    f"${pred_price:,.2f}/MWh",
)

col4.metric(
    "Absolute Error",
    f"${abs_error:,.2f}/MWh",
)


# ============================================================
# DUCK MOOD
# ============================================================

st.markdown("---")

if pred_price < 100:
    st.success("🟢🦆 Duck Mood: Calm Market — Low predicted price.")
elif pred_price < 200:
    st.warning("🟡🦆 Duck Mood: Alert Market — Normal to moderate price.")
else:
    st.error("🔴🦆 Duck Mood: Panic Market — High price / spike risk.")


# ============================================================
# LOCAL WINDOW CHART
# ============================================================

st.subheader("📈 Local Prediction Window")

window_size = st.slider(
    "Select Window Size Around Selected Point",
    min_value=12,
    max_value=240,
    value=48,
    step=12,
)

original_position = df_filtered.index.get_loc(selected.name)

start_pos = max(0, original_position - window_size)
end_pos = min(len(df_filtered), original_position + window_size)

window_df = df_filtered.iloc[start_pos:end_pos].copy()

fig = px.line(
    window_df,
    x=datetime_col,
    y=[actual_col, pred_col],
    title="Actual vs Predicted Price Around Selected Time",
    labels={
        "value": "Price ($/MWh)",
        "variable": "Series",
    },
)

fig.add_vline(
    x=selected_time,
    line_dash="dash",
    annotation_text="Selected Time",
    annotation_position="top",
)

fig.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# ERROR CONTEXT
# ============================================================

st.subheader("🎯 Error Context")

error_col1, error_col2, error_col3 = st.columns(3)

error_col1.metric(
    "Mean Absolute Error in Selected Range",
    f"${df_filtered['abs_error'].mean():,.2f}",
)

error_col2.metric(
    "Max Absolute Error in Selected Range",
    f"${df_filtered['abs_error'].max():,.2f}",
)

error_col3.metric(
    "Selected Error Direction",
    "Underprediction" if error > 0 else "Overprediction",
)

if error > 0:
    st.info(
        """
        The model underpredicted the market price.

        Meaning: actual price was higher than predicted price.
        """
    )
elif error < 0:
    st.info(
        """
        The model overpredicted the market price.

        Meaning: predicted price was higher than actual price.
        """
    )
else:
    st.success("Perfect prediction for this selected point.")


# ============================================================
# SELECTED ROW DETAILS
# ============================================================

with st.expander("Show selected row details"):
    st.dataframe(
        selected.to_frame().T,
        use_container_width=True,
    )


# ============================================================
# TOP ERROR TABLE
# ============================================================

st.subheader("🚨 Top Prediction Errors in Selected Range")

top_n = st.slider(
    "Show Top N Errors",
    min_value=5,
    max_value=50,
    value=10,
)

top_errors = (
    df_filtered
    .sort_values("abs_error", ascending=False)
    .head(top_n)
)

st.dataframe(
    top_errors[[datetime_col, actual_col, pred_col, "error", "abs_error"]],
    use_container_width=True,
)
