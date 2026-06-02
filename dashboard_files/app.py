from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Duckstradamus NZ Electricity Forecasting",
    page_icon="🦆",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #020617 100%);
    color: #F8FAFC !important;
}

html, body, p, span, label, div, li {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.12);
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

h1 {
    color: #FFFFFF !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2 {
    color: #E2E8F0 !important;
    font-weight: 750 !important;
}

h3 {
    color: #CBD5E1 !important;
    font-weight: 700 !important;
}

p {
    color: #CBD5E1 !important;
    font-size: 16px !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.13);
    border-radius: 18px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 34px !important;
    font-weight: 800 !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stToggle"] label {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] * {
    color: #111827 !important;
}

div[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] {
    background-color: white !important;
    border-radius: 12px !important;
}

.risk-duck {
    font-size: 52px;
    font-weight: 800;
    margin-top: 20px;
    color: #FFFFFF !important;
}

.calm-duck {
    animation: calmduck 5s ease-in-out infinite alternate;
}

.alert-duck {
    animation: alertduck 1.5s ease-in-out infinite alternate;
}

.panic-duck {
    animation: panicduck 0.35s ease-in-out infinite alternate;
}

@keyframes calmduck {
    from { transform: translateX(0px); }
    to { transform: translateX(35px); }
}

@keyframes alertduck {
    from { transform: translateX(0px); }
    to { transform: translateX(80px); }
}

@keyframes panicduck {
    0%   { transform: rotate(-3deg); }
    25%  { transform: rotate(3deg); }
    50%  { transform: rotate(-3deg); }
    75%  { transform: rotate(3deg); }
    100% { transform: rotate(-3deg); }
}

.panic-duck {
    animation: panicduck 1.5s ease-in-out infinite;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FUNCTIONS
# ============================================================

def show_duck_status(error_value, price_value):
    if error_value < 20 and price_value < 150:
        st.success("🟢 Duck Status: Calm Market | Good Prediction")
        st.markdown(
            '<div class="risk-duck calm-duck">🦆 Calm Duck</div>',
            unsafe_allow_html=True,
        )

    elif error_value < 60 and price_value < 250:
        st.warning("🟡 Duck Status: Alert Market | Moderate Movement")
        st.markdown(
            '<div class="risk-duck alert-duck">🦆⚠️ Alert Duck</div>',
            unsafe_allow_html=True,
        )

    else:
        st.error("🔴 Duck Status: Panic Market | High Risk or Price Spike")
        st.markdown(
            '<div class="risk-duck panic-duck">🦆🚨 Panic Duck</div>',
            unsafe_allow_html=True,
        )


def add_duck_mood_columns(data, actual_col, pred_col):
    data = data.copy()
    data["error"] = data[actual_col] - data[pred_col]
    data["abs_error"] = data["error"].abs()

    conditions = [
        (data["abs_error"] < 20) & (data[actual_col] < 150),
        (data["abs_error"] < 60) & (data[actual_col] < 250),
    ]

    choices = ["Calm", "Alert"]

    data["duck_mood"] = np.select(
        conditions,
        choices,
        default="Panic",
    )

    return data


@st.cache_data
def load_data():
    app_dir = Path(__file__).resolve().parent

    possible_paths = [
        app_dir / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
        app_dir / "xgboost_v3_test_predictions_with_dates.csv",
        app_dir.parent / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
        app_dir.parent / "Model_XGBoost" / "xgboost_v3_test_predictions_with_dates.csv",
        app_dir.parent / "notebooks" / "Anu" / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    matches = list(app_dir.parent.rglob("xgboost_v3_test_predictions_with_dates.csv"))

    if matches:
        return pd.read_csv(matches[0])

    st.error("Could not find xgboost_v3_test_predictions_with_dates.csv")
    st.stop()


def style_plot(fig, height=600):
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=height,
        font=dict(color="white", size=15),
        title=dict(font=dict(color="white", size=24)),
        legend=dict(font=dict(color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.7)",
    )
    return fig


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

datetime_col = None

for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        datetime_col = col
        break

if datetime_col is None:
    st.error("No datetime column found.")
    st.stop()

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least actual and predicted price columns.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🦆 Duckstradamus")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Forecast Viewer",
        "Error Analysis",
        "Duck Risk Meter",
    ],
)

actual_col = st.sidebar.selectbox(
    "Actual Price Column",
    numeric_cols,
    index=0,
)

pred_col = st.sidebar.selectbox(
    "Prediction Column",
    numeric_cols,
    index=min(1, len(numeric_cols) - 1),
)

df = add_duck_mood_columns(df, actual_col, pred_col)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.title("🦆⚡ Duckstradamus NZ Electricity Forecasting Control Room")

    st.markdown(
        """
        ### ⚡ Market Intelligence Powered by Ducks

        This dashboard predicts New Zealand electricity prices using the XGBoost V3 model.
        The duck reacts to market risk, model movement, and spike events.
        """
    )

    latest_price = df[actual_col].iloc[-1]
    latest_pred = df[pred_col].iloc[-1]
    latest_error = abs(latest_price - latest_pred)

    avg_price = df[actual_col].mean()
    max_price = df[actual_col].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("⚡ Current Price", f"${latest_price:,.2f}/MWh")
    col2.metric("🔮 Forecast Price", f"${latest_pred:,.2f}/MWh")
    col3.metric("📊 Average Price", f"${avg_price:,.2f}/MWh")
    col4.metric("🚨 Highest Price", f"${max_price:,.2f}/MWh")

    show_duck_status(latest_error, latest_price)

    st.markdown("---")
    st.markdown("## 🎛️ Interactive Control Panel")

    control1, control2, control3, control4 = st.columns(4)

    with control1:
        view = st.radio(
            "Time Window",
            ["24 Hours", "7 Days", "30 Days", "Full Period"],
        )

    with control2:
        duck_filter = st.radio(
            "Duck Mood Filter",
            ["All", "Calm", "Alert", "Panic"],
        )

    with control3:
        show_spikes = st.toggle("Show Only Spikes")

    with control4:
        log_scale = st.toggle("Use Log Scale")

    if view == "24 Hours":
        plot_df = df.tail(24).copy()
    elif view == "7 Days":
        plot_df = df.tail(24 * 7).copy()
    elif view == "30 Days":
        plot_df = df.tail(24 * 30).copy()
    else:
        plot_df = df.copy()

    if duck_filter != "All":
        plot_df = plot_df[plot_df["duck_mood"] == duck_filter]

    if show_spikes:
        spike_threshold = st.slider(
            "Spike Threshold ($/MWh)",
            min_value=0,
            max_value=int(df[actual_col].max()),
            value=min(300, int(df[actual_col].max())),
        )

        plot_df = plot_df[plot_df[actual_col] >= spike_threshold]

    if plot_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    st.markdown("### 📈 Interactive Forecast Chart")

    fig = px.line(
        plot_df,
        x=datetime_col,
        y=[actual_col, pred_col],
        title="Actual vs Predicted Electricity Price",
        labels={
            "value": "Price ($/MWh)",
            "variable": "Series",
        },
    )

    if log_scale:
        fig.update_yaxes(type="log")

    fig = style_plot(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🦆 Duck Error Map")

    scatter_df = df.copy()

    if view == "24 Hours":
        scatter_df = scatter_df.tail(24).copy()
    elif view == "7 Days":
        scatter_df = scatter_df.tail(24 * 7).copy()
    elif view == "30 Days":
        scatter_df = scatter_df.tail(24 * 30).copy()

    fig2 = px.scatter(
        scatter_df,
        x=actual_col,
        y="abs_error",
        color="duck_mood",
        hover_data=[datetime_col, actual_col, pred_col, "error"],
        title="Where does the model struggle?",
        labels={
            actual_col: "Actual Price ($/MWh)",
            "abs_error": "Absolute Difference",
            "duck_mood": "Duck Mood",
        },
    )

    fig2 = style_plot(fig2, height=520)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🚨 Top Duck Panic Events")

    top_events = (
        df.sort_values("abs_error", ascending=False)
        [[datetime_col, actual_col, pred_col, "error", "abs_error", "duck_mood"]]
        .head(20)
    )

    selected_event = st.selectbox(
        "Select a high-risk event",
        top_events[datetime_col].astype(str),
    )

    event_row = top_events[
        top_events[datetime_col].astype(str) == selected_event
    ].iloc[0]

    e1, e2, e3 = st.columns(3)

    e1.metric("Event Time", str(event_row[datetime_col]))
    e2.metric("Actual Price", f"${event_row[actual_col]:,.2f}")
    e3.metric("Forecast Price", f"${event_row[pred_col]:,.2f}")

    show_duck_status(event_row["abs_error"], event_row[actual_col])

    with st.expander("Show Top 20 High-Risk Events"):
        st.dataframe(top_events, use_container_width=True)


# ============================================================
# FORECAST VIEWER
# ============================================================

elif page == "Forecast Viewer":

    st.title("📈🦆 Forecast Viewer with Duck Replay")

    view = st.selectbox(
        "Select Time Window",
        ["24 Hours", "7 Days", "30 Days", "Full Period"],
    )

    if view == "24 Hours":
        plot_df = df.tail(24).copy()
    elif view == "7 Days":
        plot_df = df.tail(24 * 7).copy()
    elif view == "30 Days":
        plot_df = df.tail(24 * 30).copy()
    else:
        plot_df = df.copy()

    st.markdown("### 🦆 Move Duck Through Time")

    selected_row = st.slider(
        "Select Forecast Point",
        min_value=0,
        max_value=len(plot_df) - 1,
        value=len(plot_df) - 1,
    )

    selected_point = plot_df.iloc[selected_row]

    selected_time = selected_point[datetime_col]
    selected_actual = selected_point[actual_col]
    selected_pred = selected_point[pred_col]
    selected_error = abs(selected_actual - selected_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Time", str(selected_time))
    col2.metric("Actual Price", f"${selected_actual:,.2f}")
    col3.metric("Predicted Price", f"${selected_pred:,.2f}")

    show_duck_status(selected_error, selected_actual)

    fig = px.line(
        plot_df,
        x=datetime_col,
        y=[actual_col, pred_col],
        title=f"{view}: Actual vs Prediction",
        labels={
            "value": "Price ($/MWh)",
            "variable": "Series",
        },
    )

    fig.add_vline(
        x=selected_time,
        line_dash="dash",
        line_color="yellow",
        annotation_text="🦆 Selected Duck Point",
        annotation_position="top",
    )

    fig = style_plot(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Predictions")

    st.dataframe(
        plot_df[[datetime_col, actual_col, pred_col, "duck_mood"]].tail(50),
        use_container_width=True,
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "Error Analysis":

    st.title("🎯🦆 Error Analysis")

    st.markdown(
        """
        This page keeps the model error metrics separate from the main dashboard.
        Use this section only when you want to evaluate model performance.
        """
    )

    mean_error = df["error"].mean()
    rmse = np.sqrt(np.mean(df["error"] ** 2))
    mae = df["abs_error"].mean()
    max_abs_error = df["abs_error"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean Error", f"{mean_error:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("MAE", f"{mae:,.2f}")
    col4.metric("Max Abs Error", f"{max_abs_error:,.2f}")

    high_error_threshold = st.slider(
        "High Error Threshold",
        min_value=0,
        max_value=int(df["abs_error"].max()),
        value=int(df["abs_error"].quantile(0.90)),
    )

    high_error_df = df[df["abs_error"] >= high_error_threshold]

    st.warning(f"🦆 High-error duck events found: {len(high_error_df)}")

    fig = px.histogram(
        df,
        x="error",
        color="duck_mood",
        nbins=60,
        title="Prediction Error Distribution by Duck Mood",
    )

    fig = style_plot(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df,
        x=actual_col,
        y="abs_error",
        color="duck_mood",
        hover_data=[datetime_col, actual_col, pred_col],
        title="Actual Price vs Absolute Error",
        labels={
            actual_col: "Actual Price ($/MWh)",
            "abs_error": "Absolute Error",
        },
    )

    fig2 = style_plot(fig2, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚨 High Error Events")

    st.dataframe(
        high_error_df[[datetime_col, actual_col, pred_col, "error", "abs_error", "duck_mood"]]
        .sort_values("abs_error", ascending=False)
        .head(30),
        use_container_width=True,
    )


# ============================================================
# DUCK RISK METER
# ============================================================

elif page == "Duck Risk Meter":

    st.title("🦆 Market Risk Simulator")

    st.markdown(
        """
        Adjust the market conditions below and see how the estimated price risk changes.

        This is a business simulator, not a retrained ML model.
        """
    )

    demand = st.slider("Demand Increase (%)", -20, 50, 0)
    wind = st.slider("Wind Change (%)", -50, 50, 0)
    hydro = st.slider("Hydro Change (%)", -50, 50, 0)

    base_price = df[actual_col].mean()

    simulated_price = (
        base_price
        * (1 + demand / 100 * 0.8)
        * (1 - wind / 100 * 0.3)
        * (1 - hydro / 100 * 0.2)
    )

    simulated_error = abs(simulated_price - base_price)

    col1, col2 = st.columns(2)

    col1.metric("Base Price", f"${base_price:,.2f}/MWh")
    col2.metric("Simulated Market Price", f"${simulated_price:,.2f}/MWh")

    show_duck_status(simulated_error, simulated_price)

    if simulated_price < base_price * 0.9:
        st.success("🦆 Ducks are calm. Low market risk.")
    elif simulated_price < base_price * 1.2:
        st.warning("🦆 Ducks are alert. Medium market risk.")
    else:
        st.error("🦆 Ducks are panicking. High market risk!")
