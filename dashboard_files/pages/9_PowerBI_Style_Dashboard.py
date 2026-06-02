from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerBI Style Dashboard",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# CLEAN STYLE
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 45%, #f8fafc 100%);
    color: #111827 !important;
}

html, body, p, span, label, div, li {
    color: #111827 !important;
}

h1 {
    color: #0f172a !important;
    font-size: 42px !important;
    font-weight: 850 !important;
}

h2, h3 {
    color: #1e293b !important;
    font-weight: 800 !important;
}

p {
    color: #334155 !important;
    font-size: 16px !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #cbd5e1;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 28px !important;
    font-weight: 850 !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stToggle"] label {
    color: #111827 !important;
    font-weight: 650 !important;
}

div[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    project_dir = Path(__file__).resolve().parents[2]

    possible_paths = [
        project_dir / "Model_XGBoost" / "xgboost_v3_test_predictions_with_dates.csv",
        project_dir / "dashboard_files" / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
        project_dir / "notebooks" / "Anu" / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    matches = list(project_dir.rglob("xgboost_v3_test_predictions_with_dates.csv"))

    if matches:
        return pd.read_csv(matches[0])

    st.error("Prediction CSV file not found.")
    st.stop()


df = load_data()


# ============================================================
# BASIC PREP
# ============================================================

datetime_col = None

for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        datetime_col = col
        break

if datetime_col is None:
    st.error("No datetime column found.")
    st.stop()

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
df = df.dropna(subset=[datetime_col]).sort_values(datetime_col)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🦆 Duckstradamus Controls")

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

time_window = st.sidebar.radio(
    "Time Window",
    ["24 Hours", "7 Days", "30 Days", "Full Period"],
)

show_map = st.sidebar.toggle("Show NZ Node Map", value=True)
show_table = st.sidebar.toggle("Show Data Table", value=False)

spike_threshold = st.sidebar.slider(
    "High Price Threshold ($/MWh)",
    min_value=0,
    max_value=int(df[actual_col].max()),
    value=min(300, int(df[actual_col].max())),
)


# ============================================================
# DASHBOARD FEATURES
# ============================================================

df["error"] = df[actual_col] - df[pred_col]
df["abs_error"] = df["error"].abs()
df["is_spike"] = df[actual_col] >= spike_threshold

conditions = [
    (df["abs_error"] < 20) & (df[actual_col] < 150),
    (df["abs_error"] < 60) & (df[actual_col] < 250),
]

df["duck_mood"] = np.select(
    conditions,
    ["Calm", "Alert"],
    default="Panic",
)

if time_window == "24 Hours":
    plot_df = df.tail(24).copy()
elif time_window == "7 Days":
    plot_df = df.tail(24 * 7).copy()
elif time_window == "30 Days":
    plot_df = df.tail(24 * 30).copy()
else:
    plot_df = df.copy()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ Duckstradamus NZ Electricity Supply and Forecasting")

st.markdown(
    """
Analyze New Zealand electricity price forecasts, model behaviour, market conditions, and regional node-level patterns.
"""
)


# ============================================================
# KPI CARDS
# ============================================================

latest_price = df[actual_col].iloc[-1]
latest_pred = df[pred_col].iloc[-1]
avg_price = df[actual_col].mean()
highest_price = df[actual_col].max()
high_price_count = int(df["is_spike"].sum())

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("⚡ Current Price", f"${latest_price:,.2f}")
k2.metric("🔮 Forecast Price", f"${latest_pred:,.2f}")
k3.metric("📊 Average Price", f"${avg_price:,.2f}")
k4.metric("🚨 Highest Price", f"${highest_price:,.2f}")
k5.metric("High Price Events", f"{high_price_count:,}")


# ============================================================
# MAIN CHARTS - MODEL PERFORMANCE + FORECAST VIEW
# ============================================================

left, right = st.columns([1, 1.8])

with left:
    st.markdown("### 🏆 Model Performance Summary")

    xgb_rmse = 79.20
    xgb_mae = 66.89
    naive_rmse = 90.18
    lstm_rmse = 86.45

    improvement = ((naive_rmse - xgb_rmse) / naive_rmse) * 100

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    m1.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")
    m2.metric("XGBoost MAE", f"{xgb_mae:.2f}")
    m3.metric("Naive RMSE", f"{naive_rmse:.2f}")
    m4.metric("Improvement", f"{improvement:.1f}%")

    comparison_df = pd.DataFrame(
        {
            "Model": ["XGBoost V3", "LSTM", "Naive Baseline"],
            "RMSE": [xgb_rmse, lstm_rmse, naive_rmse],
        }
    )

    fig_models = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Model Comparison by RMSE",
    )

    fig_models.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        marker_line_width=1,
    )

    fig_models.update_layout(
        template="plotly_white",
        height=360,
        showlegend=False,
        font=dict(color="#111827", size=14),
        title=dict(font=dict(size=20, color="#111827")),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig_models, use_container_width=True)


with right:
    st.markdown("### 📈 Forecast Comparison")

    plot_df = plot_df.copy()
    plot_df["Naive Baseline"] = plot_df[actual_col].shift(24)

    comparison_cols = [actual_col, pred_col, "Naive Baseline"]

    fig_price = px.line(
        plot_df,
        x=datetime_col,
        y=comparison_cols,
        labels={
            "value": "Price ($/MWh)",
            "variable": "",
            datetime_col: "Datetime",
        },
    )

    fig_price.for_each_trace(
        lambda t: t.update(
            name={
                actual_col: "Actual Price",
                pred_col: "XGBoost Forecast",
                "Naive Baseline": "Naive Baseline",
            }.get(t.name, t.name),
            line=dict(width=3),
        )
    )

    fig_price.update_layout(
        template="plotly_white",
        height=470,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
        font=dict(color="#111827", size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=13, color="#111827"),
        ),
    )

    fig_price.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig_price.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

    st.plotly_chart(fig_price, use_container_width=True)

    st.caption(
        "Naive Baseline uses the price from 24 hours earlier as a simple benchmark forecast."
    )

# ============================================================
# NZ NODE MAP
# ============================================================

if show_map:
    st.markdown("---")
    st.markdown("## 🗺️ New Zealand Electricity Node Map")

    node_locations = pd.DataFrame(
        {
            "node": [
                "BEN2201",
                "HAY2201",
                "INV2201",
                "ISL2201",
                "KIK2201",
                "OTA2201",
                "RDF2201",
                "SFD2201",
                "WKM2201",
            ],
            "name": [
                "Benmore",
                "Haywards",
                "Invercargill",
                "Islington",
                "Kikiwa",
                "Otahuhu",
                "Redclyffe",
                "Stratford",
                "Whakamaru",
            ],
            "lat": [
                -44.563,
                -41.166,
                -46.413,
                -43.532,
                -41.760,
                -36.945,
                -39.515,
                -39.337,
                -38.421,
            ],
            "lon": [
                170.198,
                174.981,
                168.353,
                172.560,
                173.726,
                174.838,
                176.869,
                174.284,
                175.812,
            ],
        }
    )

    price_cols = [c for c in df.columns if "el_price" in c.lower()]
    latest_row = df.iloc[-1]

    node_prices = []

    for node in node_locations["node"]:
        matched_cols = [c for c in price_cols if node in c]

        if matched_cols:
            node_prices.append(float(latest_row[matched_cols[0]]))
        else:
            node_prices.append(np.nan)

    node_locations["latest_price"] = node_prices

    if node_locations["latest_price"].isna().all():
        st.warning(
            "Node-level price columns were not found in this CSV, so demo values are shown on the map."
        )

        node_locations["latest_price"] = [
            110,
            145,
            95,
            125,
            135,
            160,
            105,
            120,
            140,
        ]
    else:
        node_locations["latest_price"] = node_locations["latest_price"].fillna(
            node_locations["latest_price"].mean()
        )

    selected_node = st.selectbox(
        "Select a node to inspect",
        node_locations["node"].tolist(),
    )

    selected_info = node_locations[node_locations["node"] == selected_node].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Selected Node", selected_info["node"])
    c2.metric("Location", selected_info["name"])
    c3.metric("Latest Price", f"${selected_info['latest_price']:,.2f}")

    fig_map = px.scatter_mapbox(
        node_locations,
        lat="lat",
        lon="lon",
        size="latest_price",
        color="latest_price",
        hover_name="name",
        hover_data={
            "node": True,
            "latest_price": ":.2f",
            "lat": False,
            "lon": False,
        },
        zoom=4.4,
        height=620,
        title="Latest Electricity Price by NZ Grid Node",
        color_continuous_scale="Turbo",
        size_max=35,
    )

    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(color="#111827"),
        title=dict(font=dict(size=22, color="#111827")),
    )

    st.plotly_chart(fig_map, use_container_width=True)


# ============================================================
# ERROR ANALYSIS
# ============================================================

st.markdown("---")
st.markdown("## 🎯 Prediction Error Distribution")

fig_error = px.histogram(
    plot_df,
    x="error",
    color="duck_mood",
    nbins=50,
    title="Prediction Error by Duck Mood",
    color_discrete_map={
        "Calm": "#22c55e",
        "Alert": "#facc15",
        "Panic": "#ef4444",
    },
)

fig_error.update_layout(
    template="plotly_white",
    height=430,
    font=dict(color="#111827"),
    title=dict(font=dict(size=20, color="#111827")),
    margin=dict(l=20, r=20, t=50, b=20),
)

st.plotly_chart(fig_error, use_container_width=True)


# ============================================================
# MODEL FAILURE MAP
# ============================================================

st.markdown("---")
st.markdown("## 🧭 Where Does the Model Struggle?")

fig_fail = px.scatter(
    plot_df,
    x=actual_col,
    y="abs_error",
    color="duck_mood",
    size="abs_error",
    hover_data=[datetime_col, actual_col, pred_col, "error"],
    title="Actual Price vs Absolute Forecast Error",
    labels={
        actual_col: "Actual Price ($/MWh)",
        "abs_error": "Absolute Error",
    },
    color_discrete_map={
        "Calm": "#22c55e",
        "Alert": "#facc15",
        "Panic": "#ef4444",
    },
)

fig_fail.update_layout(
    template="plotly_white",
    height=520,
    font=dict(color="#111827"),
    title=dict(font=dict(size=22, color="#111827")),
    margin=dict(l=20, r=20, t=50, b=20),
)

st.plotly_chart(fig_fail, use_container_width=True)


# ============================================================
# FORECAST PERFORMANCE ANALYSIS
# ============================================================

st.markdown("---")
st.markdown("## 🎯 Forecast Performance Analysis")

st.markdown(
    """
This section compares the **actual price**, the **model forecast**, and a simple **baseline forecast**.
Use it in the final presentation to show how the model performs under different market conditions.
"""
)

presentation_choice = st.radio(
    "Select Scenario",
    ["Scenario A", "Scenario B"],
    horizontal=True,
)

temp_df = df.copy()
temp_df["naive_price"] = temp_df[actual_col].shift(24)
temp_df = temp_df.dropna(subset=["naive_price"])

weekly = (
    temp_df.set_index(datetime_col)
    .resample("7D")
    .agg(
        actual_mean=(actual_col, "mean"),
        abs_error=("abs_error", "mean"),
    )
    .dropna()
)

if not weekly.empty:
    if presentation_choice == "Scenario A":
        selected_week_start = weekly["abs_error"].idxmin()
    else:
        selected_week_start = weekly["abs_error"].idxmax()

    selected_week_end = selected_week_start + pd.Timedelta(days=7)

    week_df = temp_df[
        (temp_df[datetime_col] >= selected_week_start)
        & (temp_df[datetime_col] < selected_week_end)
    ].copy()

    mae_model = (week_df[actual_col] - week_df[pred_col]).abs().mean()
    mae_naive = (week_df[actual_col] - week_df["naive_price"]).abs().mean()

    rmse_model = np.sqrt(np.mean((week_df[actual_col] - week_df[pred_col]) ** 2))
    rmse_naive = np.sqrt(np.mean((week_df[actual_col] - week_df["naive_price"]) ** 2))

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Model MAE", f"{mae_model:.2f}")
    m2.metric("Baseline MAE", f"{mae_naive:.2f}")
    m3.metric("Model RMSE", f"{rmse_model:.2f}")
    m4.metric("Baseline RMSE", f"{rmse_naive:.2f}")

    fig_week = px.line(
        week_df,
        x=datetime_col,
        y=[actual_col, pred_col, "naive_price"],
        labels={
            "value": "Price ($/MWh)",
            "variable": "Series",
            datetime_col: "Datetime",
        },
    )

    fig_week.for_each_trace(
        lambda t: t.update(
            name={
                actual_col: "Actual Price",
                pred_col: "Model Forecast",
                "naive_price": "Baseline Forecast",
            }.get(t.name, t.name),
            line=dict(width=3),
        )
    )

    fig_week.update_layout(
        template="plotly_white",
        height=600,
        hovermode="x unified",
        title=None,
        legend_title_text="",
        font=dict(size=15, color="#111827"),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_week, use_container_width=True)

else:
    st.info("Not enough data to create weekly presentation view.")


# ============================================================
# DATA TABLE
# ============================================================

if show_table:
    st.markdown("---")
    st.markdown("## 📋 Dashboard Data")

    st.dataframe(
        plot_df[
            [
                datetime_col,
                actual_col,
                pred_col,
                "error",
                "abs_error",
                "duck_mood",
                "is_spike",
            ]
        ],
        use_container_width=True,
    )
