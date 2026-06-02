from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Comparison & Live Prediction Lab",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# STYLE
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

h1, h2, h3 {
    color: #0f172a !important;
    font-weight: 850 !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 850 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATHS = [
    PROJECT_DIR / "dashboard_files" / "notebooks" / "xgboost_v3_champion_model.pkl",
    PROJECT_DIR / "Model_XGBoost" / "xgboost_v3_champion_model.pkl",
    PROJECT_DIR / "notebooks" / "Anu" / "notebooks" / "xgboost_v3_champion_model.pkl",
]

FEATURE_PATHS = [
    PROJECT_DIR / "dashboard_files" / "notebooks" / "v3_features.pkl",
    PROJECT_DIR / "Model_XGBoost" / "v3_features.pkl",
    PROJECT_DIR / "notebooks" / "Anu" / "notebooks" / "v3_features.pkl",
]

TEST_DATA_PATHS = [
    PROJECT_DIR / "dashboard_files" / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
    PROJECT_DIR / "Model_XGBoost" / "xgboost_v3_test_predictions_with_dates.csv",
    PROJECT_DIR / "notebooks" / "Anu" / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv",
]


# ============================================================
# HELPERS
# ============================================================

def first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model_and_features():
    model_path = first_existing_path(MODEL_PATHS)
    feature_path = first_existing_path(FEATURE_PATHS)

    model = None
    feature_cols = None

    if model_path is not None:
        model = joblib.load(model_path)

    if feature_path is not None:
        feature_cols = joblib.load(feature_path)

    return model, feature_cols, model_path, feature_path


@st.cache_data
def load_test_data():
    test_path = first_existing_path(TEST_DATA_PATHS)

    if test_path is None:
        return None, None

    return pd.read_csv(test_path), test_path


def find_datetime_col(data):
    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def find_actual_pred_cols(data):
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    actual_col = None
    pred_col = None

    for col in data.columns:
        if col.lower() in ["actual_price", "actual", "y_test", "true_price"]:
            actual_col = col
        if col.lower() in ["predicted_price", "prediction", "pred", "y_pred"]:
            pred_col = col

    if actual_col is None and len(numeric_cols) >= 1:
        actual_col = numeric_cols[0]

    if pred_col is None and len(numeric_cols) >= 2:
        pred_col = numeric_cols[1]

    return actual_col, pred_col


def add_error_columns(data, actual_col, pred_col):
    data = data.copy()
    data["error"] = data[actual_col] - data[pred_col]
    data["absolute_error"] = data["error"].abs()
    return data


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Model Comparison & Live Prediction Lab")

st.markdown(
    """
This page compares model performance, visualizes prediction quality, and allows live prediction
when a CSV with the required engineered features is available.
"""
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

model, feature_cols, model_path, feature_path = load_model_and_features()
test_df, test_data_path = load_test_data()

if model is not None:
    st.success(f"✅ XGBoost V3 Champion Model loaded successfully")
else:
    st.warning("⚠️ XGBoost model file was not found.")

if feature_cols is not None:
    st.success(f"✅ Feature list loaded successfully: {len(feature_cols)} features")
else:
    st.warning("⚠️ Feature list file was not found.")

if test_df is not None:
    st.info(f"Test prediction data loaded: `{test_df.shape[0]:,}` rows × `{test_df.shape[1]}` columns")
else:
    st.warning("⚠️ Test prediction CSV was not found.")


# ============================================================
# MODEL COMPARISON SUMMARY
# ============================================================

st.markdown("---")
st.markdown("## 🏆 Model Comparison Summary")

model_results = pd.DataFrame(
    {
        "Model": [
            "XGBoost V3",
            "LightGBM",
            "GRU",
            "Naive Baseline",
            "Weekly Baseline",
        ],
        "RMSE": [
            79.20,
            81.09,
            86.45,
            90.18,
            116.20,
        ],
    }
)

best_model = model_results.loc[model_results["RMSE"].idxmin()]
naive_rmse = model_results.loc[model_results["Model"] == "Naive Baseline", "RMSE"].iloc[0]
improvement = ((naive_rmse - best_model["RMSE"]) / naive_rmse) * 100

c1, c2, c3 = st.columns(3)

c1.metric("Best Model", best_model["Model"])
c2.metric("Best RMSE", f"{best_model['RMSE']:.2f}")
c3.metric("Improvement vs Naive", f"{improvement:.1f}%")

fig_model = px.bar(
    model_results.sort_values("RMSE"),
    x="Model",
    y="RMSE",
    text="RMSE",
    title="Model RMSE Comparison",
)

fig_model.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig_model.update_layout(
    template="plotly_white",
    height=420,
    showlegend=False,
    font=dict(color="#111827", size=14),
    title=dict(font=dict(size=22, color="#111827")),
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_model, use_container_width=True)


# ============================================================
# PREDICTION QUALITY VISUALS
# ============================================================

if test_df is not None:

    datetime_col = find_datetime_col(test_df)
    actual_col, pred_col = find_actual_pred_cols(test_df)

    if actual_col is not None and pred_col is not None:

        plot_df = test_df.copy()

        if datetime_col is not None:
            plot_df[datetime_col] = pd.to_datetime(plot_df[datetime_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[datetime_col]).sort_values(datetime_col)

        plot_df = add_error_columns(plot_df, actual_col, pred_col)

        st.markdown("---")
        st.markdown("## 📈 Actual vs XGBoost Prediction")

        if datetime_col is not None:
            fig_pred = px.line(
                plot_df,
                x=datetime_col,
                y=[actual_col, pred_col],
                labels={
                    "value": "Price ($/MWh)",
                    "variable": "",
                    datetime_col: "Datetime",
                },
            )
        else:
            fig_pred = px.line(
                plot_df.reset_index(),
                x="index",
                y=[actual_col, pred_col],
                labels={
                    "value": "Price ($/MWh)",
                    "variable": "",
                    "index": "Row Index",
                },
            )

        fig_pred.for_each_trace(
            lambda t: t.update(
                name={
                    actual_col: "Actual Price",
                    pred_col: "XGBoost Forecast",
                }.get(t.name, t.name),
                line=dict(width=3),
            )
        )

        fig_pred.update_layout(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            legend_title_text="",
            font=dict(color="#111827", size=14),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("---")
        st.markdown("## 🎯 Prediction Error Analysis")

        e1, e2, e3, e4 = st.columns(4)

        rmse = np.sqrt(np.mean(plot_df["error"] ** 2))
        mae = plot_df["absolute_error"].mean()
        mean_error = plot_df["error"].mean()
        max_error = plot_df["absolute_error"].max()

        e1.metric("RMSE", f"{rmse:.2f}")
        e2.metric("MAE", f"{mae:.2f}")
        e3.metric("Mean Error", f"{mean_error:.2f}")
        e4.metric("Max Abs Error", f"{max_error:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            fig_error = px.histogram(
                plot_df,
                x="absolute_error",
                nbins=60,
                title="Absolute Error Distribution",
            )

            fig_error.update_layout(
                template="plotly_white",
                height=420,
                font=dict(color="#111827"),
                title=dict(font=dict(size=20, color="#111827")),
            )

            st.plotly_chart(fig_error, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                plot_df,
                x=actual_col,
                y="absolute_error",
                title="Actual Price vs Absolute Error",
                labels={
                    actual_col: "Actual Price ($/MWh)",
                    "absolute_error": "Absolute Error",
                },
            )

            fig_scatter.update_layout(
                template="plotly_white",
                height=420,
                font=dict(color="#111827"),
                title=dict(font=dict(size=20, color="#111827")),
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.markdown("## 🚨 Largest Forecast Errors")

        top_errors = (
            plot_df.sort_values("absolute_error", ascending=False)
            .head(20)
        )

        display_cols = []

        if datetime_col is not None:
            display_cols.append(datetime_col)

        display_cols += [actual_col, pred_col, "error", "absolute_error"]

        st.dataframe(
            top_errors[display_cols],
            use_container_width=True,
        )

    else:
        st.warning("Could not identify actual and predicted price columns in the test CSV.")


# ============================================================
# LIVE PREDICTION USING EXISTING TEST DATA
# ============================================================

st.markdown("---")
st.markdown("## 🦆 Try Live Prediction Using Existing Test Data")

if test_df is not None and model is not None and feature_cols is not None:

    st.write(f"Test data loaded: `{test_df.shape}`")

    required_features = list(feature_cols)

    available_features = [c for c in required_features if c in test_df.columns]
    missing_features = [c for c in required_features if c not in test_df.columns]

    if len(missing_features) > 0:
        st.warning(
            f"""
            Live model prediction cannot run from this CSV because it does not contain the full engineered feature set.

            Required features: {len(required_features)}
            Available features: {len(available_features)}
            Missing features: {len(missing_features)}

            The charts above still work because they use saved prediction results.
            """
        )

        with st.expander("Show missing feature columns"):
            st.write(missing_features)

    else:
        selected_row = st.slider(
            "Choose a row for live prediction",
            min_value=0,
            max_value=len(test_df) - 1,
            value=min(0, len(test_df) - 1),
        )

        row = test_df.iloc[[selected_row]]
        X_live = row[required_features]

        prediction = model.predict(X_live)[0]

        p1, p2 = st.columns(2)

        p1.metric("Live Model Prediction", f"${prediction:,.2f}/MWh")

        if "actual_price" in test_df.columns:
            actual_value = row["actual_price"].iloc[0]
            p2.metric("Actual Price", f"${actual_value:,.2f}/MWh")
        else:
            p2.metric("Selected Row", selected_row)

        st.markdown("### Selected Row")
        st.dataframe(row, use_container_width=True)

else:
    st.info(
        "Live prediction requires model file, feature list, and a CSV containing all engineered features."
    )


# ============================================================
# UPLOAD NEW MARKET DATA
# ============================================================

st.markdown("---")
st.markdown("## 📤 Upload New Market Data CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV with engineered model features",
    type=["csv"],
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.write(f"Uploaded data shape: `{uploaded_df.shape}`")
    st.dataframe(uploaded_df.head(), use_container_width=True)

    if model is not None and feature_cols is not None:

        required_features = list(feature_cols)

        missing_cols = [c for c in required_features if c not in uploaded_df.columns]

        if missing_cols:
            st.error(
                f"""
                This uploaded CSV cannot be used for live prediction because it is missing
                `{len(missing_cols)}` required feature columns.
                """
            )

            with st.expander("Show missing columns"):
                st.write(missing_cols)

        else:
            X_upload = uploaded_df[required_features]
            uploaded_df["live_prediction"] = model.predict(X_upload)

            st.success("Prediction completed successfully.")

            st.dataframe(uploaded_df.head(50), use_container_width=True)

            st.download_button(
                label="Download predictions as CSV",
                data=uploaded_df.to_csv(index=False),
                file_name="live_predictions.csv",
                mime="text/csv",
            )
