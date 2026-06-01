from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Live ML Prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Live ML Prediction Lab")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "notebooks" / "xgboost_v3_champion_model.pkl"
FEATURES_PATH = PROJECT_DIR / "notebooks" / "v3_features.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


model, feature_cols = load_model()

st.success("✅ XGBoost V3 Champion Model loaded successfully")

uploaded_file = st.file_uploader(
    "Upload new market data CSV",
    type=["csv"],
)

if uploaded_file is not None:

    new_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(new_df.head(), use_container_width=True)

    st.write("Uploaded columns:", list(new_df.columns))
    st.write("Required model features:", len(feature_cols))

    missing_features = [
        col for col in feature_cols
        if col not in new_df.columns
    ]

    extra_features = [
        col for col in new_df.columns
        if col not in feature_cols
    ]

    if missing_features:
        st.error("❌ Uploaded CSV is missing required model features.")
        st.write(missing_features)

    else:
        X_new = new_df[feature_cols]

        if st.button("🦆 Run Live Prediction"):

            predictions = model.predict(X_new)

            result_df = new_df.copy()
            result_df["xgboost_v3_prediction"] = predictions

            st.success("✅ Prediction completed!")

            st.subheader("Prediction Results")
            st.dataframe(result_df.head(50), use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="duckstradamus_live_predictions.csv",
                mime="text/csv",
            )

    with st.expander("Feature Check"):
        st.write("Missing features:", missing_features)
        st.write("Extra uploaded columns:", extra_features)


st.markdown("---")

st.subheader("🦆 Try Live Prediction Using Existing Test Data")

test_data_path = PROJECT_DIR / "notebooks" / "xgboost_v3_test_predictions_with_dates.csv"

test_df = pd.read_csv(test_data_path)

st.write("Test data loaded:", test_df.shape)

row_number = st.slider(
    "Choose a row for live prediction",
    0,
    len(test_df) - 1,
    0
)

selected_row = test_df.iloc[[row_number]]

st.subheader("Selected Row")
st.dataframe(selected_row)

missing_features = [
    col for col in feature_cols
    if col not in test_df.columns
]

if missing_features:
    st.error("This test file does not contain all model input features.")
    st.write("Missing features:")
    st.write(missing_features)

else:
    X_sample = selected_row[feature_cols]

    if st.button("🦆 Predict This Row"):

        pred = model.predict(X_sample)[0]

        st.metric(
            "Live Model Prediction",
            f"${pred:,.2f}/MWh"
        )

        if pred < 100:
            st.success("🟢 Duck Mood: Calm market")
        elif pred < 200:
            st.warning("🟡 Duck Mood: Normal market")
        else:
            st.error("🔴 Duck Mood: Price spike risk")
