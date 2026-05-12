"""KNN-only Streamlit website demo for NASA93 software-effort prediction.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from config import COCOMO_FEATURES, TARGET_COLUMN
from src.inference import (
    load_artifacts,
    predict_effort,
    train_and_save_artifacts,
)


DEFAULT_ARTIFACT_DIR = Path("artifacts")
SAMPLE_DATA = Path("tests/data/sample_nasa93_small.csv")

FEATURE_NAMES = {
    "prec": "Precedentedness",
    "flex": "Development flexibility",
    "resl": "Architecture/risk resolution",
    "team": "Team cohesion",
    "pmat": "Process maturity",
    "rely": "Required reliability",
    "data": "Database size",
    "cplx": "Product complexity",
    "ruse": "Required reuse",
    "docu": "Documentation match",
    "time": "Execution time constraint",
    "stor": "Storage constraint",
    "pvol": "Platform volatility",
    "acap": "Analyst capability",
    "pcap": "Programmer capability",
    "pcon": "Personnel continuity",
    "apex": "Application experience",
    "plex": "Platform experience",
    "ltex": "Language/tool experience",
    "tool": "Tool support",
    "site": "Multisite development",
    "sced": "Schedule constraint",
    "kloc": "Code size in KLOC",
}


st.set_page_config(
    page_title="NASA93 KNN Effort Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("NASA93 KNN Software Effort Prediction Demo")
st.write(
    "Upload one NASA93 training CSV, train the KNN model, then choose one row from the table "
    "and predict its software-development effort."
)


@st.cache_resource(show_spinner=False)
def cached_load_artifacts(artifact_dir: str):
    return load_artifacts(artifact_dir)


@st.cache_data(show_spinner=False)
def load_clean_training_table(artifact_dir: str) -> pd.DataFrame:
    table_path = Path(artifact_dir) / "clean_training_data.csv"
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_csv(table_path)


@st.cache_data(show_spinner=False)
def metadata_to_dataframe(metrics: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(metrics)


artifact_dir = st.sidebar.text_input("Artifact folder", value=str(DEFAULT_ARTIFACT_DIR))

st.header("1. Upload training CSV")
st.caption(
    "The CSV must contain the 23 NASA93/COCOMO feature columns and the target column `effort`. "
    "After training, the app saves `knn_model.joblib`, `preprocessor.joblib`, and `clean_training_data.csv`."
)

required_columns = COCOMO_FEATURES + [TARGET_COLUMN]
with st.expander("Required training CSV columns"):
    st.code(", ".join(required_columns), language="text")

uploaded_training_csv = st.file_uploader(
    "Upload NASA93 training CSV",
    type=["csv"],
    key="training_csv",
)

if uploaded_training_csv is not None:
    try:
        uploaded_preview = pd.read_csv(uploaded_training_csv)
        uploaded_preview.columns = [str(c).strip().lower() for c in uploaded_preview.columns]
        st.subheader("Uploaded CSV preview")
        st.dataframe(uploaded_preview.head(20), use_container_width=True, hide_index=True)

        missing_cols = [col for col in required_columns if col not in uploaded_preview.columns]
        if missing_cols:
            st.error(f"This CSV is missing required column(s): {missing_cols}")
        else:
            if st.button("Train KNN from this CSV", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_training_csv.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Training KNN model from uploaded CSV..."):
                    metadata = train_and_save_artifacts(tmp_path, artifact_dir=artifact_dir)

                st.success(
                    f"KNN training completed. Saved artifacts to `{artifact_dir}`. "
                    f"Clean rows used: {metadata['n_samples']}."
                )
                st.cache_resource.clear()
                st.cache_data.clear()
    except Exception as exc:
        st.error(f"Could not read or train from the uploaded CSV: {exc}")
else:
    st.info("Upload your NASA93 CSV above. For quick testing, you can use the sample button below.")
    if st.button("Train KNN from bundled sample CSV"):
        if not SAMPLE_DATA.exists():
            st.error("Bundled sample data not found.")
        else:
            with st.spinner("Training KNN demo model from bundled sample..."):
                metadata = train_and_save_artifacts(SAMPLE_DATA, artifact_dir=artifact_dir)
            st.success(
                f"Sample KNN training completed. Saved artifacts to `{artifact_dir}`. "
                f"Clean rows used: {metadata['n_samples']}."
            )
            st.cache_resource.clear()
            st.cache_data.clear()

st.divider()

try:
    model, preprocessor, metadata = cached_load_artifacts(artifact_dir)
except Exception as exc:
    st.warning("No trained KNN artifacts are available yet.")
    st.caption(str(exc))
    st.stop()

st.header("2. Current KNN model")
left, right = st.columns([1, 2])

with left:
    st.metric("Model", metadata.get("model_name", "KNN"))
    st.metric("Clean training rows", metadata.get("n_samples", "Unknown"))

with right:
    metrics_df = metadata_to_dataframe(metadata.get("metrics", []))
    if not metrics_df.empty:
        show_cols = [
            col
            for col in ["Model", "MMRE", "RMSE", "BRE", "MAE", "R2", "Accuracy"]
            if col in metrics_df.columns
        ]
        st.dataframe(metrics_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No KNN metrics stored in metadata.")

st.divider()
st.header("3. Choose one row from the training table")

training_df = load_clean_training_table(artifact_dir)
if training_df.empty:
    st.error("The clean training table was not found. Please train from a CSV first.")
    st.stop()

available_cols = [col for col in required_columns if col in training_df.columns]
display_df = training_df[available_cols].copy()
display_df.insert(0, "row_id", range(len(display_df)))

st.write("Select a row from this table. The selected row's 23 feature values will be sent to the KNN model.")
st.dataframe(display_df, use_container_width=True, hide_index=True, height=360)

row_ids = display_df["row_id"].tolist()

row_id = st.selectbox(
    "Choose row for prediction",
    row_ids,
    format_func=lambda idx: (
        f"Row {idx} | KLOC={training_df.loc[idx, 'kloc']:.3f}"
        + (f" | actual effort={training_df.loc[idx, TARGET_COLUMN]:.3f}" if TARGET_COLUMN in training_df.columns else "")
    ),
)

selected_row = training_df.loc[int(row_id)]
selected_features = selected_row[COCOMO_FEATURES].to_dict()

st.subheader("Selected feature values")
selected_feature_table = pd.DataFrame(
    {
        "feature": COCOMO_FEATURES,
        "meaning": [FEATURE_NAMES.get(feature, feature) for feature in COCOMO_FEATURES],
        "value": [selected_features[feature] for feature in COCOMO_FEATURES],
    }
)
st.dataframe(selected_feature_table, use_container_width=True, hide_index=True, height=360)

if st.button("Predict effort for selected row", type="primary"):
    try:
        prediction = predict_effort(selected_features, model, preprocessor, metadata)
        st.success(f"Predicted effort by KNN: {prediction:,.2f}")

        if TARGET_COLUMN in training_df.columns:
            actual_effort = float(selected_row[TARGET_COLUMN])
            abs_error = abs(prediction - actual_effort)
            percent_error = abs_error / actual_effort * 100 if actual_effort != 0 else None

            c1, c2, c3 = st.columns(3)
            c1.metric("Actual effort", f"{actual_effort:,.2f}")
            c2.metric("Absolute error", f"{abs_error:,.2f}")
            if percent_error is not None:
                c3.metric("Percent error", f"{percent_error:,.2f}%")

        st.caption("The unit follows the NASA93 `effort` column, commonly person-months.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
