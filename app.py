"""KNN-only Streamlit website demo for NASA93 software-effort prediction.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from config import COCOMO_FEATURES
from src.inference import (
    load_artifacts,
    predict_effort,
    predict_effort_batch,
    train_and_save_artifacts,
)


DEFAULT_ARTIFACT_DIR = Path("artifacts")
SAMPLE_DATA = Path("tests/data/sample_nasa93_small.csv")

FEATURE_GROUPS = {
    "Scale factors": ["prec", "flex", "resl", "team", "pmat"],
    "Product factors": ["rely", "data", "cplx", "ruse", "docu"],
    "Platform factors": ["time", "stor", "pvol"],
    "Personnel factors": ["acap", "pcap", "pcon", "apex", "plex", "ltex"],
    "Project factors": ["tool", "site", "sced"],
    "Size": ["kloc"],
}

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
    "Enter the COCOMO/NASA project features below. The app predicts software-development effort "
    "using one saved KNN regression model."
)


@st.cache_resource(show_spinner=False)
def cached_load_artifacts(artifact_dir: str):
    return load_artifacts(artifact_dir)


@st.cache_data(show_spinner=False)
def metadata_to_dataframe(metrics: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(metrics)


with st.sidebar:
    st.header("KNN model setup")
    artifact_dir = st.text_input("Artifact folder", value=str(DEFAULT_ARTIFACT_DIR))

    st.caption("Use a real NASA_93_Sheet.csv for final demo quality. The bundled sample is only for smoke testing.")
    uploaded_training_csv = st.file_uploader("Upload NASA93 training CSV", type=["csv"], key="training_csv")

    train_from_sample = st.button("Train KNN demo artifacts from bundled sample")
    train_from_upload = st.button("Train KNN artifacts from uploaded CSV", disabled=uploaded_training_csv is None)

    if train_from_sample:
        if not SAMPLE_DATA.exists():
            st.error("Bundled sample data not found.")
        else:
            with st.spinner("Training KNN demo model..."):
                metadata = train_and_save_artifacts(SAMPLE_DATA, artifact_dir=artifact_dir)
            st.success(f"KNN artifacts saved. Model: {metadata['model_name']}")
            st.cache_resource.clear()

    if train_from_upload and uploaded_training_csv is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_training_csv.getvalue())
            tmp_path = tmp.name
        with st.spinner("Training KNN model from uploaded CSV..."):
            metadata = train_and_save_artifacts(tmp_path, artifact_dir=artifact_dir)
        st.success(f"KNN artifacts saved. Model: {metadata['model_name']}")
        st.cache_resource.clear()

try:
    model, preprocessor, metadata = cached_load_artifacts(artifact_dir)
except Exception as exc:
    st.warning(str(exc))
    st.info("Train KNN artifacts from the sidebar first, or run `python train_artifacts.py --data path/to/NASA_93_Sheet.csv`.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("Current model")
    st.metric("Model", metadata.get("model_name", "KNN"))
    st.metric("Training samples", metadata.get("n_samples", "Unknown"))

with right:
    st.subheader("KNN test metrics")
    metrics_df = metadata_to_dataframe(metadata.get("metrics", []))
    if not metrics_df.empty:
        show_cols = [col for col in ["Model", "MMRE", "RMSE", "BRE", "MAE", "R2", "Accuracy"] if col in metrics_df.columns]
        st.dataframe(metrics_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No KNN metrics stored in metadata.")

st.divider()

st.subheader("Single-project prediction")
feature_defaults = metadata.get("feature_defaults", {})
feature_min = metadata.get("feature_min", {})
feature_max = metadata.get("feature_max", {})

with st.form("single_prediction_form"):
    feature_values = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(min(3, len(group_features)))
        for i, feature in enumerate(group_features):
            default = float(feature_defaults.get(feature, 1.0 if feature != "kloc" else 10.0))
            min_value = float(feature_min.get(feature, 0.0))
            max_value = float(feature_max.get(feature, max(default * 2.0, default + 1.0)))
            if max_value <= min_value:
                max_value = min_value + 1.0
            with cols[i % len(cols)]:
                feature_values[feature] = st.number_input(
                    label=f"{feature} — {FEATURE_NAMES.get(feature, feature)}",
                    min_value=0.0,
                    value=default,
                    step=0.01 if feature != "kloc" else 1.0,
                    help=f"Observed training range: {min_value:.3f} to {max_value:.3f}",
                )

    submitted = st.form_submit_button("Predict effort")

if submitted:
    try:
        prediction = predict_effort(feature_values, model, preprocessor, metadata)
        st.success(f"Predicted effort: {prediction:,.2f}")
        st.caption("Unit follows the NASA93 effort column, commonly person-months in software-effort datasets.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()

st.subheader("Batch CSV prediction")
st.write("Upload a CSV containing the same 23 feature columns. The app will append `predicted_effort_knn`.")
batch_csv = st.file_uploader("Upload feature CSV for batch prediction", type=["csv"], key="batch_csv")
if batch_csv is not None:
    try:
        batch_df = pd.read_csv(batch_csv)
        output_df = predict_effort_batch(batch_df, model, preprocessor, metadata)
        st.dataframe(output_df, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_effort_knn.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")

with st.expander("Required input columns"):
    st.code(", ".join(COCOMO_FEATURES), language="text")
