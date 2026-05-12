"""KNN-only Streamlit website demo for NASA93 software-effort prediction.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import read_csv_safely
from config import COCOMO_FEATURES, TARGET_COLUMN
from src.inference import (
    load_artifacts,
    predict_effort,
    train_and_save_artifacts,
)
from src.preprocessing import inverse_log_prediction


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
    "Upload one NASA93 training CSV, train the KNN model, inspect the plots, "
    "then choose one row from the table and predict its software-development effort."
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


def build_prediction_table(training_df: pd.DataFrame, model, preprocessor, metadata: dict) -> pd.DataFrame:
    """Predict effort for every cleaned row so the UI can draw model-performance plots."""
    features = metadata.get("features", COCOMO_FEATURES)
    X = training_df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_scaled = preprocessor.transform(X)
    pred_log = model.predict(X_scaled)
    pred_effort = inverse_log_prediction(pred_log)

    plot_df = training_df.copy()
    plot_df.insert(0, "row_id", range(len(plot_df)))
    plot_df["predicted_effort_knn"] = pred_effort

    if TARGET_COLUMN in plot_df.columns:
        plot_df["absolute_error"] = (plot_df["predicted_effort_knn"] - plot_df[TARGET_COLUMN]).abs()
        plot_df["percent_error"] = np.where(
            plot_df[TARGET_COLUMN].astype(float) != 0,
            plot_df["absolute_error"] / plot_df[TARGET_COLUMN].astype(float) * 100,
            np.nan,
        )
    return plot_df


def show_effort_distribution(plot_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(plot_df[TARGET_COLUMN].dropna(), bins=min(15, max(5, len(plot_df) // 5)))
    ax.set_title("Distribution of actual effort")
    ax.set_xlabel("Actual effort")
    ax.set_ylabel("Number of projects")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_kloc_effort_scatter(plot_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(plot_df["kloc"], plot_df[TARGET_COLUMN])
    ax.set_title("Project size versus effort")
    ax.set_xlabel("KLOC")
    ax.set_ylabel("Actual effort")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_actual_vs_predicted(plot_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(plot_df[TARGET_COLUMN], plot_df["predicted_effort_knn"])
    low = float(min(plot_df[TARGET_COLUMN].min(), plot_df["predicted_effort_knn"].min()))
    high = float(max(plot_df[TARGET_COLUMN].max(), plot_df["predicted_effort_knn"].max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Actual effort versus KNN-predicted effort")
    ax.set_xlabel("Actual effort")
    ax.set_ylabel("Predicted effort")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_error_plot(plot_df: pd.DataFrame):
    error_df = plot_df[["row_id", "absolute_error"]].sort_values("absolute_error", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(error_df["row_id"].astype(str), error_df["absolute_error"])
    ax.set_title("Top 15 rows with largest absolute prediction error")
    ax.set_xlabel("Row ID")
    ax.set_ylabel("Absolute error")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_feature_profile(selected_features: dict):
    feature_profile = pd.DataFrame(
        {
            "feature": COCOMO_FEATURES,
            "value": [float(selected_features[feature]) for feature in COCOMO_FEATURES],
        }
    ).set_index("feature")
    st.bar_chart(feature_profile, use_container_width=True)


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
        uploaded_preview = read_csv_safely(uploaded_training_csv)
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

training_df = load_clean_training_table(artifact_dir)
if training_df.empty:
    st.error("The clean training table was not found. Please train from a CSV first.")
    st.stop()

plot_df = build_prediction_table(training_df, model, preprocessor, metadata)

st.divider()
st.header("3. Visual analysis")
st.caption("These plots make the demo easier to understand after the KNN model is trained.")

summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric("Average effort", f"{plot_df[TARGET_COLUMN].mean():,.2f}")
summary_2.metric("Median effort", f"{plot_df[TARGET_COLUMN].median():,.2f}")
summary_3.metric("Average KLOC", f"{plot_df['kloc'].mean():,.2f}")
summary_4.metric("Average abs. error", f"{plot_df['absolute_error'].mean():,.2f}")

tab_overview, tab_prediction, tab_error, tab_table = st.tabs(
    ["Dataset overview", "Actual vs predicted", "Prediction error", "Prediction table"]
)

with tab_overview:
    col_a, col_b = st.columns(2)
    with col_a:
        show_effort_distribution(plot_df)
    with col_b:
        show_kloc_effort_scatter(plot_df)

with tab_prediction:
    show_actual_vs_predicted(plot_df)
    st.caption(
        "Points closer to the dashed diagonal line mean the KNN prediction is closer to the actual effort."
    )

with tab_error:
    show_error_plot(plot_df)
    st.caption("This helps you see which rows are harder for the KNN model to predict.")

with tab_table:
    pred_cols = ["row_id", "kloc", TARGET_COLUMN, "predicted_effort_knn", "absolute_error", "percent_error"]
    st.dataframe(plot_df[pred_cols], use_container_width=True, hide_index=True, height=360)

st.divider()
st.header("4. Choose one row from the training table")

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

with st.expander("Show selected-row feature profile plot"):
    show_feature_profile(selected_features)

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
