# NASA93 KNN Software Effort Prediction Demo

This project contains a simple website demo for predicting software-development effort from NASA93/COCOMO project features.

The web demo intentionally uses **one model only: KNN regression**.  This avoids confusing the user with many model choices and makes the demo easier to explain.

## What the demo does

1. Loads a NASA93 CSV dataset.
2. Selects the 23 COCOMO/NASA input features.
3. Applies preprocessing:
   - median imputation,
   - standard scaling,
   - `log1p(effort)` target transformation.
4. Trains a **KNN regressor**.
5. Saves the trained KNN model and preprocessor.
6. Opens a Streamlit website where users can:
   - enter one project manually and predict effort,
   - upload a CSV and get batch KNN predictions.

## Folder structure

```text
software_effort_knn_demo/
├── app.py
├── train_artifacts.py
├── config.py
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── inference.py
│   └── models/
│       └── knn_model.py
└── tests/
    └── data/sample_nasa93_small.csv
```

The original experiment files are still kept in the project, but the **website demo path uses only KNN**.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train the KNN artifact

Use your real NASA93 dataset:

```bash
python train_artifacts.py --data /path/to/NASA_93_Sheet.csv --artifact-dir artifacts
```

This creates:

```text
artifacts/knn_model.joblib
artifacts/preprocessor.joblib
artifacts/metadata.json
artifacts/clean_training_data.csv
```

For quick checking only, the Streamlit sidebar can also train from the bundled small sample dataset.

## Run the website

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## User input features

The website asks users to enter the same 23 NASA93/COCOMO features:

```text
prec, flex, resl, team, pmat,
rely, data, cplx, ruse, docu,
time, stor, pvol,
acap, pcap, pcon, apex, plex, ltex,
tool, site, sced,
kloc
```

The output is:

```text
predicted effort
```

The unit follows the `effort` column in the NASA93 dataset, commonly interpreted as person-months.

## Batch prediction CSV

For batch prediction, upload a CSV containing all 23 input columns. The app appends:

```text
predicted_effort_knn
```

## Run tests

```bash
pytest -q
```

## Notes

- The web demo uses `KNeighborsRegressor(n_neighbors=3, weights="distance", p=2)`.
- The model predicts `log1p(effort)` internally.
- Final predictions are converted back with `expm1()`.
- No model-selection UI is included because this demo is KNN-only.
