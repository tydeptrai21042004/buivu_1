# NASA93 KNN Software Effort Prediction Demo

This project contains a simple Streamlit website demo for predicting software-development effort from a NASA93/COCOMO CSV dataset.

The web demo intentionally uses **one model only: KNN regression**. The interface is also simplified to one workflow:

```text
Upload NASA93 training CSV
        ↓
Train KNN model
        ↓
Show clean training table
        ↓
Choose one row
        ↓
Predict effort for that row
```

## What the demo does

1. Uploads a NASA93 CSV dataset.
2. Checks that the file contains the 23 COCOMO/NASA input features and the target column `effort`.
3. Applies preprocessing:
   - median imputation,
   - standard scaling,
   - `log1p(effort)` target transformation.
4. Trains a **KNN regressor**.
5. Saves the trained KNN model and preprocessor.
6. Displays the cleaned training table.
7. Lets the user choose one row from the table.
8. Predicts the effort for the selected row.

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

## Run the website

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Training CSV format

Your CSV must contain these 23 input columns:

```text
prec, flex, resl, team, pmat,
rely, data, cplx, ruse, docu,
time, stor, pvol,
acap, pcap, pcon, apex, plex, ltex,
tool, site, sced,
kloc
```

It must also contain the target column:

```text
effort
```

So the required training columns are:

```text
prec, flex, resl, team, pmat, rely, data, cplx, ruse, docu,
time, stor, pvol, acap, pcap, pcon, apex, plex, ltex,
tool, site, sced, kloc, effort
```

## Optional CLI training

You can also train the KNN artifact from the terminal:

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

## App workflow

After opening the website:

1. Upload your NASA93 CSV.
2. Click **Train KNN from this CSV**.
3. The app shows the cleaned table.
4. Choose one row using **Choose row for prediction**.
5. Click **Predict effort for selected row**.
6. The app shows:
   - predicted effort,
   - actual effort,
   - absolute error,
   - percent error.

## Run tests

```bash
pytest -q
```

## Notes

- The web demo uses `KNeighborsRegressor(n_neighbors=3, weights="distance", p=2)`.
- The model predicts `log1p(effort)` internally.
- Final predictions are converted back with `expm1()`.
- No manual input form, batch-prediction mode, or model-selection UI is included.
