# Software Effort / Cost Prediction on NASA93

This project is a modular implementation of the paper workflow:

1. **Problem definition**: predict required software development effort/cost.
2. **Dataset**: NASA93 software project dataset.
3. **Preprocessing**: select COCOMO/NASA features, impute missing values, standardize inputs, and use `log1p(effort)` as the training target.
4. **Techniques applied**: KNN, Cascade Forward Neural Network, Elman Neural Network, plus additional classical and neural models.
5. **Evaluation**: MMRE, RMSE, BRE, MAE, R², and accuracy based on `1 - MMRE`.
6. **Comparison and conclusion**: rank models using MMRE, RMSE, and BRE.

The implementation keeps the same logic as your single-cell Kaggle code, but separates the code into reusable modules and model files.

## Folder structure

```text
software_effort_paper_project/
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── plotting.py
│   ├── pipeline.py
│   └── models/
│       ├── classical_models.py
│       ├── knn_model.py
│       ├── tree_models.py
│       ├── neural_base.py
│       ├── mlp_model.py
│       ├── cascade_model.py
│       └── sequence_models.py
└── tests/
    ├── test_metrics.py
    ├── test_data_schema.py
    ├── test_pipeline_smoke.py
    └── data/sample_nasa93_small.csv
```

## How to run on Kaggle

Upload or attach the Kaggle dataset that contains:

```text
/kaggle/input/datasets/asmasadaqat/nasa93/NASA_93_Sheet.csv
```

Then run:

```bash
python main.py --data /kaggle/input/datasets/asmasadaqat/nasa93/NASA_93_Sheet.csv --output outputs
```

To run only classical models without TensorFlow neural networks:

```bash
python main.py --data /kaggle/input/datasets/asmasadaqat/nasa93/NASA_93_Sheet.csv --output outputs --skip-neural
```

## How to run tests

```bash
pytest -q
```

The tests use a small built-in sample dataset in `tests/data/sample_nasa93_small.csv`, so they do not need the Kaggle dataset.

## Main outputs

After running `main.py`, the following files are saved in the output folder:

```text
corrected_log_effort_model_comparison.csv
corrected_best_model_predictions.csv
actual_vs_predicted.png
mmre_comparison.png
rmse_comparison.png
bre_comparison.png
```

## Notes

- The target is `effort`.
- The model is trained on `log1p(effort)` and predictions are converted back using `expm1`.
- This avoids negative or zero effort prediction problems.
- Model selection uses the average rank of MMRE, RMSE, and BRE, not RMSE alone.
