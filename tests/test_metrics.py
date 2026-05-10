import numpy as np

from src.metrics import mmre, bre, rmse, evaluate_regression
from src.preprocessing import inverse_log_prediction


def test_mmre_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert mmre(y, y) == 0.0


def test_inverse_log_prediction_is_positive():
    pred = inverse_log_prediction(np.array([-100.0, 0.0, 1.0]))
    assert np.all(pred > 0)


def test_evaluate_regression_has_required_keys():
    out = evaluate_regression([10, 20, 30], [11, 19, 29])
    for key in ["MMRE", "RMSE", "BRE", "MAE", "R2", "Accuracy (%)"]:
        assert key in out
