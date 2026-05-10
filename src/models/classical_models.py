"""Classical machine-learning models."""
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.svm import SVR
from .knn_model import build_knn


def build_classical_models() -> dict:
    return {
        "KNN": build_knn(),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Huber Regression": HuberRegressor(),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.05),
    }
