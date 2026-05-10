"""KNN model from the paper-style comparison."""
from sklearn.neighbors import KNeighborsRegressor


def build_knn() -> KNeighborsRegressor:
    return KNeighborsRegressor(n_neighbors=3, weights="distance", p=2)
