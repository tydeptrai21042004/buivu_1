"""Tree-based extension models."""
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


def build_tree_models(random_state: int = 42) -> dict:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }
