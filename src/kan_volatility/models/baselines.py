import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class NaiveVolatilityBaseline:
    """
    Naive volatility forecasting baseline.

    This model predicts that future log realized volatility will equal the
    current log realized volatility implied by a chosen realized-volatility
    feature.

    For example, if reference_column='spy_rv_5d', then:

        prediction = log(spy_rv_5d)
    
    This is a strong simple baseline because volatility tends to cluster.
    """
    def __init__(self, reference_column: str = "spy_rv_5d") -> None:
        self.reference_column = reference_column

    def fit(self, X, y=None):
        """
        Included for compatibility with sklearn-style usage.
        """
        if self.reference_column not in X.columns:
            raise ValueError(
                f"Reference column '{self.reference_column}' not found in X."
            )

        return self

    def predict(self, X):
        """
        Predict log future volatility using current realized volatility.
        """
        if self.reference_column not in X.columns:
            raise ValueError(
                f"Reference column '{self.reference_column}' not found in X."
            )
        
        refernce_vol = X[self.reference_column].to_numpy()
        # Avoid log(0)
        refernce_vol = np.maximum(refernce_vol, 1e-8)

        return np.log(refernce_vol)

def make_linear_regression_model() -> Pipeline:
    """
    Linear regression with standardized features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

def make_ridge_model(alpha: float = 1.0) -> Pipeline:
    """
    Ridge regression with standardized features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )

def make_random_forest_model(
        n_estimators: int = 300,
        max_depth: int | None = 6,
        random_state: int = 42,
) -> RandomForestRegressor:
    """
    Random forest regression baseline.

    We keep max_depth modest to reduce overfitting.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )