from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from valuation.preprocessing import NUMERIC_FEATURES, build_ols_design_matrix


@dataclass(frozen=True)
class OLSFitResult:
    model: object
    predictions: np.ndarray
    coefficients: pd.DataFrame
    feature_names: list[str]


def fit_ols_hc3(
    *,
    X_train_numeric: np.ndarray,
    X_test_numeric: np.ndarray,
    train_positions: pd.Series,
    test_positions: pd.Series,
    position_levels: list[str],
    y_train: np.ndarray,
) -> OLSFitResult:
    X_train, dummy_names = build_ols_design_matrix(
        X_train_numeric,
        train_positions,
        position_levels=position_levels,
    )
    X_test, _ = build_ols_design_matrix(
        X_test_numeric,
        test_positions,
        position_levels=position_levels,
    )
    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    model = sm.OLS(y_train, X_train).fit(cov_type="HC3")
    predictions = np.asarray(model.predict(X_test), dtype=float)
    feature_names = ["intercept", *NUMERIC_FEATURES, *dummy_names]
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.params,
            "standard_error": model.bse,
            "p_value": model.pvalues,
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    return OLSFitResult(
        model=model,
        predictions=predictions,
        coefficients=coefficients,
        feature_names=feature_names,
    )
