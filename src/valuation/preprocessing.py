from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "target_log_value_change"
POSITION_COLUMN = "position_group"

NUMERIC_FEATURES = [
    "log_previous_market_value",
    "previous_log_value_change",
    "age_at_valuation",
    "age_squared",
    "interval_days",
    "valuation_year",
    "valuation_month_sin",
    "valuation_month_cos",
    "log_minutes",
    "log_appearances",
    "start_share",
    "average_rating",
    "recency_weighted_rating",
    "rating_last_90_days",
    "rating_volatility",
    "recent_rating_trend",
    "threat_component_average",
    "creation_component_average",
    "progression_component_average",
    "retention_component_average",
    "attacking_xpv_component_average",
    "defensive_component_average",
    "finishing_component_average",
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xgot_per90",
    "xa_per90",
    "goals_over_xg_per90",
    "assists_over_xa_per90",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp


@dataclass(frozen=True)
class PreparedMatrices:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_position_code: np.ndarray
    test_position_code: np.ndarray
    position_levels: list[str]
    train_player_code: np.ndarray
    test_player_code: np.ndarray
    player_levels: list[int]
    preprocessor: Pipeline
    split_date: pd.Timestamp


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
) -> ChronologicalSplit:
    if not 0.05 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.05 and 0.50")

    data = frame.copy()
    data["valuation_date"] = pd.to_datetime(data["valuation_date"])
    unique_dates = np.array(sorted(data["valuation_date"].dropna().unique()))
    if len(unique_dates) < 5:
        raise ValueError("Need at least five unique valuation dates")

    split_index = min(
        len(unique_dates) - 2,
        max(1, int(len(unique_dates) * (1.0 - test_fraction))),
    )
    split_date = pd.Timestamp(unique_dates[split_index])
    train = data[data["valuation_date"] <= split_date].copy()
    test = data[data["valuation_date"] > split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set")
    return ChronologicalSplit(train=train, test=test, split_date=split_date)


def build_numeric_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_matrices(
    frame: pd.DataFrame,
    *,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    target_column: str = TARGET_COLUMN,
    test_fraction: float = 0.20,
) -> PreparedMatrices:
    missing = sorted(
        set(numeric_features)
        .union(
            {
                target_column,
                POSITION_COLUMN,
                "valuation_date",
                "transfermarkt_player_id",
            }
        )
        .difference(frame.columns)
    )
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")

    split = chronological_split(frame, test_fraction=test_fraction)
    preprocessor = build_numeric_preprocessor()
    X_train = preprocessor.fit_transform(split.train[list(numeric_features)])
    X_test = preprocessor.transform(split.test[list(numeric_features)])

    # Match the validated notebook: position categories come from the complete
    # modeling frame, while all numeric transforms remain train-only. Reading
    # category labels does not expose the target and prevents a holdout-only
    # position from being silently mapped to the first training category.
    position_levels = sorted(frame[POSITION_COLUMN].dropna().astype(str).unique())
    position_to_code = {value: index for index, value in enumerate(position_levels)}
    fallback_code = 0
    train_codes = (
        split.train[POSITION_COLUMN]
        .astype(str)
        .map(position_to_code)
        .fillna(fallback_code)
        .to_numpy(dtype=np.int32)
    )
    test_codes = (
        split.test[POSITION_COLUMN]
        .astype(str)
        .map(position_to_code)
        .fillna(fallback_code)
        .to_numpy(dtype=np.int32)
    )
    player_levels = sorted(
        split.train["transfermarkt_player_id"].dropna().astype(int).unique().tolist()
    )
    player_to_code = {value: index for index, value in enumerate(player_levels)}
    unknown_player_code = len(player_levels)
    train_player_codes = (
        split.train["transfermarkt_player_id"]
        .astype(int)
        .map(player_to_code)
        .to_numpy(dtype=np.int32)
    )
    test_player_codes = (
        split.test["transfermarkt_player_id"]
        .astype(int)
        .map(player_to_code)
        .fillna(unknown_player_code)
        .to_numpy(dtype=np.int32)
    )

    return PreparedMatrices(
        train_frame=split.train,
        test_frame=split.test,
        X_train=np.asarray(X_train, dtype=np.float64),
        X_test=np.asarray(X_test, dtype=np.float64),
        y_train=split.train[target_column].to_numpy(dtype=np.float64),
        y_test=split.test[target_column].to_numpy(dtype=np.float64),
        train_position_code=train_codes,
        test_position_code=test_codes,
        position_levels=position_levels,
        train_player_code=train_player_codes,
        test_player_code=test_player_codes,
        player_levels=player_levels,
        preprocessor=preprocessor,
        split_date=split.split_date,
    )


def build_ols_design_matrix(
    numeric_matrix: np.ndarray,
    positions: pd.Series,
    *,
    position_levels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    categories = pd.Categorical(positions.astype(str), categories=list(position_levels))
    dummies = pd.get_dummies(categories, prefix="position", drop_first=True, dtype=float)
    matrix = np.column_stack([numeric_matrix, dummies.to_numpy(dtype=float)])
    return matrix, list(dummies.columns)
