from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _movement_class(values: np.ndarray) -> np.ndarray:
    """Classify market-value movement as decrease, stable, or increase."""
    values = np.asarray(values, dtype=float)
    increase_threshold = np.log(1.10)
    decrease_threshold = np.log(0.90)

    return np.where(
        values > increase_threshold,
        1,
        np.where(values < decrease_threshold, -1, 0),
    )


def regression_metrics(
    name: str, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int | str]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    correlation = (
        float("nan")
        if np.ptp(actual) <= 1e-12 or np.ptp(predicted) <= 1e-12
        else pd.Series(actual).corr(pd.Series(predicted), method="spearman")
    )
    return {
        "model": name,
        "observations": len(actual),
        "mae_log_change": float(mae),
        "approx_mae_percent": float(np.expm1(mae)),
        "rmse_log_change": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "direction_accuracy": float(
            np.mean(_movement_class(actual) == _movement_class(predicted))
        ),
        "spearman_correlation": None if pd.isna(correlation) else float(correlation),
        "mean_error": float(np.mean(actual - predicted)),
    }


def calibration_by_decile(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    frame["prediction_decile"] = pd.qcut(
        frame["predicted"], q=10, labels=False, duplicates="drop"
    )
    return (
        frame.groupby("prediction_decile", observed=True)
        .agg(
            observations=("actual", "size"),
            mean_predicted_change=("predicted", "mean"),
            mean_actual_change=("actual", "mean"),
            actual_increase_rate=("actual", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )


def bootstrap_mae_difference(
    actual: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    *,
    samples: int = 5_000,
    seed: int = 42,
) -> dict[str, float]:
    actual = np.asarray(actual)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(actual), size=(samples, len(actual)))
    difference = np.mean(
        np.abs(actual[indices] - predictions_a[indices]), axis=1
    ) - np.mean(np.abs(actual[indices] - predictions_b[indices]), axis=1)
    return {
        "mean_mae_difference": float(difference.mean()),
        "lower_95": float(np.quantile(difference, 0.025)),
        "upper_95": float(np.quantile(difference, 0.975)),
        "probability_model_a_better": float((difference < 0).mean()),
    }


def save_evaluation_plots(
    *,
    actual: np.ndarray,
    selected_predictions: np.ndarray,
    model_predictions: Mapping[str, np.ndarray],
    calibration: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, selected_predictions, alpha=0.3)
    lower = min(float(np.min(actual)), float(np.min(selected_predictions)))
    upper = max(float(np.max(actual)), float(np.max(selected_predictions)))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual log valuation change")
    plt.ylabel("Predicted log valuation change")
    plt.title("Actual versus predicted valuation change")
    plt.tight_layout()
    plt.savefig(output_directory / "actual_vs_predicted.png", dpi=160)
    plt.close()

    residuals = actual - selected_predictions
    plt.figure(figsize=(9, 5))
    plt.hist(residuals, bins=50, density=True)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Selected model residual distribution")
    plt.tight_layout()
    plt.savefig(output_directory / "residual_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        calibration["prediction_decile"],
        calibration["mean_predicted_change"],
        marker="o",
        label="Predicted",
    )
    plt.plot(
        calibration["prediction_decile"],
        calibration["mean_actual_change"],
        marker="o",
        label="Actual",
    )
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Prediction decile")
    plt.ylabel("Mean log valuation change")
    plt.title("Calibration by prediction decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "calibration_by_decile.png", dpi=160)
    plt.close()

    metric_rows = []
    for name, prediction in model_predictions.items():
        metric_rows.append((name, mean_absolute_error(actual, prediction)))
    metric_rows.sort(key=lambda item: item[1])
    plt.figure(figsize=(9, 5))
    plt.bar([row[0] for row in metric_rows], [row[1] for row in metric_rows])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("MAE, log valuation change")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(output_directory / "model_comparison.png", dpi=160)
    plt.close()
