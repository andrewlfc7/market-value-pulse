from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from ingestion.common import sha256_file, write_json
from valuation.artifacts import create_version_directory, promote_model
from valuation.bayesian import BayesianFitConfig, fit_position_hierarchical_bayesian
from valuation.evaluation import (
    bootstrap_mae_difference,
    calibration_by_decile,
    regression_metrics,
    save_evaluation_plots,
)
from valuation.ols import fit_ols_hc3
from valuation.preprocessing import (
    NUMERIC_FEATURES,
    POSITION_COLUMN,
    TARGET_COLUMN,
    prepare_matrices,
)


@dataclass(frozen=True)
class TrainModelResult:
    model_version: str
    artifact_directory: Path
    metrics_path: Path
    selected_prediction: str
    selected_mae: float
    promoted: bool


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    pl.from_pandas(frame).write_parquet(path, compression="zstd")


def train_valuation_model(
    *,
    dataset_path: Path,
    model_root: Path,
    test_fraction: float = 0.20,
    bayesian_config: BayesianFitConfig | None = None,
    promote: bool = True,
) -> TrainModelResult:
    bayesian_config = bayesian_config or BayesianFitConfig()
    model_root.mkdir(parents=True, exist_ok=True)
    model_version, artifact_directory = create_version_directory(model_root)

    dataset = pl.read_parquet(dataset_path)
    model_frame = dataset.to_pandas()
    matrices = prepare_matrices(model_frame, test_fraction=test_fraction)

    ols = fit_ols_hc3(
        X_train_numeric=matrices.X_train,
        X_test_numeric=matrices.X_test,
        train_positions=matrices.train_frame[POSITION_COLUMN],
        test_positions=matrices.test_frame[POSITION_COLUMN],
        position_levels=matrices.position_levels,
        y_train=matrices.y_train,
    )

    bayesian = fit_position_hierarchical_bayesian(
        X_train=matrices.X_train,
        y_train=matrices.y_train,
        train_position_code=matrices.train_position_code,
        train_player_code=matrices.train_player_code,
        X_test=matrices.X_test,
        test_position_code=matrices.test_position_code,
        test_player_code=matrices.test_player_code,
        feature_names=NUMERIC_FEATURES,
        position_levels=matrices.position_levels,
        player_levels=matrices.player_levels,
        config=bayesian_config,
    )

    mean_metrics = regression_metrics(
        "bayesian_hierarchical_mean",
        matrices.y_test,
        bayesian.mean_predictions,
    )
    median_metrics = regression_metrics(
        "bayesian_hierarchical_median",
        matrices.y_test,
        bayesian.median_predictions,
    )
    if median_metrics["mae_log_change"] <= mean_metrics["mae_log_change"]:
        selected_predictions = bayesian.median_predictions
        selected_prediction_name = "bayesian_hierarchical_median"
        selected_metrics = median_metrics
    else:
        selected_predictions = bayesian.mean_predictions
        selected_prediction_name = "bayesian_hierarchical_mean"
        selected_metrics = mean_metrics

    previous_change = (
        matrices.test_frame["previous_log_value_change"].fillna(0.0).to_numpy()
    )
    predictions = {
        "zero_change_baseline": np.zeros_like(matrices.y_test),
        "previous_change_baseline": previous_change,
        "ols_hc3": ols.predictions,
        "bayesian_hierarchical_mean": bayesian.mean_predictions,
        "bayesian_hierarchical_median": bayesian.median_predictions,
    }
    metric_rows = [
        regression_metrics(name, matrices.y_test, values)
        for name, values in predictions.items()
    ]
    metrics = pd.DataFrame(metric_rows).sort_values("mae_log_change")

    coverage_90 = float(
        np.mean(
            (matrices.y_test >= bayesian.predictive_lower_90)
            & (matrices.y_test <= bayesian.predictive_upper_90)
        )
    )
    calibration = calibration_by_decile(matrices.y_test, selected_predictions)
    bayesian_vs_ols = bootstrap_mae_difference(
        matrices.y_test,
        selected_predictions,
        ols.predictions,
    )

    prediction_columns = [
        column
        for column in [
            "interval_id",
            "transfermarkt_player_id",
            "whoscored_player_id",
            "player_name",
            "club_name",
            "position_group",
            "previous_valuation_date",
            "valuation_date",
            "previous_market_value_eur",
            "market_value_eur",
            TARGET_COLUMN,
        ]
        if column in matrices.test_frame.columns
    ]
    test_predictions = matrices.test_frame[prediction_columns].copy()
    test_predictions["ols_predicted_log_change"] = ols.predictions
    test_predictions["bayesian_mean_log_change"] = bayesian.mean_predictions
    test_predictions["bayesian_median_log_change"] = bayesian.median_predictions
    test_predictions["selected_predicted_log_change"] = selected_predictions
    test_predictions["predicted_pct_change"] = np.expm1(selected_predictions)
    test_predictions["predictive_lower_90"] = bayesian.predictive_lower_90
    test_predictions["predictive_upper_90"] = bayesian.predictive_upper_90
    test_predictions["probability_value_increase"] = (
        bayesian.expected_samples > 0
    ).mean(axis=0)
    test_predictions["predicted_market_value_eur"] = (
        test_predictions["previous_market_value_eur"] * np.exp(selected_predictions)
    )
    test_predictions["predicted_value_lower_90"] = (
        test_predictions["previous_market_value_eur"]
        * np.exp(bayesian.predictive_lower_90)
    )
    test_predictions["predicted_value_upper_90"] = (
        test_predictions["previous_market_value_eur"]
        * np.exp(bayesian.predictive_upper_90)
    )

    joblib.dump(matrices.preprocessor, artifact_directory / "preprocessor.joblib")
    ols.model.save(artifact_directory / "ols_model.pkl")
    np.savez_compressed(
        artifact_directory / "posterior_samples.npz",
        **bayesian.posterior_samples,
    )
    _to_parquet(metrics, artifact_directory / "model_metrics.parquet")
    _to_parquet(calibration, artifact_directory / "calibration_by_decile.parquet")
    _to_parquet(test_predictions, artifact_directory / "test_predictions.parquet")
    _to_parquet(ols.coefficients, artifact_directory / "ols_coefficients.parquet")
    _to_parquet(
        bayesian.global_coefficients,
        artifact_directory / "global_coefficients.parquet",
    )
    _to_parquet(
        bayesian.position_effects,
        artifact_directory / "position_effects.parquet",
    )
    _to_parquet(
        bayesian.player_effects,
        artifact_directory / "player_effects.parquet",
    )

    feature_schema = {
        "target": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "position_column": POSITION_COLUMN,
        "position_levels": matrices.position_levels,
        "age_feature": "age_at_valuation",
        "form_feature": "recency_weighted_rating",
        "player_id_column": "transfermarkt_player_id",
        "player_effects_artifact": "player_effects.parquet",
        "rating_version": "post_match_v2",
        "ewm_half_life_days": 90.0,
        "rolling_windows": [3, 20],
        "selected_prediction": selected_prediction_name,
    }
    write_json(artifact_directory / "feature_schema.json", feature_schema)

    diagnostics = {
        **bayesian.diagnostics,
        "predictive_coverage_90": coverage_90,
        "median_predictive_interval_width": float(
            np.median(bayesian.predictive_upper_90 - bayesian.predictive_lower_90)
        ),
        "bayesian_vs_ols_bootstrap": bayesian_vs_ols,
    }
    write_json(artifact_directory / "model_diagnostics.json", diagnostics)

    zero_metrics = next(
        row for row in metric_rows if row["model"] == "zero_change_baseline"
    )
    ols_metrics = next(row for row in metric_rows if row["model"] == "ols_hc3")
    maximum_r_hat = bayesian.diagnostics.get("maximum_r_hat")
    promotion_checks = {
        "beats_zero_change_mae": (
            selected_metrics["mae_log_change"] < zero_metrics["mae_log_change"]
        ),
        "beats_zero_change_direction": (
            selected_metrics["direction_accuracy"]
            > zero_metrics["direction_accuracy"]
        ),
        "beats_ols_mae": (
            selected_metrics["mae_log_change"] < ols_metrics["mae_log_change"]
        ),
        "positive_holdout_r2": selected_metrics["r2"] > 0.0,
        "predictive_coverage_at_least_80_percent": coverage_90 >= 0.80,
        "divergences_at_most_five": bayesian.diagnostics["divergences"] <= 5,
        "maximum_r_hat_at_most_1_05": (
            maximum_r_hat is None or maximum_r_hat <= 1.05
        ),
    }
    promotion_passed = all(promotion_checks.values())
    write_json(
        artifact_directory / "promotion_checks.json",
        {
            "passed": promotion_passed,
            "checks": promotion_checks,
            "selected_mae": selected_metrics["mae_log_change"],
            "zero_change_mae": zero_metrics["mae_log_change"],
            "ols_mae": ols_metrics["mae_log_change"],
            "selected_direction_accuracy": selected_metrics["direction_accuracy"],
            "zero_change_direction_accuracy": zero_metrics["direction_accuracy"],
        },
    )

    training_manifest = {
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "training_start_date": str(matrices.train_frame["valuation_date"].min()),
        "training_end_date": str(matrices.train_frame["valuation_date"].max()),
        "test_start_date": str(matrices.test_frame["valuation_date"].min()),
        "test_end_date": str(matrices.test_frame["valuation_date"].max()),
        "training_rows": len(matrices.train_frame),
        "test_rows": len(matrices.test_frame),
        "training_players": int(
            matrices.train_frame["transfermarkt_player_id"].nunique()
        ),
        "test_fraction": test_fraction,
        "bayesian_config": asdict(bayesian_config),
    }
    write_json(artifact_directory / "training_manifest.json", training_manifest)

    model_summary = {
        "selected_model": "position_hierarchical_bayesian_student_t",
        "selected_prediction": selected_prediction_name,
        "model_version": model_version,
        "target": TARGET_COLUMN,
        "training_observations": len(matrices.train_frame),
        "test_observations": len(matrices.test_frame),
        "players": int(model_frame["transfermarkt_player_id"].nunique()),
        "split_date": str(matrices.split_date),
        "mae_log_change": selected_metrics["mae_log_change"],
        "approx_mae_percent": selected_metrics["approx_mae_percent"],
        "rmse_log_change": selected_metrics["rmse_log_change"],
        "r2": selected_metrics["r2"],
        "direction_accuracy": selected_metrics["direction_accuracy"],
        "spearman_correlation": selected_metrics["spearman_correlation"],
        "predictive_coverage_90": coverage_90,
        "divergences": bayesian.diagnostics["divergences"],
        "ols_mae_log_change": float(
            metrics.loc[metrics["model"] == "ols_hc3", "mae_log_change"].iloc[0]
        ),
        "zero_change_mae_log_change": zero_metrics["mae_log_change"],
        "promotion_checks_passed": promotion_passed,
        "promotion_checks": promotion_checks,
    }
    write_json(artifact_directory / "model_summary.json", model_summary)

    save_evaluation_plots(
        actual=matrices.y_test,
        selected_predictions=selected_predictions,
        model_predictions={
            "Zero baseline": predictions["zero_change_baseline"],
            "OLS HC3": ols.predictions,
            "Bayesian hierarchy": selected_predictions,
        },
        calibration=calibration,
        output_directory=artifact_directory / "plots",
    )

    active_path = None
    if promote and promotion_passed:
        active_path = promote_model(
            model_root=model_root,
            model_version=model_version,
            artifact_directory=artifact_directory,
        )

    return TrainModelResult(
        model_version=model_version,
        artifact_directory=artifact_directory,
        metrics_path=artifact_directory / "model_metrics.parquet",
        selected_prediction=selected_prediction_name,
        selected_mae=float(selected_metrics["mae_log_change"]),
        promoted=active_path is not None,
    )
