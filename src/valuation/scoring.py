from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from ingestion.common import write_json
from valuation.artifacts import resolve_model_directory


@dataclass(frozen=True)
class ScoreResult:
    output_path: Path
    summary_path: Path
    model_directory: Path
    rows: int


def score_valuation_features(
    *,
    features_path: Path,
    model_root: Path,
    model_version: str,
    output_path: Path,
    random_seed: int = 42,
) -> ScoreResult:
    model_directory = resolve_model_directory(model_root, model_version)
    schema = json.loads(
        (model_directory / "feature_schema.json").read_text(encoding="utf-8")
    )
    preprocessor = joblib.load(model_directory / "preprocessor.joblib")
    posterior = {
        key: np.asarray(value)
        for key, value in np.load(model_directory / "posterior_samples.npz").items()
    }

    frame = pl.read_parquet(features_path)
    pandas_frame = frame.to_pandas()
    numeric_features = schema["numeric_features"]
    missing = sorted(set(numeric_features).union({schema["position_column"]}).difference(pandas_frame.columns))
    if missing:
        raise ValueError(f"Scoring feature table is missing columns: {missing}")

    X = np.asarray(preprocessor.transform(pandas_frame[numeric_features]), dtype=float)
    position_levels = schema["position_levels"]
    position_to_code = {value: index for index, value in enumerate(position_levels)}
    position_codes = (
        pandas_frame[schema["position_column"]]
        .astype(str)
        .map(position_to_code)
    )
    if position_codes.isna().any():
        unknown = sorted(
            pandas_frame.loc[position_codes.isna(), schema["position_column"]]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Model has no position effects for: {unknown}")
    position_codes = position_codes.to_numpy(dtype=np.int32)

    position_intercept = (
        posterior["position_intercept_raw"]
        * posterior["position_intercept_scale"][:, None]
    )
    position_age = (
        posterior["position_age_raw"] * posterior["position_age_scale"][:, None]
    )
    position_form = (
        posterior["position_form_raw"] * posterior["position_form_scale"][:, None]
    )
    player_effect = np.zeros(
        (len(posterior["global_intercept"]), len(pandas_frame)), dtype=float
    )
    if {
        "player_intercept_raw",
        "player_intercept_scale",
    }.issubset(posterior):
        player_intercept = (
            posterior["player_intercept_raw"]
            * posterior["player_intercept_scale"][:, None]
        )
        effects_path = model_directory / schema.get(
            "player_effects_artifact", "player_effects.parquet"
        )
        if effects_path.exists():
            player_levels = (
                pl.read_parquet(effects_path)["transfermarkt_player_id"]
                .cast(pl.Int64)
                .to_list()
            )
            player_to_code = {
                int(player_id): index for index, player_id in enumerate(player_levels)
            }
            player_column = schema.get(
                "player_id_column", "transfermarkt_player_id"
            )
            for row_index, player_id in enumerate(pandas_frame[player_column]):
                if pd.isna(player_id):
                    continue
                code = player_to_code.get(int(player_id))
                if code is not None:
                    player_effect[:, row_index] = player_intercept[:, code]
    age_index = numeric_features.index(schema["age_feature"])
    form_index = numeric_features.index(schema["form_feature"])

    expected = (
        posterior["global_intercept"][:, None]
        + posterior["global_beta"] @ X.T
        + position_intercept[:, position_codes]
        + player_effect
        + position_age[:, position_codes] * X[:, age_index][None, :]
        + position_form[:, position_codes] * X[:, form_index][None, :]
    )
    if schema.get("selected_prediction") == "bayesian_hierarchical_median":
        predicted_log_change = np.median(expected, axis=0)
    else:
        predicted_log_change = expected.mean(axis=0)

    generator = np.random.default_rng(random_seed)
    noise = generator.standard_t(
        df=posterior["nu"][:, None],
        size=(len(posterior["nu"]), X.shape[0]),
    )
    predictive = expected + posterior["sigma"][:, None] * noise
    lower = np.quantile(predictive, 0.05, axis=0)
    upper = np.quantile(predictive, 0.95, axis=0)
    probability_increase = (expected > 0).mean(axis=0)

    output = frame.with_columns(
        pl.lit(model_directory.name.removeprefix("model_version=")).alias(
            "valuation_model_version"
        ),
        pl.Series("predicted_log_value_change", predicted_log_change),
        pl.Series("predicted_pct_value_change", np.expm1(predicted_log_change)),
        pl.Series("probability_value_increase", probability_increase),
        pl.Series("predictive_log_change_lower_90", lower),
        pl.Series("predictive_log_change_upper_90", upper),
    )
    if "previous_market_value_eur" in output.columns:
        output = output.with_columns(
            (
                pl.col("previous_market_value_eur")
                * pl.col("predicted_log_value_change").exp()
            ).alias("predicted_market_value_eur"),
            (
                pl.col("previous_market_value_eur")
                * pl.col("predictive_log_change_lower_90").exp()
            ).alias("predicted_market_value_lower_90_eur"),
            (
                pl.col("previous_market_value_eur")
                * pl.col("predictive_log_change_upper_90").exp()
            ).alias("predicted_market_value_upper_90_eur"),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.write_parquet(temporary, compression="zstd")
    temporary.replace(output_path)

    summary_path = output_path.with_name("valuation_predictions_summary.json")
    write_json(
        summary_path,
        {
            "model_directory": str(model_directory),
            "features_path": str(features_path),
            "output_path": str(output_path),
            "rows": output.height,
            "mean_predicted_pct_change": float(output["predicted_pct_value_change"].mean()),
            "mean_probability_value_increase": float(output["probability_value_increase"].mean()),
        },
    )
    return ScoreResult(
        output_path=output_path,
        summary_path=summary_path,
        model_directory=model_directory,
        rows=output.height,
    )
