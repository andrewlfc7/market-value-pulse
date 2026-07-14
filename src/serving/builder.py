from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ingestion.common import write_json


class ServingBuildError(RuntimeError):
    """Raised when API serving tables cannot be built consistently."""


@dataclass(frozen=True)
class ServingBuildResult:
    output_root: Path
    players: int
    valuation_rows: int
    match_impact_rows: int


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _latest_by(frame: pl.DataFrame, key: str, order: str) -> pl.DataFrame:
    return frame.sort([key, order]).group_by(key, maintain_order=True).tail(1)


def _reason(row: dict[str, Any]) -> str:
    candidates = [
        ("Shot threat", row.get("threat_component")),
        ("Chance creation", row.get("creation_component")),
        ("Ball progression", row.get("progression_component")),
        ("Pass retention", row.get("retention_component")),
        ("Attacking possession value", row.get("attacking_xpv_component")),
        ("Defensive threat prevention", row.get("defensive_component")),
        ("Finishing", row.get("finishing_component")),
        ("Shot stopping", row.get("goalkeeper_component")),
    ]
    available = [
        (label, float(value))
        for label, value in candidates
        if value is not None
    ]
    if not available:
        return "Position-adjusted match performance"
    available.sort(key=lambda item: abs(item[1]), reverse=True)
    return " · ".join(label for label, _ in available[:2])


def _prediction_columns(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions
    id_column = (
        "whoscored_player_id"
        if "whoscored_player_id" in predictions.columns
        else "player_id"
    )
    columns = [
        pl.col(id_column).cast(pl.Int64).alias("whoscored_player_id")
    ]
    aliases = {
        "predicted_market_value_eur": "estimated_value_eur",
        "predicted_market_value_lower_90_eur": "estimated_lower_eur",
        "predicted_market_value_upper_90_eur": "estimated_upper_eur",
        "predicted_pct_value_change": "predicted_pct_change",
        "probability_value_increase": "probability_value_increase",
        "valuation_model_version": "valuation_model_version",
    }
    for source, target in aliases.items():
        if source in predictions.columns:
            columns.append(pl.col(source).alias(target))
    return predictions.select(columns).unique("whoscored_player_id", keep="last")


def build_serving_tables(
    *,
    ratings_path: Path,
    valuations_path: Path,
    mapping_path: Path,
    output_root: Path = Path("data/serving"),
    predictions_path: Path | None = None,
) -> ServingBuildResult:
    ratings = pl.read_parquet(ratings_path)
    valuations = pl.read_parquet(valuations_path).filter(
        pl.col("is_valid_for_model").fill_null(True)
        if "is_valid_for_model" in pl.read_parquet_schema(valuations_path)
        else pl.lit(True)
    )
    mapping = pl.read_parquet(mapping_path)
    required_mapping = {"whoscored_player_id", "transfermarkt_player_id"}
    missing = sorted(required_mapping.difference(mapping.columns))
    if missing:
        raise ServingBuildError(f"Player mapping is missing columns: {missing}")
    if ratings.is_empty() or valuations.is_empty() or mapping.is_empty():
        raise ServingBuildError("Ratings, valuations, and mapping must all contain rows")

    latest_ratings = _latest_by(
        ratings, "whoscored_player_id", "match_datetime"
    )
    latest_values = _latest_by(
        valuations, "transfermarkt_player_id", "valuation_date"
    ).select(
        "transfermarkt_player_id",
        pl.col("valuation_date").alias("latest_valuation_date"),
        pl.col("market_value_eur").alias("current_market_value_eur"),
    )
    players = (
        latest_ratings.join(mapping, on="whoscored_player_id", how="inner")
        .join(latest_values, on="transfermarkt_player_id", how="left")
        .with_columns(
            pl.col("whoscored_player_id").alias("player_id"),
            pl.coalesce("player_name", "whoscored_player_name").alias("display_name"),
            pl.col("position_group").alias("position"),
            pl.col("form_rating_ewm").alias("current_form_rating"),
            pl.col("match_method").alias("mapping_method"),
            pl.col("confidence").alias("mapping_confidence"),
        )
    )
    if predictions_path is not None and predictions_path.exists():
        predictions = _prediction_columns(pl.read_parquet(predictions_path))
        players = players.join(predictions, on="whoscored_player_id", how="left")
    for column, dtype in (
        ("estimated_value_eur", pl.Float64),
        ("estimated_lower_eur", pl.Float64),
        ("estimated_upper_eur", pl.Float64),
        ("predicted_pct_change", pl.Float64),
        ("probability_value_increase", pl.Float64),
    ):
        if column not in players.columns:
            players = players.with_columns(pl.lit(None).cast(dtype).alias(column))
    if "valuation_model_version" not in players.columns:
        players = players.with_columns(
            pl.lit(None).cast(pl.String).alias("valuation_model_version")
        )
    players = players.with_columns(
        pl.when(pl.col("probability_value_increase") >= 0.60)
        .then(pl.lit("rising"))
        .when(pl.col("probability_value_increase") <= 0.40)
        .then(pl.lit("falling"))
        .when(pl.col("probability_value_increase").is_not_null())
        .then(pl.lit("stable"))
        .otherwise(pl.lit("unscored"))
        .alias("direction"),
        pl.when(pl.col("probability_value_increase").is_not_null())
        .then(
            pl.max_horizontal(
                pl.col("probability_value_increase"),
                1.0 - pl.col("probability_value_increase"),
            )
        )
        .otherwise(None)
        .alias("confidence"),
        pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("refreshed_at"),
    ).select(
        "player_id",
        "whoscored_player_id",
        "transfermarkt_player_id",
        "mapping_method",
        "mapping_confidence",
        "display_name",
        "team_id",
        "position",
        "current_form_rating",
        "rolling_3_match_rating",
        "rolling_20_match_rating",
        "latest_valuation_date",
        "current_market_value_eur",
        "estimated_value_eur",
        "estimated_lower_eur",
        "estimated_upper_eur",
        "predicted_pct_change",
        "probability_value_increase",
        "valuation_model_version",
        "direction",
        "confidence",
        "refreshed_at",
    ).sort("display_name")


    canonical_player_ids = players.select("player_id").unique()

    valuation_history = (
        valuations.join(mapping, on="transfermarkt_player_id", how="inner")
        .with_columns(pl.col("whoscored_player_id").alias("player_id"))
        .join(
            canonical_player_ids,
            on="player_id",
            how="semi",
        )
        .select(
            "player_id",
            "transfermarkt_player_id",
            "valuation_date",
            pl.col("market_value_eur").alias("value_eur"),
            pl.lit("transfermarkt").alias("source"),
        )
        .sort(["player_id", "valuation_date"])
    )



    impact_rows = []
    for row in ratings.sort("match_datetime").to_dicts():
        rating = row.get("post_match_rating")
        impact_rows.append(
            {
                "player_id": int(row["whoscored_player_id"]),
                "match_id": int(row["match_id"]),
                "match_datetime": row["match_datetime"],
                "rating": rating,
                "performance_impact_score": (
                    float(rating) - 6.0 if rating is not None else None
                ),
                "impact_direction": (
                    "positive"
                    if rating is not None and float(rating) > 6.25
                    else "negative"
                    if rating is not None and float(rating) < 5.75
                    else "neutral"
                ),
                "explanation": _reason(row),
                "minutes": row.get("minutes"),
                "opponent": None,
                "estimated_value_delta_eur": None,
                "replay_run_id": None,
                "replay_sequence": None,
                "valuation_update_status": None,
                "valuation_estimate_eur": None,
                "valuation_lower_90_eur": None,
                "valuation_upper_90_eur": None,
                "probability_value_increase": None,
                "rating_version": row.get("rating_version"),
            }
        )
    match_impacts = pl.DataFrame(impact_rows, infer_schema_length=None).sort(
        ["player_id", "match_datetime"], descending=[False, True]
    )

    _atomic_parquet(players, output_root / "players.parquet")
    _atomic_parquet(valuation_history, output_root / "valuation_history.parquet")
    _atomic_parquet(match_impacts, output_root / "match_impacts.parquet")
    write_json(
        output_root / "serving_build_summary.json",
        {
            "ratings_path": str(ratings_path),
            "valuations_path": str(valuations_path),
            "mapping_path": str(mapping_path),
            "predictions_path": str(predictions_path) if predictions_path else None,
            "players": players.height,
            "valuation_rows": valuation_history.height,
            "match_impact_rows": match_impacts.height,
            "note": (
                "Per-match estimated euro deltas remain null until the valuation replay "
                "scores before/after state; performance impact is not mislabeled as value movement."
            ),
        },
    )
    return ServingBuildResult(
        output_root=output_root,
        players=players.height,
        valuation_rows=valuation_history.height,
        match_impact_rows=match_impacts.height,
    )
