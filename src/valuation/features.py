from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ingestion.common import write_json


class ValuationFeatureError(RuntimeError):
    """Raised when valuation model features cannot be built safely."""


COMPONENT_COLUMNS = [
    "threat_component",
    "creation_component",
    "progression_component",
    "retention_component",
    "attacking_xpv_component",
    "defensive_component",
    "finishing_component",
]


@dataclass(frozen=True)
class ValuationFeatureConfig:
    minimum_interval_days: int = 14
    maximum_interval_days: int = 365
    minimum_interval_minutes: float = 180.0
    ewm_half_life_days: float = 90.0
    rolling_short_matches: int = 3
    rolling_long_matches: int = 20


@dataclass(frozen=True)
class ValuationFeatureResult:
    output_path: Path
    summary_path: Path
    observations: int
    players: int
    first_valuation_date: str
    last_valuation_date: str


def read_rating_history(path: Path) -> pl.DataFrame:
    """Read one rating file or a partitioned multi-season rating directory."""
    if path.is_file():
        return pl.read_parquet(path)
    if not path.exists():
        raise FileNotFoundError(path)
    paths = sorted(path.glob("season=*/player_match_ratings.parquet"))
    if not paths:
        paths = sorted(path.rglob("player_match_ratings.parquet"))
    if not paths:
        raise ValuationFeatureError(f"No player-match rating partitions under {path}")
    return pl.concat(
        [pl.read_parquet(candidate) for candidate in paths],
        how="diagonal_relaxed",
    ).unique(
        subset=["season", "match_id", "whoscored_player_id"],
        keep="last",
    )


def _resolve_column(
    frame: pl.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValuationFeatureError(
            f"Could not find any of {list(candidates)}. Available columns: {frame.columns}"
        )
    return None


def clean_valuations(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"transfermarkt_player_id", "valuation_date", "market_value_eur"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValuationFeatureError(f"Valuation table is missing columns: {missing}")

    filters: list[pl.Expr] = [pl.col("market_value_eur") > 0]
    if "is_valid_for_model" in frame.columns:
        filters.append(pl.col("is_valid_for_model").fill_null(False))
    if "is_terminal_record" in frame.columns:
        filters.append(~pl.col("is_terminal_record").fill_null(False))
    if "is_future_dated" in frame.columns:
        filters.append(~pl.col("is_future_dated").fill_null(False))

    columns = [
        "transfermarkt_player_id",
        "valuation_date",
        "market_value_eur",
    ]
    for optional in [
        "club_name",
        "age_at_valuation",
        "competition_id",
        "source_run_id",
        "source_snapshot_date",
    ]:
        if optional in frame.columns:
            columns.append(optional)

    output = frame.filter(pl.all_horizontal(filters)).select(columns)
    if "age_at_valuation" not in output.columns:
        output = output.with_columns(pl.lit(None, dtype=pl.Float64).alias("age_at_valuation"))
    else:
        output = output.with_columns(
            pl.col("age_at_valuation").cast(pl.Float64, strict=False)
        )
    if "club_name" not in output.columns:
        output = output.with_columns(pl.lit(None, dtype=pl.String).alias("club_name"))

    return (
        output.with_columns(
            pl.col("transfermarkt_player_id").cast(pl.Int64),
            pl.col("valuation_date").cast(pl.Date),
            pl.col("market_value_eur").cast(pl.Float64),
        )
        .unique(
            subset=["transfermarkt_player_id", "valuation_date"],
            keep="last",
        )
        .sort(["transfermarkt_player_id", "valuation_date"])
    )


def clean_crosswalk(frame: pl.DataFrame) -> pl.DataFrame:
    tm_column = _resolve_column(
        frame,
        ["transfermarkt_player_id", "tm_player_id", "player_id"],
    )
    ws_column = _resolve_column(
        frame,
        ["whoscored_player_id", "ws_player_id", "matched_whoscored_player_id"],
    )

    output = (
        frame.select(
            pl.col(tm_column).cast(pl.Int64, strict=False).alias("transfermarkt_player_id"),
            pl.col(ws_column).cast(pl.Int64, strict=False).alias("whoscored_player_id"),
        )
        .drop_nulls()
        .unique()
    )

    ambiguous_tm = (
        output.group_by("transfermarkt_player_id")
        .agg(pl.col("whoscored_player_id").n_unique().alias("count"))
        .filter(pl.col("count") > 1)
    )
    ambiguous_ws = (
        output.group_by("whoscored_player_id")
        .agg(pl.col("transfermarkt_player_id").n_unique().alias("count"))
        .filter(pl.col("count") > 1)
    )
    if ambiguous_tm.height or ambiguous_ws.height:
        raise ValuationFeatureError(
            "Crosswalk must be one-to-one; ambiguous player mappings were found"
        )
    return output


def prepare_ratings(frame: pl.DataFrame) -> pl.DataFrame:
    rating_column = _resolve_column(
        frame,
        ["post_match_rating"],
    )
    required = {
        "season",
        "match_id",
        "match_datetime",
        "whoscored_player_id",
        "position_group",
        "minutes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValuationFeatureError(f"Rating table is missing columns: {missing}")

    output = frame
    default_numeric = {
        "started": 0,
        "goals": 0.0,
        "assists": 0.0,
        "xg": 0.0,
        "xgot": 0.0,
        "xa": 0.0,
        "big_chances_missed": 0.0,
        "big_chance_xg_missed": 0.0,
        "own_goals": 0.0,
    }
    for column, default in default_numeric.items():
        if column not in output.columns:
            output = output.with_columns(pl.lit(default).alias(column))
    for column in COMPONENT_COLUMNS:
        legacy = f"{column}_v3"
        if column not in output.columns:
            output = output.with_columns(
                (
                    pl.col(legacy).cast(pl.Float64, strict=False)
                    if legacy in output.columns
                    else pl.lit(None, dtype=pl.Float64)
                ).alias(column)
            )
    if "player_name" not in output.columns:
        output = output.with_columns(pl.lit(None, dtype=pl.String).alias("player_name"))

    return (
        output.filter(pl.col(rating_column).is_not_null())
        .select(
            pl.col("season").cast(pl.String),
            pl.col("match_id").cast(pl.Int64),
            pl.col("match_datetime").cast(pl.Datetime),
            pl.col("match_datetime").cast(pl.Datetime).dt.date().alias("match_date"),
            pl.col("whoscored_player_id").cast(pl.Int64),
            pl.col("player_name").cast(pl.String),
            pl.col("position_group").cast(pl.String),
            pl.col("minutes").cast(pl.Float64).fill_null(0.0),
            pl.col("started").cast(pl.Int64, strict=False).fill_null(0),
            pl.col(rating_column).cast(pl.Float64).alias("post_match_rating"),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in COMPONENT_COLUMNS
            ],
            *[
                pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column)
                for column in [
                    "goals",
                    "assists",
                    "xg",
                    "xgot",
                    "xa",
                    "big_chances_missed",
                    "big_chance_xg_missed",
                    "own_goals",
                ]
            ],
        )
        .unique(subset=["season", "match_id", "whoscored_player_id"], keep="last")
    )


def build_valuation_intervals(
    valuations: pl.DataFrame,
    crosswalk: pl.DataFrame,
    rating_players: pl.DataFrame,
    *,
    config: ValuationFeatureConfig,
) -> pl.DataFrame:
    mapped_players = (
        valuations.select("transfermarkt_player_id")
        .unique()
        .join(crosswalk, on="transfermarkt_player_id", how="inner")
        .join(
            rating_players.select("whoscored_player_id").unique(),
            on="whoscored_player_id",
            how="inner",
        )
    )

    mapped = (
        valuations.join(mapped_players, on="transfermarkt_player_id", how="inner")
        .sort(["transfermarkt_player_id", "valuation_date"])
    )

    intervals = (
        mapped.with_columns(
            pl.col("valuation_date")
            .shift(1)
            .over("transfermarkt_player_id")
            .alias("previous_valuation_date"),
            pl.col("market_value_eur")
            .shift(1)
            .over("transfermarkt_player_id")
            .alias("previous_market_value_eur"),
            pl.col("market_value_eur")
            .shift(2)
            .over("transfermarkt_player_id")
            .alias("two_updates_back_value_eur"),
        )
        .drop_nulls(["previous_valuation_date", "previous_market_value_eur"])
        .with_columns(
            (pl.col("valuation_date") - pl.col("previous_valuation_date"))
            .dt.total_days()
            .alias("interval_days"),
            (pl.col("market_value_eur") / pl.col("previous_market_value_eur"))
            .log()
            .alias("target_log_value_change"),
            (
                pl.col("market_value_eur") / pl.col("previous_market_value_eur") - 1.0
            ).alias("target_pct_value_change"),
            pl.col("previous_market_value_eur")
            .log()
            .alias("log_previous_market_value"),
        )
        .with_columns(
            pl.when(pl.col("two_updates_back_value_eur").is_not_null())
            .then(
                (
                    pl.col("previous_market_value_eur")
                    / pl.col("two_updates_back_value_eur")
                ).log()
            )
            .otherwise(None)
            .alias("previous_log_value_change")
        )
        .filter(
            pl.col("interval_days").is_between(
                config.minimum_interval_days,
                config.maximum_interval_days,
                closed="both",
            )
        )
        .with_row_index("interval_id")
    )
    return intervals


def _weighted_mean(column: str, weight: str, alias: str, condition: pl.Expr | None = None) -> pl.Expr:
    valid = pl.col(column).is_not_null()
    if condition is not None:
        valid = valid & condition
    numerator = (
        pl.when(valid)
        .then(pl.col(column) * pl.col(weight))
        .otherwise(0.0)
        .sum()
    )
    denominator = pl.when(valid).then(pl.col(weight)).otherwise(0.0).sum()
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None).alias(alias)


def aggregate_interval_performance(
    intervals: pl.DataFrame,
    ratings: pl.DataFrame,
    *,
    config: ValuationFeatureConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    joined = (
        intervals.select(
            "interval_id",
            "whoscored_player_id",
            "previous_valuation_date",
            "valuation_date",
        )
        .join(ratings, on="whoscored_player_id", how="inner")
        .filter(pl.col("match_date") < pl.col("valuation_date"))
        .with_columns(
            (pl.col("valuation_date") - pl.col("match_date"))
            .dt.total_days()
            .alias("days_before_valuation")
        )
        .with_columns(
            (
                pl.col("minutes").clip(1.0, 120.0)
                * (
                    -np.log(2.0)
                    * pl.col("days_before_valuation")
                    / config.ewm_half_life_days
                ).exp()
            ).alias("ewm_weight"),
            pl.col("match_date")
            .rank(method="ordinal", descending=True)
            .over("interval_id")
            .alias("match_recency_rank"),
        )
    )

    rolling = joined.group_by("interval_id").agg(
        _weighted_mean("post_match_rating", "ewm_weight", "form_rating_ewm"),
        _weighted_mean(
            "post_match_rating",
            "minutes",
            "rolling_3_match_rating",
            pl.col("match_recency_rank") <= config.rolling_short_matches,
        ),
        _weighted_mean(
            "post_match_rating",
            "minutes",
            "rolling_20_match_rating",
            pl.col("match_recency_rank") <= config.rolling_long_matches,
        ),
        pl.col("match_recency_rank")
        .filter(pl.col("match_recency_rank") <= config.rolling_short_matches)
        .count()
        .alias("rolling_3_matches_available"),
        pl.col("match_recency_rank")
        .filter(pl.col("match_recency_rank") <= config.rolling_long_matches)
        .count()
        .alias("rolling_20_matches_available"),
    )

    interval_matches = joined.filter(
        pl.col("match_date") > pl.col("previous_valuation_date")
    )

    interval_performance = interval_matches.group_by("interval_id").agg(
        pl.col("match_id").n_unique().alias("appearances"),
        pl.col("started").sum().alias("starts"),
        pl.col("minutes").sum().alias("minutes"),
        pl.col("position_group").drop_nulls().last().alias("position_group"),
        pl.col("player_name").drop_nulls().last().alias("player_name"),
        _weighted_mean("post_match_rating", "minutes", "average_rating"),
        _weighted_mean(
            "post_match_rating", "ewm_weight", "recency_weighted_rating"
        ),
        _weighted_mean(
            "post_match_rating",
            "minutes",
            "rating_last_90_days",
            pl.col("days_before_valuation") <= 90,
        ),
        pl.col("post_match_rating").std().fill_null(0.0).alias("rating_volatility"),
        *[
            _weighted_mean(column, "minutes", f"{column}_average")
            for column in COMPONENT_COLUMNS
        ],
        *[
            pl.col(column).sum().alias(column)
            for column in [
                "goals",
                "assists",
                "xg",
                "xgot",
                "xa",
                "big_chances_missed",
                "big_chance_xg_missed",
                "own_goals",
            ]
        ],
    )

    return interval_performance, rolling


def finalize_model_table(
    intervals: pl.DataFrame,
    interval_performance: pl.DataFrame,
    rolling: pl.DataFrame,
    *,
    config: ValuationFeatureConfig,
) -> pl.DataFrame:
    output = (
        intervals.join(interval_performance, on="interval_id", how="inner")
        .join(rolling, on="interval_id", how="left")
        .filter(pl.col("minutes") >= config.minimum_interval_minutes)
        .with_columns(
            pl.col("minutes").log1p().alias("log_minutes"),
            pl.col("appearances").log1p().alias("log_appearances"),
            (pl.col("starts") / pl.col("appearances")).alias("start_share"),
            pl.col("age_at_valuation").pow(2).alias("age_squared"),
            pl.col("valuation_date").dt.year().cast(pl.Float64).alias("valuation_year"),
            (
                2.0 * np.pi * pl.col("valuation_date").dt.month() / 12.0
            ).sin().alias("valuation_month_sin"),
            (
                2.0 * np.pi * pl.col("valuation_date").dt.month() / 12.0
            ).cos().alias("valuation_month_cos"),
            (
                pl.col("rating_last_90_days") - pl.col("average_rating")
            ).alias("recent_rating_trend"),
        )
    )

    per90_columns = [
        "goals",
        "assists",
        "xg",
        "xgot",
        "xa",
        "big_chances_missed",
        "big_chance_xg_missed",
        "own_goals",
    ]
    output = output.with_columns(
        *[
            (pl.col(column) * 90.0 / pl.col("minutes")).alias(f"{column}_per90")
            for column in per90_columns
        ],
        ((pl.col("goals") - pl.col("xg")) * 90.0 / pl.col("minutes")).alias(
            "goals_over_xg_per90"
        ),
        ((pl.col("assists") - pl.col("xa")) * 90.0 / pl.col("minutes")).alias(
            "assists_over_xa_per90"
        ),
    )
    return output.sort("valuation_date")


def build_valuation_model_dataset(
    *,
    valuations_path: Path,
    mapping_path: Path,
    ratings_path: Path,
    output_path: Path,
    config: ValuationFeatureConfig | None = None,
) -> ValuationFeatureResult:
    config = config or ValuationFeatureConfig()
    valuations = clean_valuations(pl.read_parquet(valuations_path))
    crosswalk = clean_crosswalk(pl.read_parquet(mapping_path))
    ratings = prepare_ratings(read_rating_history(ratings_path))

    intervals = build_valuation_intervals(
        valuations,
        crosswalk,
        ratings,
        config=config,
    )
    interval_performance, rolling = aggregate_interval_performance(
        intervals,
        ratings,
        config=config,
    )
    dataset = finalize_model_table(
        intervals,
        interval_performance,
        rolling,
        config=config,
    )

    if dataset.is_empty():
        raise ValuationFeatureError("No model observations remained after feature filters")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataset.write_parquet(temporary_path, compression="zstd")
    temporary_path.replace(output_path)

    summary_path = output_path.with_name("valuation_model_dataset_summary.json")
    first_date = dataset["valuation_date"].min()
    last_date = dataset["valuation_date"].max()
    summary = {
        "output_path": str(output_path),
        "observations": dataset.height,
        "players": dataset["transfermarkt_player_id"].n_unique(),
        "first_valuation_date": str(first_date),
        "last_valuation_date": str(last_date),
        "target": "target_log_value_change",
        "rating_column": "post_match_rating",
        "config": asdict(config),
        "source_paths": {
            "valuations": str(valuations_path),
            "mapping": str(mapping_path),
            "ratings": str(ratings_path),
        },
    }
    write_json(summary_path, summary)

    return ValuationFeatureResult(
        output_path=output_path,
        summary_path=summary_path,
        observations=dataset.height,
        players=dataset["transfermarkt_player_id"].n_unique(),
        first_valuation_date=str(first_date),
        last_valuation_date=str(last_date),
    )


def build_current_scoring_dataset(
    *,
    valuations_path: Path,
    mapping_path: Path,
    ratings_path: Path,
    output_path: Path,
    as_of_date,
    config: ValuationFeatureConfig | None = None,
) -> ValuationFeatureResult:
    """Build one current, unlabeled feature row per mapped player.

    The latest known Transfermarkt valuation is treated as the interval start and
    `as_of_date` as the hypothetical next valuation date. This supports scoring
    new match data without retraining the posterior.
    """
    from datetime import date as date_type

    config = config or ValuationFeatureConfig()
    if isinstance(as_of_date, str):
        as_of = date_type.fromisoformat(as_of_date)
    elif isinstance(as_of_date, date_type):
        as_of = as_of_date
    else:
        raise TypeError("as_of_date must be a date or YYYY-MM-DD string")

    valuations = clean_valuations(pl.read_parquet(valuations_path)).filter(
        pl.col("valuation_date") < pl.lit(as_of).cast(pl.Date)
    )
    if valuations.is_empty():
        raise ValuationFeatureError(
            f"No valid valuation existed before the scoring date {as_of}"
        )
    crosswalk = clean_crosswalk(pl.read_parquet(mapping_path))
    ratings = prepare_ratings(read_rating_history(ratings_path))

    mapped = valuations.join(crosswalk, on="transfermarkt_player_id", how="inner")
    mapped = mapped.join(
        ratings.select("whoscored_player_id").unique(),
        on="whoscored_player_id",
        how="inner",
    ).sort(["transfermarkt_player_id", "valuation_date"])

    latest = mapped.group_by("transfermarkt_player_id", maintain_order=True).agg(
        pl.col("whoscored_player_id").last(),
        pl.col("valuation_date").last().alias("previous_valuation_date"),
        pl.col("market_value_eur").last().alias("previous_market_value_eur"),
        pl.col("market_value_eur").shift(1).last().alias("two_updates_back_value_eur"),
        pl.col("age_at_valuation").last(),
        pl.col("club_name").last(),
    )

    pseudo_intervals = (
        latest.with_columns(pl.lit(as_of).cast(pl.Date).alias("valuation_date"))
        .with_columns(
            (pl.col("valuation_date") - pl.col("previous_valuation_date"))
            .dt.total_days()
            .alias("interval_days"),
            pl.col("previous_market_value_eur")
            .log()
            .alias("log_previous_market_value"),
            pl.when(pl.col("two_updates_back_value_eur") > 0)
            .then(
                (
                    pl.col("previous_market_value_eur")
                    / pl.col("two_updates_back_value_eur")
                ).log()
            )
            .otherwise(None)
            .alias("previous_log_value_change"),
            (
                pl.col("age_at_valuation")
                + (
                    pl.col("valuation_date") - pl.col("previous_valuation_date")
                ).dt.total_days()
                / 365.25
            ).alias("age_at_valuation"),
            pl.lit(None, dtype=pl.Float64).alias("market_value_eur"),
            pl.lit(None, dtype=pl.Float64).alias("target_log_value_change"),
            pl.lit(None, dtype=pl.Float64).alias("target_pct_value_change"),
        )
        .filter(pl.col("interval_days") > 0)
        .with_row_index("interval_id")
    )

    interval_performance, rolling = aggregate_interval_performance(
        pseudo_intervals,
        ratings,
        config=config,
    )
    dataset = finalize_model_table(
        pseudo_intervals,
        interval_performance,
        rolling,
        config=config,
    )
    if dataset.is_empty():
        raise ValuationFeatureError("No current scoring rows could be constructed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataset.write_parquet(temporary_path, compression="zstd")
    temporary_path.replace(output_path)

    summary_path = output_path.with_name("current_scoring_features_summary.json")
    write_json(
        summary_path,
        {
            "output_path": str(output_path),
            "as_of_date": str(as_of),
            "observations": dataset.height,
            "players": dataset["transfermarkt_player_id"].n_unique(),
            "config": asdict(config),
        },
    )
    return ValuationFeatureResult(
        output_path=output_path,
        summary_path=summary_path,
        observations=dataset.height,
        players=dataset["transfermarkt_player_id"].n_unique(),
        first_valuation_date=str(as_of),
        last_valuation_date=str(as_of),
    )
