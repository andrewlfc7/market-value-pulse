"""Pure xT/xPV grid scoring for successful passes and inferred carries."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl


ACTION_COLUMNS = [
    "match_id",
    "tournament_id",
    "season_id",
    "season_name",
    "start_date",
    "action_type",
    "action_event_id",
    "event_uid",
    "event_id",
    "source_event_id",
    "target_event_id",
    "team_id",
    "player_id",
    "period_value",
    "minute",
    "second",
    "event_time_seconds",
    "x",
    "y",
    "end_x",
    "end_y",
]


def _add_time(frame: pl.DataFrame) -> pl.DataFrame:
    if "event_time_seconds" in frame.columns:
        return frame
    return frame.with_columns(
        (
            pl.col("expanded_minute").fill_null(pl.col("minute")).cast(pl.Float64)
            * 60.0
            + pl.col("second").fill_null(0.0).cast(pl.Float64)
        ).alias("event_time_seconds")
    )


def _ensure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        pl.lit(None).alias(column)
        for column in ACTION_COLUMNS
        if column not in frame.columns
    ]
    return frame.with_columns(expressions) if expressions else frame


def build_actions(passes: pl.DataFrame, carries: pl.DataFrame) -> pl.DataFrame:
    """Create one canonical action frame without querying an external store."""
    frames: list[pl.DataFrame] = []
    if not passes.is_empty():
        prepared = _add_time(
            passes.filter(
                (pl.col("success") == 1)
                & pl.col("x").is_not_null()
                & pl.col("y").is_not_null()
                & pl.col("end_x").is_not_null()
                & pl.col("end_y").is_not_null()
            )
        ).with_columns(
                pl.lit("pass").alias("action_type"),
                pl.col("event_id").cast(pl.Float64).alias("action_event_id"),
                pl.lit(None).cast(pl.Int64).alias("source_event_id"),
                pl.lit(None).cast(pl.Int64).alias("target_event_id"),
            )
        frames.append(_ensure_columns(prepared).select(ACTION_COLUMNS))
    if not carries.is_empty():
        prepared = carries.filter(
            pl.col("x").is_not_null()
            & pl.col("y").is_not_null()
            & pl.col("end_x").is_not_null()
            & pl.col("end_y").is_not_null()
        ).with_columns(
            pl.lit("carry").alias("action_type"),
            pl.col("event_id").cast(pl.Float64).alias("action_event_id"),
            pl.col("source_event_id").cast(pl.Int64, strict=False),
            pl.col("target_event_id").cast(pl.Int64, strict=False),
        )
        frames.append(_ensure_columns(prepared).select(ACTION_COLUMNS))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _zone_columns(frame: pl.DataFrame, *, nx: int, ny: int) -> pl.DataFrame:
    return frame.with_columns(
        ((pl.col("x").clip(0.0, 99.999) / 100.0 * nx).floor().cast(pl.Int64)).alias(
            "start_zone_x"
        ),
        ((pl.col("y").clip(0.0, 99.999) / 100.0 * ny).floor().cast(pl.Int64)).alias(
            "start_zone_y"
        ),
        ((pl.col("end_x").clip(0.0, 99.999) / 100.0 * nx).floor().cast(pl.Int64)).alias(
            "end_zone_x"
        ),
        ((pl.col("end_y").clip(0.0, 99.999) / 100.0 * ny).floor().cast(pl.Int64)).alias(
            "end_zone_y"
        ),
    ).with_columns(
        (pl.col("start_zone_x") + pl.col("start_zone_y") * nx).alias("start_zone"),
        (pl.col("end_zone_x") + pl.col("end_zone_y") * nx).alias("end_zone"),
    )


def score_actions(
    actions: pl.DataFrame,
    grid: np.ndarray,
    *,
    metric: str,
    model_name: str,
    model_version: str,
) -> pl.DataFrame:
    """Score a canonical action frame against a two-dimensional value grid."""
    if actions.is_empty():
        return pl.DataFrame()
    if grid.ndim != 2 or not np.isfinite(grid).all():
        raise ValueError(f"{metric} grid must be a finite two-dimensional matrix")
    ny, nx = grid.shape
    value_column = metric
    lookup = pl.DataFrame(
        {"zone": np.arange(grid.size, dtype=np.int64), value_column: grid.reshape(-1)}
    )
    start_column = f"{metric}_start"
    end_column = f"{metric}_end"
    added_column = f"{metric}_added"
    model_prefix = "xt" if metric == "xT" else "xpv"
    return (
        _zone_columns(actions, nx=nx, ny=ny)
        .join(
            lookup.rename({"zone": "start_zone", value_column: start_column}),
            on="start_zone",
            how="left",
        )
        .join(
            lookup.rename({"zone": "end_zone", value_column: end_column}),
            on="end_zone",
            how="left",
        )
        .with_columns(
            pl.col(start_column).fill_null(0.0),
            pl.col(end_column).fill_null(0.0),
        )
        .with_columns(
            (pl.col(end_column) - pl.col(start_column)).alias(added_column),
            pl.max_horizontal(
                pl.col(end_column) - pl.col(start_column), pl.lit(0.0)
            ).alias(f"{added_column}_positive"),
            pl.min_horizontal(
                pl.col(end_column) - pl.col(start_column), pl.lit(0.0)
            ).alias(f"{added_column}_negative"),
            pl.lit(model_name).alias(f"{model_prefix}_model_name"),
            pl.lit(model_version).alias(f"{model_prefix}_model_version"),
            pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("created_at"),
        )
    )
