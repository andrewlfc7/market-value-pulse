"""Pure expected-assist scoring kernels used by Market Value Pulse."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

# Reuse feature logic from train script.
from .xa_features import (
    FEATURE_COLS_NUM,
    FEATURE_COLS_CAT,
    add_pass_features,
)


def build_actual_targets(shots: pl.DataFrame) -> pl.DataFrame:
    if shots.height == 0:
        return pl.DataFrame(
            schema={
                "match_id": pl.Int64,
                "team_id": pl.Int64,
                "event_id": pl.Int64,
                "shot_created": pl.Int64,
                "xA_target": pl.Float64,
                "linked_shots": pl.Int64,
                "goals_created": pl.Int64,
            }
        )

    return (
        shots
        .with_columns(
            pl.col("match_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64),
            pl.col("related_event_id").cast(pl.Int64).alias("event_id"),
            pl.col("xg").cast(pl.Float64),
            pl.col("is_goal").fill_null(0).cast(pl.Int64),
        )
        .group_by(["match_id", "team_id", "event_id"])
        .agg(
            pl.len().alias("linked_shots"),
            pl.sum("xg").alias("xA_target"),
            pl.sum("is_goal").alias("goals_created"),
        )
        .with_columns(
            (pl.col("linked_shots") > 0).cast(pl.Int64).alias("shot_created")
        )
    )


def score_passes(passes: pl.DataFrame, targets: pl.DataFrame, artifact: dict, args) -> pl.DataFrame:
    if passes.height == 0:
        return pl.DataFrame()

    clf = artifact["classifier"]
    reg = artifact["regressor"]
    calibration_factor = float(artifact.get("calibration_factor", 1.0))

    passes_feat = add_pass_features(passes)

    model_df = (
        passes_feat
        .filter(
            (pl.col("success") == 1)
            & pl.col("x").is_not_null()
            & pl.col("y").is_not_null()
            & pl.col("end_x").is_not_null()
            & pl.col("end_y").is_not_null()
        )
        .join(
            targets.select(
                "match_id",
                "team_id",
                "event_id",
                "shot_created",
                "xA_target",
                "linked_shots",
                "goals_created",
            ),
            on=["match_id", "team_id", "event_id"],
            how="left",
        )
        .with_columns(
            pl.col("shot_created").fill_null(0).cast(pl.Int64),
            pl.col("xA_target").fill_null(0.0).cast(pl.Float64),
            pl.col("linked_shots").fill_null(0).cast(pl.Int64),
            pl.col("goals_created").fill_null(0).cast(pl.Int64),
        )
    )
    if model_df.is_empty():
        return pl.DataFrame()

    base_cols = [
        "match_id",
        "tournament_id",
        "season_id",
        "season_name",
        "start_date",
        "event_id",
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
        "pass_type",
        "shot_created",
        "xA_target",
        "linked_shots",
        "goals_created",
    ]

    # Avoid duplicate Polars projection names.
    # x/y/end_x/end_y are both base columns and model features.
    # pass_type is both a base column and categorical feature.
    scoring_cols = list(dict.fromkeys(base_cols + FEATURE_COLS_NUM + FEATURE_COLS_CAT))

    pdf = (
        model_df
        .select(scoring_cols)
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
    )

    X = pdf[FEATURE_COLS_NUM + FEATURE_COLS_CAT]

    p_shot_created = clf.predict_proba(X)[:, 1]
    conditional_shot_xg = np.clip(reg.predict(X), 0.0, 1.0)
    xa_raw = p_shot_created * conditional_shot_xg
    xa_model = np.clip(xa_raw * calibration_factor, 0.0, 1.0)

    pdf["p_shot_created"] = p_shot_created
    pdf["conditional_shot_xg"] = conditional_shot_xg
    pdf["xa_model_raw"] = xa_raw
    pdf["xa_model"] = xa_model
    pdf["xa_model_name"] = args.model_name
    pdf["xa_model_version"] = args.model_version
    pdf["created_at"] = datetime.utcnow()

    return pl.from_pandas(pdf).select(
        "match_id",
        "tournament_id",
        "season_id",
        "season_name",
        "start_date",
        "event_id",
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
        "pass_type",
        "p_shot_created",
        "conditional_shot_xg",
        "xa_model_raw",
        "xa_model",
        "shot_created",
        "xA_target",
        "linked_shots",
        "goals_created",
        "xa_model_name",
        "xa_model_version",
        "created_at",
    )


# ---------------------------------------------------------------------------
# Inputs (per partition):
#   passes derived from normalized events   (actions to score)
#   feature   shots_with_models.parquet     (xG-scored shots -> xA targets)
# Output:
#   feature   passes_with_xa.parquet
# xa therefore depends on the xg step having produced shots_with_models.parquet.
# ---------------------------------------------------------------------------


def build_actual_targets_from_shots(shots: pl.DataFrame) -> pl.DataFrame:
    """Build linked-shot xA targets from a shots_with_models feature frame.

    Reads xG-scored shot columns and builds the pass-to-shot targets.
    """
    if shots.height == 0:
        return build_actual_targets(pl.DataFrame())

    needed = {"xg", "assisted", "related_event_id", "is_goal", "match_id",
              "team_id", "period_value"}
    missing = needed - set(shots.columns)
    if missing:
        # Without these we cannot link assists; return empty targets.
        return build_actual_targets(pl.DataFrame())

    linked = shots.filter(
        pl.col("xg").is_not_null()
        & (pl.col("assisted").cast(pl.Int64, strict=False) == 1)
        & pl.col("related_event_id").is_not_null()
        & (pl.col("related_event_id").cast(pl.Int64, strict=False) > 0)
    )
    return build_actual_targets(linked)
