"""Pure xG and xGOT scoring kernels used by Market Value Pulse."""

from __future__ import annotations

from datetime import datetime
import polars as pl

from .artifacts import load_artifact

from .shot_features import (
    XGOT_TYPES,
    XG_NUMERIC,
    XG_CATEGORICAL,
    XGOT_NUMERIC,
    XGOT_CATEGORICAL,
    add_xg_base_features,
    add_xg_buckets,
    apply_clip_bounds,
    add_xgot_features,
    add_xgot_trajectory_features,
    to_lgb_frame,
)


def prepare_xg_scoring_features(
    shots: pl.DataFrame,
    xg_metadata: dict,
) -> pl.DataFrame:
    df = add_xg_base_features(shots)
    df = apply_clip_bounds(df, xg_metadata["clip_bounds"])
    df = add_xg_buckets(df)
    return df


def xg_scope(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("is_own_goal") == 0)
        & (~pl.col("is_penalty"))
        & pl.col("distance_to_goal").is_not_null()
        & pl.col("angle_to_goal").is_not_null()
        & pl.col("x").is_not_null()
        & pl.col("y").is_not_null()
    )


def xgot_scope(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns([
            pl.col("type_name").is_in(XGOT_TYPES).alias("is_xgot_type"),
            (
                pl.col("goal_mouth_y").is_not_null()
                & pl.col("goal_mouth_z").is_not_null()
            ).alias("has_goalmouth"),
        ])
        .filter(
            (pl.col("is_own_goal") == 0)
            & (~pl.col("is_penalty"))
            & pl.col("is_xgot_type")
            & pl.col("has_goalmouth")
        )
    )


def predict_xg(
    xg_model,
    xg_feature_df: pl.DataFrame,
    xg_metadata: dict,
    penalty_xg: float,
    own_goal_xg: float,
) -> pl.DataFrame:
    id_cols = [
        "shot_uid",
        "match_id",
        "event_id",
        "minute",
        "second",
        "team_id",
        "player_id",
        "type_name",
        "x",
        "y",
        "is_goal",
        "is_own_goal",
        "is_penalty",
    ]

    pred_parts = []

    normal_scope = xg_feature_df.filter(
        (pl.col("is_own_goal") == 0)
        & (~pl.col("is_penalty"))
        & pl.col("distance_to_goal").is_not_null()
        & pl.col("angle_to_goal").is_not_null()
        & pl.col("x").is_not_null()
        & pl.col("y").is_not_null()
    )

    if normal_scope.height > 0:
        X = to_lgb_frame(
            normal_scope,
            numeric=xg_metadata["numeric_features"],
            categorical=xg_metadata["categorical_features"],
            target=None,
        )

        p = xg_model.predict_proba(X)[:, 1]

        pred_parts.append(
            normal_scope
            .select(id_cols)
            .with_columns([
                pl.Series("xg", p),
            ])
        )

    penalty_scope = xg_feature_df.filter(
        (pl.col("is_own_goal") == 0)
        & pl.col("is_penalty")
    )

    if penalty_scope.height > 0:
        pred_parts.append(
            penalty_scope
            .select(id_cols)
            .with_columns([
                pl.lit(float(penalty_xg)).cast(pl.Float64).alias("xg"),
            ])
        )

    own_goal_scope = xg_feature_df.filter(pl.col("is_own_goal") == 1)

    if own_goal_scope.height > 0:
        pred_parts.append(
            own_goal_scope
            .select(id_cols)
            .with_columns([
                pl.lit(float(own_goal_xg)).cast(pl.Float64).alias("xg"),
            ])
        )

    if not pred_parts:
        return pl.DataFrame()

    return (
        pl.concat(pred_parts, how="vertical")
        .with_columns([
            pl.col("is_penalty").cast(pl.UInt8),
            pl.col("is_own_goal").cast(pl.UInt8),
            pl.col("is_goal").cast(pl.UInt8),
        ])
        .unique(subset=["shot_uid"], keep="first", maintain_order=True)
    )

def predict_xgot(
    xgot_model,
    xg_feature_df: pl.DataFrame,
    xg_pred_df: pl.DataFrame,
    xgot_metadata: dict,
) -> pl.DataFrame:
    if xg_pred_df.height == 0:
        return pl.DataFrame()

    base = (
        xg_feature_df
        .join(
            xg_pred_df.select(["shot_uid", "xg"]),
            on="shot_uid",
            how="inner",
        )
    )

    scoped = xgot_scope(base)

    if scoped.height == 0:
        return pl.DataFrame()

    feat = add_xgot_features(scoped, xgot_metadata["goalframe_params"])
    feat = add_xgot_trajectory_features(feat, xgot_metadata["goalframe_params"])

    X = to_lgb_frame(
        feat,
        numeric=xgot_metadata["numeric_features"],
        categorical=xgot_metadata["categorical_features"],
        target=None,
    )

    p = xgot_model.predict_proba(X)[:, 1]

    return (
        feat
        .select(["shot_uid"])
        .with_columns([
            pl.Series("xgot", p),
            pl.lit(1).cast(pl.UInt8).alias("xgot_available"),
        ])
    )


def build_prediction_rows(
    xg_pred_df: pl.DataFrame,
    xgot_pred_df: pl.DataFrame,
    model_version: str,
    xg_model_name: str,
    xgot_model_name: str,
) -> pl.DataFrame:
    if xg_pred_df.height == 0:
        return pl.DataFrame()

    if xgot_pred_df.height > 0:
        out = xg_pred_df.join(xgot_pred_df, on="shot_uid", how="left")
    else:
        out = xg_pred_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("xgot"),
            pl.lit(None).cast(pl.UInt8).alias("xgot_available"),
        ])

    scored_at = datetime.utcnow().replace(microsecond=0)

    out = (
        out
        .with_columns([
            pl.col("xgot_available").fill_null(0).cast(pl.UInt8),
            pl.lit(model_version).alias("model_version"),
            pl.lit(xg_model_name).alias("xg_model_name"),
            pl.lit(xgot_model_name).alias("xgot_model_name"),
            pl.lit(scored_at).alias("scored_at"),
        ])
        .select([
            "model_version",
            "shot_uid",
            "match_id",
            "event_id",
            "minute",
            "second",
            "team_id",
            "player_id",
            "type_name",
            "x",
            "y",
            "is_goal",
            "is_own_goal",
            "is_penalty",
            "xg",
            "xgot",
            "xgot_available",
            "xg_model_name",
            "xgot_model_name",
            "scored_at",
        ])
    )

    return out


# ---------------------------------------------------------------------------
# Reads normalized shots, scores xG + xGOT, and writes two feature parquets:
#   shot_model_predictions.parquet  (per-shot predictions, model output grain)
#   shots_with_models.parquet       (normalized shots LEFT JOIN predictions)
# ---------------------------------------------------------------------------

SHOT_UID_FIELDS = [
    "match_id",
    "team_id",
    "event_id",
    "minute",
    "second",
    "player_id",
    "type_name",
    "x",
    "y",
]

XG_FAMILY = "xg"
XGOT_FAMILY = "xgot"


def add_shot_uid(shots: pl.DataFrame) -> pl.DataFrame:
    """Add a stable per-shot UID, preferring the normalized source identity."""
    if "event_uid" in shots.columns:
        key = pl.concat_str(
            [pl.col("match_id").cast(pl.Utf8), pl.col("event_uid").cast(pl.Utf8)],
            separator="|",
        )
        return shots.with_columns(key.hash(seed=0).cast(pl.UInt64).alias("shot_uid"))
    parts = []
    for col in SHOT_UID_FIELDS:
        if col in shots.columns:
            parts.append(pl.col(col).cast(pl.Utf8).fill_null("∅"))
        else:
            parts.append(pl.lit("∅"))
    key = pl.concat_str(parts, separator="|")
    return shots.with_columns(key.hash(seed=0).cast(pl.UInt64).alias("shot_uid"))


def load_shot_artifacts():
    """Resolve xG + xGOT artifacts from ``models/features``."""
    xg_art = load_artifact(XG_FAMILY)
    xgot_art = load_artifact(XGOT_FAMILY)

    xg_model = xg_art.load_model()
    xgot_model = xgot_art.load_model()
    xg_metadata = xg_art.load_extra_joblib("metadata.joblib")
    xgot_metadata = xgot_art.load_extra_joblib("metadata.joblib")

    return xg_art, xgot_art, xg_model, xgot_model, xg_metadata, xgot_metadata


def score_shots_frame(
    shots: pl.DataFrame,
    xg_model,
    xgot_model,
    xg_metadata: dict,
    xgot_metadata: dict,
    xg_model_name: str,
    xgot_model_name: str,
    model_version: str,
    penalty_xg: float,
    own_goal_xg: float,
) -> pl.DataFrame:
    """Pure scoring: normalized shots frame -> prediction rows frame."""
    if shots.height == 0:
        return pl.DataFrame()

    shots = add_shot_uid(shots)
    xg_features = prepare_xg_scoring_features(shots, xg_metadata)
    xg_pred = predict_xg(
        xg_model, xg_features, xg_metadata,
        penalty_xg=penalty_xg, own_goal_xg=own_goal_xg,
    )
    xgot_pred = predict_xgot(xgot_model, xg_features, xg_pred, xgot_metadata)
    return build_prediction_rows(
        xg_pred_df=xg_pred,
        xgot_pred_df=xgot_pred,
        model_version=model_version,
        xg_model_name=xg_model_name,
        xgot_model_name=xgot_model_name,
    )
