"""Pure shot feature transformations used by Market Value Pulse."""

from __future__ import annotations

import json
from dataclasses import dataclass
try:  # Needed for training and when joblib reconstructs saved LightGBM estimators.
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
import numpy as np
import pandas as pd
import polars as pl


MODEL_VERSION = "xg_lgb_v1__xgot_lgb_v2_trajectory"
XG_MODEL_NAME = "xg_lightgbm_v1_prod"
XGOT_MODEL_NAME = "xgot_lightgbm_v2_trajectory_prod"

XGOT_TYPES = ["Goal", "SavedShot", "ShotOnPost"]
EPS = 1e-6


@dataclass(frozen=True)
class ZoneConfig:
    box_x_min: float = 83.0
    box_y_min: float = 20.0
    box_y_max: float = 80.0

    six_x_min: float = 94.0
    six_y_min: float = 36.0
    six_y_max: float = 64.0

    zone14_x_min: float = 77.0
    zone14_x_max: float = 83.0
    zone14_y_min: float = 36.0
    zone14_y_max: float = 64.0

    central_y_min: float = 36.0
    central_y_max: float = 64.0

    penalty_spot_x: float = 89.5
    penalty_spot_y: float = 50.0
    penalty_spot_radius: float = 5.0


ZONE = ZoneConfig()


XG_NUMERIC = [
    "distance_to_goal",
    "log_distance",
    "inverse_distance",
    "angle_to_goal",
    "x",
    "centrality_score",

    "is_inside_box",
    "is_six_yard_box",
    "is_zone_14",
    "is_central_box",
    "is_wide_box",
    "is_penalty_spot_zone",

    "angle_inside_box",
    "angle_outside_box",
    "angle_six_yard_box",
    "angle_zone_14",

    "distance_inside_box",
    "distance_outside_box",
    "distance_six_yard_box",
    "distance_zone_14",

    "distance_angle_ratio",
    "central_close",
    "wide_angle_penalty",
    "six_yard_central",
    "box_centrality",
]

XG_CATEGORICAL = [
    "body_part_clean",
    "situation_clean",
    "shot_location_name_clean",
    "distance_bucket",
    "angle_bucket",
    "centrality_bucket",
]


XGOT_NUMERIC = [
    "xg",

    "distance_to_goal",
    "log_distance",
    "inverse_distance",
    "angle_to_goal",
    "x",
    "centrality_score",

    "is_inside_box",
    "is_six_yard_box",
    "is_zone_14",
    "is_central_box",
    "is_wide_box",
    "is_penalty_spot_zone",

    "gm_y_norm",
    "abs_gm_y_norm",
    "gm_z_norm",
    "placement_dist_from_center",
    "dist_to_nearest_post",
    "dist_to_nearest_corner",
    "dist_to_nearest_top_corner",
    "dist_to_nearest_bottom_corner",

    "is_central_placement",
    "is_wide_placement",
    "is_high_placement",
    "is_low_placement",
    "is_top_corner_placement",
    "is_bottom_corner_placement",

    "xg_top_corner",
    "xg_bottom_corner",
    "xg_central_placement",
    "xg_placement_dist",

    "is_across_goal",
    "is_near_side_finish",
    "shot_to_goal_lateral_delta",
    "abs_shot_to_goal_lateral_delta",
    "vertical_lift_per_distance",
    "wide_placement_per_distance",
    "placement_dist_per_distance",
    "vertical_trajectory_angle",
    "lateral_trajectory_angle",
    "across_goal_wide",
    "near_side_wide",
    "across_goal_low",
    "across_goal_high",

    "xg_across_goal",
    "xg_near_side",
    "xg_vertical_angle",
    "xg_lateral_angle",
    "xg_placement_dist_per_distance",
]

XGOT_CATEGORICAL = [
    "body_part_clean",
    "situation_clean",
    "shot_location_name_clean",
    "distance_bucket",
    "angle_bucket",
    "centrality_bucket",
]


def add_zone_features(df: pl.DataFrame, zone: ZoneConfig = ZONE) -> pl.DataFrame:
    is_inside_box = (
        (pl.col("x") >= zone.box_x_min)
        & (pl.col("y") >= zone.box_y_min)
        & (pl.col("y") <= zone.box_y_max)
    )

    is_six_yard_box = (
        (pl.col("x") >= zone.six_x_min)
        & (pl.col("y") >= zone.six_y_min)
        & (pl.col("y") <= zone.six_y_max)
    )

    is_zone_14 = (
        (pl.col("x") >= zone.zone14_x_min)
        & (pl.col("x") < zone.zone14_x_max)
        & (pl.col("y") >= zone.zone14_y_min)
        & (pl.col("y") <= zone.zone14_y_max)
    )

    is_central_box = (
        is_inside_box
        & (pl.col("y") >= zone.central_y_min)
        & (pl.col("y") <= zone.central_y_max)
    )

    penalty_spot_dist = (
        ((pl.col("x") - zone.penalty_spot_x) ** 2)
        + ((pl.col("y") - zone.penalty_spot_y) ** 2)
    ).sqrt()

    is_penalty_spot_zone = penalty_spot_dist <= zone.penalty_spot_radius

    return (
        df
        .with_columns([
            is_inside_box.fill_null(False).cast(pl.Int8).alias("is_inside_box"),
            is_six_yard_box.fill_null(False).cast(pl.Int8).alias("is_six_yard_box"),
            is_zone_14.fill_null(False).cast(pl.Int8).alias("is_zone_14"),
            is_central_box.fill_null(False).cast(pl.Int8).alias("is_central_box"),
            is_penalty_spot_zone.fill_null(False).cast(pl.Int8).alias("is_penalty_spot_zone"),
            penalty_spot_dist.alias("penalty_spot_dist"),
        ])
        .with_columns([
            (
                pl.col("is_inside_box").cast(pl.Boolean)
                & ~pl.col("is_central_box").cast(pl.Boolean)
            ).cast(pl.Int8).alias("is_wide_box")
        ])
    )


def add_xg_base_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_zone_features(df)

    return (
        df
        .with_columns([
            pl.col("body_part").fill_null("Unknown").cast(pl.Utf8).alias("body_part_clean"),
            pl.col("situation").fill_null("Unknown").cast(pl.Utf8).alias("situation_clean"),
            pl.col("shot_location_name").fill_null("Unknown").cast(pl.Utf8).alias("shot_location_name_clean"),

            pl.col("distance_to_goal").cast(pl.Float64),
            pl.col("angle_to_goal").cast(pl.Float64),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        ])
        .with_columns([
            pl.col("distance_to_goal").log1p().alias("log_distance"),
            (1.0 / (pl.col("distance_to_goal") + EPS)).alias("inverse_distance"),

            (pl.col("y") - 50.0).abs().alias("abs_y_from_center"),
            (1.0 - ((pl.col("y") - 50.0).abs() / 50.0)).clip(0.0, 1.0).alias("centrality_score"),

            (pl.col("angle_to_goal") / (pl.col("distance_to_goal") + EPS)).alias("distance_angle_ratio"),
        ])
        .with_columns([
            (pl.col("angle_to_goal") * pl.col("is_inside_box")).alias("angle_inside_box"),
            (pl.col("angle_to_goal") * (1 - pl.col("is_inside_box"))).alias("angle_outside_box"),
            (pl.col("angle_to_goal") * pl.col("is_six_yard_box")).alias("angle_six_yard_box"),
            (pl.col("angle_to_goal") * pl.col("is_zone_14")).alias("angle_zone_14"),

            (pl.col("distance_to_goal") * pl.col("is_inside_box")).alias("distance_inside_box"),
            (pl.col("distance_to_goal") * (1 - pl.col("is_inside_box"))).alias("distance_outside_box"),
            (pl.col("distance_to_goal") * pl.col("is_six_yard_box")).alias("distance_six_yard_box"),
            (pl.col("distance_to_goal") * pl.col("is_zone_14")).alias("distance_zone_14"),

            (pl.col("is_central_box") * pl.col("inverse_distance")).alias("central_close"),
            (pl.col("is_wide_box") * pl.col("angle_to_goal")).alias("wide_angle_penalty"),
            (pl.col("is_six_yard_box") * pl.col("centrality_score")).alias("six_yard_central"),
            (pl.col("is_inside_box") * pl.col("centrality_score")).alias("box_centrality"),

            pl.col("situation_clean")
            .str.to_lowercase()
            .str.contains("pen")
            .fill_null(False)
            .alias("is_penalty"),
        ])
    )


def get_clip_bounds(
    df: pl.DataFrame,
    cols: list[str],
    lo_q: float = 0.001,
    hi_q: float = 0.999,
) -> dict[str, tuple[float, float]]:
    bounds = {}
    for c in cols:
        row = df.select([
            pl.col(c).quantile(lo_q).alias("lo"),
            pl.col(c).quantile(hi_q).alias("hi"),
        ]).row(0, named=True)
        bounds[c] = (float(row["lo"]), float(row["hi"]))
    return bounds


def apply_clip_bounds(
    df: pl.DataFrame,
    clip_bounds: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    exprs = [
        pl.col(c).clip(lo, hi).alias(c)
        for c, (lo, hi) in clip_bounds.items()
    ]
    return df.with_columns(exprs)


def add_xg_buckets(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns([
            pl.when(pl.col("distance_to_goal") <= 3).then(pl.lit("0_3"))
            .when(pl.col("distance_to_goal") <= 6).then(pl.lit("3_6"))
            .when(pl.col("distance_to_goal") <= 9).then(pl.lit("6_9"))
            .when(pl.col("distance_to_goal") <= 12).then(pl.lit("9_12"))
            .when(pl.col("distance_to_goal") <= 15).then(pl.lit("12_15"))
            .when(pl.col("distance_to_goal") <= 18).then(pl.lit("15_18"))
            .when(pl.col("distance_to_goal") <= 21).then(pl.lit("18_21"))
            .when(pl.col("distance_to_goal") <= 24).then(pl.lit("21_24"))
            .when(pl.col("distance_to_goal") <= 30).then(pl.lit("24_30"))
            .when(pl.col("distance_to_goal") <= 40).then(pl.lit("30_40"))
            .otherwise(pl.lit("40_plus"))
            .alias("distance_bucket"),

            pl.when(pl.col("angle_to_goal") <= 0.15).then(pl.lit("0_015"))
            .when(pl.col("angle_to_goal") <= 0.30).then(pl.lit("015_030"))
            .when(pl.col("angle_to_goal") <= 0.50).then(pl.lit("030_050"))
            .when(pl.col("angle_to_goal") <= 0.75).then(pl.lit("050_075"))
            .when(pl.col("angle_to_goal") <= 1.00).then(pl.lit("075_100"))
            .when(pl.col("angle_to_goal") <= 1.25).then(pl.lit("100_125"))
            .otherwise(pl.lit("125_plus"))
            .alias("angle_bucket"),

            pl.when(pl.col("centrality_score") <= 0.40).then(pl.lit("0_040"))
            .when(pl.col("centrality_score") <= 0.60).then(pl.lit("040_060"))
            .when(pl.col("centrality_score") <= 0.75).then(pl.lit("060_075"))
            .when(pl.col("centrality_score") <= 0.85).then(pl.lit("075_085"))
            .when(pl.col("centrality_score") <= 0.92).then(pl.lit("085_092"))
            .when(pl.col("centrality_score") <= 0.97).then(pl.lit("092_097"))
            .otherwise(pl.lit("097_100"))
            .alias("centrality_bucket"),
        ])
    )


def prepare_xg_features(shots: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, tuple[float, float]]]:
    base = add_xg_base_features(shots)

    xg_scope = base.filter(
        (pl.col("is_own_goal") == 0)
        & (~pl.col("is_penalty"))
        & pl.col("distance_to_goal").is_not_null()
        & pl.col("angle_to_goal").is_not_null()
        & pl.col("x").is_not_null()
        & pl.col("y").is_not_null()
    )

    clip_bounds = get_clip_bounds(
        xg_scope,
        ["inverse_distance", "distance_angle_ratio", "central_close"],
        lo_q=0.001,
        hi_q=0.999,
    )

    out = apply_clip_bounds(base, clip_bounds)
    out = add_xg_buckets(out)

    return out, clip_bounds


def to_lgb_frame(df: pl.DataFrame, numeric: list[str], categorical: list[str], target: str | None):
    cols = ([target] if target else []) + numeric + categorical
    pdf = df.select(cols).to_pandas()

    for c in categorical:
        pdf[c] = pdf[c].astype("string").fillna("Unknown").astype("category")

    for c in numeric:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)

    if target:
        y = pdf[target].astype(int).values
        X = pdf[numeric + categorical]
        return X, y

    return pdf[numeric + categorical]


def train_xg_model(xg_feature_df: pl.DataFrame, xg_n_estimators: int) -> tuple[lgb.LGBMClassifier, pl.DataFrame]:
    xg_train = xg_feature_df.filter(
        (pl.col("is_own_goal") == 0)
        & (~pl.col("is_penalty"))
        & pl.col("distance_to_goal").is_not_null()
        & pl.col("angle_to_goal").is_not_null()
        & pl.col("x").is_not_null()
        & pl.col("y").is_not_null()
    )

    X_train, y_train = to_lgb_frame(
        xg_train,
        numeric=XG_NUMERIC,
        categorical=XG_CATEGORICAL,
        target="is_goal",
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=xg_n_estimators,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=XG_CATEGORICAL,
    )

    p = model.predict_proba(X_train)[:, 1]

    xg_pred = xg_train.select([
        "shot_row_id",
        "match_id",
        "event_id",
        "minute",
        "second",
        "team_id",
        "player_id",
        "type_name",
        "is_goal",
        "is_own_goal",
        "is_penalty",
    ]).with_columns([
        pl.Series("xg", p),
    ])

    print(
        json.dumps(
            {
                "model": XG_MODEL_NAME,
                "n_train": int(len(y_train)),
                "train_goal_rate": float(np.mean(y_train)),
                "train_pred_mean": float(np.mean(p)),
                "train_actual_goals": int(np.sum(y_train)),
                "train_pred_xg": float(np.sum(p)),
                "n_estimators": xg_n_estimators,
            },
            indent=2,
        )
    )

    return model, xg_pred


def get_goalframe_params(df: pl.DataFrame) -> dict[str, float]:
    row = df.select([
        pl.col("goal_mouth_y").quantile(0.01).alias("gm_y_p01"),
        pl.col("goal_mouth_y").quantile(0.50).alias("gm_y_p50"),
        pl.col("goal_mouth_y").quantile(0.99).alias("gm_y_p99"),
        pl.col("goal_mouth_z").quantile(0.99).alias("gm_z_p99"),
    ]).row(0, named=True)

    gm_y_low = float(row["gm_y_p01"])
    gm_y_center = float(row["gm_y_p50"])
    gm_y_high = float(row["gm_y_p99"])
    gm_y_half_width = max(gm_y_high - gm_y_center, gm_y_center - gm_y_low)
    gm_z_high = float(row["gm_z_p99"])

    return {
        "gm_y_low": gm_y_low,
        "gm_y_center": gm_y_center,
        "gm_y_high": gm_y_high,
        "gm_y_half_width": gm_y_half_width,
        "gm_z_low": 0.0,
        "gm_z_high": gm_z_high,
    }


def add_xgot_features(df: pl.DataFrame, goalframe: dict[str, float]) -> pl.DataFrame:
    gm_y_center = goalframe["gm_y_center"]
    gm_y_half_width = goalframe["gm_y_half_width"]
    gm_z_high = goalframe["gm_z_high"]

    return (
        df
        .with_columns([
            pl.col("goal_mouth_y").cast(pl.Float64),
            pl.col("goal_mouth_z").cast(pl.Float64),
            pl.col("xg").cast(pl.Float64),
        ])
        .with_columns([
            (pl.col("goal_mouth_y") - gm_y_center).alias("gm_y_centered"),
            (pl.col("goal_mouth_y") - gm_y_center).abs().alias("abs_gm_y_centered"),
            ((pl.col("goal_mouth_y") - gm_y_center) / gm_y_half_width)
            .clip(-1.5, 1.5)
            .alias("gm_y_norm"),
            (pl.col("goal_mouth_z") / gm_z_high).clip(0.0, 1.25).alias("gm_z_norm"),
        ])
        .with_columns([
            pl.col("gm_y_norm").abs().alias("abs_gm_y_norm"),
            (
                (pl.col("gm_y_norm") ** 2)
                + ((pl.col("gm_z_norm") - 0.5) ** 2)
            ).sqrt().alias("placement_dist_from_center"),
            pl.min_horizontal(
                (pl.col("gm_y_norm") + 1.0).abs(),
                (pl.col("gm_y_norm") - 1.0).abs(),
            ).alias("dist_to_nearest_post"),
            (
                ((pl.col("gm_y_norm") + 1.0) ** 2)
                + ((pl.col("gm_z_norm") - 0.0) ** 2)
            ).sqrt().alias("dist_bottom_left"),
            (
                ((pl.col("gm_y_norm") - 1.0) ** 2)
                + ((pl.col("gm_z_norm") - 0.0) ** 2)
            ).sqrt().alias("dist_bottom_right"),
            (
                ((pl.col("gm_y_norm") + 1.0) ** 2)
                + ((pl.col("gm_z_norm") - 1.0) ** 2)
            ).sqrt().alias("dist_top_left"),
            (
                ((pl.col("gm_y_norm") - 1.0) ** 2)
                + ((pl.col("gm_z_norm") - 1.0) ** 2)
            ).sqrt().alias("dist_top_right"),
        ])
        .with_columns([
            pl.min_horizontal(
                "dist_bottom_left",
                "dist_bottom_right",
                "dist_top_left",
                "dist_top_right",
            ).alias("dist_to_nearest_corner"),
            pl.min_horizontal(
                "dist_top_left",
                "dist_top_right",
            ).alias("dist_to_nearest_top_corner"),
            pl.min_horizontal(
                "dist_bottom_left",
                "dist_bottom_right",
            ).alias("dist_to_nearest_bottom_corner"),
        ])
        .with_columns([
            (pl.col("abs_gm_y_norm") <= 0.25).cast(pl.Int8).alias("is_central_placement"),
            (pl.col("abs_gm_y_norm") >= 0.70).cast(pl.Int8).alias("is_wide_placement"),
            (pl.col("gm_z_norm") >= 0.70).cast(pl.Int8).alias("is_high_placement"),
            (pl.col("gm_z_norm") <= 0.25).cast(pl.Int8).alias("is_low_placement"),
            (
                (pl.col("abs_gm_y_norm") >= 0.70)
                & (pl.col("gm_z_norm") >= 0.70)
            ).cast(pl.Int8).alias("is_top_corner_placement"),
            (
                (pl.col("abs_gm_y_norm") >= 0.70)
                & (pl.col("gm_z_norm") <= 0.25)
            ).cast(pl.Int8).alias("is_bottom_corner_placement"),
        ])
        .with_columns([
            (pl.col("xg") * pl.col("is_top_corner_placement")).alias("xg_top_corner"),
            (pl.col("xg") * pl.col("is_bottom_corner_placement")).alias("xg_bottom_corner"),
            (pl.col("xg") * pl.col("is_central_placement")).alias("xg_central_placement"),
            (pl.col("xg") * pl.col("placement_dist_from_center")).alias("xg_placement_dist"),
        ])
    )


def add_xgot_trajectory_features(df: pl.DataFrame, goalframe: dict[str, float]) -> pl.DataFrame:
    gm_y_center = goalframe["gm_y_center"]
    gm_y_half_width = goalframe["gm_y_half_width"]
    gm_z_high = goalframe["gm_z_high"]

    return (
        df
        .with_columns([
            (pl.col("y") - 50.0).alias("shot_y_centered"),
            (pl.col("y") - 50.0).abs().alias("abs_shot_y_centered"),
            (pl.col("goal_mouth_y") - gm_y_center).alias("goal_y_centered"),
            ((pl.col("goal_mouth_y") - gm_y_center) / gm_y_half_width)
            .clip(-1.5, 1.5)
            .alias("goal_y_norm"),
            (pl.col("goal_mouth_z") / gm_z_high).clip(0.0, 1.25).alias("goal_z_norm"),
        ])
        .with_columns([
            pl.when(pl.col("shot_y_centered") > 0).then(pl.lit(1.0))
            .when(pl.col("shot_y_centered") < 0).then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .alias("shot_side_sign"),

            pl.when(pl.col("goal_y_centered") > 0).then(pl.lit(1.0))
            .when(pl.col("goal_y_centered") < 0).then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .alias("goal_side_sign"),
        ])
        .with_columns([
            (
                (pl.col("shot_side_sign") != 0)
                & (pl.col("goal_side_sign") != 0)
                & (pl.col("shot_side_sign") != pl.col("goal_side_sign"))
            ).cast(pl.Int8).alias("is_across_goal"),

            (
                (pl.col("shot_side_sign") != 0)
                & (pl.col("goal_side_sign") != 0)
                & (pl.col("shot_side_sign") == pl.col("goal_side_sign"))
            ).cast(pl.Int8).alias("is_near_side_finish"),
        ])
        .with_columns([
            (
                pl.col("goal_y_norm")
                - (pl.col("shot_y_centered") / 50.0).clip(-1.0, 1.0)
            ).alias("shot_to_goal_lateral_delta"),

            (
                pl.col("goal_y_norm")
                - (pl.col("shot_y_centered") / 50.0).clip(-1.0, 1.0)
            ).abs().alias("abs_shot_to_goal_lateral_delta"),

            (pl.col("goal_z_norm") / (pl.col("distance_to_goal") + EPS)).alias("vertical_lift_per_distance"),
            (pl.col("abs_gm_y_norm") / (pl.col("distance_to_goal") + EPS)).alias("wide_placement_per_distance"),
            (pl.col("placement_dist_from_center") / (pl.col("distance_to_goal") + EPS)).alias("placement_dist_per_distance"),
        ])
        .with_columns([
            pl.arctan2(
                pl.col("goal_z_norm"),
                pl.col("distance_to_goal") + EPS,
            ).alias("vertical_trajectory_angle"),

            pl.arctan2(
                pl.col("abs_shot_to_goal_lateral_delta"),
                pl.col("distance_to_goal") + EPS,
            ).alias("lateral_trajectory_angle"),

            (pl.col("is_across_goal") * pl.col("is_wide_placement")).alias("across_goal_wide"),
            (pl.col("is_near_side_finish") * pl.col("is_wide_placement")).alias("near_side_wide"),
            (pl.col("is_across_goal") * pl.col("is_low_placement")).alias("across_goal_low"),
            (pl.col("is_across_goal") * pl.col("is_high_placement")).alias("across_goal_high"),
        ])
        .with_columns([
            (pl.col("xg") * pl.col("is_across_goal")).alias("xg_across_goal"),
            (pl.col("xg") * pl.col("is_near_side_finish")).alias("xg_near_side"),
            (pl.col("xg") * pl.col("vertical_trajectory_angle")).alias("xg_vertical_angle"),
            (pl.col("xg") * pl.col("lateral_trajectory_angle")).alias("xg_lateral_angle"),
            (pl.col("xg") * pl.col("placement_dist_per_distance")).alias("xg_placement_dist_per_distance"),
        ])
    )


def prepare_xgot_training_df(
    xg_feature_df: pl.DataFrame,
    xg_pred_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float]]:
    base = (
        xg_feature_df
        .join(
            xg_pred_df.select(["shot_row_id", "xg"]),
            on="shot_row_id",
            how="inner",
        )
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

    goalframe = get_goalframe_params(base)
    out = add_xgot_features(base, goalframe)
    out = add_xgot_trajectory_features(out, goalframe)

    return out, goalframe


def train_xgot_model(xgot_df: pl.DataFrame, xgot_n_estimators: int) -> lgb.LGBMClassifier:
    X_train, y_train = to_lgb_frame(
        xgot_df,
        numeric=XGOT_NUMERIC,
        categorical=XGOT_CATEGORICAL,
        target="is_goal",
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=xgot_n_estimators,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.08,
        reg_lambda=0.35,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=XGOT_CATEGORICAL,
    )

    p = model.predict_proba(X_train)[:, 1]

    print(
        json.dumps(
            {
                "model": XGOT_MODEL_NAME,
                "n_train": int(len(y_train)),
                "train_goal_rate": float(np.mean(y_train)),
                "train_pred_mean": float(np.mean(p)),
                "train_actual_goals": int(np.sum(y_train)),
                "train_pred_xgot": float(np.sum(p)),
                "n_estimators": xgot_n_estimators,
            },
            indent=2,
        )
    )

    return model


