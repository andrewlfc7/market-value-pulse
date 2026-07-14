"""Pure expected-assist feature transformations used by Market Value Pulse."""

from __future__ import annotations

import polars as pl


FEATURE_COLS_NUM = [
    "x",
    "y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "dx_m",
    "dy_m",
    "pass_length",
    "pass_angle",
    "pass_angle_calc",
    "pass_angle_sin",
    "pass_angle_cos",
    "pass_distance_100",
    "pass_distance_m",
    "forward_progress_100",
    "forward_progress_m",
    "lateral_movement_100",
    "lateral_movement_m",
    "start_abs_y_from_center",
    "end_abs_y_from_center",
    "end_centrality",
    "start_distance_to_goal_m",
    "end_distance_to_goal_m",
    "distance_to_goal_reduction_m",
    "end_angle_to_goal",
    "verticality",
    "horizontality",
    "ends_final_third",
    "enters_final_third",
    "ends_in_box",
    "enters_box",
    "ends_zone14",
    "ends_left_halfspace",
    "ends_right_halfspace",
    "ends_wide_final_third",
    "is_cutback_proxy",
    "is_switch_proxy",
    "is_cross",
    "is_through_ball",
    "is_long_ball",
    "is_corner",
    "is_free_kick",
    "is_throw_in",
    "is_goal_kick",
]

FEATURE_COLS_CAT = [
    "pass_type",
    "start_third",
    "end_third",
    "end_channel",
]


def add_pass_features(df: pl.DataFrame) -> pl.DataFrame:
    eps = 1e-9

    return (
        df
        .with_columns(
            pl.col("minute").fill_null(0).cast(pl.Int64),
            pl.col("second").fill_null(0).cast(pl.Int64),
            pl.col("expanded_minute").fill_null(pl.col("minute")).cast(pl.Int64),
            pl.col("period_value").fill_null(0).cast(pl.Int64),

            pl.col("match_id").cast(pl.Int64),
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("season_id").cast(pl.Int64, strict=False),
            pl.col("event_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64, strict=False),

            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("end_x").cast(pl.Float64),
            pl.col("end_y").cast(pl.Float64),
            pl.col("pass_length").cast(pl.Float64, strict=False),
            pl.col("pass_angle").cast(pl.Float64, strict=False),

            pl.col("success").fill_null(0).cast(pl.Int64),
            pl.col("is_cross").fill_null(0).cast(pl.Int64),
            pl.col("is_through_ball").fill_null(0).cast(pl.Int64),
            pl.col("is_long_ball").fill_null(0).cast(pl.Int64),
            pl.col("is_corner").fill_null(0).cast(pl.Int64),
            pl.col("is_free_kick").fill_null(0).cast(pl.Int64),
            pl.col("is_throw_in").fill_null(0).cast(pl.Int64),
            pl.col("is_goal_kick").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (pl.col("expanded_minute") * 60 + pl.col("second")).alias("event_time_seconds"),

            (pl.col("end_x") - pl.col("x")).alias("dx"),
            (pl.col("end_y") - pl.col("y")).alias("dy"),

            (105.0 * (pl.col("end_x") - pl.col("x")) / 100.0).alias("dx_m"),
            (68.0 * (pl.col("end_y") - pl.col("y")) / 100.0).alias("dy_m"),

            (105.0 * pl.col("x") / 100.0).alias("start_x_m"),
            (68.0 * pl.col("y") / 100.0).alias("start_y_m"),
            (105.0 * pl.col("end_x") / 100.0).alias("end_x_m"),
            (68.0 * pl.col("end_y") / 100.0).alias("end_y_m"),
        )
        .with_columns(
            ((pl.col("dx") ** 2 + pl.col("dy") ** 2) ** 0.5).alias("pass_distance_100"),
            ((pl.col("dx_m") ** 2 + pl.col("dy_m") ** 2) ** 0.5).alias("pass_distance_m"),

            pl.col("dx").alias("forward_progress_100"),
            pl.col("dx_m").alias("forward_progress_m"),
            pl.col("dy").abs().alias("lateral_movement_100"),
            pl.col("dy_m").abs().alias("lateral_movement_m"),

            (pl.col("y") - 50.0).abs().alias("start_abs_y_from_center"),
            (pl.col("end_y") - 50.0).abs().alias("end_abs_y_from_center"),
            (50.0 - (pl.col("end_y") - 50.0).abs()).alias("end_centrality"),

            (((105.0 - pl.col("start_x_m")) ** 2 + (34.0 - pl.col("start_y_m")) ** 2) ** 0.5)
            .alias("start_distance_to_goal_m"),
            (((105.0 - pl.col("end_x_m")) ** 2 + (34.0 - pl.col("end_y_m")) ** 2) ** 0.5)
            .alias("end_distance_to_goal_m"),
        )
        .with_columns(
            (pl.col("start_distance_to_goal_m") - pl.col("end_distance_to_goal_m"))
            .alias("distance_to_goal_reduction_m"),

            (pl.col("dx_m") / (pl.col("pass_distance_m") + eps)).alias("verticality"),
            (pl.col("dy_m").abs() / (pl.col("pass_distance_m") + eps)).alias("horizontality"),

            pl.arctan2(
                7.32 * (105.0 - pl.col("end_x_m")),
                ((105.0 - pl.col("end_x_m")) ** 2 + (34.0 - pl.col("end_y_m")) ** 2 - (7.32 / 2.0) ** 2),
            ).abs().alias("end_angle_to_goal"),

            pl.arctan2(pl.col("dy_m"), pl.col("dx_m")).alias("pass_angle_calc"),
        )
        .with_columns(
            pl.col("pass_angle_calc").sin().alias("pass_angle_sin"),
            pl.col("pass_angle_calc").cos().alias("pass_angle_cos"),

            (pl.col("end_x") >= 66.7).cast(pl.Int8).alias("ends_final_third"),
            ((pl.col("x") < 66.7) & (pl.col("end_x") >= 66.7)).cast(pl.Int8).alias("enters_final_third"),

            ((pl.col("end_x") >= 85.0) & (pl.col("end_y").is_between(22.5, 77.5)))
            .cast(pl.Int8)
            .alias("ends_in_box"),

            (
                ~((pl.col("x") >= 85.0) & (pl.col("y").is_between(22.5, 77.5)))
                & ((pl.col("end_x") >= 85.0) & (pl.col("end_y").is_between(22.5, 77.5)))
            ).cast(pl.Int8).alias("enters_box"),

            ((pl.col("end_x") >= 66.7) & (pl.col("end_x") < 85.0) & (pl.col("end_y").is_between(35.0, 65.0)))
            .cast(pl.Int8)
            .alias("ends_zone14"),

            ((pl.col("end_x") >= 66.7) & (pl.col("end_y").is_between(18.0, 35.0)))
            .cast(pl.Int8)
            .alias("ends_left_halfspace"),

            ((pl.col("end_x") >= 66.7) & (pl.col("end_y").is_between(65.0, 82.0)))
            .cast(pl.Int8)
            .alias("ends_right_halfspace"),

            ((pl.col("end_x") >= 66.7) & ((pl.col("end_y") < 18.0) | (pl.col("end_y") > 82.0)))
            .cast(pl.Int8)
            .alias("ends_wide_final_third"),

            (
                (pl.col("x") >= 85.0)
                & ((pl.col("y") <= 22.5) | (pl.col("y") >= 77.5))
                & (pl.col("end_x") <= pl.col("x") + 2.0)
                & (pl.col("end_x") >= 75.0)
                & (pl.col("end_y").is_between(25.0, 75.0))
            ).cast(pl.Int8).alias("is_cutback_proxy"),

            (
                (pl.col("lateral_movement_100") >= 35.0)
                & (pl.col("pass_distance_100") >= 35.0)
            ).cast(pl.Int8).alias("is_switch_proxy"),
        )
        .with_columns(
            pl.when(pl.col("end_y") < 18.0).then(pl.lit("left_wide"))
            .when(pl.col("end_y") < 35.0).then(pl.lit("left_halfspace"))
            .when(pl.col("end_y") <= 65.0).then(pl.lit("central"))
            .when(pl.col("end_y") <= 82.0).then(pl.lit("right_halfspace"))
            .otherwise(pl.lit("right_wide"))
            .alias("end_channel"),

            pl.when(pl.col("x") < 33.3).then(pl.lit("defensive_third"))
            .when(pl.col("x") < 66.7).then(pl.lit("middle_third"))
            .otherwise(pl.lit("final_third"))
            .alias("start_third"),

            pl.when(pl.col("end_x") < 33.3).then(pl.lit("defensive_third"))
            .when(pl.col("end_x") < 66.7).then(pl.lit("middle_third"))
            .otherwise(pl.lit("final_third"))
            .alias("end_third"),

            pl.when(pl.col("is_corner") == 1).then(pl.lit("corner"))
            .when(pl.col("is_free_kick") == 1).then(pl.lit("free_kick"))
            .when(pl.col("is_cross") == 1).then(pl.lit("cross"))
            .when(pl.col("is_through_ball") == 1).then(pl.lit("through_ball"))
            .when(pl.col("is_long_ball") == 1).then(pl.lit("long_ball"))
            .when(pl.col("is_throw_in") == 1).then(pl.lit("throw_in"))
            .when(pl.col("is_goal_kick") == 1).then(pl.lit("goal_kick"))
            .when(pl.col("is_cutback_proxy") == 1).then(pl.lit("cutback_proxy"))
            .when(pl.col("is_switch_proxy") == 1).then(pl.lit("switch_proxy"))
            .otherwise(pl.lit("pass_other"))
            .alias("pass_type"),
        )
    )


