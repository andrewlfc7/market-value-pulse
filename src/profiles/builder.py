#!/usr/bin/env python3
"""Build role-relative player profiles and similar-player serving tables.

Example:
    uv run python scripts/build_player_profiles.py \
        --competition EPL \
        --season 2025-2026

The command writes partitioned outputs and, by default, publishes aliases at:

    data/serving/player_profiles.parquet
    data/serving/player_similarities.parquet

The API reads those aliases. Partitioned copies are retained under:

    data/serving/profiles/competition=<competition>/season=<season>/
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl


PROFILE_MINUTES_DEFAULT = 270
BENCHMARK_MINUTES_DEFAULT = 900
TOP_N_DEFAULT = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: Path) -> Path:
    """Resolve relative paths from the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path

PASS_ATTEMPT_MINIMUM = 200
DRIBBLE_ATTEMPT_MINIMUM = 15
DUEL_ATTEMPT_MINIMUM = 30
AERIAL_ATTEMPT_MINIMUM = 20
TACKLE_ATTEMPT_MINIMUM = 20

LOWER_WINSOR = 0.02
UPPER_WINSOR = 0.98


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    phase: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class BuildResult:
    profiles_path: Path
    similarities_path: Path
    profile_players: int
    profile_rows: int
    similarity_rows: int


def metric(
    key: str,
    label: str,
    phase: str,
    higher_is_better: bool = True,
) -> MetricSpec:
    return MetricSpec(
        key=key,
        label=label,
        phase=phase,
        higher_is_better=higher_is_better,
    )


ROLE_METRICS: dict[str, list[MetricSpec]] = {
    "ATTACKING_MIDFIELDER": [
        metric("open_play_shots_90", "Open-play shots", "Scoring"),
        metric("non_penalty_goals_90", "Non-penalty goals", "Scoring"),
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("non_penalty_xgot_90", "Non-penalty xGOT", "Scoring"),
        metric("finishing_above_expected_90", "Finishing above xG", "Scoring"),
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("big_chances_created_90", "Big chances created", "Creation"),
        metric("passes_into_box_90", "Passes into box", "Creation"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Creation"),
        metric("pass_completion_pct", "Pass completion", "Progression"),
        metric("progressive_passes_90", "Progressive passes", "Progression"),
        metric("xt_pass_added_90", "xT from passing", "Progression"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("takeon_completed_90", "Dribbles completed", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("ball_recovery_90", "Recoveries", "Defending"),
        metric("duel_win_pct", "Duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
    ],
    "WINGER": [
        metric("open_play_shots_90", "Open-play shots", "Scoring"),
        metric("non_penalty_goals_90", "Non-penalty goals", "Scoring"),
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("non_penalty_xgot_90", "Non-penalty xGOT", "Scoring"),
        metric("shot_placement_above_expected_90", "Shot placement", "Scoring"),
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("big_chances_created_90", "Big chances created", "Creation"),
        metric("passes_into_box_90", "Passes into box", "Creation"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Creation"),
        metric("pass_completion_pct", "Pass completion", "Progression"),
        metric("progressive_passes_90", "Progressive passes", "Progression"),
        metric("xt_pass_added_90", "xT from passing", "Progression"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("carries_into_box_90", "Carries into box", "Progression"),
        metric("takeon_completed_90", "Dribbles completed", "Progression"),
        metric("dribble_success_pct", "Dribble success", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("duel_win_pct", "Duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
    ],
    "STRIKER": [
        metric("open_play_shots_90", "Open-play shots", "Scoring"),
        metric("open_play_shots_on_target_90", "Open-play shots on target", "Scoring"),
        metric("non_penalty_goals_90", "Non-penalty goals", "Scoring"),
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("non_penalty_xgot_90", "Non-penalty xGOT", "Scoring"),
        metric("average_xg_per_shot", "Average xG per shot", "Scoring"),
        metric("finishing_above_expected_90", "Finishing above xG", "Scoring"),
        metric("shot_placement_above_expected_90", "Shot placement", "Scoring"),
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("big_chances_created_90", "Big chances created", "Creation"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Creation"),
        metric("carries_into_box_90", "Carries into box", "Progression"),
        metric("takeon_completed_90", "Dribbles completed", "Progression"),
        metric("xpv_total_added_90", "Goal probability added", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("aerial_duel_win_pct", "Aerial duel win rate", "Defending"),
        metric("ground_duel_win_pct", "Ground duel win rate", "Defending"),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
    ],
    "CENTRAL_MIDFIELDER": [
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("big_chances_created_90", "Big chances created", "Creation"),
        metric("passes_into_box_90", "Passes into box", "Creation"),
        metric("pass_attempt_90", "Pass volume", "Passing"),
        metric("pass_completion_pct", "Pass completion", "Passing"),
        metric(
            "pass_completion_above_expected_pct",
            "Completion above expected",
            "Passing",
        ),
        metric("progressive_passes_90", "Progressive passes", "Passing"),
        metric("passes_into_final_third_90", "Passes into final third", "Passing"),
        metric("xt_pass_added_90", "xT from passing", "Passing"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Passing"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("final_third_carries_90", "Final-third carries", "Progression"),
        metric("takeon_completed_90", "Dribbles completed", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("ball_recovery_90", "Recoveries", "Defending"),
        metric("duel_win_pct", "Duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
    ],
    "DEFENSIVE_MIDFIELDER": [
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("passes_into_box_90", "Passes into box", "Creation"),
        metric("pass_attempt_90", "Pass volume", "Passing"),
        metric("pass_completion_pct", "Pass completion", "Passing"),
        metric(
            "pass_completion_above_expected_pct",
            "Completion above expected",
            "Passing",
        ),
        metric("progressive_passes_90", "Progressive passes", "Passing"),
        metric("progressive_pass_share_pct", "Progressive pass share", "Passing"),
        metric("passes_into_final_third_90", "Passes into final third", "Passing"),
        metric("xt_pass_added_90", "xT from passing", "Passing"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("tackle_won_90", "Tackles won", "Defending"),
        metric("interception_90", "Interceptions", "Defending"),
        metric("ball_recovery_90", "Recoveries", "Defending"),
        metric("duel_win_pct", "Duel win rate", "Defending"),
        metric("aerial_duel_win_pct", "Aerial duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
        metric(
            "defensive_net_threat_reduction_90",
            "Net threat reduction",
            "Defending",
        ),
    ],
    "FULL_BACK": [
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("xa_90", "Expected assists", "Creation"),
        metric("key_passes_90", "Key passes", "Creation"),
        metric("big_chances_created_90", "Big chances created", "Creation"),
        metric("passes_into_box_90", "Passes into box", "Creation"),
        metric("pass_completion_pct", "Pass completion", "Passing"),
        metric("progressive_passes_90", "Progressive passes", "Passing"),
        metric("passes_into_final_third_90", "Passes into final third", "Passing"),
        metric("xt_pass_added_90", "xT from passing", "Passing"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Passing"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("final_third_carries_90", "Final-third carries", "Progression"),
        metric("carries_into_box_90", "Carries into box", "Progression"),
        metric("takeon_completed_90", "Dribbles completed", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("ball_recovery_90", "Recoveries", "Defending"),
        metric("duel_win_pct", "Duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
        metric(
            "defensive_net_threat_reduction_90",
            "Net threat reduction",
            "Defending",
        ),
    ],
    "CENTRE_BACK": [
        metric("non_penalty_xg_90", "Non-penalty xG", "Scoring"),
        metric("open_play_shots_90", "Open-play shots", "Scoring"),
        metric("non_penalty_goals_90", "Non-penalty goals", "Scoring"),
        metric("pass_attempt_90", "Pass volume", "Passing"),
        metric("pass_completion_pct", "Pass completion", "Passing"),
        metric(
            "pass_completion_above_expected_pct",
            "Completion above expected",
            "Passing",
        ),
        metric("progressive_passes_90", "Progressive passes", "Passing"),
        metric("progressive_pass_share_pct", "Progressive pass share", "Passing"),
        metric("passes_into_final_third_90", "Passes into final third", "Passing"),
        metric("xt_pass_added_90", "xT from passing", "Passing"),
        metric("progressive_carries_90", "Progressive carries", "Progression"),
        metric("turnovers_90", "Ball security", "Progression", False),
        metric("defensive_actions_90", "Defensive actions", "Defending"),
        metric("tackle_won_90", "Tackles won", "Defending"),
        metric("interception_90", "Interceptions", "Defending"),
        metric("ball_recovery_90", "Recoveries", "Defending"),
        metric("blocked_pass_90", "Blocks", "Defending"),
        metric("aerial_duel_win_pct", "Aerial duel win rate", "Defending"),
        metric("opponent_threat_prevented_90", "Threat prevented", "Defending"),
        metric("error_leading_to_shot_90", "Error avoidance", "Defending", False),
    ],
    "GOALKEEPER": [
        metric("shots_on_target_faced_90", "Shots on target faced", "Shot stopping"),
        metric("xgot_faced_90", "xGOT faced", "Shot stopping"),
        metric(
            "goals_conceded_90",
            "Goals conceded prevention",
            "Shot stopping",
            False,
        ),
        metric("pass_attempt_90", "Pass volume", "Distribution"),
        metric("pass_completion_pct", "Pass completion", "Distribution"),
        metric("progressive_passes_90", "Progressive passes", "Distribution"),
        metric("xt_pass_added_90", "xT from passing", "Distribution"),
        metric("xpv_pass_added_90", "Goal probability added: pass", "Distribution"),
        metric("turnovers_90", "Ball security", "Distribution", False),
        metric("ball_recovery_90", "Recoveries", "Sweeping"),
        metric("clearance_90", "Clearances", "Sweeping"),
        metric("aerial_duel_win_pct", "Aerial duel win rate", "Sweeping"),
    ],
}


def role_expression(column: str = "position") -> pl.Expr:
    position = (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z]", "")
    )
    return (
        pl.when(position == "GK")
        .then(pl.lit("GOALKEEPER"))
        .when(position == "DC")
        .then(pl.lit("CENTRE_BACK"))
        .when(position.is_in(["DL", "DR", "DML", "DMR"]))
        .then(pl.lit("FULL_BACK"))
        .when(position == "DMC")
        .then(pl.lit("DEFENSIVE_MIDFIELDER"))
        .when(position == "MC")
        .then(pl.lit("CENTRAL_MIDFIELDER"))
        .when(position == "AMC")
        .then(pl.lit("ATTACKING_MIDFIELDER"))
        .when(position.is_in(["ML", "MR", "AML", "AMR", "FWL", "FWR"]))
        .then(pl.lit("WINGER"))
        .when(position == "FW")
        .then(pl.lit("STRIKER"))
        .otherwise(pl.lit("IGNORE"))
        .alias("role_group")
    )


def _existing_columns(path: Path, requested: Iterable[str]) -> list[str]:
    schema = pl.read_parquet_schema(path)
    return [column for column in requested if column in schema]


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _sum_player_frames(
    frames: list[pl.DataFrame],
    *,
    id_column: str,
) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame({id_column: []}, schema={id_column: pl.Int64})

    combined = pl.concat(frames, how="diagonal_relaxed")
    value_columns = [
        column for column in combined.columns if column != id_column
    ]
    return combined.group_by(id_column).agg(
        *[pl.col(column).fill_null(0).sum().alias(column) for column in value_columns]
    )


def _flag(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)

    dtype = frame.schema[column]
    if dtype == pl.Boolean:
        return pl.col(column).fill_null(False)

    return (
        pl.col(column).is_not_null()
        & (
            pl.col(column)
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
            != ""
        )
    )


def _aggregate_events(path: Path) -> pl.DataFrame:
    requested = [
        "player_id",
        "type_display_name",
        "outcome_type_display_name",
        "x",
        "y",
        "end_x",
        "end_y",
        "q_CornerTaken",
        "q_FromCorner",
        "q_FreekickTaken",
        "q_IndirectFreekickTaken",
        "q_ThrowIn",
        "q_GoalKick",
        "q_SetPiece",
        "q_ThrowinSetPiece",
        "q_KeyPass",
        "q_BigChanceCreated",
        "q_LeadingToAttempt",
        "q_LeadingToGoal",
    ]
    frame = pl.read_parquet(path, columns=_existing_columns(path, requested))

    required_defaults: dict[str, Any] = {
        "player_id": None,
        "type_display_name": "",
        "outcome_type_display_name": "",
        "x": None,
        "y": None,
        "end_x": None,
        "end_y": None,
    }
    missing_expressions = [
        pl.lit(value).alias(column)
        for column, value in required_defaults.items()
        if column not in frame.columns
    ]
    if missing_expressions:
        frame = frame.with_columns(*missing_expressions)

    successful = (
        pl.col("outcome_type_display_name")
        .cast(pl.String)
        .fill_null("")
        .str.to_lowercase()
        == "successful"
    )
    event_type = pl.col("type_display_name").cast(pl.String).fill_null("")

    is_set_piece_pass = (
        _flag(frame, "q_CornerTaken")
        | _flag(frame, "q_FromCorner")
        | _flag(frame, "q_FreekickTaken")
        | _flag(frame, "q_IndirectFreekickTaken")
        | _flag(frame, "q_ThrowIn")
        | _flag(frame, "q_GoalKick")
        | _flag(frame, "q_SetPiece")
        | _flag(frame, "q_ThrowinSetPiece")
    )

    end_in_final_third = pl.col("end_x").cast(pl.Float64, strict=False).fill_null(0) >= 66.67
    start_before_final_third = pl.col("x").cast(pl.Float64, strict=False).fill_null(0) < 66.67

    end_in_box = (
        (pl.col("end_x").cast(pl.Float64, strict=False).fill_null(0) >= 83.0)
        & pl.col("end_y")
        .cast(pl.Float64, strict=False)
        .fill_null(-1)
        .is_between(21.0, 79.0)
    )
    start_in_box = (
        (pl.col("x").cast(pl.Float64, strict=False).fill_null(0) >= 83.0)
        & pl.col("y")
        .cast(pl.Float64, strict=False)
        .fill_null(-1)
        .is_between(21.0, 79.0)
    )

    prepared = frame.with_columns(
        (
            (event_type == "Pass")
            & successful
            & ~is_set_piece_pass
            & start_before_final_third
            & end_in_final_third
        ).cast(pl.Int64).alias("passes_into_final_third"),
        (
            (event_type == "Pass")
            & successful
            & ~is_set_piece_pass
            & ~start_in_box
            & end_in_box
        ).cast(pl.Int64).alias("passes_into_box"),
        ((event_type == "Pass") & _flag(frame, "q_KeyPass"))
        .cast(pl.Int64)
        .alias("event_key_pass"),
        ((event_type == "Pass") & _flag(frame, "q_BigChanceCreated"))
        .cast(pl.Int64)
        .alias("event_big_chance_created"),
        (event_type == "Pass").cast(pl.Int64).alias("pass_attempt"),
        ((event_type == "Pass") & successful)
        .cast(pl.Int64)
        .alias("pass_completed"),
        (event_type == "TakeOn").cast(pl.Int64).alias("takeon_attempt"),
        ((event_type == "TakeOn") & successful)
        .cast(pl.Int64)
        .alias("takeon_completed"),
        ((event_type == "TakeOn") & ~successful)
        .cast(pl.Int64)
        .alias("failed_takeon"),
        (event_type == "Dispossessed").cast(pl.Int64).alias("dispossessed"),
        ((event_type == "BallTouch") & ~successful)
        .cast(pl.Int64)
        .alias("miscontrol_proxy"),
        (event_type == "Tackle").cast(pl.Int64).alias("tackle_attempt"),
        ((event_type == "Tackle") & successful)
        .cast(pl.Int64)
        .alias("tackle_won"),
        (event_type == "Aerial").cast(pl.Int64).alias("aerial_attempt"),
        ((event_type == "Aerial") & successful)
        .cast(pl.Int64)
        .alias("aerial_won"),
        (event_type == "Interception").cast(pl.Int64).alias("interception"),
        (event_type == "BallRecovery").cast(pl.Int64).alias("ball_recovery"),
        (event_type == "BlockedPass").cast(pl.Int64).alias("blocked_pass"),
        (event_type == "Clearance").cast(pl.Int64).alias("clearance"),
        (
            (event_type == "Error")
            & _flag(frame, "q_LeadingToAttempt")
        ).cast(pl.Int64).alias("error_leading_to_shot"),
        (
            (event_type == "Error")
            & _flag(frame, "q_LeadingToGoal")
        ).cast(pl.Int64).alias("error_leading_to_goal"),
    )

    aggregate_columns = [
        "passes_into_final_third",
        "passes_into_box",
        "event_key_pass",
        "event_big_chance_created",
        "pass_attempt",
        "pass_completed",
        "takeon_attempt",
        "takeon_completed",
        "failed_takeon",
        "dispossessed",
        "miscontrol_proxy",
        "tackle_attempt",
        "tackle_won",
        "aerial_attempt",
        "aerial_won",
        "interception",
        "ball_recovery",
        "blocked_pass",
        "clearance",
        "error_leading_to_shot",
        "error_leading_to_goal",
    ]

    return (
        prepared.filter(pl.col("player_id").is_not_null())
        .group_by("player_id")
        .agg(
            *[
                pl.col(column).sum().alias(column)
                for column in aggregate_columns
            ]
        )
    )


def _aggregate_shots(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    defaults: dict[str, tuple[Any, pl.DataType]] = {
        "player_id": (None, pl.Int64),
        "type_name": ("", pl.String),
        "situation": ("", pl.String),
        "is_penalty": (0, pl.Int64),
        "is_goal": (0, pl.Int64),
        "xg": (0.0, pl.Float64),
        "xgot": (0.0, pl.Float64),
        "q_BigChance": (False, pl.Boolean),
    }
    for column, (value, dtype) in defaults.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(value).cast(dtype).alias(column))

    open_play = pl.col("situation").cast(pl.String).is_in(["RegularPlay", "FastBreak"])
    non_penalty = pl.col("is_penalty").cast(pl.Int64, strict=False).fill_null(0) == 0
    on_target = (
        pl.col("is_goal").cast(pl.Boolean, strict=False).fill_null(False)
        | (pl.col("type_name").cast(pl.String) == "SavedShot")
    )

    prepared = frame.with_columns(
        open_play.cast(pl.Int64).alias("open_play_shots"),
        (open_play & on_target)
        .cast(pl.Int64)
        .alias("open_play_shots_on_target"),
        non_penalty.cast(pl.Int64).alias("non_penalty_shots"),
        (
            non_penalty
            & pl.col("is_goal").cast(pl.Boolean, strict=False).fill_null(False)
        ).cast(pl.Int64).alias("non_penalty_goals"),
        pl.when(non_penalty)
        .then(pl.col("xg").cast(pl.Float64, strict=False).fill_null(0))
        .otherwise(0.0)
        .alias("non_penalty_xg"),
        pl.when(non_penalty)
        .then(pl.col("xgot").cast(pl.Float64, strict=False).fill_null(0))
        .otherwise(0.0)
        .alias("non_penalty_xgot"),
        _flag(frame, "q_BigChance")
        .cast(pl.Int64)
        .alias("big_chance_shots"),
    )

    columns = [
        "open_play_shots",
        "open_play_shots_on_target",
        "non_penalty_shots",
        "non_penalty_goals",
        "non_penalty_xg",
        "non_penalty_xgot",
        "big_chance_shots",
    ]
    return (
        prepared.filter(pl.col("player_id").is_not_null())
        .group_by("player_id")
        .agg(*[pl.col(column).sum().alias(column) for column in columns])
    )


def _find_column(frame: pl.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        exact = lookup.get(candidate.lower())
        if exact is not None:
            return exact
    return None


def _aggregate_action_values(path: Path, prefix: str) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    player_column = _find_column(frame, ["player_id", "whoscored_player_id"])
    action_column = _find_column(frame, ["action_type", "type"])
    value_column = _find_column(
        frame,
        [
            f"{prefix}_added",
            f"{prefix.lower()}_added",
            "value_added",
        ],
    )

    if player_column is None or action_column is None or value_column is None:
        return pl.DataFrame(
            {"player_id": []},
            schema={"player_id": pl.Int64},
        )

    normalized_prefix = prefix.lower()
    return (
        frame.filter(pl.col(player_column).is_not_null())
        .group_by(player_column)
        .agg(
            pl.when(pl.col(action_column).cast(pl.String).str.to_lowercase() == "pass")
            .then(pl.col(value_column).cast(pl.Float64, strict=False).fill_null(0))
            .otherwise(0.0)
            .sum()
            .alias(f"{normalized_prefix}_pass_added"),
            pl.when(pl.col(action_column).cast(pl.String).str.to_lowercase() == "carry")
            .then(pl.col(value_column).cast(pl.Float64, strict=False).fill_null(0))
            .otherwise(0.0)
            .sum()
            .alias(f"{normalized_prefix}_carry_added"),
            pl.col(value_column)
            .cast(pl.Float64, strict=False)
            .fill_null(0)
            .sum()
            .alias(f"{normalized_prefix}_total_added"),
        )
        .rename({player_column: "player_id"})
    )


def _build_roles(ratings: pl.DataFrame) -> pl.DataFrame:
    positioned = ratings.with_columns(role_expression())
    role_minutes = (
        positioned.filter(pl.col("role_group") != "IGNORE")
        .group_by(["whoscored_player_id", "role_group"])
        .agg(pl.col("minutes").fill_null(0).sum().alias("role_minutes"))
        .sort(
            ["whoscored_player_id", "role_minutes"],
            descending=[False, True],
        )
    )

    records: list[dict[str, Any]] = []
    for player_id, group in role_minutes.to_pandas().groupby(
        "whoscored_player_id",
        sort=False,
    ):
        ordered = group.sort_values("role_minutes", ascending=False).reset_index(drop=True)
        total = float(ordered["role_minutes"].sum())
        primary = ordered.iloc[0]
        secondary = ordered.iloc[1] if len(ordered) > 1 else None
        primary_share = float(primary["role_minutes"]) / total if total > 0 else math.nan

        records.append(
            {
                "whoscored_player_id": int(player_id),
                "primary_role": str(primary["role_group"]),
                "primary_role_minutes": float(primary["role_minutes"]),
                "primary_role_share": primary_share,
                "secondary_role": (
                    str(secondary["role_group"])
                    if secondary is not None
                    else None
                ),
                "secondary_role_minutes": (
                    float(secondary["role_minutes"])
                    if secondary is not None
                    else 0.0
                ),
                "is_hybrid_role": bool(primary_share < 0.60),
            }
        )

    if not records:
        raise ValueError("No usable detailed WhoScored positions were found")

    return pl.from_pandas(pd.DataFrame(records))


def _build_player_base(ratings: pl.DataFrame, roles: pl.DataFrame) -> pl.DataFrame:
    sum_candidates = [
        "shots",
        "goals",
        "xg",
        "xgot",
        "passes",
        "completed_passes",
        "key_passes",
        "progressive_passes",
        "assists",
        "xa",
        "carries",
        "progressive_carries",
        "final_third_carries",
        "carries_into_box",
        "xt_added",
        "xpv_added",
        "opponent_threat_prevented",
        "defensive_net_threat_reduction",
        "big_chances_created",
        "big_chances_missed",
        "big_chance_xg_missed",
        "shots_on_target_faced",
        "xgot_faced",
        "goals_conceded",
    ]
    available = [column for column in sum_candidates if column in ratings.columns]

    aggregate_expressions: list[pl.Expr] = [
        pl.col("player_name").drop_nulls().last().alias("player_name"),
        pl.col("team_id").drop_nulls().last().alias("team_id"),
        pl.col("season").drop_nulls().last().alias("season"),
        pl.col("minutes").fill_null(0).sum().alias("minutes"),
        pl.col("match_id").n_unique().alias("appearances"),
        pl.col("started").cast(pl.Int64, strict=False).fill_null(0).sum().alias("starts"),
    ]
    aggregate_expressions.extend(
        pl.col(column).cast(pl.Float64, strict=False).fill_null(0).sum().alias(column)
        for column in available
    )

    if {
        "pass_completion_above_expected",
        "passes",
    }.issubset(ratings.columns):
        aggregate_expressions.append(
            (
                pl.col("pass_completion_above_expected")
                .cast(pl.Float64, strict=False)
                .fill_null(0)
                * pl.col("passes").cast(pl.Float64, strict=False).fill_null(0)
            )
            .sum()
            .alias("pass_completion_above_expected_weighted")
        )
    else:
        aggregate_expressions.append(
            pl.lit(0.0).alias("pass_completion_above_expected_weighted")
        )

    base = (
        ratings.group_by("whoscored_player_id")
        .agg(*aggregate_expressions)
        .join(roles, on="whoscored_player_id", how="inner")
    )

    for column in sum_candidates:
        if column not in base.columns:
            base = base.with_columns(pl.lit(0.0).alias(column))

    return base


def _add_derived_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    count_defaults = [
        "open_play_shots",
        "open_play_shots_on_target",
        "non_penalty_shots",
        "non_penalty_goals",
        "non_penalty_xg",
        "non_penalty_xgot",
        "big_chance_shots",
        "passes_into_final_third",
        "passes_into_box",
        "event_key_pass",
        "event_big_chance_created",
        "pass_attempt",
        "pass_completed",
        "takeon_attempt",
        "takeon_completed",
        "failed_takeon",
        "dispossessed",
        "miscontrol_proxy",
        "tackle_attempt",
        "tackle_won",
        "aerial_attempt",
        "aerial_won",
        "interception",
        "ball_recovery",
        "blocked_pass",
        "clearance",
        "error_leading_to_shot",
        "error_leading_to_goal",
        "xt_pass_added",
        "xt_carry_added",
        "xt_total_added",
        "xpv_pass_added",
        "xpv_carry_added",
        "xpv_total_added",
    ]
    for column in count_defaults:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0).alias(column))

    frame = frame.with_columns(
        (
            pl.col("dispossessed")
            + pl.col("failed_takeon")
            + pl.col("miscontrol_proxy")
        ).alias("turnovers"),
        (
            pl.col("tackle_attempt")
            + pl.col("takeon_attempt")
        ).alias("ground_duels_attempted"),
        (
            pl.col("tackle_won")
            + pl.col("takeon_completed")
        ).alias("ground_duels_won"),
        (
            pl.col("aerial_attempt")
            + pl.col("tackle_attempt")
            + pl.col("takeon_attempt")
        ).alias("duels_attempted"),
        (
            pl.col("aerial_won")
            + pl.col("tackle_won")
            + pl.col("takeon_completed")
        ).alias("duels_won"),
        (
            pl.col("tackle_won")
            + pl.col("interception")
            + pl.col("ball_recovery")
            + pl.col("blocked_pass")
            + pl.col("clearance")
        ).alias("defensive_actions"),
    )

    per_90_columns = [
        "open_play_shots",
        "open_play_shots_on_target",
        "non_penalty_shots",
        "non_penalty_goals",
        "non_penalty_xg",
        "non_penalty_xgot",
        "big_chance_shots",
        "xa",
        "key_passes",
        "big_chances_created",
        "passes_into_box",
        "passes_into_final_third",
        "pass_attempt",
        "pass_completed",
        "progressive_passes",
        "takeon_attempt",
        "takeon_completed",
        "progressive_carries",
        "final_third_carries",
        "carries_into_box",
        "xt_pass_added",
        "xt_carry_added",
        "xt_total_added",
        "xpv_pass_added",
        "xpv_carry_added",
        "xpv_total_added",
        "turnovers",
        "dispossessed",
        "miscontrol_proxy",
        "defensive_actions",
        "tackle_attempt",
        "tackle_won",
        "interception",
        "ball_recovery",
        "blocked_pass",
        "clearance",
        "error_leading_to_shot",
        "error_leading_to_goal",
        "opponent_threat_prevented",
        "defensive_net_threat_reduction",
        "shots_on_target_faced",
        "xgot_faced",
        "goals_conceded",
    ]

    frame = frame.with_columns(
        *[
            (
                pl.col(column).cast(pl.Float64, strict=False).fill_null(0)
                * 90.0
                / pl.col("minutes").cast(pl.Float64).clip(lower_bound=1.0)
            ).alias(f"{column}_90")
            for column in per_90_columns
        ],
        (
            (pl.col("non_penalty_goals") - pl.col("non_penalty_xg"))
            * 90.0
            / pl.col("minutes").clip(lower_bound=1.0)
        ).alias("finishing_above_expected_90"),
        (
            (pl.col("non_penalty_xgot") - pl.col("non_penalty_xg"))
            * 90.0
            / pl.col("minutes").clip(lower_bound=1.0)
        ).alias("shot_placement_above_expected_90"),
        pl.when(pl.col("non_penalty_shots") > 0)
        .then(pl.col("non_penalty_xg") / pl.col("non_penalty_shots"))
        .otherwise(None)
        .alias("average_xg_per_shot"),
        pl.when(pl.col("pass_attempt") >= PASS_ATTEMPT_MINIMUM)
        .then(pl.col("pass_completed") * 100.0 / pl.col("pass_attempt"))
        .otherwise(None)
        .alias("pass_completion_pct"),
        pl.when(pl.col("passes") > 0)
        .then(
            pl.col("pass_completion_above_expected_weighted")
            * 100.0
            / pl.col("passes")
        )
        .otherwise(None)
        .alias("pass_completion_above_expected_pct"),
        pl.when(pl.col("passes") > 0)
        .then(pl.col("progressive_passes") * 100.0 / pl.col("passes"))
        .otherwise(None)
        .alias("progressive_pass_share_pct"),
        pl.when(pl.col("takeon_attempt") >= DRIBBLE_ATTEMPT_MINIMUM)
        .then(pl.col("takeon_completed") * 100.0 / pl.col("takeon_attempt"))
        .otherwise(None)
        .alias("dribble_success_pct"),
        pl.when(pl.col("tackle_attempt") >= TACKLE_ATTEMPT_MINIMUM)
        .then(pl.col("tackle_won") * 100.0 / pl.col("tackle_attempt"))
        .otherwise(None)
        .alias("tackle_win_pct"),
        pl.when(pl.col("aerial_attempt") >= AERIAL_ATTEMPT_MINIMUM)
        .then(pl.col("aerial_won") * 100.0 / pl.col("aerial_attempt"))
        .otherwise(None)
        .alias("aerial_duel_win_pct"),
        pl.when(pl.col("ground_duels_attempted") >= DUEL_ATTEMPT_MINIMUM)
        .then(
            pl.col("ground_duels_won")
            * 100.0
            / pl.col("ground_duels_attempted")
        )
        .otherwise(None)
        .alias("ground_duel_win_pct"),
        pl.when(pl.col("duels_attempted") >= DUEL_ATTEMPT_MINIMUM)
        .then(pl.col("duels_won") * 100.0 / pl.col("duels_attempted"))
        .otherwise(None)
        .alias("duel_win_pct"),
        (
            pl.col("starts")
            / pl.col("appearances").cast(pl.Float64).clip(lower_bound=1.0)
        ).alias("start_share"),
    )
    return frame


def _percentile(
    values: pd.Series,
    benchmark: pd.Series,
) -> np.ndarray:
    benchmark_values = benchmark.dropna().to_numpy(dtype=float)
    benchmark_values = benchmark_values[np.isfinite(benchmark_values)]
    result = np.full(len(values), np.nan)

    if benchmark_values.size == 0:
        return result

    benchmark_values.sort()
    input_values = values.to_numpy(dtype=float)
    finite = np.isfinite(input_values)
    left = np.searchsorted(
        benchmark_values,
        input_values[finite],
        side="left",
    )
    right = np.searchsorted(
        benchmark_values,
        input_values[finite],
        side="right",
    )
    result[finite] = 100.0 * (left + right) / (2.0 * benchmark_values.size)
    return np.clip(result, 0.0, 100.0)


def _add_percentiles_and_zscores(
    profiles: pd.DataFrame,
    *,
    benchmark_minutes: int,
) -> pd.DataFrame:
    result = profiles.copy()

    for role, specs in ROLE_METRICS.items():
        role_mask = result["primary_role"] == role
        benchmark_mask = role_mask & (result["minutes"] >= benchmark_minutes)

        for spec in specs:
            if spec.key not in result.columns:
                continue

            benchmark = result.loc[benchmark_mask, spec.key].dropna()
            if benchmark.empty:
                continue

            lower = benchmark.quantile(LOWER_WINSOR)
            upper = benchmark.quantile(UPPER_WINSOR)
            benchmark_clipped = benchmark.clip(lower, upper)
            role_values = result.loc[role_mask, spec.key].clip(lower, upper)

            percentiles = _percentile(role_values, benchmark_clipped)
            if not spec.higher_is_better:
                percentiles = 100.0 - percentiles

            result.loc[
                role_mask,
                f"{spec.key}_percentile",
            ] = percentiles

            mean = float(benchmark_clipped.mean())
            standard_deviation = float(benchmark_clipped.std(ddof=0))
            if standard_deviation > 0:
                result.loc[
                    role_mask,
                    f"{spec.key}_zscore",
                ] = (role_values - mean) / standard_deviation
            else:
                result.loc[role_mask, f"{spec.key}_zscore"] = 0.0

    return result


def _phase_weights(specs: list[MetricSpec]) -> dict[str, float]:
    by_phase: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        by_phase[spec.phase].append(spec.key)

    phase_weight = 1.0 / len(by_phase)
    weights: dict[str, float] = {}
    for keys in by_phase.values():
        metric_weight = phase_weight / len(keys)
        for key in keys:
            weights[key] = metric_weight
    return weights


def _build_profile_rows(
    profiles: pd.DataFrame,
    *,
    competition: str,
    season: str,
    benchmark_minutes: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, player in profiles.iterrows():
        specs = ROLE_METRICS[str(player["primary_role"])]
        for display_order, spec in enumerate(specs, start=1):
            raw_value = player.get(spec.key)
            percentile_value = player.get(f"{spec.key}_percentile")

            rows.append(
                {
                    "player_id": int(player["whoscored_player_id"]),
                    "whoscored_player_id": int(player["whoscored_player_id"]),
                    "player_name": player["player_name"],
                    "competition": competition,
                    "season": season,
                    "primary_role": player["primary_role"],
                    "secondary_role": player.get("secondary_role"),
                    "primary_role_share": float(player["primary_role_share"]),
                    "is_hybrid_role": bool(player["is_hybrid_role"]),
                    "minutes": float(player["minutes"]),
                    "appearances": int(player["appearances"]),
                    "sample_status": player["sample_status"],
                    "phase": spec.phase,
                    "metric_key": spec.key,
                    "metric_label": spec.label,
                    "metric_value": (
                        None if pd.isna(raw_value) else float(raw_value)
                    ),
                    "percentile": (
                        None
                        if pd.isna(percentile_value)
                        else float(percentile_value)
                    ),
                    "higher_is_better": spec.higher_is_better,
                    "display_order": display_order,
                    "benchmark_minutes": benchmark_minutes,
                }
            )

    return pl.from_pandas(pd.DataFrame(rows))


def _build_similarity_rows(
    profiles: pd.DataFrame,
    *,
    competition: str,
    season: str,
    benchmark_minutes: int,
    top_n: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, selected in profiles.iterrows():
        role = str(selected["primary_role"])
        specs = ROLE_METRICS[role]
        weights = _phase_weights(specs)

        candidates = profiles[
            (profiles["primary_role"] == role)
            & (profiles["minutes"] >= benchmark_minutes)
            & (
                profiles["whoscored_player_id"]
                != selected["whoscored_player_id"]
            )
        ]

        scored: list[dict[str, Any]] = []
        for _, candidate in candidates.iterrows():
            weighted_error = 0.0
            available_weight = 0.0
            metrics_used = 0

            for spec in specs:
                z_column = f"{spec.key}_zscore"
                selected_value = selected.get(z_column)
                candidate_value = candidate.get(z_column)

                if pd.isna(selected_value) or pd.isna(candidate_value):
                    continue

                weight = weights[spec.key]
                weighted_error += (
                    weight
                    * (float(candidate_value) - float(selected_value)) ** 2
                )
                available_weight += weight
                metrics_used += 1

            if available_weight <= 0 or metrics_used == 0:
                continue

            normalized_mse = weighted_error / available_weight
            similarity = 100.0 * math.exp(-0.5 * normalized_mse)

            scored.append(
                {
                    "similar_player_id": int(
                        candidate["whoscored_player_id"]
                    ),
                    "similar_player_name": candidate["player_name"],
                    "primary_role": candidate["primary_role"],
                    "secondary_role": candidate.get("secondary_role"),
                    "minutes": float(candidate["minutes"]),
                    "appearances": int(candidate["appearances"]),
                    "metrics_used": metrics_used,
                    "similarity": similarity,
                    "profile_similarity": similarity,
                }
            )

        scored.sort(key=lambda row: float(row["similarity"]), reverse=True)

        for rank, candidate in enumerate(scored[:top_n], start=1):
            rows.append(
                {
                    "player_id": int(selected["whoscored_player_id"]),
                    "whoscored_player_id": int(
                        selected["whoscored_player_id"]
                    ),
                    "player_name": selected["player_name"],
                    "similar_player_id": candidate["similar_player_id"],
                    "similar_player_name": candidate["similar_player_name"],
                    "competition": competition,
                    "season": season,
                    "primary_role": candidate["primary_role"],
                    "secondary_role": candidate["secondary_role"],
                    "minutes": candidate["minutes"],
                    "appearances": candidate["appearances"],
                    "metrics_used": candidate["metrics_used"],
                    "similarity": candidate["similarity"],
                    "profile_similarity": candidate["profile_similarity"],
                    "rank": rank,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "player_id": pl.Int64,
                "whoscored_player_id": pl.Int64,
                "player_name": pl.String,
                "similar_player_id": pl.Int64,
                "similar_player_name": pl.String,
                "competition": pl.String,
                "season": pl.String,
                "primary_role": pl.String,
                "secondary_role": pl.String,
                "minutes": pl.Float64,
                "appearances": pl.Int64,
                "metrics_used": pl.Int64,
                "similarity": pl.Float64,
                "profile_similarity": pl.Float64,
                "rank": pl.Int64,
            }
        )

    return pl.from_pandas(pd.DataFrame(rows))


def build_player_profiles(
    *,
    competition: str,
    season: str,
    ratings_path: Path,
    normalized_match_root: Path,
    feature_match_root: Path,
    output_root: Path,
    profile_minutes: int = PROFILE_MINUTES_DEFAULT,
    benchmark_minutes: int = BENCHMARK_MINUTES_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
    publish_latest: bool = True,
) -> BuildResult:
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    if not normalized_match_root.exists():
        raise FileNotFoundError(
            f"Normalized match directory not found: {normalized_match_root}"
        )
    if not feature_match_root.exists():
        raise FileNotFoundError(
            f"Feature match directory not found: {feature_match_root}"
        )

    print(f"Loading ratings: {ratings_path}")
    ratings = pl.read_parquet(ratings_path)
    required_rating_columns = {
        "whoscored_player_id",
        "player_name",
        "position",
        "minutes",
        "match_id",
        "season",
        "started",
    }
    missing_ratings = sorted(required_rating_columns.difference(ratings.columns))
    if missing_ratings:
        raise ValueError(
            f"Ratings table is missing required columns: {missing_ratings}"
        )

    roles = _build_roles(ratings)
    player_base = _build_player_base(ratings, roles)

    event_paths = sorted(normalized_match_root.rglob("events.parquet"))
    if not event_paths:
        raise FileNotFoundError(
            f"No events.parquet files found under {normalized_match_root}"
        )
    print(f"Aggregating {len(event_paths):,} normalized event partitions")
    player_events = _sum_player_frames(
        [_aggregate_events(path) for path in event_paths],
        id_column="player_id",
    ).rename({"player_id": "whoscored_player_id"})

    shot_paths = sorted(feature_match_root.rglob("shots_with_models.parquet"))
    if not shot_paths:
        raise FileNotFoundError(
            f"No shots_with_models.parquet files found under {feature_match_root}"
        )
    print(f"Aggregating {len(shot_paths):,} modeled shot partitions")
    player_shots = _sum_player_frames(
        [_aggregate_shots(path) for path in shot_paths],
        id_column="player_id",
    ).rename({"player_id": "whoscored_player_id"})

    xt_paths = sorted(feature_match_root.rglob("xthreat_actions.parquet"))
    if not xt_paths:
        raise FileNotFoundError(
            f"No xthreat_actions.parquet files found under {feature_match_root}"
        )
    print(f"Aggregating {len(xt_paths):,} xT action partitions")
    player_xt = _sum_player_frames(
        [_aggregate_action_values(path, "xt") for path in xt_paths],
        id_column="player_id",
    ).rename({"player_id": "whoscored_player_id"})

    xpv_paths = sorted(feature_match_root.rglob("xpv_actions.parquet"))
    if not xpv_paths:
        raise FileNotFoundError(
            f"No xpv_actions.parquet files found under {feature_match_root}"
        )
    print(f"Aggregating {len(xpv_paths):,} xPV action partitions")
    player_xpv = _sum_player_frames(
        [_aggregate_action_values(path, "xpv") for path in xpv_paths],
        id_column="player_id",
    ).rename({"player_id": "whoscored_player_id"})

    combined = (
        player_base
        .join(player_events, on="whoscored_player_id", how="left")
        .join(player_shots, on="whoscored_player_id", how="left")
        .join(player_xt, on="whoscored_player_id", how="left")
        .join(player_xpv, on="whoscored_player_id", how="left")
    )

    combined = _add_derived_metrics(combined)
    profiles = combined.to_pandas()
    profiles = profiles[
        (profiles["minutes"] >= profile_minutes)
        & profiles["primary_role"].isin(ROLE_METRICS)
    ].copy()

    if profiles.empty:
        raise ValueError(
            "No players met the profile eligibility threshold "
            f"of {profile_minutes} minutes"
        )

    profiles["sample_status"] = np.where(
        profiles["minutes"] >= benchmark_minutes,
        "benchmark_eligible",
        "limited_sample",
    )
    profiles = _add_percentiles_and_zscores(
        profiles,
        benchmark_minutes=benchmark_minutes,
    )

    profile_rows = _build_profile_rows(
        profiles,
        competition=competition,
        season=season,
        benchmark_minutes=benchmark_minutes,
    )
    similarity_rows = _build_similarity_rows(
        profiles,
        competition=competition,
        season=season,
        benchmark_minutes=benchmark_minutes,
        top_n=top_n,
    )

    partition_root = (
        output_root
        / "profiles"
        / f"competition={competition}"
        / f"season={season}"
    )
    profiles_path = partition_root / "player_profiles.parquet"
    similarities_path = partition_root / "player_similarities.parquet"

    _atomic_write(profile_rows, profiles_path)
    _atomic_write(similarity_rows, similarities_path)

    if publish_latest:
        output_root.mkdir(parents=True, exist_ok=True)
        latest_profiles = output_root / "player_profiles.parquet"
        latest_similarities = output_root / "player_similarities.parquet"
        shutil.copy2(profiles_path, latest_profiles)
        shutil.copy2(similarities_path, latest_similarities)
        print(f"Published API profile alias: {latest_profiles}")
        print(f"Published API similarity alias: {latest_similarities}")

    return BuildResult(
        profiles_path=profiles_path,
        similarities_path=similarities_path,
        profile_players=int(profile_rows["player_id"].n_unique()),
        profile_rows=profile_rows.height,
        similarity_rows=similarity_rows.height,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build role-relative player profiles and similar-player serving tables."
        )
    )
    parser.add_argument("--competition", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument(
        "--ratings",
        type=Path,
        default=None,
        help="Override the canonical season ratings Parquet.",
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("data/normalized/whoscored"),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features/whoscored"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/serving"),
    )
    parser.add_argument(
        "--profile-minutes",
        type=int,
        default=PROFILE_MINUTES_DEFAULT,
    )
    parser.add_argument(
        "--benchmark-minutes",
        type=int,
        default=BENCHMARK_MINUTES_DEFAULT,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_DEFAULT,
    )
    parser.add_argument(
        "--publish-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Publish root serving aliases consumed by the current API "
            "(default: enabled)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    competition = str(args.competition)
    season = str(args.season)

    normalized_root = _project_path(args.normalized_root)
    features_root = _project_path(args.features_root)
    output_root = _project_path(args.output_root)

    ratings_path = (
        _project_path(args.ratings)
        if args.ratings is not None
        else (
            PROJECT_ROOT
            / "data/features/ratings"
            / f"competition={competition}"
            / f"season={season}"
            / "player_match_ratings.parquet"
        )
    )
    normalized_match_root = (
        normalized_root
        / f"competition={competition}"
        / f"season={season}"
        / "matches"
    )
    feature_match_root = (
        features_root
        / f"competition={competition}"
        / f"season={season}"
        / "matches"
    )

    result = build_player_profiles(
        competition=competition,
        season=season,
        ratings_path=ratings_path,
        normalized_match_root=normalized_match_root,
        feature_match_root=feature_match_root,
        output_root=output_root,
        profile_minutes=max(1, int(args.profile_minutes)),
        benchmark_minutes=max(1, int(args.benchmark_minutes)),
        top_n=max(1, int(args.top_n)),
        publish_latest=bool(args.publish_latest),
    )

    print("\nPlayer-profile build completed")
    print(f"Players: {result.profile_players:,}")
    print(f"Profile metric rows: {result.profile_rows:,}")
    print(f"Similarity rows: {result.similarity_rows:,}")
    print(f"Profiles: {result.profiles_path}")
    print(f"Similarities: {result.similarities_path}")


if __name__ == "__main__":
    main()
