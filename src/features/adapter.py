from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl

from ingestion.whoscored.appearances import (
    derive_appearance_minutes,
    reconcile_position_groups,
)


class FeatureCompatibilityError(RuntimeError):
    """Raised when a normalized WhoScored match cannot satisfy feature inference."""


@dataclass(frozen=True)
class CompatibilityBundle:
    match_id: int
    matches: pl.DataFrame
    player_matches: pl.DataFrame
    events: pl.DataFrame
    passes: pl.DataFrame
    shots: pl.DataFrame
    checks: tuple[dict[str, object], ...]


def _first_existing(frame: pl.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _ensure_column(
    frame: pl.DataFrame,
    name: str,
    dtype: pl.DataType,
    *,
    default: object = None,
) -> pl.DataFrame:
    if name in frame.columns:
        return frame
    return frame.with_columns(pl.lit(default).cast(dtype).alias(name))


def _alias(frame: pl.DataFrame, target: str, candidates: Iterable[str]) -> pl.DataFrame:
    if target in frame.columns:
        return frame
    source = _first_existing(frame, candidates)
    if source is None:
        return frame
    return frame.with_columns(pl.col(source).alias(target))


def _qualifier_flag(frame: pl.DataFrame, names: Iterable[str]) -> pl.Expr:
    expressions: list[pl.Expr] = []
    for name in names:
        column = f"q_{name}"
        if column in frame.columns:
            expressions.append(pl.col(column).is_not_null())
        if "qualifier_names" in frame.columns:
            expressions.append(pl.col("qualifier_names").list.contains(name))
    return pl.any_horizontal(expressions).fill_null(False) if expressions else pl.lit(False)


def _match_start_date(matches: pl.DataFrame) -> date | None:
    if matches.is_empty() or "start_date" not in matches.columns:
        return None
    value = matches["start_date"][0]
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _prepare_events(
    frame: pl.DataFrame,
    *,
    competition_id: int,
    season_id: int | None,
    season: str,
    start_date: date | None,
) -> pl.DataFrame:
    output = frame
    output = _alias(output, "type_name", ["type_display_name", "event_type"])
    output = _alias(output, "outcome_name", ["outcome_type_display_name", "outcome"])
    output = _alias(output, "outcome_value", ["outcome_type_value"])
    output = _alias(output, "period_name", ["period_display_name"])

    defaults: dict[str, tuple[pl.DataType, object]] = {
        "event_id": (pl.Int64, None),
        "minute": (pl.Int64, 0),
        "second": (pl.Float64, 0.0),
        "expanded_minute": (pl.Int64, 0),
        "period_value": (pl.Int64, 0),
        "team_id": (pl.Int64, None),
        "player_id": (pl.Int64, None),
        "x": (pl.Float64, None),
        "y": (pl.Float64, None),
        "end_x": (pl.Float64, None),
        "end_y": (pl.Float64, None),
        "type_name": (pl.String, None),
        "outcome_name": (pl.String, None),
        "outcome_value": (pl.Int64, None),
        "related_event_id": (pl.Int64, None),
        "related_player_id": (pl.Int64, None),
        "is_touch": (pl.Boolean, False),
        "is_shot": (pl.Boolean, False),
        "is_goal": (pl.Boolean, False),
    }
    for column, (dtype, default) in defaults.items():
        output = _ensure_column(output, column, dtype, default=default)

    output = _ensure_column(output, "persistent_id", pl.String)
    output = output.with_row_index("_event_row").with_columns(
        pl.lit(competition_id).cast(pl.Int64).alias("tournament_id"),
        pl.lit(season_id).cast(pl.Int64).alias("season_id"),
        pl.lit(season).alias("season_name"),
        pl.lit(start_date).cast(pl.Date).alias("start_date"),
        pl.col("minute").fill_null(0).cast(pl.Int64),
        pl.col("second").fill_null(0.0).cast(pl.Float64),
        pl.col("expanded_minute").fill_null(pl.col("minute")).cast(pl.Int64),
        pl.col("period_value").fill_null(0).cast(pl.Int64),
        pl.col("event_id").cast(pl.Int64, strict=False),
        pl.col("team_id").cast(pl.Int64, strict=False),
        pl.col("player_id").cast(pl.Int64, strict=False),
        pl.col("x").cast(pl.Float64, strict=False),
        pl.col("y").cast(pl.Float64, strict=False),
        pl.col("end_x").cast(pl.Float64, strict=False),
        pl.col("end_y").cast(pl.Float64, strict=False),
        pl.col("is_touch").fill_null(False).cast(pl.Int64),
        pl.col("is_shot").fill_null(False).cast(pl.Int64),
        pl.col("is_goal").fill_null(False).cast(pl.Int64),
    )
    fallback_key = pl.concat_str(
        [
            pl.col("match_id").cast(pl.String),
            pl.col("team_id").cast(pl.String).fill_null("null"),
            pl.col("event_id").cast(pl.String).fill_null("null"),
            pl.col("period_value").cast(pl.String),
            pl.col("minute").cast(pl.String),
            pl.col("second").cast(pl.String),
            pl.col("player_id").cast(pl.String).fill_null("null"),
            pl.col("type_name").cast(pl.String).fill_null("null"),
            pl.col("_event_row").cast(pl.String),
        ],
        separator="|",
    )
    return output.with_columns(
        pl.when(pl.col("persistent_id").is_not_null())
        .then(
            pl.concat_str(
                [
                    pl.col("match_id").cast(pl.String),
                    pl.col("persistent_id").cast(pl.String),
                ],
                separator=":",
            )
        )
        .otherwise(
            pl.concat_str(
                [pl.lit("fallback"), fallback_key.hash(seed=0).cast(pl.String)],
                separator=":",
            )
        )
        .alias("event_uid")
    ).drop("_event_row")


def _prepare_passes(events: pl.DataFrame) -> pl.DataFrame:
    passes = events.filter(pl.col("type_name") == "Pass")
    if passes.is_empty():
        return passes

    flag_names = {
        "is_key_pass": ("KeyPass",),
        "is_assist": ("Assist", "IntentionalAssist"),
        "is_intentional_assist": ("IntentionalAssist",),
        "is_cross": ("Cross",),
        "is_through_ball": ("Throughball", "ThroughBall"),
        "is_long_ball": ("Longball", "LongBall"),
        "is_corner": ("CornerTaken", "FromCorner"),
        "is_free_kick": ("FreekickTaken", "FreeKickTaken"),
        "is_throw_in": ("ThrowIn",),
        "is_goal_kick": ("GoalKick",),
    }
    passes = passes.with_columns(
        (
            (pl.col("outcome_value") == 1)
            | (pl.col("outcome_name").cast(pl.String).str.to_lowercase() == "successful")
        ).fill_null(False).cast(pl.Int64).alias("success"),
        *[
            _qualifier_flag(passes, names).cast(pl.Int64).alias(column)
            for column, names in flag_names.items()
        ],
    )

    dx_m = (pl.col("end_x") - pl.col("x")) * 1.05
    dy_m = (pl.col("end_y") - pl.col("y")) * 0.68
    return passes.with_columns(
        (dx_m.pow(2) + dy_m.pow(2)).sqrt().alias("pass_length"),
        pl.arctan2(dy_m, dx_m).alias("pass_angle"),
    )


def _prepare_shots(events: pl.DataFrame, normalized_shots: pl.DataFrame) -> pl.DataFrame:
    shots = normalized_shots if not normalized_shots.is_empty() else events.filter(pl.col("is_shot") == 1)
    shots = _prepare_events(
        shots,
        competition_id=int(events["tournament_id"][0]),
        season_id=events["season_id"][0],
        season=str(events["season_name"][0]),
        start_date=events["start_date"][0],
    )
    shots = _alias(shots, "goal_mouth_y", ["q_GoalMouthY"])
    shots = _alias(shots, "goal_mouth_z", ["q_GoalMouthZ"])

    shot_flags = {
        "is_header": ("Head",),
        "is_right_foot": ("RightFoot",),
        "is_left_foot": ("LeftFoot",),
        "is_other_body_part": ("OtherBodyPart",),
        "is_penalty": ("Penalty",),
        "is_direct_free_kick": ("DirectFreekick",),
        "is_from_corner": ("FromCorner",),
        "is_set_piece": ("SetPiece",),
        "is_fast_break": ("FastBreak",),
        "is_own_goal": ("OwnGoal",),
        "assisted": ("Assisted", "IntentionalAssist"),
        "intentional_assist": ("IntentionalAssist",),
    }
    additions = []
    for column, names in shot_flags.items():
        if column in shots.columns:
            additions.append(pl.col(column).fill_null(False).cast(pl.Int64).alias(column))
        else:
            additions.append(_qualifier_flag(shots, names).cast(pl.Int64).alias(column))
    shots = shots.with_columns(additions)

    x_m = pl.col("x") * 1.05
    y_m = pl.col("y") * 0.68
    goal_dx = 105.0 - x_m
    goal_dy = 34.0 - y_m
    shots = shots.with_columns(
        (goal_dx.pow(2) + goal_dy.pow(2)).sqrt().alias("distance_to_goal"),
        pl.arctan2(
            7.32 * goal_dx,
            goal_dx.pow(2) + goal_dy.pow(2) - (7.32 / 2.0) ** 2,
        ).abs().alias("angle_to_goal"),
        pl.when(pl.col("is_header") == 1).then(pl.lit("Head"))
        .when(pl.col("is_left_foot") == 1).then(pl.lit("LeftFoot"))
        .when(pl.col("is_right_foot") == 1).then(pl.lit("RightFoot"))
        .otherwise(pl.lit("OtherBodyPart")).alias("body_part"),
        pl.when(pl.col("is_penalty") == 1).then(pl.lit("Penalty"))
        .when(pl.col("is_direct_free_kick") == 1).then(pl.lit("DirectFreekick"))
        .when(pl.col("is_from_corner") == 1).then(pl.lit("FromCorner"))
        .when(pl.col("is_set_piece") == 1).then(pl.lit("SetPiece"))
        .when(pl.col("is_fast_break") == 1).then(pl.lit("FastBreak"))
        .otherwise(pl.lit("RegularPlay")).alias("situation"),
        pl.when(pl.col("x") >= 94.0).then(pl.lit("SixYardBox"))
        .when(pl.col("x") >= 83.0).then(pl.lit("PenaltyArea"))
        .otherwise(pl.lit("OutOfBox")).alias("shot_location_name"),
    )
    for column in ("goal_mouth_y", "goal_mouth_z"):
        shots = _ensure_column(shots, column, pl.Float64)
    shots = shots.with_columns(
        pl.col("goal_mouth_y").cast(pl.Float64, strict=False),
        pl.col("goal_mouth_z").cast(pl.Float64, strict=False),
    )
    for column in ("body_part_qualifier_id", "situation_qualifier_id", "location_zone"):
        shots = _ensure_column(shots, column, pl.Int64)
    return shots


def _checks(events: pl.DataFrame, passes: pl.DataFrame, shots: pl.DataFrame) -> tuple[dict[str, object], ...]:
    duplicate_events = events.select(
        pl.struct(["match_id", "event_uid"]).is_duplicated().sum()
    ).item()
    duplicate_numeric_ids = events.filter(pl.col("event_id").is_not_null()).select(
        pl.struct(["match_id", "event_id"]).is_duplicated().sum()
    ).item()
    missing_persistent_ids = events.filter(pl.col("persistent_id").is_null()).height
    coordinate_violations = events.filter(
        pl.any_horizontal(
            [
                (pl.col(column) < 0) | (pl.col(column) > 100)
                for column in ("x", "y", "end_x", "end_y")
            ]
        )
    ).height
    return (
        {"check": "event_uid_unique", "passed": duplicate_events == 0, "value": int(duplicate_events)},
        {
            "check": "numeric_event_id_duplicates",
            "passed": True,
            "value": int(duplicate_numeric_ids),
            "severity": "info",
        },
        {
            "check": "persistent_event_id_coverage",
            "passed": True,
            "value": int(events.height - missing_persistent_ids),
            "missing": int(missing_persistent_ids),
            "severity": "info",
        },
        {"check": "coordinates_in_range", "passed": coordinate_violations == 0, "value": coordinate_violations},
        {"check": "passes_derived", "passed": passes.height > 0, "value": passes.height},
        {"check": "shots_available", "passed": shots.height > 0, "value": shots.height, "severity": "warning"},
    )


def build_compatibility_bundle(
    match_directory: Path,
    *,
    competition_id: int,
    season: str,
    season_id: int | None = None,
) -> CompatibilityBundle:
    required = ["matches.parquet", "player_matches.parquet", "events.parquet", "shots.parquet", "_SUCCESS.json"]
    missing = [name for name in required if not (match_directory / name).exists()]
    if missing:
        raise FeatureCompatibilityError(f"Match partition is incomplete: {missing}")

    matches = pl.read_parquet(match_directory / "matches.parquet")
    player_matches = pl.read_parquet(match_directory / "player_matches.parquet")
    raw_events = pl.read_parquet(match_directory / "events.parquet")
    normalized_shots = pl.read_parquet(match_directory / "shots.parquet")
    if matches.is_empty() or raw_events.is_empty():
        raise FeatureCompatibilityError(f"Match partition has no match/events rows: {match_directory}")
    match_id = int(matches["match_id"][0])
    events = _prepare_events(
        raw_events,
        competition_id=competition_id,
        season_id=season_id,
        season=season,
        start_date=_match_start_date(matches),
    )
    player_matches, derived_minutes = derive_appearance_minutes(
        player_matches, events
    )
    player_matches, derived_positions = reconcile_position_groups(
        player_matches, events
    )
    passes = _prepare_passes(events)
    shots = _prepare_shots(events, normalized_shots)
    checks = (
        *_checks(events, passes, shots),
        {
            "check": "appearance_minutes_reconciled",
            "passed": True,
            "value": derived_minutes,
            "severity": "warning" if derived_minutes else "info",
            "method": "provider_or_lineup_substitution_event_clock",
        },
        {
            "check": "position_groups_reconciled",
            "passed": True,
            "value": derived_positions,
            "severity": "warning" if derived_positions else "info",
            "method": "provider_or_linked_replacement_position",
        },
    )
    failed = [check for check in checks if not check["passed"] and check.get("severity") != "warning"]
    if failed:
        raise FeatureCompatibilityError(f"Feature compatibility checks failed: {failed}")
    return CompatibilityBundle(
        match_id=match_id,
        matches=matches,
        player_matches=player_matches,
        events=events,
        passes=passes,
        shots=shots,
        checks=checks,
    )
