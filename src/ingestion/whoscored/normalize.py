from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import orjson
import polars as pl

from ingestion.whoscored.appearances import (
    derive_appearance_minutes,
    reconcile_position_groups,
)
from ingestion.whoscored.qualifiers import flatten_qualifiers, qualifier_names
from ingestion.whoscored.raw_events import raw_events_frame
from ingestion.whoscored.schema import EVENT_RENAME, SHOT_QUALIFIER_FLAGS


@dataclass(frozen=True)
class NormalizedMatch:
    raw_events: pl.DataFrame
    matches: pl.DataFrame
    teams: pl.DataFrame
    player_matches: pl.DataFrame
    events: pl.DataFrame
    shots: pl.DataFrame
    metadata: dict[str, Any]


def _display_name(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("displayName") is not None:
        return str(value["displayName"])
    return None


def _display_value(value: Any) -> int | None:
    if isinstance(value, dict) and value.get("value") is not None:
        return int(value["value"])
    return None


def _json(value: Any) -> str:
    return orjson.dumps(value).decode("utf-8")


def normalize_event(
    event: dict[str, Any],
    *,
    match_id: int,
    source_url: str,
) -> dict[str, Any]:
    output = {EVENT_RENAME.get(key, key): value for key, value in event.items()}
    persistent_id = event.get("id")
    output.update(
        {
            "match_id": match_id,
            "source_url": source_url,
            "persistent_id": str(persistent_id) if persistent_id is not None else None,
            "period_value": _display_value(event.get("period")),
            "period_display_name": _display_name(event.get("period")),
            "type_value": _display_value(event.get("type")),
            "type_display_name": _display_name(event.get("type")),
            "outcome_type_value": _display_value(event.get("outcomeType")),
            "outcome_type_display_name": _display_name(event.get("outcomeType")),
            "card_type_value": _display_value(event.get("cardType")),
            "card_type_display_name": _display_name(event.get("cardType")),
            "qualifier_names": qualifier_names(event.get("qualifiers") or []),
            "qualifiers_json": _json(event.get("qualifiers") or []),
            "satisfied_events_types_json": _json(
                event.get("satisfiedEventsTypes") or []
            ),
            "is_shot": bool(event.get("isShot", False)),
            "is_goal": bool(event.get("isGoal", False)),
            "is_touch": bool(event.get("isTouch", False)),
        }
    )
    output.update(flatten_qualifiers(event.get("qualifiers") or []))
    for nested in (
        "period",
        "type",
        "outcome_type",
        "card_type",
        "qualifiers",
        "satisfied_events_types",
    ):
        output.pop(nested, None)
    return output


def _add_shot_features(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    frame = frame.with_columns(
        (100.0 - pl.col("x").cast(pl.Float64, strict=False)).alias(
            "dist_x_to_goal"
        ),
        (pl.col("y").cast(pl.Float64, strict=False) - 50.0)
        .abs()
        .alias("dist_y_to_goal"),
    ).with_columns(
        (
            pl.col("dist_x_to_goal").pow(2)
            + pl.col("dist_y_to_goal").pow(2)
        )
        .sqrt()
        .alias("shot_distance_pct")
    )
    expressions: list[pl.Expr] = []
    for qualifier, flag in SHOT_QUALIFIER_FLAGS.items():
        column = f"q_{qualifier}"
        expressions.append(
            pl.col(column).is_not_null().alias(flag)
            if column in frame.columns
            else pl.lit(False).alias(flag)
        )
    return frame.with_columns(expressions)


def _normalization_key(event: dict[str, Any]) -> str:
    if event.get("id") is not None:
        return f"id:{event['id']}"
    if event.get("eventId") is not None:
        return f"event:{event['eventId']}"
    return "fallback:" + _json(event)


def _team_rows(match_data: dict[str, Any], match_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = match_data.get(side) or {}
        team_id = team.get("teamId")
        if team_id is None:
            continue
        rows.append(
            {
                "match_id": match_id,
                "team_id": int(team_id),
                "side": side,
                "team_name": team.get("name"),
                "formation_name": team.get("formationName"),
                "field": team.get("field"),
            }
        )
    return rows


def _player_rows(match_data: dict[str, Any], match_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = match_data.get(side) or {}
        team_id = team.get("teamId")
        for player in team.get("players") or []:
            player_id = player.get("playerId", player.get("id"))
            if player_id is None:
                continue
            stats = player.get("stats") or {}
            minutes = stats.get("minutesPlayed", stats.get("minutes"))
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": int(player_id),
                    "team_id": int(team_id) if team_id is not None else None,
                    "side": side,
                    "player_name": player.get("name"),
                    "position": player.get("position"),
                    "shirt_number": player.get("shirtNo"),
                    "started": bool(player.get("isFirstEleven", False)),
                    "is_man_of_match": bool(player.get("isManOfTheMatch", False)),
                    "minutes": float(minutes) if minutes is not None else None,
                    "player_json": _json(player),
                }
            )
    return rows


def normalize_match_data(
    match_data: dict[str, Any],
    *,
    match_id: int,
    source_url: str,
) -> NormalizedMatch:
    raw_events = raw_events_frame(
        match_data,
        match_id=match_id,
        source_url=source_url,
    )

    event_objects: list[dict[str, Any]] = list(match_data.get("events") or [])
    for side in ("home", "away"):
        event_objects.extend(
            list((match_data.get(side) or {}).get("incidentEvents") or [])
        )
    unique_events: dict[str, dict[str, Any]] = {}
    for event in event_objects:
        if isinstance(event, dict):
            unique_events.setdefault(_normalization_key(event), event)
    event_rows = [
        normalize_event(event, match_id=match_id, source_url=source_url)
        for event in unique_events.values()
    ]
    events = (
        pl.DataFrame(event_rows, infer_schema_length=None)
        if event_rows
        else pl.DataFrame()
    )
    shots = (
        _add_shot_features(events.filter(pl.col("is_shot")))
        if not events.is_empty()
        else pl.DataFrame()
    )

    home = match_data.get("home") or {}
    away = match_data.get("away") or {}
    match_row = {
        "match_id": match_id,
        "source_url": source_url,
        "start_date": match_data.get("startDate"),
        "start_time": match_data.get("startTime"),
        "status_code": match_data.get("statusCode"),
        "period_code": match_data.get("periodCode"),
        "score": match_data.get("score"),
        "half_time_score": match_data.get("htScore"),
        "full_time_score": match_data.get("ftScore"),
        "venue_name": match_data.get("venueName"),
        "attendance": match_data.get("attendance"),
        "home_team_id": home.get("teamId"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("teamId"),
        "away_team_name": away.get("name"),
    }
    matches = pl.DataFrame([match_row], infer_schema_length=None)
    team_rows = _team_rows(match_data, match_id)
    player_rows = _player_rows(match_data, match_id)
    teams = pl.DataFrame(team_rows, infer_schema_length=None) if team_rows else pl.DataFrame()
    player_matches = (
        pl.DataFrame(player_rows, infer_schema_length=None).unique(
            ["match_id", "player_id"], keep="last"
        )
        if player_rows
        else pl.DataFrame()
    )
    player_matches, derived_minutes = derive_appearance_minutes(
        player_matches, events
    )
    player_matches, derived_positions = reconcile_position_groups(
        player_matches, events
    )
    metadata = {
        **match_row,
        "top_level_keys": sorted(match_data),
        "raw_event_rows": raw_events.height,
        "normalized_event_rows": events.height,
        "shot_rows": shots.height,
        "player_match_rows": player_matches.height,
        "derived_appearance_minutes_rows": derived_minutes,
        "derived_position_group_rows": derived_positions,
    }
    return NormalizedMatch(
        raw_events=raw_events,
        matches=matches,
        teams=teams,
        player_matches=player_matches,
        events=events,
        shots=shots,
        metadata=metadata,
    )
