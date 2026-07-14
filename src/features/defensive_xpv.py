from __future__ import annotations

import math
from datetime import datetime, UTC
from typing import Any

import numpy as np
import polars as pl


DEFENSIVE_ACTION_TYPES = {
    "Tackle",
    "Interception",
    "BallRecovery",
    "Clearance",
    "BlockedPass",
    "Challenge",
    "Save",
    "KeeperPickup",
    "KeeperSweeper",
    "Aerial",
}

IGNORE_NEXT_PREV_TYPES = {
    "SubstitutionOff",
    "SubstitutionOn",
    "Card",
    "FormationChange",
}


def add_time_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize event clock fields and derive a sortable match timestamp."""
    return (
        df.with_columns(
            pl.col("minute").fill_null(0).cast(pl.Int64),
            pl.col("second").fill_null(0).cast(pl.Float64),
            pl.col("expanded_minute").fill_null(pl.col("minute")).cast(pl.Int64),
            pl.col("period_value").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (
                pl.col("expanded_minute").cast(pl.Float64) * 60.0
                + pl.col("second").fill_null(0.0)
            ).alias("event_time_seconds")
        )
    )


def build_match_team_map(events_df: pl.DataFrame) -> dict[int, dict[int, int]]:
    pairs: dict[int, dict[int, int]] = {}

    team_rows = (
        events_df
        .filter(pl.col("team_id").is_not_null())
        .group_by("match_id")
        .agg(pl.col("team_id").unique().alias("teams"))
        .to_dicts()
    )

    for row in team_rows:
        match_id = int(row["match_id"])
        teams = [int(t) for t in row["teams"] if t is not None]

        if len(teams) == 2:
            pairs[match_id] = {
                teams[0]: teams[1],
                teams[1]: teams[0],
            }

    return pairs


def xpv_lookup_from_xy(
    x: float | None,
    y: float | None,
    grid: np.ndarray,
) -> float:
    if x is None or y is None:
        return 0.0

    if not np.isfinite(x) or not np.isfinite(y):
        return 0.0

    ny, nx = grid.shape

    x_clip = min(max(float(x), 0.0), 99.999)
    y_clip = min(max(float(y), 0.0), 99.999)

    zx = int(math.floor((x_clip / 100.0) * nx))
    zy = int(math.floor((y_clip / 100.0) * ny))

    return float(grid[zy, zx])


def zone_from_xy(
    x: float | None,
    y: float | None,
    grid: np.ndarray,
) -> tuple[int | None, int | None, int | None]:
    if x is None or y is None:
        return None, None, None

    if not np.isfinite(x) or not np.isfinite(y):
        return None, None, None

    ny, nx = grid.shape

    x_clip = min(max(float(x), 0.0), 99.999)
    y_clip = min(max(float(y), 0.0), 99.999)

    zx = int(math.floor((x_clip / 100.0) * nx))
    zy = int(math.floor((y_clip / 100.0) * ny))
    z = zx + zy * nx

    return zx, zy, z


def event_state_xy_for_opponent_event(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    For opponent events, coordinates are already in that team's attacking direction.
    Prefer the event endpoint if available, else use start location.
    """
    end_x = row.get("end_x")
    end_y = row.get("end_y")

    if end_x is not None and end_y is not None and np.isfinite(end_x) and np.isfinite(end_y):
        return float(end_x), float(end_y)

    x = row.get("x")
    y = row.get("y")

    if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
        return float(x), float(y)

    return None, None


def defensive_action_xy_to_opponent_perspective(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Defensive event coordinates are from defender perspective.
    Mirror them into the opponent attacking frame.
    """
    x = row.get("x")
    y = row.get("y")

    if x is None or y is None:
        return None, None

    if not np.isfinite(x) or not np.isfinite(y):
        return None, None

    return 100.0 - float(x), 100.0 - float(y)


def build_defensive_xpv_v2_for_match(
    match_events: pl.DataFrame,
    xpv_grid: np.ndarray,
    match_team_map: dict[int, dict[int, int]],
    max_prev_gap_s: float,
    max_next_gap_s: float,
) -> pl.DataFrame:
    sort_columns = ["period_value", "event_time_seconds", "event_id"]
    if "event_uid" in match_events.columns:
        sort_columns.append("event_uid")
    rows = (
        match_events
        .filter(
            pl.col("team_id").is_not_null()
            & pl.col("period_value").is_not_null()
            & pl.col("event_time_seconds").is_not_null()
        )
        .sort(sort_columns)
        .to_dicts()
    )

    if not rows:
        return pl.DataFrame()

    match_id = int(rows[0]["match_id"])
    opponent_map = match_team_map.get(match_id, {})

    out: list[dict[str, Any]] = []
    n = len(rows)

    for i, row in enumerate(rows):
        action_type = row.get("type_name")

        if action_type not in DEFENSIVE_ACTION_TYPES:
            continue

        team_id = row.get("team_id")
        if team_id is None:
            continue

        team_id = int(team_id)
        opponent_team_id = opponent_map.get(team_id)

        if opponent_team_id is None:
            continue

        t = float(row["event_time_seconds"])
        period = row["period_value"]

        # Previous opponent state.
        prev_opp = None
        j = i - 1

        while j >= 0:
            prev = rows[j]

            if prev.get("period_value") != period:
                break

            if prev.get("type_name") in IGNORE_NEXT_PREV_TYPES:
                j -= 1
                continue

            dt = t - float(prev["event_time_seconds"])
            if dt > max_prev_gap_s:
                break

            if prev.get("team_id") == opponent_team_id:
                prev_opp = prev
                break

            j -= 1

        if prev_opp is not None:
            before_x, before_y = event_state_xy_for_opponent_event(prev_opp)
            before_source = "previous_opponent_event"
            before_event_id = prev_opp.get("event_id")
        else:
            before_x, before_y = defensive_action_xy_to_opponent_perspective(row)
            before_source = "mirrored_defensive_location"
            before_event_id = None

        opponent_xpv_before = xpv_lookup_from_xy(before_x, before_y, xpv_grid)
        before_zone_x, before_zone_y, before_zone = zone_from_xy(before_x, before_y, xpv_grid)

        # Next valid state after defensive action.
        next_valid = None
        k = i + 1

        while k < n:
            nxt = rows[k]

            if nxt.get("period_value") != period:
                break

            if nxt.get("type_name") in IGNORE_NEXT_PREV_TYPES:
                k += 1
                continue

            dt_next = float(nxt["event_time_seconds"]) - t
            if dt_next > max_next_gap_s:
                break

            if nxt.get("team_id") is not None:
                next_valid = nxt
                break

            k += 1

        if next_valid is None:
            if row.get("outcome_name") == "Successful" or action_type in {
                "Clearance",
                "Save",
                "KeeperPickup",
                "KeeperSweeper",
            }:
                after_x, after_y = None, None
                opponent_xpv_after = 0.0
                after_source = "no_next_event_successful_stop"
                after_team_id = None
                after_event_id = None
            else:
                after_x, after_y = before_x, before_y
                opponent_xpv_after = opponent_xpv_before
                after_source = "no_next_event_no_credit"
                after_team_id = None
                after_event_id = None

        else:
            after_team_id = next_valid.get("team_id")
            after_event_id = next_valid.get("event_id")

            if after_team_id == team_id:
                after_x, after_y = None, None
                opponent_xpv_after = 0.0
                after_source = "defending_team_next_event"

            elif after_team_id == opponent_team_id:
                after_x, after_y = event_state_xy_for_opponent_event(next_valid)
                opponent_xpv_after = xpv_lookup_from_xy(after_x, after_y, xpv_grid)
                after_source = "opponent_next_event"

            else:
                after_x, after_y = before_x, before_y
                opponent_xpv_after = opponent_xpv_before
                after_source = "unknown_next_team"

        after_zone_x, after_zone_y, after_zone = zone_from_xy(after_x, after_y, xpv_grid)

        net_threat_reduction = opponent_xpv_before - opponent_xpv_after
        opponent_threat_prevented = max(net_threat_reduction, 0.0)
        opponent_threat_increased = max(-net_threat_reduction, 0.0)

        out.append(
            {
                "match_id": match_id,
                "tournament_id": row.get("tournament_id"),
                "season_id": row.get("season_id"),
                "season_name": row.get("season_name"),
                "start_date": row.get("start_date"),

                "event_id": row.get("event_id"),
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "player_id": row.get("player_id"),

                "period_value": period,
                "minute": row.get("minute"),
                "second": row.get("second"),
                "event_time_seconds": t,

                "type_name": action_type,
                "outcome_name": row.get("outcome_name"),
                "x": row.get("x"),
                "y": row.get("y"),
                "end_x": row.get("end_x"),
                "end_y": row.get("end_y"),

                "before_source": before_source,
                "before_event_id": before_event_id,
                "before_opp_x": before_x,
                "before_opp_y": before_y,
                "before_zone_x": before_zone_x,
                "before_zone_y": before_zone_y,
                "before_zone": before_zone,
                "opponent_xPV_before": opponent_xpv_before,

                "after_source": after_source,
                "after_event_id": after_event_id,
                "after_team_id": after_team_id,
                "after_opp_x": after_x,
                "after_opp_y": after_y,
                "after_zone_x": after_zone_x,
                "after_zone_y": after_zone_y,
                "after_zone": after_zone,
                "opponent_xPV_after": opponent_xpv_after,

                "net_threat_reduction": net_threat_reduction,
                "opponent_threat_prevented": opponent_threat_prevented,
                "opponent_threat_increased": opponent_threat_increased,
            }
        )

    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def score_all_matches(
    events: pl.DataFrame,
    grid: np.ndarray,
    match_team_map: dict[int, dict[int, int]],
    max_prev_gap_s: float,
    max_next_gap_s: float,
    model_name: str,
    model_version: str,
) -> pl.DataFrame:
    match_ids = events.select("match_id").unique().to_series().to_list()
    total = len(match_ids)
    frames = []

    for i, match_id in enumerate(match_ids, start=1):
        if i == 1 or i % 250 == 0 or i == total:
            print(f"Scoring defensive xPV: match {i}/{total}")

        scored = build_defensive_xpv_v2_for_match(
            match_events=events.filter(pl.col("match_id") == match_id),
            xpv_grid=grid,
            match_team_map=match_team_map,
            max_prev_gap_s=max_prev_gap_s,
            max_next_gap_s=max_next_gap_s,
        )

        if scored.height > 0:
            frames.append(scored)

    if not frames:
        return pl.DataFrame()

    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            pl.lit(model_name).alias("xpv_model_name"),
            pl.lit(model_version).alias("xpv_model_version"),
            pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("created_at"),
        )
        .sort(["match_id", "period_value", "event_time_seconds", "event_id"])
    )


# ---------------------------------------------------------------------------
# Input:  normalized events.parquet
# Output: feature   defensive_xpv_actions.parquet
# Uses the versioned xPV grid from ``models/features``.
# ---------------------------------------------------------------------------


def prepare_events_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add time columns + type-coerce (mirrors load_events_for_match_batch)."""
    if df.height == 0:
        return df
    return (
        add_time_cols(df)
        .with_columns(
            pl.col("match_id").cast(pl.Int64),
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("season_id").cast(pl.Int64, strict=False),
            pl.col("event_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            pl.col("period_value").cast(pl.Int64),
            pl.col("x").cast(pl.Float64, strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            pl.col("end_x").cast(pl.Float64, strict=False),
            pl.col("end_y").cast(pl.Float64, strict=False),
            pl.col("is_touch").fill_null(0).cast(pl.Int64),
            pl.col("is_shot").fill_null(0).cast(pl.Int64),
            pl.col("is_goal").fill_null(0).cast(pl.Int64),
        )
    )
