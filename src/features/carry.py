from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import polars as pl

MODEL_VERSION = "carry_inference_v1_min5m"
CARRY_FAMILY = "carry"

VALID_PREV_TYPES = {
    "Pass",
    "TakeOn",
    "BallTouch",
    "BallRecovery",
    "Interception",
    "Tackle",
    "KeeperPickup",
    "KeeperSweeper",
    "Clearance",
    "BlockedPass",
}

VALID_NEXT_TYPES = {
    "Pass",
    "Shot",
    "SavedShot",
    "MissedShots",
    "Goal",
    "TakeOn",
    "BallTouch",
    "Dispossessed",
}

IGNORE_TYPES = {
    "SubstitutionOff",
    "SubstitutionOn",
    "Card",
    "FormationChange",
}


def carry_distance_meters(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = 105.0 * (x2 - x1) / 100.0
    dy = 68.0 * (y2 - y1) / 100.0
    return math.sqrt(dx * dx + dy * dy)


def carry_distance_100(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def goal_distance_120_80(x: float, y: float) -> float:
    x_120 = 120.0 * x / 100.0
    y_80 = 80.0 * y / 100.0
    return math.sqrt((120.0 - x_120) ** 2 + (40.0 - y_80) ** 2)


def is_progressive_action(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    x_start_120 = 120.0 * start_x / 100.0
    x_end_120 = 120.0 * end_x / 100.0

    delta_goal_dist = goal_distance_120_80(start_x, start_y) - goal_distance_120_80(end_x, end_y)

    if x_start_120 < 60.0 and x_end_120 < 60.0 and delta_goal_dist >= 32.8:
        return True
    if x_start_120 < 60.0 and x_end_120 >= 60.0 and delta_goal_dist >= 16.4:
        return True
    if x_start_120 >= 60.0 and x_end_120 >= 60.0 and delta_goal_dist >= 10.94:
        return True

    return False


def is_box_entry(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    start_in_box = start_x >= 85.0 and 22.5 <= start_y <= 77.5
    end_in_box = end_x >= 85.0 and 22.5 <= end_y <= 77.5
    return (not start_in_box) and end_in_box


def is_final_third_entry(start_x: float, end_x: float) -> bool:
    return start_x < 66.7 and end_x >= 66.7


def zone_id(x: float, y: float, nx: int = 12, ny: int = 8) -> tuple[int, int, int]:
    x_clip = min(max(float(x), 0.0), 99.999)
    y_clip = min(max(float(y), 0.0), 99.999)

    zx = int(math.floor((x_clip / 100.0) * nx))
    zy = int(math.floor((y_clip / 100.0) * ny))
    z = zx + zy * nx

    return zx, zy, z


def infer_carries_for_match(
    match_events: pl.DataFrame,
    min_carry_m: float,
    max_carry_m: float,
    min_duration_s: float,
    max_duration_s: float,
    nx: int,
    ny: int,
) -> pl.DataFrame:
    sort_columns = ["period_value", "event_time_seconds", "event_id"]
    if "event_uid" in match_events.columns:
        sort_columns.append("event_uid")
    rows = (
        match_events
        .filter(
            pl.col("match_id").is_not_null()
            & pl.col("team_id").is_not_null()
            & pl.col("player_id").is_not_null()
            & pl.col("period_value").is_not_null()
            & pl.col("event_time_seconds").is_not_null()
        )
        .sort(sort_columns)
        .to_dicts()
    )

    carries: list[dict[str, Any]] = []

    n = len(rows)

    for idx, prev in enumerate(rows):
        if idx >= n - 1:
            continue

        prev_type = prev.get("type_name")
        prev_outcome = prev.get("outcome_name")

        if prev_type in IGNORE_TYPES:
            continue

        if prev_type == "BallTouch":
            continue

        if prev_type not in VALID_PREV_TYPES:
            continue

        if prev_outcome != "Successful":
            continue

        if prev.get("end_x") is None or prev.get("end_y") is None:
            continue

        take_ons = 0
        next_idx = idx + 1
        initial_next = rows[next_idx]
        next_valid = None

        while next_idx < n:
            candidate = rows[next_idx]

            if candidate.get("period_value") != prev.get("period_value"):
                break

            candidate_type = candidate.get("type_name")
            candidate_outcome = candidate.get("outcome_name")
            candidate_team = candidate.get("team_id")

            if candidate_type in IGNORE_TYPES:
                next_idx += 1
                continue

            skip = False

            if candidate_type == "TakeOn" and candidate_outcome == "Successful":
                take_ons += 1
                skip = True

            elif candidate_type == "TakeOn" and candidate_outcome == "Unsuccessful":
                skip = True

            elif (
                candidate_team != prev.get("team_id")
                and candidate_type == "Challenge"
                and candidate_outcome == "Unsuccessful"
            ):
                skip = True

            elif candidate_type == "Foul":
                skip = True

            if skip:
                next_idx += 1
                continue

            next_valid = candidate
            break

        if next_valid is None:
            continue

        if next_valid.get("team_id") != prev.get("team_id"):
            continue

        if next_valid.get("type_name") not in VALID_NEXT_TYPES:
            continue

        if next_valid.get("x") is None or next_valid.get("y") is None:
            continue
        if prev.get("event_id") is None or next_valid.get("event_id") is None:
            continue

        start_x = float(prev["end_x"])
        start_y = float(prev["end_y"])
        end_x = float(next_valid["x"])
        end_y = float(next_valid["y"])

        duration_s = float(next_valid["event_time_seconds"]) - float(prev["event_time_seconds"])
        dist_m = carry_distance_meters(start_x, start_y, end_x, end_y)
        dist_100 = carry_distance_100(start_x, start_y, end_x, end_y)

        if not (min_duration_s <= duration_s <= max_duration_s):
            continue

        if not (min_carry_m <= dist_m <= max_carry_m):
            continue

        carry_time = (
            float(prev["event_time_seconds"]) + float(initial_next["event_time_seconds"])
        ) / 2.0

        carry_minute = int(carry_time // 60)
        carry_second = float(carry_time - carry_minute * 60)

        start_zone_x, start_zone_y, start_zone = zone_id(start_x, start_y, nx, ny)
        end_zone_x, end_zone_y, end_zone = zone_id(end_x, end_y, nx, ny)

        is_progressive = is_progressive_action(start_x, start_y, end_x, end_y)
        is_final_third = is_final_third_entry(start_x, end_x)
        is_into_box = is_box_entry(start_x, start_y, end_x, end_y)

        carries.append(
            {
                "match_id": int(prev["match_id"]),
                "tournament_id": int(prev["tournament_id"]) if prev.get("tournament_id") is not None else None,
                "season_id": int(prev["season_id"]) if prev.get("season_id") is not None else None,
                "season_name": prev.get("season_name"),
                "start_date": prev.get("start_date"),

                "event_id": float(prev["event_id"]) + 0.5,
                "event_uid": (
                    f"carry:{prev.get('event_uid') or prev['event_id']}:"
                    f"{next_valid.get('event_uid') or next_valid['event_id']}"
                ),
                "source_event_id": int(prev["event_id"]),
                "target_event_id": int(next_valid["event_id"]),

                "period_value": int(next_valid["period_value"]),
                "minute": carry_minute,
                "second": carry_second,
                "event_time_seconds": carry_time,

                "team_id": int(next_valid["team_id"]),
                "player_id": int(next_valid["player_id"]),

                "x": start_x,
                "y": start_y,
                "end_x": end_x,
                "end_y": end_y,

                "start_zone_x": start_zone_x,
                "start_zone_y": start_zone_y,
                "start_zone": start_zone,
                "end_zone_x": end_zone_x,
                "end_zone_y": end_zone_y,
                "end_zone": end_zone,

                "carry_duration_seconds": duration_s,
                "carry_distance_m": dist_m,
                "carry_distance_100": dist_100,
                "take_ons": int(take_ons),

                "is_progressive_carry": int(is_progressive),
                "is_final_third_carry": int(is_final_third),
                "is_carry_into_box": int(is_into_box),

                # value fields filled by later xT/xPV scoring scripts
                "xT_start": None,
                "xT_end": None,
                "xT_added": None,
                "xT_added_positive": None,
                "xT_added_negative": None,

                "xPV_start": None,
                "xPV_end": None,
                "xPV_added": None,
                "xPV_added_positive": None,
                "xPV_added_negative": None,

                "model_version": MODEL_VERSION,
                "created_at": datetime.utcnow(),
            }
        )

    return pl.DataFrame(carries) if carries else pl.DataFrame()


def infer_carries(
    events: pl.DataFrame,
    min_carry_m: float,
    max_carry_m: float,
    min_duration_s: float,
    max_duration_s: float,
    nx: int,
    ny: int,
) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()

    match_ids = events.select("match_id").unique().to_series().to_list()

    out = []
    total = len(match_ids)

    for i, match_id in enumerate(match_ids, start=1):
        if i % 100 == 0 or i == 1 or i == total:
            print(f"Infer carries: match {i}/{total}")

        match_events = events.filter(pl.col("match_id") == match_id)

        carries = infer_carries_for_match(
            match_events=match_events,
            min_carry_m=min_carry_m,
            max_carry_m=max_carry_m,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            nx=nx,
            ny=ny,
        )

        if carries.height > 0:
            out.append(carries)

    if not out:
        return pl.DataFrame()

    return (
        pl.concat(out, how="diagonal_relaxed")
        .sort(["match_id", "period_value", "event_time_seconds", "event_id"])
    )


# ---------------------------------------------------------------------------
# Input:  normalized events.parquet
# Output: feature   carries.parquet
# ---------------------------------------------------------------------------


def prepare_events_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add the time columns infer_carries expects (mirrors load_events)."""
    if df.height == 0:
        return df
    return (
        df.with_columns(
            pl.col("minute").fill_null(0).cast(pl.Int64),
            pl.col("second").fill_null(0).cast(pl.Float64),
            pl.col("expanded_minute").fill_null(pl.col("minute")).cast(pl.Int64),
            pl.col("period_value").fill_null(0).cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
            pl.col("match_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            pl.col("x").cast(pl.Float64, strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            pl.col("end_x").cast(pl.Float64, strict=False),
            pl.col("end_y").cast(pl.Float64, strict=False),
        )
        .with_columns(
            (
                pl.col("expanded_minute").cast(pl.Float64) * 60.0
                + pl.col("second").fill_null(0.0)
            ).alias("event_time_seconds"),
            (
                pl.col("expanded_minute").cast(pl.Float64)
                + pl.col("second").fill_null(0.0) / 60.0
            ).alias("cumulative_mins"),
        )
    )
