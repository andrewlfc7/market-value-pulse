from __future__ import annotations

from typing import Any

import orjson
import polars as pl


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


def _looks_like_event(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and ("eventId" in value or "id" in value)
        and ("minute" in value or "expandedMinute" in value)
        and isinstance(value.get("type"), dict)
    )


def _walk(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if _looks_like_event(value):
            found.append((path, value))
        for key, child in value.items():
            found.extend(_walk(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, f"{path}[{index}]"))
    return found


def raw_events_frame(
    match_data: dict[str, Any],
    *,
    match_id: int,
    source_url: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_path, event in _walk(match_data):
        persistent_id = event.get("id")
        rows.append(
            {
                "match_id": match_id,
                "source_url": source_url,
                "source_path": source_path,
                "persistent_id": str(persistent_id) if persistent_id is not None else None,
                "event_id": event.get("eventId"),
                "minute": event.get("minute"),
                "second": event.get("second"),
                "expanded_minute": event.get("expandedMinute"),
                "team_id": event.get("teamId"),
                "player_id": event.get("playerId"),
                "related_event_id": event.get("relatedEventId"),
                "related_player_id": event.get("relatedPlayerId"),
                "x": event.get("x"),
                "y": event.get("y"),
                "end_x": event.get("endX"),
                "end_y": event.get("endY"),
                "period_value": _display_value(event.get("period")),
                "period_display_name": _display_name(event.get("period")),
                "type_value": _display_value(event.get("type")),
                "type_display_name": _display_name(event.get("type")),
                "outcome_type_value": _display_value(event.get("outcomeType")),
                "outcome_type_display_name": _display_name(event.get("outcomeType")),
                "is_touch": bool(event.get("isTouch", False)),
                "is_shot": bool(event.get("isShot", False)),
                "is_goal": bool(event.get("isGoal", False)),
                "qualifiers_json": _json(event.get("qualifiers") or []),
                "raw_event_json": _json(event),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
