"""Appearance-minute reconciliation for WhoScored player-match rows."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl


_NON_ACTION_TYPES = {
    "card",
    "formationchange",
    "start",
    "end",
    "substitutionoff",
    "substitutionon",
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_minute(row: dict[str, Any]) -> float:
    # Use the regulation match clock for appearance minutes. Expanded minutes
    # include accumulated stoppage time (for example minute=45 can have
    # expanded_minute=49), which would under-count a second-half substitute.
    value = _number(row.get("minute"))
    if value is None:
        value = _number(row.get("expanded_minute"))
    return max(value or 0.0, 0.0)


def _event_type(row: dict[str, Any]) -> str:
    value = row.get("type_name") or row.get("type_display_name")
    return str(value or "").casefold().replace(" ", "").replace("_", "")


def _is_dismissal(row: dict[str, Any]) -> bool:
    card = str(
        row.get("card_type_display_name")
        or row.get("card_type_name")
        or ""
    ).casefold().replace(" ", "").replace("_", "")
    return "red" in card or "secondyellow" in card


_POSITION_GROUPS = {"Goalkeeper", "Defender", "Midfielder", "Forward"}

_DEFENDER_POSITIONS = {
    "DC",
    "DL",
    "DR",
    "DCL",
    "DCR",
    "WB",
    "WBL",
    "WBR",
}
_MIDFIELDER_POSITIONS = {"DMC", "MC", "ML", "MR", "MCL", "MCR"}
_FORWARD_POSITIONS = {"AMC", "AML", "AMR", "FW", "FWL", "FWR", "ST", "CF"}


def _position_group(value: Any) -> str | None:
    raw = str(value or "").strip()
    if raw in _POSITION_GROUPS:
        return raw
    code = raw.upper()
    if not code or code == "SUB":
        return None
    if code.startswith("GK"):
        return "Goalkeeper"
    if code in _DEFENDER_POSITIONS:
        return "Defender"
    if code in _MIDFIELDER_POSITIONS:
        return "Midfielder"
    if code in _FORWARD_POSITIONS:
        return "Forward"
    # Preserve a conservative fallback for provider variants not in the
    # explicit WhoScored position registry.
    if code.startswith("D"):
        return "Defender"
    if code.startswith("M"):
        return "Midfielder"
    if code.startswith("F") or code.startswith("W"):
        return "Forward"
    return None


def reconcile_position_groups(
    player_matches: pl.DataFrame,
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Resolve broad position groups, including WhoScored ``Sub`` players.

    WhoScored exposes the tactical position for starters but labels every bench
    player as ``Sub``. Used substitutes inherit the broad group of the linked
    outgoing player. The substitution relationship is present both in events
    and in the preserved player JSON, and chains are resolved iteratively for
    players who enter and are later substituted themselves.
    """
    if player_matches.is_empty():
        return player_matches, 0

    rows = player_matches.to_dicts()
    groups: dict[int, str | None] = {
        int(row["player_id"]): _position_group(
            row.get("position_group") or row.get("position")
        )
        for row in rows
    }
    sources: dict[int, str] = {
        int(row["player_id"]): (
            str(row.get("position_group_source") or "provider_position")
            if groups[int(row["player_id"])] is not None
            else "unresolved"
        )
        for row in rows
    }
    replacement_pairs: set[tuple[int, int]] = set()

    for event in events.to_dicts() if not events.is_empty() else []:
        event_type = _event_type(event)
        player = _number(event.get("player_id"))
        related = _number(event.get("related_player_id"))
        if player is None or related is None:
            continue
        player_id = int(player)
        related_id = int(related)
        if event_type == "substitutionon":
            replacement_pairs.add((player_id, related_id))
        elif event_type == "substitutionoff":
            replacement_pairs.add((related_id, player_id))

    for row in rows:
        payload = row.get("player_json")
        if not isinstance(payload, str) or not payload:
            continue
        try:
            player = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        player_id = int(row["player_id"])
        outgoing = _number(player.get("subbedOutPlayerId"))
        incoming = _number(player.get("subbedInPlayerId"))
        if outgoing is not None:
            replacement_pairs.add((player_id, int(outgoing)))
        if incoming is not None:
            replacement_pairs.add((int(incoming), player_id))

    for _ in range(len(replacement_pairs) + 1):
        changed = False
        for incoming, outgoing in replacement_pairs:
            if groups.get(incoming) is None and groups.get(outgoing) is not None:
                groups[incoming] = groups[outgoing]
                sources[incoming] = "replacement_player_position"
                changed = True
        if not changed:
            break

    reconciled = [
        {
            "match_id": int(row["match_id"]),
            "player_id": int(row["player_id"]),
            "position_group": groups.get(int(row["player_id"])) or "Unknown",
            "position_group_source": sources.get(int(row["player_id"]), "unresolved"),
        }
        for row in rows
    ]
    derived_count = sum(
        record["position_group_source"] == "replacement_player_position"
        for record in reconciled
    )
    lookup = pl.DataFrame(reconciled, infer_schema_length=None)
    existing = [
        column
        for column in ("position_group", "position_group_source")
        if column in player_matches.columns
    ]
    base = player_matches.drop(existing) if existing else player_matches
    output = base.join(
        lookup,
        on=["match_id", "player_id"],
        how="left",
        validate="1:1",
    )
    return output, int(derived_count)


def derive_appearance_minutes(
    player_matches: pl.DataFrame,
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Fill missing participant minutes from lineup and event timing.

    Provider-supplied positive minutes always win. For missing values, starters
    begin at minute zero; substitutes begin at their substitution-on event (or
    the related player on a substitution-off event) and end at substitution-off,
    dismissal, or the nominal match end. A first on-pitch action is the final
    fallback.

    Returns the reconciled frame and the number of positive appearances whose
    minutes were derived rather than supplied by the provider.
    """
    if player_matches.is_empty():
        return player_matches, 0

    event_rows = events.to_dicts() if not events.is_empty() else []
    observed_end = max((_event_minute(row) for row in event_rows), default=90.0)
    has_extra_time = observed_end > 105.0 or any(
        "extra" in str(
            row.get("period_name") or row.get("period_display_name") or ""
        ).casefold()
        or _number(row.get("period_value")) in {3.0, 4.0}
        for row in event_rows
    )
    match_end = 120.0 if has_extra_time else 90.0

    on_minutes: dict[int, float] = {}
    off_minutes: dict[int, float] = {}
    first_actions: dict[int, float] = {}

    for row in event_rows:
        minute = _event_minute(row)
        event_type = _event_type(row)
        player = _number(row.get("player_id"))
        related = _number(row.get("related_player_id"))
        player_id = int(player) if player is not None else None
        related_id = int(related) if related is not None else None

        if event_type == "substitutionon":
            if player_id is not None:
                on_minutes[player_id] = min(on_minutes.get(player_id, minute), minute)
            if related_id is not None:
                off_minutes[related_id] = min(off_minutes.get(related_id, minute), minute)
        elif event_type == "substitutionoff":
            if player_id is not None:
                off_minutes[player_id] = min(off_minutes.get(player_id, minute), minute)
            if related_id is not None:
                on_minutes[related_id] = min(on_minutes.get(related_id, minute), minute)
        elif player_id is not None and _is_dismissal(row):
            off_minutes[player_id] = min(
                off_minutes.get(player_id, minute), minute
            )
        elif player_id is not None and event_type not in _NON_ACTION_TYPES:
            first_actions[player_id] = min(
                first_actions.get(player_id, minute), minute
            )

    reconciled: list[dict[str, object]] = []
    derived_count = 0
    for row in player_matches.to_dicts():
        player_id = int(row["player_id"])
        supplied = _number(row.get("minutes"))
        started = bool(row.get("started"))
        participated = (
            started
            or player_id in on_minutes
            or player_id in off_minutes
            or player_id in first_actions
        )

        if supplied is not None and supplied > 0.0:
            minutes = supplied
            source = "provider"
        elif participated:
            start = 0.0 if started else on_minutes.get(
                player_id, first_actions.get(player_id, match_end)
            )
            end = off_minutes.get(player_id, match_end)
            start = min(max(start, 0.0), match_end)
            end = min(max(end, start), match_end)
            minutes = min(max(end - start, 1.0), 130.0)
            source = "derived_lineup_events"
            derived_count += 1
        else:
            minutes = max(supplied or 0.0, 0.0)
            source = "provider" if supplied is not None else "not_played"

        reconciled.append(
            {
                "match_id": int(row["match_id"]),
                "player_id": player_id,
                "minutes": float(minutes),
                "minutes_source": source,
            }
        )

    lookup = pl.DataFrame(reconciled, infer_schema_length=None)
    existing = [
        column
        for column in ("minutes", "minutes_source")
        if column in player_matches.columns
    ]
    base = player_matches.drop(existing) if existing else player_matches
    output = base.join(
        lookup,
        on=["match_id", "player_id"],
        how="left",
        validate="1:1",
    )
    return output, derived_count
