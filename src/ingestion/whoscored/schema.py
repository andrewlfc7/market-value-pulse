from __future__ import annotations

EVENT_RENAME: dict[str, str] = {
    "eventId": "event_id",
    "expandedMinute": "expanded_minute",
    "teamId": "team_id",
    "playerId": "player_id",
    "endX": "end_x",
    "endY": "end_y",
    "outcomeType": "outcome_type",
    "isTouch": "is_touch",
    "isShot": "is_shot",
    "isGoal": "is_goal",
    "cardType": "card_type",
    "goalMouthY": "goal_mouth_y",
    "goalMouthZ": "goal_mouth_z",
    "blockedX": "blocked_x",
    "blockedY": "blocked_y",
    "relatedEventId": "related_event_id",
    "relatedPlayerId": "related_player_id",
    "satisfiedEventsTypes": "satisfied_events_types",
}

SHOT_QUALIFIER_FLAGS: dict[str, str] = {
    "Head": "is_header",
    "RightFoot": "is_right_foot",
    "LeftFoot": "is_left_foot",
    "OtherBodyPart": "is_other_body_part",
    "RegularPlay": "is_regular_play",
    "SetPiece": "is_set_piece",
    "FromCorner": "is_from_corner",
    "Penalty": "is_penalty",
    "DirectFreekick": "is_direct_free_kick",
    "Assisted": "is_assisted",
    "IntentionalAssist": "is_intentional_assist",
    "BigChance": "is_big_chance",
    "FastBreak": "is_fast_break",
    "FirstTouch": "is_first_touch",
    "Volley": "is_volley",
    "OwnGoal": "is_own_goal",
}

NORMALIZED_DATASETS = (
    "matches",
    "teams",
    "player_matches",
    "events",
    "shots",
)
