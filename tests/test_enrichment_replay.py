from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from features.adapter import build_compatibility_bundle
from features.pipeline import enrich_match, enrich_season
from ingestion.whoscored.normalize import normalize_match_data
from pipelines.replay import run_historical_replay, select_replay_matches


def _payload(*, start_date: str, player_id: int = 10) -> dict:
    return {
        "startDate": start_date,
        "score": "1 : 0",
        "ftScore": "1 : 0",
        "statusCode": 6,
        "home": {
            "teamId": 1,
            "name": "Home",
            "players": [
                {
                    "playerId": player_id,
                    "name": "Player A",
                    "position": "FW",
                    "isFirstEleven": True,
                    "stats": {"minutesPlayed": 90},
                }
            ],
        },
        "away": {
            "teamId": 2,
            "name": "Away",
            "players": [
                {
                    "playerId": 20,
                    "name": "Player B",
                    "position": "DC",
                    "isFirstEleven": True,
                    "stats": {"minutesPlayed": 90},
                }
            ],
        },
        "events": [
            {
                "id": 100,
                "eventId": 1,
                "minute": 12,
                "second": 10,
                "expandedMinute": 12,
                "teamId": 1,
                "playerId": player_id,
                "x": 50,
                "y": 45,
                "endX": 88,
                "endY": 50,
                "isTouch": True,
                "type": {"value": 1, "displayName": "Pass"},
                "outcomeType": {"value": 1, "displayName": "Successful"},
                "qualifiers": [],
            },
            {
                "id": 102,
                "eventId": 1,
                "minute": 12,
                "second": 11,
                "expandedMinute": 12,
                "teamId": 2,
                "playerId": 20,
                "x": 35,
                "y": 55,
                "endX": 45,
                "endY": 50,
                "isTouch": True,
                "type": {"value": 1, "displayName": "Pass"},
                "outcomeType": {"value": 1, "displayName": "Successful"},
                "qualifiers": [],
            },
            {
                "id": 101,
                "eventId": 2,
                "minute": 12,
                "second": 14,
                "expandedMinute": 12,
                "teamId": 1,
                "playerId": player_id,
                "x": 90,
                "y": 50,
                "isShot": True,
                "isGoal": True,
                "isTouch": True,
                "relatedEventId": 1,
                "type": {"value": 16, "displayName": "Goal"},
                "outcomeType": {"value": 1, "displayName": "Successful"},
                "qualifiers": [
                    {"type": {"value": 72, "displayName": "RightFoot"}},
                    {"type": {"value": 55, "displayName": "RegularPlay"}},
                    {"type": {"value": 154, "displayName": "IntentionalAssist"}},
                ],
            },
        ],
    }


def _write_match(
    normalized_root: Path,
    *,
    match_id: int,
    start_date: str,
    player_id: int = 10,
) -> Path:
    partition = (
        normalized_root
        / "competition=EPL"
        / "season=2025-2026"
        / "matches"
        / f"match_id={match_id}"
    )
    partition.mkdir(parents=True)
    bundle = normalize_match_data(
        _payload(start_date=start_date, player_id=player_id),
        match_id=match_id,
        source_url=f"https://www.whoscored.com/Matches/{match_id}/live/test",
    )
    for name in ("matches", "player_matches", "events", "shots"):
        getattr(bundle, name).write_parquet(partition / f"{name}.parquet")
    (partition / "_SUCCESS.json").write_text(
        json.dumps({"status": "succeeded", "match_id": match_id}),
        encoding="utf-8",
    )
    return partition


def test_adapter_and_prepare_only_are_idempotent(tmp_path: Path) -> None:
    normalized_root = tmp_path / "normalized"
    partition = _write_match(
        normalized_root,
        match_id=1903444,
        start_date="2026-04-12T15:00:00",
    )
    bundle = build_compatibility_bundle(
        partition,
        competition_id=2,
        season="2025-2026",
        season_id=10743,
    )

    assert bundle.passes.height == 2
    assert bundle.shots.height == 1
    assert bundle.passes["success"].to_list() == [1, 1]
    assert bundle.events["event_uid"].n_unique() == bundle.events.height
    numeric_id_check = next(
        check for check in bundle.checks if check["check"] == "numeric_event_id_duplicates"
    )
    assert numeric_id_check["value"] == 2

    output_root = tmp_path / "enriched"
    first = enrich_match(
        partition,
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        output_root=output_root,
        prepare_only=True,
    )
    second = enrich_match(
        partition,
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        output_root=output_root,
        prepare_only=True,
    )

    assert first.status == "prepared"
    assert second.status == "skipped_existing"
    assert (first.output_directory / "adapted_events.parquet").exists()
    assert (first.output_directory / "adapted_shots.parquet").exists()
    assert (first.output_directory / "passes.parquet").exists()
    assert (first.output_directory / "_PREPARED.json").exists()
    marker = json.loads((first.output_directory / "_PREPARED.json").read_text())
    assert marker["input_signature"]

    events_path = partition / "events.parquet"
    changed_events = pl.read_parquet(events_path).with_columns(
        pl.when(pl.col("event_id") == 2)
        .then(pl.lit(13))
        .otherwise(pl.col("minute"))
        .alias("minute")
    )
    changed_events.write_parquet(events_path)
    changed = enrich_match(
        partition,
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        output_root=output_root,
        prepare_only=True,
    )
    assert changed.status == "prepared"
    changed_marker = json.loads(
        (changed.output_directory / "_PREPARED.json").read_text()
    )
    assert changed_marker["input_signature"] != marker["input_signature"]


def test_season_enrichment_only_prepares_new_partitions(tmp_path: Path) -> None:
    normalized_root = tmp_path / "normalized"
    for offset in range(3):
        _write_match(
            normalized_root,
            match_id=1903444 + offset,
            start_date=f"2026-04-{10 + offset:02d}T15:00:00",
        )
    output_root = tmp_path / "enriched"
    first = enrich_season(
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        normalized_root=normalized_root,
        output_root=output_root,
        max_matches=2,
        prepare_only=True,
    )
    second = enrich_season(
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        normalized_root=normalized_root,
        output_root=output_root,
        max_matches=2,
        prepare_only=True,
    )

    assert first.selected == first.processed == 2
    assert second.skipped == 2
    assert second.selected == second.processed == 1


def test_replay_selects_and_runs_last_real_matches(tmp_path: Path) -> None:
    normalized_root = tmp_path / "normalized"
    for offset in range(10):
        _write_match(
            normalized_root,
            match_id=1903400 + offset,
            start_date=f"2026-04-{offset + 1:02d}T15:00:00",
            player_id=10 if offset != 9 else 99,
        )

    selected = select_replay_matches(
        normalized_root=normalized_root,
        competition="EPL",
        season="2025-2026",
        match_count=8,
        player_id=10,
    )
    assert [row["match_id"] for row in selected] == list(range(1903401, 1903409))
    assert [row["replay_sequence"] for row in selected] == list(range(1, 9))

    result = run_historical_replay(
        competition="EPL",
        competition_id=2,
        season="2025-2026",
        match_count=8,
        player_id=10,
        normalized_root=normalized_root,
        replay_root=tmp_path / "replays",
        prepare_only=True,
    )
    rows = pl.read_parquet(result.results_path)

    assert result.selected_matches == result.completed_matches == 8
    assert result.failed_matches == 0
    assert rows["match_id"].to_list() == list(range(1903401, 1903409))
    assert set(rows["rating_update_status"].to_list()) == {"pending_rating_model"}
    assert (result.run_directory / "replay_state.json").exists()
