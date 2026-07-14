from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.whoscored.catalog import (
    normalize_season_label,
    parse_seasons,
    parse_stages,
    resolve_fixture_stage,
    resolve_season,
)
from ingestion.whoscored.competitions import Competition, resolve_competition
from ingestion.whoscored.discovery import discover_match_links
from ingestion.whoscored.extract import parse_match_centre_data
from ingestion.whoscored.fetch import MatchFetchResult
from ingestion.whoscored.normalize import normalize_match_data
from ingestion.whoscored.scope import ScopeDiscoveryResult
from ingestion.whoscored.validation import validate_normalized_match


def _payload() -> dict:
    return {
        "startDate": "2026-04-12T15:00:00",
        "score": "1 : 0",
        "ftScore": "1 : 0",
        "statusCode": 6,
        "home": {"teamId": 1, "name": "Home", "players": [{"playerId": 10, "name": "A", "isFirstEleven": True, "stats": {"minutesPlayed": 90}}]},
        "away": {"teamId": 2, "name": "Away", "players": [{"playerId": 20, "name": "B", "stats": {"minutesPlayed": 65}}], "incidentEvents": [{"id": 101, "eventId": 2, "minute": 40, "teamId": 2, "playerId": 20, "x": 70, "y": 40, "isShot": False, "type": {"value": 1, "displayName": "Pass"}}]},
        "events": [
            {"id": 100, "eventId": 1, "minute": 12, "teamId": 1, "playerId": 10, "x": 90, "y": 48, "isShot": True, "isGoal": True, "type": {"value": 16, "displayName": "Goal"}, "outcomeType": {"value": 1, "displayName": "Successful"}, "qualifiers": []},
            {"id": 101, "eventId": 2, "minute": 40, "teamId": 2, "playerId": 20, "x": 70, "y": 40, "isShot": False, "type": {"value": 1, "displayName": "Pass"}}
        ]
    }


def test_extract_and_normalize_match_centre_data() -> None:
    html = f"<script>require.config.params[\"args\"] = {{matchId: 1903444, matchCentreData: {json.dumps(_payload())}}};</script>"
    parsed = parse_match_centre_data(html)
    bundle = normalize_match_data(parsed, match_id=1903444, source_url="https://www.whoscored.com/Matches/1903444/live/test")
    assert bundle.matches.height == 1
    assert bundle.teams.height == 2
    assert bundle.player_matches.height == 2
    assert bundle.events.height == 2  # duplicated incident event is idempotently removed
    assert bundle.shots.height == 1
    assert all(check.passed or check.severity == "warning" for check in validate_normalized_match(bundle))


def test_missing_minutes_are_derived_from_lineup_and_substitutions() -> None:
    payload = {
        "startDate": "2026-04-12T15:00:00",
        "score": "0 : 0",
        "ftScore": "0 : 0",
        "statusCode": 6,
        "home": {
            "teamId": 1,
            "name": "Home",
            "players": [
                {"playerId": 10, "name": "Starter", "position": "FW", "isFirstEleven": True},
                {"playerId": 11, "name": "Substitute", "position": "Sub", "isFirstEleven": False},
            ],
        },
        "away": {
            "teamId": 2,
            "name": "Away",
            "players": [
                {"playerId": 20, "name": "Opponent", "position": "DC", "isFirstEleven": True}
            ],
        },
        "events": [
            {
                "id": 1,
                "eventId": 1,
                "minute": 5,
                "teamId": 1,
                "playerId": 10,
                "type": {"value": 1, "displayName": "Pass"},
            },
            {
                "id": 2,
                "eventId": 2,
                "minute": 60,
                "expandedMinute": 64,
                "teamId": 1,
                "playerId": 10,
                "relatedPlayerId": 11,
                "type": {"value": 18, "displayName": "SubstitutionOff"},
            },
            {
                "id": 3,
                "eventId": 3,
                "minute": 61,
                "teamId": 1,
                "playerId": 11,
                "type": {"value": 1, "displayName": "Pass"},
            },
            {
                "id": 4,
                "eventId": 4,
                "minute": 80,
                "teamId": 2,
                "playerId": 20,
                "type": {"value": 17, "displayName": "Card"},
                "cardType": {"value": 2, "displayName": "Red"},
            },
        ],
    }
    bundle = normalize_match_data(
        payload,
        match_id=1903444,
        source_url="https://www.whoscored.com/Matches/1903444/live/test",
    )
    appearances = {
        row["player_id"]: (
            row["minutes"],
            row["minutes_source"],
            row["position_group"],
            row["position_group_source"],
        )
        for row in bundle.player_matches.to_dicts()
    }

    assert appearances == {
        10: (60.0, "derived_lineup_events", "Forward", "provider_position"),
        11: (
            30.0,
            "derived_lineup_events",
            "Forward",
            "replacement_player_position",
        ),
        20: (80.0, "derived_lineup_events", "Defender", "provider_position"),
    }
    assert bundle.metadata["derived_appearance_minutes_rows"] == 3
    assert bundle.metadata["derived_position_group_rows"] == 1


def test_discover_match_links_deduplicates_ids() -> None:
    html = '<a href="/Matches/123/live/a">one</a><a href="/matches/123/matchreport/a">two</a>'
    rows = discover_match_links(html, "fixture-page")
    assert len(rows) == 1
    assert rows[0].match_id == 123
    assert rows[0].page_kind == "live"


def test_discover_match_links_reads_fixture_date() -> None:
    html = """
    <section>
      <h3>Tuesday, May 19 2026</h3>
      <div><a href="/Matches/123/live/a">one</a></div>
    </section>
    """
    rows = discover_match_links(html, "fixture-page")
    assert rows[0].fixture_date == "2026-05-19"


def test_resolve_competition_season_and_fixture_stage() -> None:
    html = """
    <select id="seasons">
      <option value="/regions/252/tournaments/2/seasons/10316/england-premier-league-2024-2025">2024/2025</option>
      <option selected value="/regions/252/tournaments/2/seasons/10743/england-premier-league-2025-2026">2025/2026</option>
    </select>
    <a class="selected" href="/regions/252/tournaments/2/seasons/10743/stages/24533/fixtures/england-premier-league-2025-2026">Fixtures</a>
    """
    seasons = parse_seasons(html)
    season = resolve_season(seasons, "2025-26")
    stage = resolve_fixture_stage(parse_stages(html))
    assert normalize_season_label("2025/2026") == "2025-2026"
    assert season.season_id == 10743
    assert stage.stage_id == 24533
    assert resolve_competition("premier league").key == "EPL"


def test_match_completion_guard() -> None:
    from ingestion.whoscored.runner import _is_completed_match

    assert _is_completed_match({"statusCode": 6, "ftScore": "2 : 1"})
    assert not _is_completed_match({"statusCode": 3, "ftScore": ""})


def test_automatic_ingestion_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingestion.whoscored import runner

    competition = Competition(
        key="EPL",
        name="Premier League",
        aliases=("epl",),
        region_id=252,
        tournament_id=2,
        slug="england-premier-league",
        expected_matches=380,
        stage_overrides={},
    )
    source_url = "https://www.whoscored.com/Matches/1903444/live/test"
    discovery = ScopeDiscoveryResult(
        competition=competition,
        requested_season="2025-2026",
        canonical_season="2025-2026",
        season_id=10743,
        stage_id=24533,
        fixture_url="https://www.whoscored.com/fixtures/test",
        matches=[{"match_id": 1903444, "match_url": source_url}],
        pages_collected=10,
        warnings=(),
    )
    page_html = (
        '<script>require.config.params["args"] = {'
        f'matchId: 1903444, matchCentreData: {json.dumps(_payload())}'
        "};</script>"
    )

    monkeypatch.setattr(runner, "discover_competition_season", lambda **_: discovery)

    async def fake_fetch(requests, **_):
        return [
            MatchFetchResult(
                match_id=request.match_id,
                match_url=request.match_url,
                html=page_html,
                status_code=200,
                elapsed_ms=10,
                attempts=1,
                error=None,
            )
            for request in requests
        ]

    monkeypatch.setattr(runner, "fetch_match_pages", fake_fetch)
    raw_root = tmp_path / "raw"
    normalized_root = tmp_path / "normalized"
    first = runner.ingest_whoscored(
        competition="EPL",
        season="2025-2026",
        raw_root=raw_root,
        normalized_root=normalized_root,
    )
    second = runner.ingest_whoscored(
        competition="EPL",
        season="2025-2026",
        raw_root=raw_root,
        normalized_root=normalized_root,
    )
    assert first.processed == 1
    assert second.processed == 0
    assert second.skipped == 1
    assert (
        normalized_root
        / "competition=EPL/season=2025-2026/matches/match_id=1903444/_SUCCESS.json"
    ).exists()


def test_ingestion_selects_newest_unprocessed_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ingestion.whoscored import runner

    competition = Competition(
        key="EPL",
        name="Premier League",
        aliases=("epl",),
        region_id=252,
        tournament_id=2,
        slug="england-premier-league",
        expected_matches=380,
        stage_overrides={},
    )
    match_ids = [1903444, 1903445, 1903446, 1903447]
    fixture_dates = {
        1903444: "2026-04-10",
        1903445: "2026-04-11",
        1903446: "2026-04-12",
        1903447: "2999-01-01",
    }
    discovery = ScopeDiscoveryResult(
        competition=competition,
        requested_season="2025-2026",
        canonical_season="2025-2026",
        season_id=10743,
        stage_id=24533,
        fixture_url="https://www.whoscored.com/fixtures/test",
        matches=[
            {
                "match_id": match_id,
                "match_url": f"https://www.whoscored.com/Matches/{match_id}/live/test",
                "fixture_date": fixture_dates[match_id],
            }
            for match_id in match_ids
        ],
        pages_collected=1,
        warnings=(),
    )
    monkeypatch.setattr(runner, "discover_competition_season", lambda **_: discovery)
    fetched_ids: list[int] = []

    async def fake_fetch(requests, **kwargs):
        results = []
        for request in requests:
            fetched_ids.append(request.match_id)
            payload = {**_payload(), "startDate": f"2026-04-{request.match_id % 20 + 1:02d}T15:00:00"}
            html = (
                '<script>require.config.params["args"] = {'
                f'matchId: {request.match_id}, matchCentreData: {json.dumps(payload)}'
                "};</script>"
            )
            result = MatchFetchResult(
                match_id=request.match_id,
                match_url=request.match_url,
                html=html,
                status_code=200,
                elapsed_ms=10,
                attempts=1,
                error=None,
            )
            results.append(result)
            if callback := kwargs.get("on_result"):
                callback(result)
        return results

    monkeypatch.setattr(runner, "fetch_match_pages", fake_fetch)
    raw_root = tmp_path / "raw"
    normalized_root = tmp_path / "normalized"

    first = runner.ingest_whoscored(
        competition="EPL",
        season="2025-2026",
        raw_root=raw_root,
        normalized_root=normalized_root,
        max_new_matches=1,
    )
    second = runner.ingest_whoscored(
        competition="EPL",
        season="2025-2026",
        raw_root=raw_root,
        normalized_root=normalized_root,
        max_new_matches=1,
    )

    assert fetched_ids == [1903446, 1903445]
    assert first.requested == first.processed == 1
    assert second.requested == second.processed == 1
    assert second.skipped == 1
    assert first.deferred == second.deferred == 1
