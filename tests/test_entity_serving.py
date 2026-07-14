from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from api.repository import ParquetRepository
from entity_resolution.player_mapping import build_player_mapping
from serving.builder import build_serving_tables
from pipelines.replay import _publish_replay_impacts


def _write_whoscored_players(root: Path) -> None:
    partition = (
        root
        / "competition=EPL"
        / "season=2025-2026"
        / "matches"
        / "match_id=1"
    )
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            "player_id": [10, 20],
            "player_name": ["Player Exact", "Shared Name"],
            "team_id": [1, 2],
            "position": ["FW", "DC"],
        }
    ).write_parquet(partition / "player_matches.parquet")


def test_mapping_review_serving_and_parquet_api(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    _write_whoscored_players(normalized)
    transfermarkt_players = tmp_path / "players.parquet"
    pl.DataFrame(
        {
            "transfermarkt_player_id": [100, 200, 201],
            "player_name": ["Player Exact", "Shared Name", "Shared Name"],
            "normalized_player_name": ["player exact", "shared name", "shared name"],
            "date_of_birth": [date(2000, 1, 1), date(1999, 1, 1), date(1998, 1, 1)],
            "club_name": ["A", "B", "C"],
            "position": ["Forward", "Defender", "Defender"],
        }
    ).write_parquet(transfermarkt_players)
    mapping_path = tmp_path / "mapping" / "player_mapping_exact.parquet"
    mapping = build_player_mapping(
        transfermarkt_players_path=transfermarkt_players,
        whoscored_normalized_root=normalized,
        competition="EPL",
        season="2025-2026",
        output_path=mapping_path,
    )

    assert mapping.mapped_players == 1
    assert mapping.review_players == 1

    ratings_path = tmp_path / "player_match_ratings.parquet"
    pl.DataFrame(
        {
            "season": ["2025-2026", "2025-2026"],
            "match_id": [1, 2],
            "match_datetime": [datetime(2026, 4, 1, 15), datetime(2026, 4, 8, 15)],
            "whoscored_player_id": [10, 10],
            "player_name": ["Player Exact", "Player Exact"],
            "team_id": [1, 1],
            "position_group": ["Forward", "Forward"],
            "minutes": [90.0, 90.0],
            "post_match_rating": [7.2, 7.8],
            "rating_version": ["post_match_v1", "post_match_v1"],
            "form_rating_ewm": [7.2, 7.5],
            "rolling_3_match_rating": [7.2, 7.5],
            "rolling_20_match_rating": [7.2, 7.5],
            "threat_component": [1.1, 1.4],
            "creation_component": [0.4, 0.6],
        }
    ).write_parquet(ratings_path)
    valuations_path = tmp_path / "player_valuations.parquet"
    pl.DataFrame(
        {
            "transfermarkt_player_id": [100, 100],
            "valuation_date": [date(2025, 8, 1), date(2026, 4, 1)],
            "market_value_eur": [10_000_000, 12_000_000],
            "is_valid_for_model": [True, True],
        }
    ).write_parquet(valuations_path)
    serving_root = tmp_path / "serving"
    serving = build_serving_tables(
        ratings_path=ratings_path,
        valuations_path=valuations_path,
        mapping_path=mapping_path,
        output_root=serving_root,
    )

    assert serving.players == 1
    repository = ParquetRepository(serving_root)
    player = repository.player("10")
    assert player is not None
    assert player["display_name"] == "Player Exact"
    assert len(player["valuation_history"]) == 2
    assert player["match_impacts"][0]["match_id"] == 2
    assert player["match_impacts"][0]["estimated_value_delta_eur"] is None

    published = _publish_replay_impacts(
        results=[
            {
                "match_id": 2,
                "replay_sequence": 2,
                "valuation_update_status": "succeeded",
                "estimated_player_value_eur": 12_500_000.0,
                "estimated_player_lower_90_eur": 11_500_000.0,
                "estimated_player_upper_90_eur": 13_500_000.0,
                "estimated_player_delta_eur": 500_000.0,
                "probability_value_increase": 0.76,
            }
        ],
        run_id="replay-test",
        player_id=10,
        serving_root=serving_root,
    )
    assert published == 1
    refreshed = repository.player("10")
    assert refreshed is not None
    assert refreshed["match_impacts"][0]["estimated_value_delta_eur"] == 500_000.0
