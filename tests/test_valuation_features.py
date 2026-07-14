from datetime import date, datetime

import polars as pl

from valuation.features import (
    ValuationFeatureConfig,
    aggregate_interval_performance,
    build_valuation_intervals,
    clean_crosswalk,
    clean_valuations,
    finalize_model_table,
    prepare_ratings,
    read_rating_history,
)


def test_rolling_form_uses_only_matches_before_valuation() -> None:
    valuations_raw = pl.DataFrame(
        {
            "transfermarkt_player_id": [1, 1, 1],
            "valuation_date": [date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1)],
            "market_value_eur": [1_000_000, 1_200_000, 1_500_000],
            "age_at_valuation": [22, 22, 23],
            "club_name": ["A", "A", "A"],
            "is_valid_for_model": [True, True, True],
            "is_terminal_record": [False, False, False],
            "is_future_dated": [False, False, False],
        }
    )
    mapping_raw = pl.DataFrame(
        {"transfermarkt_player_id": [1], "whoscored_player_id": [10]}
    )
    ratings_raw = pl.DataFrame(
        {
            "season": ["2024-2025"] * 5,
            "match_id": [1, 2, 3, 4, 5],
            "match_datetime": [
                datetime(2025, 1, 15),
                datetime(2025, 2, 15),
                datetime(2025, 3, 15),
                datetime(2025, 4, 15),
                datetime(2025, 7, 2),  # after the second modeled valuation
            ],
            "whoscored_player_id": [10] * 5,
            "player_name": ["Player"] * 5,
            "position_group": ["forward"] * 5,
            "minutes": [90.0] * 5,
            "started": [1] * 5,
            "post_match_rating": [5.0, 6.0, 7.0, 8.0, 10.0],
            "goals": [0, 0, 1, 1, 3],
            "assists": [0] * 5,
            "xg": [0.1] * 5,
            "xgot": [0.1] * 5,
            "xa": [0.0] * 5,
        }
    )

    config = ValuationFeatureConfig(minimum_interval_minutes=90.0)
    valuations = clean_valuations(valuations_raw)
    mapping = clean_crosswalk(mapping_raw)
    ratings = prepare_ratings(ratings_raw)
    intervals = build_valuation_intervals(
        valuations,
        mapping,
        ratings,
        config=config,
    )
    interval_performance, rolling = aggregate_interval_performance(
        intervals,
        ratings,
        config=config,
    )
    output = finalize_model_table(
        intervals,
        interval_performance,
        rolling,
        config=config,
    ).sort("valuation_date")

    first = output.row(0, named=True)
    second = output.row(1, named=True)
    assert first["rolling_3_match_rating"] == 6.0
    assert second["rolling_3_match_rating"] == 7.0
    assert second["rolling_20_match_rating"] == 6.5
    assert second["rolling_3_match_rating"] < 10.0
    assert first["recency_weighted_rating"] > first["average_rating"]
    assert second["rating_last_90_days"] == 8.0
    assert second["recent_rating_trend"] == 0.0


def test_rating_history_reads_multiple_season_partitions(tmp_path) -> None:
    competition = tmp_path / "competition=EPL"
    for season, match_id in (("2024-2025", 1), ("2025-2026", 2)):
        directory = competition / f"season={season}"
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "season": [season],
                "match_id": [match_id],
                "whoscored_player_id": [10],
            }
        ).write_parquet(directory / "player_match_ratings.parquet")

    history = read_rating_history(competition).sort("match_id")
    assert history.height == 2
    assert history["season"].to_list() == ["2024-2025", "2025-2026"]
