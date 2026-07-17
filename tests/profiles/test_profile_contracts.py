from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from profiles.builder import (
    ROLE_METRICS,
    _add_percentiles_and_zscores,
    _build_profile_rows,
    _build_roles,
    _build_similarity_rows,
    _percentile,
    role_expression,
)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("GK", "GOALKEEPER"),
        ("DC", "CENTRE_BACK"),
        ("DL", "FULL_BACK"),
        ("DR", "FULL_BACK"),
        ("DMC", "DEFENSIVE_MIDFIELDER"),
        ("MC", "CENTRAL_MIDFIELDER"),
        ("AMC", "ATTACKING_MIDFIELDER"),
        ("AML", "WINGER"),
        ("AMR", "WINGER"),
        ("FW", "STRIKER"),
    ],
)
def test_detailed_position_mapping(
    position: str,
    expected: str,
) -> None:
    frame = pl.DataFrame({"position": [position]}).with_columns(
        role_expression()
    )

    assert frame.item(0, "role_group") == expected


def test_primary_role_uses_minutes_not_appearance_count() -> None:
    ratings = pl.DataFrame(
        {
            "whoscored_player_id": [1, 1, 1],
            "position": ["AMC", "AMC", "FW"],
            "minutes": [20.0, 20.0, 90.0],
        }
    )

    roles = _build_roles(ratings)

    assert roles.item(0, "primary_role") == "STRIKER"
    assert roles.item(0, "secondary_role") == (
        "ATTACKING_MIDFIELDER"
    )
    assert roles.item(0, "primary_role_share") == pytest.approx(
        90 / 130
    )


def test_percentile_midrank_handles_ties() -> None:
    values = pd.Series([1.0, 2.0, 2.0, 3.0])
    percentiles = _percentile(values, values)

    assert np.allclose(percentiles, [12.5, 50.0, 50.0, 87.5])


def test_lower_is_better_metric_reverses_percentile() -> None:
    players = pd.DataFrame(
        {
            "whoscored_player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "primary_role": ["STRIKER"] * 3,
            "minutes": [1000.0] * 3,
            "turnovers_90": [1.0, 2.0, 3.0],
        }
    )

    result = _add_percentiles_and_zscores(
        players,
        benchmark_minutes=900,
    )

    assert (
        result.loc[0, "turnovers_90_percentile"]
        > result.loc[2, "turnovers_90_percentile"]
    )


def test_profile_rows_have_unique_metric_keys() -> None:
    player = {
        "whoscored_player_id": 1,
        "player_name": "Example",
        "primary_role": "STRIKER",
        "secondary_role": None,
        "primary_role_share": 1.0,
        "is_hybrid_role": False,
        "minutes": 1000.0,
        "appearances": 20,
        "sample_status": "benchmark",
    }
    for spec in ROLE_METRICS["STRIKER"]:
        player[spec.key] = 1.0
        player[f"{spec.key}_percentile"] = 50.0

    frame = _build_profile_rows(
        pd.DataFrame([player]),
        competition="EPL",
        season="2025-2026",
        benchmark_minutes=900,
    )

    assert frame.height == len(ROLE_METRICS["STRIKER"])
    assert (
        frame.select(
            pl.struct(
                [
                    "player_id",
                    "competition",
                    "season",
                    "metric_key",
                ]
            ).n_unique()
        ).item()
        == frame.height
    )
    assert frame["percentile"].is_between(0, 100).all()


def test_similarity_excludes_selected_player() -> None:
    metric_key = ROLE_METRICS["STRIKER"][0].key
    players = pd.DataFrame(
        {
            "whoscored_player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "primary_role": ["STRIKER"] * 3,
            "secondary_role": [None] * 3,
            "minutes": [1000.0] * 3,
            "appearances": [20] * 3,
            f"{metric_key}_zscore": [0.0, 0.1, 2.0],
        }
    )

    result = _build_similarity_rows(
        players,
        competition="EPL",
        season="2025-2026",
        benchmark_minutes=900,
        top_n=10,
    )

    assert result.height > 0
    assert not (
        result["player_id"] == result["similar_player_id"]
    ).any()
    assert result["similarity"].is_between(0, 100).all()
