from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from features.pipeline import _pass_features
from ingestion.whoscored.appearances import _position_group
from ratings.model import (
    RatingModelConfig,
    _decisive_action_bonus,
    _fit_pass_priors,
    _outfield_base_rating,
)
from ratings.model import fit_rating_artifacts, score_rating_features
from ratings.pipeline import fit_and_score_rating_season, update_rating_season


def _features(match_id: int, day: int, multiplier: float = 1.0) -> pl.DataFrame:
    positions = ["Forward", "Midfielder", "Defender"]
    rows = []
    for offset, position in enumerate(positions, start=1):
        rows.append(
            {
                "season": "2025-2026",
                "match_id": match_id,
                "match_datetime": datetime(2026, 4, day, 15),
                "whoscored_player_id": offset,
                "player_id": offset,
                "player_name": f"Player {offset}",
                "team_id": 10,
                "position_group": position,
                "minutes": 90.0 if offset != 2 else 45.0,
                "started": True,
                "shots": offset * multiplier,
                "goals": 1.0 if offset == 1 and multiplier > 1 else 0.0,
                "xg": 0.25 * offset * multiplier,
                "xgot": 0.20 * offset * multiplier,
                "xa": 0.10 * offset * multiplier,
                "key_passes": offset * multiplier,
                "big_chances_created": 1.0 if offset == 2 else 0.0,
                "progressive_passes": 2.0 * offset * multiplier,
                "progressive_carries": offset * multiplier,
                "final_third_carries": 1.0 * multiplier,
                "passes": 20.0 * offset,
                "completed_passes": 16.0 * offset,
                "assists": 1.0 if offset == 2 and multiplier > 1 else 0.0,
                "xpv_added": 0.05 * offset * multiplier,
                "opponent_threat_prevented": 0.03 * offset,
                "defensive_net_threat_reduction": 0.02 * offset,
                "yellow_cards": 1.0 if offset == 3 and multiplier > 1 else 0.0,
                "red_cards": 0.0,
                "big_chances_missed": 1.0 if offset == 1 and multiplier < 1 else 0.0,
                "big_chance_xg_missed": 0.40 if offset == 1 and multiplier < 1 else 0.0,
                "own_goals": 0.0,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _write_partition(root: Path, match_id: int, day: int, multiplier: float) -> None:
    partition = (
        root
        / "competition=EPL"
        / "season=2025-2026"
        / "matches"
        / f"match_id={match_id}"
    )
    partition.mkdir(parents=True)
    _features(match_id, day, multiplier).write_parquet(
        partition / "player_match_features.parquet"
    )
    (partition / "_SUCCESS.json").write_text(
        json.dumps({"status": "succeeded", "match_id": match_id}), encoding="utf-8"
    )


def test_native_rating_formula_saves_versioned_artifacts(tmp_path: Path) -> None:
    frame = pl.concat([_features(1, 1, 0.7), _features(2, 8, 1.4)])
    artifact = tmp_path / "models" / "ratings" / "post_match_v2"
    fit_rating_artifacts(frame, artifact)
    scored = score_rating_features(frame, artifact)

    assert (artifact / "rating_model_config.json").exists()
    assert (artifact / "feature_schema.json").exists()
    assert (artifact / "zscore_stats.parquet").exists()
    assert (artifact / "pass_completion_priors.parquet").exists()
    assert scored.height == frame.height
    assert scored["post_match_rating"].is_between(1.0, 10.0).all()
    assert set(scored["rating_version"].to_list()) == {"post_match_v2"}
    assert "threat_component" in scored.columns
    assert "decisive_action_bonus" in scored.columns


def test_v2_rating_curve_reserves_extreme_scores_for_exceptional_matches() -> None:
    config = RatingModelConfig()
    assert _outfield_base_rating(0.0, config) == 6.0
    assert 8.0 < _outfield_base_rating(2.5, config) < 8.2
    assert _outfield_base_rating(5.0, config) < 9.1
    one_assist = _decisive_action_bonus(
        {"goals": 0.0, "assists": 1.0}, 0.7, config
    )
    assert one_assist < 0.10


def test_rating_input_semantics_match_historical_v3_build() -> None:
    assert _position_group("AMC") == "Forward"
    assert _position_group("AML") == "Forward"
    assert _position_group("DMC") == "Midfielder"
    assert _position_group("WBR") == "Defender"

    passes = pl.DataFrame(
        {
            "player_id": [1, 1, 1],
            "x": [20.0, 20.0, 20.0],
            "end_x": [50.0, 50.0, 50.0],
            "success": [1, 0, 1],
            "is_key_pass": [0, 0, 1],
            "is_assist": [0, 0, 1],
            "is_corner": [0, 0, 1],
            "is_free_kick": [0, 0, 0],
            "is_throw_in": [0, 0, 0],
            "is_goal_kick": [0, 0, 0],
        }
    )
    result = _pass_features(passes).row(0, named=True)
    assert result["passes"] == 2
    assert result["completed_passes"] == 1
    assert result["progressive_passes"] == 1
    # Assists remain match events even when the pass is a set piece.
    assert result["assists"] == 1

    priors = _fit_pass_priors(
        [
            {
                "season": "2025-2026",
                "position_group": "Forward",
                "passes": 2,
                "completed_passes": 0,
            },
            {
                "season": "2025-2026",
                "position_group": "Forward",
                "passes": 10,
                "completed_passes": 8,
            },
        ]
    ).row(0, named=True)
    assert priors["match_rows"] == 1
    assert priors["pass_completion_prior"] == 0.8


def test_incremental_rating_scores_only_new_feature_partitions(tmp_path: Path) -> None:
    features_root = tmp_path / "features"
    output_root = tmp_path / "ratings"
    state_root = tmp_path / "state"
    artifact = tmp_path / "model"
    _write_partition(features_root, 1, 1, 0.7)
    _write_partition(features_root, 2, 8, 1.4)

    first = fit_and_score_rating_season(
        competition="EPL",
        season="2025-2026",
        features_root=features_root,
        output_root=output_root,
        state_root=state_root,
        artifact_directory=artifact,
    )
    _write_partition(features_root, 3, 15, 1.1)
    second = update_rating_season(
        competition="EPL",
        season="2025-2026",
        features_root=features_root,
        output_root=output_root,
        state_root=state_root,
        artifact_directory=artifact,
    )
    third = update_rating_season(
        competition="EPL",
        season="2025-2026",
        features_root=features_root,
        output_root=output_root,
        state_root=state_root,
        artifact_directory=artifact,
    )

    assert first.processed_matches == 2
    assert (artifact / "season_primary_positions.parquet").exists()
    assert (artifact / "career_primary_positions.parquet").exists()
    assert second.processed_matches == 1
    assert second.skipped_matches == 2
    assert third.processed_matches == 0
    assert third.skipped_matches == 3
    ratings = pl.read_parquet(second.output_path)
    assert ratings.height == 9
    assert ratings["form_rating_ewm"].null_count() == 0
    assert second.form_state_path.exists()
    assert second.processed_matches_path.exists()


def test_changed_historical_partition_refits_season_statistics(
    tmp_path: Path,
) -> None:
    features_root = tmp_path / "features"
    output_root = tmp_path / "ratings"
    state_root = tmp_path / "state"
    artifact = tmp_path / "model"
    _write_partition(features_root, 1, 1, 0.7)
    _write_partition(features_root, 2, 8, 1.4)
    fit_and_score_rating_season(
        competition="EPL",
        season="2025-2026",
        features_root=features_root,
        output_root=output_root,
        state_root=state_root,
        artifact_directory=artifact,
    )

    partition = (
        features_root
        / "competition=EPL"
        / "season=2025-2026"
        / "matches"
        / "match_id=1"
    )
    _features(1, 1, 2.0).write_parquet(
        partition / "player_match_features.parquet"
    )
    result = update_rating_season(
        competition="EPL",
        season="2025-2026",
        features_root=features_root,
        output_root=output_root,
        state_root=state_root,
        artifact_directory=artifact,
    )

    assert result.mode == "fit_full_history"
    assert result.processed_matches == 2
