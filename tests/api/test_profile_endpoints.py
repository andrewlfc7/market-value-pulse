from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from api.profile_store import ProfileStore


def _write_profile_artifacts(root: Path) -> None:
    pl.DataFrame(
        {
            "player_id": [1, 1],
            "whoscored_player_id": [1, 1],
            "player_name": ["Example", "Example"],
            "competition": ["EPL", "EPL"],
            "season": ["2025-2026", "2025-2026"],
            "primary_role": ["STRIKER", "STRIKER"],
            "secondary_role": [None, None],
            "primary_role_share": [1.0, 1.0],
            "is_hybrid_role": [False, False],
            "minutes": [1000.0, 1000.0],
            "appearances": [20, 20],
            "sample_status": ["benchmark", "benchmark"],
            "phase": ["Scoring", "Creation"],
            "metric_key": ["non_penalty_xg_90", "xa_90"],
            "metric_label": ["Non-penalty xG", "Expected assists"],
            "metric_value": [0.5, 0.2],
            "percentile": [80.0, 60.0],
            "higher_is_better": [True, True],
            "display_order": [1, 2],
            "benchmark_minutes": [900, 900],
        }
    ).write_parquet(root / "player_profiles.parquet")

    pl.DataFrame(
        {
            "player_id": [1, 1, 1],
            "whoscored_player_id": [1, 1, 1],
            "player_name": ["Example"] * 3,
            "similar_player_id": [2, 3, 4],
            "similar_player_name": ["B", "C", "D"],
            "competition": ["EPL"] * 3,
            "season": ["2025-2026"] * 3,
            "primary_role": ["STRIKER"] * 3,
            "secondary_role": [None] * 3,
            "minutes": [1000.0] * 3,
            "appearances": [20] * 3,
            "metrics_used": [20] * 3,
            "similarity": [90.0, 80.0, 70.0],
            "profile_similarity": [90.0, 80.0, 70.0],
            "rank": [1, 2, 3],
        }
    ).write_parquet(root / "player_similarities.parquet")


def test_profile_endpoint_and_similarity_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_profile_artifacts(tmp_path)
    app_module = importlib.import_module("api.app")
    monkeypatch.setattr(
        app_module,
        "profile_store",
        ProfileStore(tmp_path),
    )
    client = TestClient(app_module.app)

    profile_response = client.get("/api/players/1/profile")
    assert profile_response.status_code == 200
    assert profile_response.json()["data"]["primary_role"] == (
        "STRIKER"
    )
    assert len(profile_response.json()["data"]["metrics"]) == 2

    similarity_response = client.get(
        "/api/players/1/similar-players?limit=2"
    )
    assert similarity_response.status_code == 200
    assert similarity_response.json()["count"] == 2


def test_missing_profile_returns_404(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_module = importlib.import_module("api.app")
    monkeypatch.setattr(
        app_module,
        "profile_store",
        ProfileStore(tmp_path),
    )
    client = TestClient(app_module.app)

    response = client.get("/api/players/999/profile")

    assert response.status_code == 404
    assert response.json()["detail"] == "Player profile not found"
