from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.transfermarkt.normalize import (
    TransfermarktNormalizationError,
    normalize_name,
    normalize_transfermarkt_run,
)


def test_normalize_name_removes_accents_and_punctuation() -> None:
    assert normalize_name("Radu Drăgușin") == "radu dragusin"


def test_normalize_name_collapses_whitespace() -> None:
    assert normalize_name("  Micky   van-de Ven ") == "micky van de ven"


def _raw_run(root: Path, *, status: str = "succeeded") -> Path:
    run = root / "run_date=2026-07-13" / "run_id=test" / "competition=GB1" / "season=2025"
    (run / "valuations").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test",
                "status": status,
                "started_at": "2026-07-13T12:00:00+00:00",
                "scope": {"competition_id": "GB1", "season": 2025},
            }
        ),
        encoding="utf-8",
    )
    (run / "players.jsonl").write_text(
        json.dumps(
            {
                "player_id": 10,
                "player_name": "Player A",
                "position": "Centre-Forward",
                "club_id": 1,
                "club_name": "Club A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "valuations" / "10.json").write_text(
        json.dumps(
            {
                "list": [
                    {
                        "datum_mw": "01/07/2025",
                        "y": 10000000,
                        "mw": "€10.00m",
                        "verein": "Club A",
                        "x": 1751328000000,
                        "age": 24,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return run


def test_normalization_is_idempotent_for_an_unchanged_raw_run(tmp_path: Path) -> None:
    run = _raw_run(tmp_path / "raw")
    output = tmp_path / "normalized"
    first = normalize_transfermarkt_run(run_directory=run, output_root=output)
    second = normalize_transfermarkt_run(run_directory=run, output_root=output)

    assert first.output_directory == second.output_directory
    assert first.player_count == second.player_count == 1
    assert first.valuation_count == second.valuation_count == 1


def test_normalization_rejects_a_running_raw_manifest(tmp_path: Path) -> None:
    run = _raw_run(tmp_path / "raw", status="running")
    with pytest.raises(TransfermarktNormalizationError, match="not complete"):
        normalize_transfermarkt_run(
            run_directory=run, output_root=tmp_path / "normalized"
        )
