import json
from pathlib import Path

import pytest

from ingestion.common import sha256_file, validate_jsonl
from ingestion.progress import ProgressEmitter, ProgressUpdate


def test_validate_jsonl_counts_objects(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    assert validate_jsonl(path) == 2


def test_validate_jsonl_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        validate_jsonl(path)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"id": 1}), encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)


def test_progress_emitter_persists_and_forwards_events(tmp_path: Path) -> None:
    received: list[ProgressUpdate] = []
    path = tmp_path / "progress.jsonl"
    emitter = ProgressEmitter(run_id="run-1", log_path=path, callback=received.append)
    update = ProgressUpdate(
        stage="fetch",
        state="advanced",
        description="Fetching matches",
        completed=2,
        total=8,
        current="match_id=123",
        succeeded=2,
    )

    emitter.emit(update)
    row = json.loads(path.read_text(encoding="utf-8"))

    assert received == [update]
    assert row["run_id"] == "run-1"
    assert row["stage"] == "fetch"
    assert row["completed"] == 2
    assert row["total"] == 8
