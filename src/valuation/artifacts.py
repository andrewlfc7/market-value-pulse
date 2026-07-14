from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingestion.common import write_json


def create_model_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_version_directory(root: Path, model_version: str | None = None) -> tuple[str, Path]:
    version = model_version or create_model_version()
    directory = root / f"model_version={version}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plots").mkdir()
    return version, directory


def promote_model(*, model_root: Path, model_version: str, artifact_directory: Path) -> Path:
    active_path = model_root / "active.json"
    write_json(
        active_path,
        {
            "model_version": model_version,
            "artifact_directory": str(artifact_directory),
            "selected_model": "position_hierarchical_bayesian_student_t",
            "promoted_at": datetime.now(UTC).isoformat(),
        },
    )
    return active_path


def resolve_model_directory(model_root: Path, version: str) -> Path:
    if version == "latest":
        candidates = sorted(
            path for path in model_root.glob("model_version=*") if path.is_dir()
        )
        if not candidates:
            raise FileNotFoundError(f"No model candidates found under {model_root}")
        return candidates[-1]
    if version != "active":
        path = model_root / f"model_version={version}"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    active_path = model_root / "active.json"
    if not active_path.exists():
        raise FileNotFoundError(
            f"No active model pointer found at {active_path}. The latest candidate "
            "may have failed promotion checks; inspect it with "
            "`mvp model summary --model-version latest`."
        )
    payload = json.loads(active_path.read_text(encoding="utf-8"))
    path = Path(payload["artifact_directory"])
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path
