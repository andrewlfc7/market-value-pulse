from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal


ProgressState = Literal["started", "advanced", "completed", "failed", "info"]


@dataclass(frozen=True)
class ProgressUpdate:
    """A source-agnostic progress event emitted by long-running workflows."""

    stage: str
    state: ProgressState
    description: str
    completed: int = 0
    total: int | None = None
    current: str | None = None
    succeeded: int = 0
    skipped: int = 0
    deferred: int = 0
    failed: int = 0


ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressEmitter:
    """Forward progress to the terminal and persist the same events as JSONL."""

    def __init__(
        self,
        *,
        run_id: str,
        log_path: Path,
        callback: ProgressCallback | None = None,
    ) -> None:
        self._run_id = run_id
        self._log_path = log_path
        self._callback = callback
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, update: ProgressUpdate) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self._run_id,
            **asdict(update),
        }
        with self._log_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, sort_keys=True))
            destination.write("\n")
        if self._callback is not None:
            self._callback(update)


class TerminalProgress:
    """Rich terminal renderer with per-stage elapsed time, rate, and ETA."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._progress = None
        self._task_ids: dict[str, int] = {}

    def __enter__(self) -> "TerminalProgress":
        if not self._enabled:
            return self
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("elapsed"),
            TextColumn("• ETA"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}", justify="left"),
            refresh_per_second=8,
        )
        self._progress.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def __call__(self, update: ProgressUpdate) -> None:
        if not self._enabled:
            return
        if self._progress is None:
            raise RuntimeError("TerminalProgress must be used as a context manager")

        detail = self._detail(update)
        task_id = self._task_ids.get(update.stage)
        if task_id is None:
            task_id = self._progress.add_task(
                update.description,
                total=update.total,
                completed=update.completed,
                detail=detail,
            )
            self._task_ids[update.stage] = task_id

        fields: dict[str, object] = {
            "description": update.description,
            "completed": update.completed,
            "detail": detail,
        }
        if update.total is not None:
            fields["total"] = update.total
        if update.state in {"completed", "failed"}:
            fields["completed"] = update.total or update.completed
        self._progress.update(task_id, **fields)
        if update.state in {"completed", "failed"}:
            self._progress.stop_task(task_id)

    @staticmethod
    def _detail(update: ProgressUpdate) -> str:
        counters = []
        if update.succeeded:
            counters.append(f"ok={update.succeeded}")
        if update.skipped:
            counters.append(f"skip={update.skipped}")
        if update.deferred:
            counters.append(f"deferred={update.deferred}")
        if update.failed:
            counters.append(f"failed={update.failed}")
        if update.current:
            counters.append(update.current)
        return (" • " + " ".join(counters)) if counters else ""
