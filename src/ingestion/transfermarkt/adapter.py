from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    stdout_path: Path
    stderr_path: Path
    return_code: int


class TransfermarktCommandError(RuntimeError):
    """Raised when a transfermarkt-scraper process exits unsuccessfully."""


def build_tfmkt_command(
    *,
    crawler: str,
    parent_file: Path,
    season: int,
    base_url: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tfmkt",
        crawler,
        "--parents",
        str(parent_file),
        "--season",
        str(season),
    ]
    if base_url:
        command.extend(["--base-url", base_url])
    return command


def run_command(
    *,
    command: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    """Run a source connector command and capture its raw output."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        stdout_path.open("wb") as stdout_file,
        stderr_path.open("wb") as stderr_file,
    ):
        completed = subprocess.run(
            list(command),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )

    result = CommandResult(
        command=list(command),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        return_code=completed.returncode,
    )

    if result.return_code != 0:
        raise TransfermarktCommandError(
            f"Command exited with status {result.return_code}: "
            f"{' '.join(result.command)}. See {stderr_path}."
        )

    return result
