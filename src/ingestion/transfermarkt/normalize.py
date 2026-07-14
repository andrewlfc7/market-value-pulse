from __future__ import annotations

import json
import os
import re
import shutil
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from ingestion.common import sha256_file, write_json


class TransfermarktNormalizationError(RuntimeError):
    """Raised when a completed raw run cannot be normalized."""


@dataclass(frozen=True)
class TransfermarktNormalizationResult:
    output_directory: Path
    summary_path: Path
    player_count: int
    raw_valuation_count: int
    valuation_count: int
    model_valid_count: int
    issue_count: int


PLAYER_SCHEMA = {
    "transfermarkt_player_id": pl.Int64,
    "player_name": pl.String,
    "normalized_player_name": pl.String,
    "position": pl.String,
    "date_of_birth": pl.Date,
    "nationalities": pl.List(pl.String),
    "roster_status": pl.String,
    "source_section": pl.String,
    "destination_club_name": pl.String,
    "club_id": pl.Int64,
    "club_name": pl.String,
    "season_start_year": pl.Int64,
    "profile_path": pl.String,
    "competition_id": pl.String,
    "source_run_id": pl.String,
    "source_snapshot_date": pl.Date,
    "source_file": pl.String,
}

VALUATION_SCHEMA = {
    "transfermarkt_player_id": pl.Int64,
    "valuation_date": pl.Date,
    "market_value_eur": pl.Int64,
    "source_market_value_eur": pl.Int64,
    "market_value_display": pl.String,
    "club_name": pl.String,
    "age_at_valuation": pl.Int64,
    "source_timestamp_ms": pl.Int64,
    "badge_url": pl.String,
    "is_terminal_record": pl.Boolean,
    "is_future_dated": pl.Boolean,
    "is_valid_for_model": pl.Boolean,
    "competition_id": pl.String,
    "season_start_year": pl.Int64,
    "source_run_id": pl.String,
    "source_snapshot_date": pl.Date,
    "source_file": pl.String,
    "source_list_index": pl.Int64,
}

ISSUE_SCHEMA = {
    "issue_type": pl.String,
    "severity": pl.String,
    "transfermarkt_player_id": pl.Int64,
    "record_key": pl.String,
    "source_file": pl.String,
    "details": pl.String,
}


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    cleaned = re.sub(r"[^a-z0-9]+", " ", without_marks.casefold())
    return " ".join(cleaned.split())


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                record = json.loads(value)
            except json.JSONDecodeError as exc:
                raise TransfermarktNormalizationError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TransfermarktNormalizationError(
                    f"{path} line {line_number} is not a JSON object."
                )
            records.append(record)
    return records



def _parse_source_date(value: object) -> date | None:
    """Parse Transfermarkt DD/MM/YYYY date values."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return datetime.strptime(value.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def _snapshot_date(manifest: dict[str, Any], run_directory: Path) -> date:
    started_at = manifest.get("started_at")
    if isinstance(started_at, str):
        try:
            return datetime.fromisoformat(started_at.replace("Z", "+00:00")).date()
        except ValueError:
            pass

    for part in run_directory.parts:
        if part.startswith("run_date="):
            try:
                return date.fromisoformat(part.split("=", 1)[1])
            except ValueError:
                break

    raise TransfermarktNormalizationError("Could not determine source snapshot date.")


def _frame(records: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records, schema_overrides=schema)


def _write_parquet(dataframe: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.write_parquet(temporary_path, compression="zstd")
    os.replace(temporary_path, path)


def _raw_signature(
    manifest_path: Path, players_path: Path, valuations_directory: Path
) -> str:
    # Keep identity independent of the repository's absolute path so a raw
    # snapshot can be copied into a new checkout without changing its content
    # signature.
    objects = [
        ("manifest.json", manifest_path),
        (players_path.name, players_path),
        *[
            (f"valuations/{path.name}", path)
            for path in sorted(valuations_directory.glob("*.json"))
        ],
    ]
    payload = [(name, sha256_file(path)) for name, path in objects]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _existing_result(
    output_directory: Path, *, source_signature: str
) -> TransfermarktNormalizationResult | None:
    summary_path = output_directory / "normalization_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("status") != "succeeded" or summary.get(
        "source_signature"
    ) != source_signature:
        return None
    counts = summary.get("counts") or {}
    required = (
        output_directory / "players.parquet",
        output_directory / "player_valuations.parquet",
        output_directory / "data_quality_issues.parquet",
    )
    if not all(path.exists() for path in required):
        return None
    return TransfermarktNormalizationResult(
        output_directory=output_directory,
        summary_path=summary_path,
        player_count=int(counts.get("normalized_players", 0)),
        raw_valuation_count=int(counts.get("raw_valuation_records", 0)),
        valuation_count=int(counts.get("normalized_valuation_records", 0)),
        model_valid_count=int(counts.get("model_valid_valuation_records", 0)),
        issue_count=int(counts.get("data_quality_issues", 0)),
    )


def _promote_directory(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex[:8]}")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def normalize_transfermarkt_run(
    *,
    run_directory: Path,
    output_root: Path,
) -> TransfermarktNormalizationResult:
    manifest_path = run_directory / "manifest.json"
    reparsed_players_path = run_directory / "players.reparsed.jsonl"
    players_path = (
        reparsed_players_path
        if reparsed_players_path.exists()
        else run_directory / "players.jsonl"
    )
    valuations_directory = run_directory / "valuations"

    if not manifest_path.exists():
        raise TransfermarktNormalizationError(f"Missing manifest: {manifest_path}")
    if not players_path.exists():
        raise TransfermarktNormalizationError(f"Missing players file: {players_path}")
    if not valuations_directory.is_dir():
        raise TransfermarktNormalizationError(
            f"Missing valuations directory: {valuations_directory}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_status = str(manifest.get("status") or "")
    if manifest_status not in {"succeeded", "partial"}:
        raise TransfermarktNormalizationError(
            "Raw Transfermarkt run is not complete: "
            f"status={manifest_status or 'missing'} ({manifest_path})"
        )
    scope = manifest.get("scope") or {}
    competition_id = str(scope.get("competition_id") or "")
    season_start_year = _safe_int(scope.get("season"))
    source_run_id = str(manifest.get("run_id") or "")
    source_snapshot_date = _snapshot_date(manifest, run_directory)

    if not competition_id or season_start_year is None or not source_run_id:
        raise TransfermarktNormalizationError(
            "Manifest is missing competition_id, season, or run_id."
        )

    final_output_directory = (
        output_root
        / f"competition={competition_id}"
        / f"season={season_start_year}"
        / f"run_id={source_run_id}"
    )
    source_signature = _raw_signature(
        manifest_path, players_path, valuations_directory
    )
    existing = _existing_result(
        final_output_directory, source_signature=source_signature
    )
    if existing is not None:
        return existing
    output_directory = final_output_directory.with_name(
        f".{final_output_directory.name}.tmp-{uuid4().hex[:8]}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    issue_rows: list[dict[str, Any]] = []
    raw_player_records = _read_jsonl(players_path)
    player_rows: list[dict[str, Any]] = []
    player_ids: set[int] = set()

    for record in raw_player_records:
        player_id = _safe_int(record.get("player_id"))
        player_name = str(record.get("player_name") or "").strip()

        if player_id is None or not player_name:
            issue_rows.append(
                {
                    "issue_type": "invalid_player_record",
                    "severity": "error",
                    "transfermarkt_player_id": player_id,
                    "record_key": None,
                    "source_file": str(players_path),
                    "details": "Missing player_id or player_name.",
                }
            )
            continue

        if player_id in player_ids:
            issue_rows.append(
                {
                    "issue_type": "duplicate_player_id",
                    "severity": "warning",
                    "transfermarkt_player_id": player_id,
                    "record_key": str(player_id),
                    "source_file": str(players_path),
                    "details": "Duplicate player ID; first row retained.",
                }
            )
            continue

        player_ids.add(player_id)
        player_rows.append(
            {
                "transfermarkt_player_id": player_id,
                "player_name": player_name,
                "normalized_player_name": normalize_name(player_name),
                "position": str(record.get("position") or "").strip() or None,
                "date_of_birth": _parse_source_date(
                    record.get("date_of_birth")
                ),
                "nationalities": [
                    str(value).strip()
                    for value in (record.get("nationalities") or [])
                    if str(value).strip()
                ],
                "roster_status": (
                    str(record.get("roster_status") or "current_squad")
                    .strip()
                ),
                "source_section": (
                    str(record.get("source_section") or "squad_table")
                    .strip()
                ),
                "destination_club_name": (
                    str(record.get("destination_club_name") or "").strip()
                    or None
                ),
                "club_id": _safe_int(record.get("club_id")),
                "club_name": str(record.get("club_name") or "").strip() or None,
                "season_start_year": season_start_year,
                "profile_path": str(record.get("profile_path") or "").strip() or None,
                "competition_id": competition_id,
                "source_run_id": source_run_id,
                "source_snapshot_date": source_snapshot_date,
                "source_file": str(players_path),
            }
        )

    raw_valuation_count = 0
    valuation_file_ids: set[int] = set()
    valuation_rows_by_key: dict[tuple[int, date], dict[str, Any]] = {}

    for valuation_path in sorted(valuations_directory.glob("*.json")):
        player_id = _safe_int(valuation_path.stem)
        if player_id is None:
            issue_rows.append(
                {
                    "issue_type": "invalid_valuation_filename",
                    "severity": "error",
                    "transfermarkt_player_id": None,
                    "record_key": valuation_path.name,
                    "source_file": str(valuation_path),
                    "details": "Filename is not a numeric player ID.",
                }
            )
            continue

        valuation_file_ids.add(player_id)

        try:
            payload = json.loads(valuation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issue_rows.append(
                {
                    "issue_type": "invalid_valuation_json",
                    "severity": "error",
                    "transfermarkt_player_id": player_id,
                    "record_key": str(player_id),
                    "source_file": str(valuation_path),
                    "details": str(exc),
                }
            )
            continue

        history = payload.get("list")
        if not isinstance(history, list):
            issue_rows.append(
                {
                    "issue_type": "missing_valuation_list",
                    "severity": "error",
                    "transfermarkt_player_id": player_id,
                    "record_key": str(player_id),
                    "source_file": str(valuation_path),
                    "details": "Response list field is missing or invalid.",
                }
            )
            continue

        if not history:
            issue_rows.append(
                {
                    "issue_type": "empty_valuation_history",
                    "severity": "info",
                    "transfermarkt_player_id": player_id,
                    "record_key": str(player_id),
                    "source_file": str(valuation_path),
                    "details": "Player has no historical valuation rows.",
                }
            )

        for list_index, item in enumerate(history):
            raw_valuation_count += 1
            if not isinstance(item, dict):
                issue_rows.append(
                    {
                        "issue_type": "invalid_valuation_record",
                        "severity": "error",
                        "transfermarkt_player_id": player_id,
                        "record_key": f"{player_id}:{list_index}",
                        "source_file": str(valuation_path),
                        "details": "History entry is not an object.",
                    }
                )
                continue

            valuation_date = _parse_date(item.get("datum_mw"))
            if valuation_date is None:
                issue_rows.append(
                    {
                        "issue_type": "invalid_valuation_date",
                        "severity": "error",
                        "transfermarkt_player_id": player_id,
                        "record_key": f"{player_id}:{list_index}",
                        "source_file": str(valuation_path),
                        "details": f"Could not parse datum_mw={item.get('datum_mw')!r}.",
                    }
                )
                continue

            source_value = _safe_int(item.get("y"))
            display_value = str(item.get("mw") or "").strip() or None
            club_name = str(item.get("verein") or "").strip() or None
            source_timestamp_ms = _safe_int(item.get("x"))
            age_at_valuation = _safe_int(item.get("age"))

            is_terminal_record = (
                (club_name or "").casefold() == "retired"
                or (display_value == "-" and source_value == 0)
            )
            is_future_dated = valuation_date > source_snapshot_date
            market_value_eur = (
                source_value
                if source_value is not None
                and source_value > 0
                and display_value != "-"
                else None
            )
            is_valid_for_model = (
                market_value_eur is not None
                and not is_terminal_record
                and not is_future_dated
            )

            row = {
                "transfermarkt_player_id": player_id,
                "valuation_date": valuation_date,
                "market_value_eur": market_value_eur,
                "source_market_value_eur": source_value,
                "market_value_display": display_value,
                "club_name": club_name,
                "age_at_valuation": age_at_valuation,
                "source_timestamp_ms": source_timestamp_ms,
                "badge_url": str(item.get("wappen") or "").strip() or None,
                "is_terminal_record": is_terminal_record,
                "is_future_dated": is_future_dated,
                "is_valid_for_model": is_valid_for_model,
                "competition_id": competition_id,
                "season_start_year": season_start_year,
                "source_run_id": source_run_id,
                "source_snapshot_date": source_snapshot_date,
                "source_file": str(valuation_path),
                "source_list_index": list_index,
            }

            key = (player_id, valuation_date)
            previous = valuation_rows_by_key.get(key)
            if previous is not None:
                issue_rows.append(
                    {
                        "issue_type": "duplicate_player_valuation_date",
                        "severity": "warning",
                        "transfermarkt_player_id": player_id,
                        "record_key": f"{player_id}:{valuation_date.isoformat()}",
                        "source_file": str(valuation_path),
                        "details": "Duplicate player/date; latest source row retained.",
                    }
                )
                previous_rank = (
                    previous.get("source_timestamp_ms") or -1,
                    previous.get("source_list_index") or -1,
                )
                current_rank = (source_timestamp_ms or -1, list_index)
                if current_rank >= previous_rank:
                    valuation_rows_by_key[key] = row
            else:
                valuation_rows_by_key[key] = row

    for player_id in sorted(player_ids - valuation_file_ids):
        issue_rows.append(
            {
                "issue_type": "missing_player_valuation_file",
                "severity": "error",
                "transfermarkt_player_id": player_id,
                "record_key": str(player_id),
                "source_file": str(valuations_directory),
                "details": "Player has no valuation JSON file.",
            }
        )

    for player_id in sorted(valuation_file_ids - player_ids):
        issue_rows.append(
            {
                "issue_type": "valuation_player_not_in_roster",
                "severity": "warning",
                "transfermarkt_player_id": player_id,
                "record_key": str(player_id),
                "source_file": str(valuations_directory / f"{player_id}.json"),
                "details": "Valuation exists but player is absent from players.jsonl.",
            }
        )

    valuation_rows = sorted(
        valuation_rows_by_key.values(),
        key=lambda row: (row["transfermarkt_player_id"], row["valuation_date"]),
    )

    players_df = _frame(player_rows, PLAYER_SCHEMA).sort("transfermarkt_player_id")
    valuations_df = _frame(valuation_rows, VALUATION_SCHEMA).sort(
        ["transfermarkt_player_id", "valuation_date"]
    )
    issues_df = _frame(issue_rows, ISSUE_SCHEMA).sort(
        ["severity", "issue_type", "transfermarkt_player_id"],
        nulls_last=True,
    )

    players_output = output_directory / "players.parquet"
    valuations_output = output_directory / "player_valuations.parquet"
    issues_output = output_directory / "data_quality_issues.parquet"
    summary_path = output_directory / "normalization_summary.json"

    _write_parquet(players_df, players_output)
    _write_parquet(valuations_df, valuations_output)
    _write_parquet(issues_df, issues_output)

    model_valid_count = valuations_df.filter(pl.col("is_valid_for_model")).height
    terminal_count = valuations_df.filter(pl.col("is_terminal_record")).height
    future_count = valuations_df.filter(pl.col("is_future_dated")).height

    write_json(
        summary_path,
        {
            "status": "succeeded",
            "source_signature": source_signature,
            "source_manifest_status": manifest_status,
            "source_run_id": source_run_id,
            "source_run_directory": str(run_directory),
            "competition_id": competition_id,
            "season_start_year": season_start_year,
            "source_snapshot_date": source_snapshot_date.isoformat(),
            "counts": {
                "raw_player_records": len(raw_player_records),
                "normalized_players": players_df.height,
                "valuation_response_files": len(valuation_file_ids),
                "raw_valuation_records": raw_valuation_count,
                "normalized_valuation_records": valuations_df.height,
                "model_valid_valuation_records": model_valid_count,
                "terminal_records": terminal_count,
                "future_dated_records": future_count,
                "data_quality_issues": issues_df.height,
            },
            "outputs": {
                "players": str(final_output_directory / "players.parquet"),
                "player_valuations": str(
                    final_output_directory / "player_valuations.parquet"
                ),
                "data_quality_issues": str(
                    final_output_directory / "data_quality_issues.parquet"
                ),
            },
        },
    )

    _promote_directory(output_directory, final_output_directory)

    return TransfermarktNormalizationResult(
        output_directory=final_output_directory,
        summary_path=final_output_directory / "normalization_summary.json",
        player_count=players_df.height,
        raw_valuation_count=raw_valuation_count,
        valuation_count=valuations_df.height,
        model_valid_count=model_valid_count,
        issue_count=issues_df.height,
    )
