from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
import polars as pl

from ingestion.common import atomic_write_bytes, sha256_file, write_json
from ingestion.progress import ProgressCallback, ProgressEmitter, ProgressUpdate
from ingestion.whoscored.extract import (
    extract_match_id_from_html,
    extract_match_id_from_url,
    parse_match_centre_data,
)
from ingestion.whoscored.fetch import (
    MatchFetchRequest,
    MatchFetchResult,
    fetch_match_pages,
)
from ingestion.whoscored.normalize import NormalizedMatch, normalize_match_data
from ingestion.whoscored.catalog import normalize_season_label
from ingestion.whoscored.competitions import resolve_competition
from ingestion.whoscored.scope import discover_competition_season
from ingestion.whoscored.validation import validate_normalized_match


class WhoScoredIngestionError(RuntimeError):
    """Raised when a WhoScored run cannot produce any usable matches."""


@dataclass(frozen=True)
class WhoScoredIngestionResult:
    run_id: str
    run_directory: Path
    manifest_path: Path
    discovered: int
    requested: int
    processed: int
    skipped: int
    deferred: int
    failed: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _read_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        frame = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pl.read_csv(path)
    else:
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        frame = pl.DataFrame({"match_url": urls})

    if "match_url" not in frame.columns:
        for candidate in ("url", "source_url"):
            if candidate in frame.columns:
                frame = frame.rename({candidate: "match_url"})
                break
    if "match_url" not in frame.columns:
        raise WhoScoredIngestionError("Manifest must contain match_url or url")
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        url = str(row["match_url"])
        match_id = row.get("match_id") or extract_match_id_from_url(url)
        if match_id is None:
            continue
        rows.append({**row, "match_id": int(match_id), "match_url": url})
    deduplicated = list({row["match_id"]: row for row in rows}.values())
    deduplicated.sort(key=lambda row: int(row["match_id"]))
    return deduplicated[:limit] if limit else deduplicated


def _partition_directory(
    normalized_root: Path,
    *,
    competition: str,
    season: str,
    match_id: int,
) -> Path:
    safe_season = season.replace("/", "-")
    return (
        normalized_root
        / f"competition={competition}"
        / f"season={safe_season}"
        / "matches"
        / f"match_id={match_id}"
    )


def _success_path(
    normalized_root: Path,
    *,
    competition: str,
    season: str,
    match_id: int,
) -> Path:
    return _partition_directory(
        normalized_root,
        competition=competition,
        season=season,
        match_id=match_id,
    ) / "_SUCCESS.json"


def _fixture_date(row: dict[str, Any]) -> date | None:
    value = row.get("fixture_date")
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _write_normalized_bundle(
    bundle: NormalizedMatch,
    *,
    output_directory: Path,
    source_page: Path,
    competition: str,
    season: str,
) -> Path:
    checks = validate_normalized_match(bundle)
    output_directory.mkdir(parents=True, exist_ok=True)
    datasets = {
        "raw_events": bundle.raw_events,
        "matches": bundle.matches,
        "teams": bundle.teams,
        "player_matches": bundle.player_matches,
        "events": bundle.events,
        "shots": bundle.shots,
    }
    objects: dict[str, dict[str, object]] = {}
    for name, frame in datasets.items():
        if frame.is_empty() and name not in {"teams", "player_matches", "shots"}:
            raise WhoScoredIngestionError(f"{name} is empty")
        path = output_directory / f"{name}.parquet"
        _atomic_parquet(path, frame)
        objects[name] = {"path": str(path), "rows": frame.height}

    write_json(
        output_directory / "data_quality.json",
        {"checks": [check.as_dict() for check in checks]},
    )
    success_path = output_directory / "_SUCCESS.json"
    write_json(
        success_path,
        {
            "status": "succeeded",
            "match_id": bundle.metadata["match_id"],
            "competition": competition,
            "season": season,
            "normalized_at": _utc_now().isoformat(),
            "source_page": str(source_page),
            "source_page_sha256": sha256_file(source_page),
            "objects": objects,
            "metadata": bundle.metadata,
        },
    )
    return success_path


def normalize_saved_match(
    *,
    page_path: Path,
    source_url: str,
    normalized_root: Path,
    competition: str,
    season: str,
    match_id: int | None = None,
) -> Path:
    html = page_path.read_text(encoding="utf-8")
    resolved_match_id = (
        match_id
        or extract_match_id_from_url(source_url)
        or extract_match_id_from_html(html)
    )
    if resolved_match_id is None:
        raise WhoScoredIngestionError("Could not determine the match ID")
    payload = parse_match_centre_data(html)
    bundle = normalize_match_data(
        payload,
        match_id=resolved_match_id,
        source_url=source_url,
    )
    partition = _partition_directory(
        normalized_root,
        competition=competition,
        season=season,
        match_id=resolved_match_id,
    )
    return _write_normalized_bundle(
        bundle,
        output_directory=partition,
        source_page=page_path,
        competition=competition,
        season=season,
    )


def _result_row(
    result: MatchFetchResult,
    *,
    status: str,
    raw_page_path: Path | None = None,
    normalized_success_path: Path | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "match_id": result.match_id,
        "match_url": result.match_url,
        "status": status,
        "status_code": result.status_code,
        "elapsed_ms": result.elapsed_ms,
        "attempts": result.attempts,
        "raw_page_path": str(raw_page_path) if raw_page_path else None,
        "normalized_success_path": (
            str(normalized_success_path) if normalized_success_path else None
        ),
        "error": error or result.error,
    }


def _is_completed_match(payload: dict[str, Any]) -> bool:
    status = payload.get("statusCode")
    full_time_score = str(payload.get("ftScore") or "").strip()
    return status in {6, "6", "FT", "Finished", "finished"} or bool(full_time_score)


def ingest_whoscored(
    *,
    competition: str,
    season: str,
    manifest_path: Path | None = None,
    raw_root: Path = Path("data/raw/whoscored"),
    normalized_root: Path = Path("data/normalized/whoscored"),
    registry_path: Path = Path("config/whoscored/competitions.json"),
    workers: int = 2,
    timeout_ms: int = 45_000,
    delay_ms: int = 1_000,
    max_retries: int = 2,
    discovery_wait_seconds: float = 1.0,
    max_previous_windows: int = 60,
    max_next_windows: int = 12,
    max_new_matches: int | None = None,
    force: bool = False,
    headful: bool = False,
    progress: ProgressCallback | None = None,
) -> WhoScoredIngestionResult:
    started_at = _utc_now()
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    effective_competition = competition
    effective_season = normalize_season_label(season)
    if manifest_path is None:
        try:
            effective_competition = resolve_competition(
                competition, registry_path=registry_path
            ).key
        except (ValueError, FileNotFoundError) as exc:
            raise WhoScoredIngestionError(str(exc)) from exc
    run_directory = (
        raw_root
        / f"run_date={started_at.date().isoformat()}"
        / f"run_id={run_id}"
        / f"competition={effective_competition}"
        / f"season={effective_season}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest_output = run_directory / "manifest.json"
    progress_emitter = ProgressEmitter(
        run_id=run_id,
        log_path=run_directory / "progress.jsonl",
        callback=progress,
    )
    source_manifest = str(manifest_path) if manifest_path else None
    source_manifest_sha256 = sha256_file(manifest_path) if manifest_path else None
    write_json(
        manifest_output,
        {
            "run_id": run_id,
            "pipeline": "whoscored_match_ingestion",
            "status": "running",
            "started_at": started_at.isoformat(),
            "competition": effective_competition,
            "season": effective_season,
            "discovery_mode": "manifest" if manifest_path else "competition_season",
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_manifest_sha256,
        },
    )

    discovery_warnings: list[str] = []
    progress_emitter.emit(
        ProgressUpdate(
            stage="discovery",
            state="started",
            description="Discovering WhoScored fixtures",
            current=f"{effective_competition} {effective_season}",
        )
    )
    try:
        if manifest_path is not None:
            rows = _read_manifest(manifest_path)
        else:
            discovery = discover_competition_season(
                competition_name=competition,
                season=season,
                raw_directory=run_directory / "discovery",
                registry_path=registry_path,
                max_previous=max_previous_windows,
                max_next=max_next_windows,
                timeout_ms=max(timeout_ms, 60_000),
                wait_seconds=discovery_wait_seconds,
                headful=headful,
            )
            effective_competition = discovery.competition.key
            effective_season = discovery.canonical_season
            rows = discovery.matches
            discovery_warnings.extend(discovery.warnings)
            discovery_frame = pl.DataFrame(discovery.matches, infer_schema_length=None)
            _atomic_parquet(run_directory / "discovered_matches.parquet", discovery_frame)
            discovery_frame.write_csv(run_directory / "discovered_matches.csv")
        if not rows:
            raise ValueError("Discovery produced no valid match URLs")
        progress_emitter.emit(
            ProgressUpdate(
                stage="discovery",
                state="completed",
                description="Discovered WhoScored fixtures",
                completed=len(rows),
                total=len(rows),
                succeeded=len(rows),
            )
        )
    except Exception as exc:
        progress_emitter.emit(
            ProgressUpdate(
                stage="discovery",
                state="failed",
                description="WhoScored fixture discovery failed",
                failed=1,
                current=str(exc),
            )
        )
        write_json(
            manifest_output,
            {
                "run_id": run_id,
                "pipeline": "whoscored_match_ingestion",
                "status": "failed_discovery",
                "started_at": started_at.isoformat(),
                "completed_at": _utc_now().isoformat(),
                "competition": effective_competition,
                "season": effective_season,
                "discovery_mode": "manifest" if manifest_path else "competition_season",
                "source_manifest": source_manifest,
                "source_manifest_sha256": source_manifest_sha256,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise WhoScoredIngestionError(
            f"WhoScored match discovery failed; see {manifest_output}: {exc}"
        ) from exc

    skipped_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        match_id = int(row["match_id"])
        success = _success_path(
            normalized_root,
            competition=effective_competition,
            season=effective_season,
            match_id=match_id,
        )
        if success.exists() and not force:
            skipped_rows.append(
                {
                    "match_id": match_id,
                    "match_url": row["match_url"],
                    "status": "skipped_existing",
                    "status_code": None,
                    "elapsed_ms": 0,
                    "attempts": 0,
                    "raw_page_path": None,
                    "normalized_success_path": str(success),
                    "error": None,
                }
            )
            continue
        fixture_date = _fixture_date(row)
        if fixture_date is not None and fixture_date > started_at.date() and not force:
            future_rows.append(
                {
                    "match_id": match_id,
                    "match_url": row["match_url"],
                    "status": "deferred_future_fixture",
                    "status_code": None,
                    "elapsed_ms": 0,
                    "attempts": 0,
                    "raw_page_path": None,
                    "normalized_success_path": None,
                    "error": f"Fixture date {fixture_date.isoformat()} is in the future",
                }
            )
            continue
        candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda row: (
            _fixture_date(row) or date.min,
            int(row["match_id"]),
        ),
        reverse=True,
    )
    selected_rows = (
        candidate_rows[:max_new_matches]
        if max_new_matches is not None
        else candidate_rows
    )
    selected_ids = {int(row["match_id"]) for row in selected_rows}
    limited_rows = [
        {
            "match_id": int(row["match_id"]),
            "match_url": row["match_url"],
            "status": "not_selected_limit",
            "status_code": None,
            "elapsed_ms": 0,
            "attempts": 0,
            "raw_page_path": None,
            "normalized_success_path": None,
            "error": None,
        }
        for row in candidate_rows
        if int(row["match_id"]) not in selected_ids
    ]
    requests: list[MatchFetchRequest] = []
    for row in selected_rows:
        match_id = int(row["match_id"])
        requests.append(
            MatchFetchRequest(
                match_id=match_id,
                match_url=str(row["match_url"]),
                metadata=row,
            )
        )

    progress_emitter.emit(
        ProgressUpdate(
            stage="selection",
            state="completed",
            description="Selected newest unprocessed matches",
            completed=len(rows),
            total=len(rows),
            succeeded=len(requests),
            skipped=len(skipped_rows),
            deferred=len(future_rows),
            current=(
                f"selected={len(requests)} pending={len(limited_rows)} "
                f"future={len(future_rows)}"
            ),
        )
    )

    fetch_completed = 0
    fetch_succeeded = 0
    fetch_failed = 0

    def on_fetch_result(result: MatchFetchResult) -> None:
        nonlocal fetch_completed, fetch_succeeded, fetch_failed
        fetch_completed += 1
        fetch_succeeded += int(result.succeeded)
        fetch_failed += int(not result.succeeded)
        progress_emitter.emit(
            ProgressUpdate(
                stage="fetch",
                state=("completed" if fetch_completed == len(requests) else "advanced"),
                description="Fetching WhoScored match pages",
                completed=fetch_completed,
                total=len(requests),
                current=f"match_id={result.match_id}",
                succeeded=fetch_succeeded,
                failed=fetch_failed,
            )
        )

    if requests:
        progress_emitter.emit(
            ProgressUpdate(
                stage="fetch",
                state="started",
                description="Fetching WhoScored match pages",
                total=len(requests),
            )
        )

    fetched = asyncio.run(
        fetch_match_pages(
            requests,
            workers=workers,
            timeout_ms=timeout_ms,
            delay_ms=delay_ms,
            max_retries=max_retries,
            headful=headful,
            on_result=on_fetch_result,
        )
    )
    result_rows = [*skipped_rows, *future_rows, *limited_rows]
    normalization_succeeded = 0
    normalization_deferred = 0
    normalization_failed = 0
    if fetched:
        progress_emitter.emit(
            ProgressUpdate(
                stage="normalize",
                state="started",
                description="Normalizing WhoScored matches",
                total=len(fetched),
            )
        )
    for index, result in enumerate(fetched, start=1):
        match_raw_directory = run_directory / "matches" / f"match_id={result.match_id}"
        page_path = match_raw_directory / "page.html"
        if result.html:
            atomic_write_bytes(page_path, result.html.encode("utf-8"))
        if not result.succeeded:
            result_rows.append(
                _result_row(
                    result,
                    status="failed_fetch",
                    raw_page_path=page_path if page_path.exists() else None,
                )
            )
            normalization_failed += 1
            progress_emitter.emit(
                ProgressUpdate(
                    stage="normalize",
                    state=("completed" if index == len(fetched) else "advanced"),
                    description="Normalizing WhoScored matches",
                    completed=index,
                    total=len(fetched),
                    current=f"match_id={result.match_id}",
                    succeeded=normalization_succeeded,
                    deferred=normalization_deferred,
                    failed=normalization_failed,
                )
            )
            continue
        try:
            payload = parse_match_centre_data(result.html)
            atomic_write_bytes(
                match_raw_directory / "matchCentreData.json",
                orjson.dumps(payload, option=orjson.OPT_INDENT_2),
            )
            if not _is_completed_match(payload):
                result_rows.append(
                    _result_row(
                        result,
                        status="deferred_incomplete",
                        raw_page_path=page_path,
                        error="Match is not final; it will be retried on the next run",
                    )
                )
                normalization_deferred += 1
                continue
            bundle = normalize_match_data(
                payload,
                match_id=result.match_id,
                source_url=result.match_url,
            )
            normalized_success = _write_normalized_bundle(
                bundle,
                output_directory=_partition_directory(
                    normalized_root,
                    competition=effective_competition,
                    season=effective_season,
                    match_id=result.match_id,
                ),
                source_page=page_path,
                competition=effective_competition,
                season=effective_season,
            )
            result_rows.append(
                _result_row(
                    result,
                    status="succeeded",
                    raw_page_path=page_path,
                    normalized_success_path=normalized_success,
                )
            )
            normalization_succeeded += 1
        except Exception as exc:
            result_rows.append(
                _result_row(
                    result,
                    status="failed_normalization",
                    raw_page_path=page_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            normalization_failed += 1
        finally:
            progress_emitter.emit(
                ProgressUpdate(
                    stage="normalize",
                    state=("completed" if index == len(fetched) else "advanced"),
                    description="Normalizing WhoScored matches",
                    completed=index,
                    total=len(fetched),
                    current=f"match_id={result.match_id}",
                    succeeded=normalization_succeeded,
                    deferred=normalization_deferred,
                    failed=normalization_failed,
                )
            )

    results_frame = pl.DataFrame(result_rows, infer_schema_length=None)
    _atomic_parquet(run_directory / "match_results.parquet", results_frame)
    processed = results_frame.filter(pl.col("status") == "succeeded").height
    skipped = results_frame.filter(pl.col("status") == "skipped_existing").height
    deferred = results_frame.filter(pl.col("status").str.starts_with("deferred_")).height
    failed = results_frame.filter(pl.col("status").str.starts_with("failed")).height
    retry_manifest_path: Path | None = None
    if failed:
        retry_frame = results_frame.filter(pl.col("status").str.starts_with("failed"))
        retry_manifest_path = run_directory / "failed_matches.csv"
        retry_frame.write_csv(retry_manifest_path)
    final_status = (
        "succeeded"
        if failed == 0
        else ("partial" if processed or skipped or deferred else "failed")
    )
    write_json(
        manifest_output,
        {
            "run_id": run_id,
            "pipeline": "whoscored_match_ingestion",
            "status": final_status,
            "started_at": started_at.isoformat(),
            "completed_at": _utc_now().isoformat(),
            "competition": effective_competition,
            "season": effective_season,
            "discovery_mode": "manifest" if manifest_path else "competition_season",
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_manifest_sha256,
            "warnings": discovery_warnings,
            "counts": {
                "discovered": len(rows),
                "selected_for_fetch": len(requests),
                "requested": len(requests),
                "not_selected_limit": len(limited_rows),
                "future_fixtures": len(future_rows),
                "processed": processed,
                "skipped": skipped,
                "deferred": deferred,
                "failed": failed,
            },
            "match_results": str(run_directory / "match_results.parquet"),
            "progress_log": str(run_directory / "progress.jsonl"),
            "retry_manifest": str(retry_manifest_path) if retry_manifest_path else None,
        },
    )
    if processed == 0 and skipped == 0 and deferred == 0:
        raise WhoScoredIngestionError(
            f"All {failed} requested matches failed; see {manifest_output}"
        )
    return WhoScoredIngestionResult(
        run_id=run_id,
        run_directory=run_directory,
        manifest_path=manifest_output,
        discovered=len(rows),
        requested=len(requests),
        processed=processed,
        skipped=skipped,
        deferred=deferred,
        failed=failed,
    )
