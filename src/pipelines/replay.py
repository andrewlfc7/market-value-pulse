from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

from features.pipeline import enrich_match, load_feature_runtime
from ingestion.common import write_json
from ingestion.progress import ProgressCallback, ProgressEmitter, ProgressUpdate
from ratings.model import load_rating_config, score_rating_features
from ratings.pipeline import add_form_history, resolve_rating_positions
from valuation.features import (
    ValuationFeatureConfig,
    build_current_scoring_dataset,
)
from valuation.scoring import score_valuation_features
from valuation.artifacts import resolve_model_directory


class ReplayError(RuntimeError):
    """Raised when a historical replay cannot be constructed or executed."""


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    run_directory: Path
    manifest_path: Path
    results_path: Path
    selected_matches: int
    completed_matches: int
    failed_matches: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _match_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value or "").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError as exc:
        raise ReplayError(f"Invalid replay match datetime: {value!r}") from exc


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _publish_replay_impacts(
    *,
    results: list[dict[str, object]],
    run_id: str,
    player_id: int | None,
    serving_root: Path | None,
) -> int:
    """Attach real before/after valuation changes to serving match rows."""
    if player_id is None or serving_root is None:
        return 0
    path = serving_root / "match_impacts.parquet"
    if not path.exists():
        return 0
    updates = {int(row["match_id"]): row for row in results}
    output_rows: list[dict[str, object]] = []
    published = 0
    defaults = {
        "replay_run_id": None,
        "replay_sequence": None,
        "valuation_update_status": None,
        "valuation_estimate_eur": None,
        "valuation_lower_90_eur": None,
        "valuation_upper_90_eur": None,
        "estimated_value_delta_eur": None,
        "probability_value_increase": None,
    }
    for source in pl.read_parquet(path).to_dicts():
        row = {**defaults, **source}
        if int(row["player_id"]) == player_id:
            update = updates.get(int(row["match_id"]))
            if update is not None:
                row.update(
                    {
                        "replay_run_id": run_id,
                        "replay_sequence": update.get("replay_sequence"),
                        "valuation_update_status": update.get("valuation_update_status"),
                        "valuation_estimate_eur": update.get("estimated_player_value_eur"),
                        "valuation_lower_90_eur": update.get(
                            "estimated_player_lower_90_eur"
                        ),
                        "valuation_upper_90_eur": update.get(
                            "estimated_player_upper_90_eur"
                        ),
                        "estimated_value_delta_eur": update.get(
                            "estimated_player_delta_eur"
                        ),
                        "probability_value_increase": update.get(
                            "probability_value_increase"
                        ),
                    }
                )
                published += 1
        output_rows.append(row)
    _atomic_parquet(pl.DataFrame(output_rows, infer_schema_length=None), path)
    return published


def _metadata(partition: Path, player_id: int | None) -> dict[str, object] | None:
    matches_path = partition / "matches.parquet"
    players_path = partition / "player_matches.parquet"
    if not matches_path.exists() or not players_path.exists():
        return None
    match = pl.read_parquet(matches_path)
    players = pl.read_parquet(players_path)
    if match.is_empty():
        return None
    if player_id is not None and (
        players.is_empty()
        or player_id not in set(players["player_id"].drop_nulls().to_list())
    ):
        return None
    row = match.row(0, named=True)
    return {
        "match_id": int(row["match_id"]),
        "match_datetime": str(row.get("start_date") or ""),
        "home_team_id": row.get("home_team_id"),
        "home_team_name": row.get("home_team_name"),
        "away_team_id": row.get("away_team_id"),
        "away_team_name": row.get("away_team_name"),
        "affected_players": players.height,
        "player_id": player_id,
        "source_partition": str(partition),
    }


def select_replay_matches(
    *,
    normalized_root: Path,
    competition: str,
    season: str,
    match_count: int,
    player_id: int | None = None,
) -> list[dict[str, object]]:
    if match_count < 1:
        raise ReplayError("match_count must be at least 1")
    source = normalized_root / f"competition={competition}" / f"season={season}" / "matches"
    if not source.exists():
        raise ReplayError(f"Normalized season does not exist: {source}")
    rows = [
        row
        for partition in source.glob("match_id=*")
        if (partition / "_SUCCESS.json").exists()
        if (row := _metadata(partition, player_id)) is not None
    ]
    rows.sort(key=lambda row: (str(row["match_datetime"]), int(row["match_id"])))
    selected = rows[-match_count:]
    if not selected:
        scope = f"player_id={player_id}" if player_id is not None else "season"
        raise ReplayError(f"No completed matches found for {scope}")
    return [
        {**row, "replay_sequence": sequence}
        for sequence, row in enumerate(selected, start=1)
    ]


def run_historical_replay(
    *,
    competition: str,
    competition_id: int,
    season: str,
    match_count: int = 8,
    player_id: int | None = None,
    normalized_root: Path = Path("data/normalized/whoscored"),
    replay_root: Path = Path("data/replays"),
    models_root: Path = Path("models/features"),
    rating_artifact_directory: Path = Path("models/ratings/post_match_v2"),
    valuations_path: Path | None = None,
    mapping_path: Path | None = None,
    valuation_model_root: Path = Path("data/modeling/valuation_model"),
    valuation_model_version: str = "active",
    serving_root: Path | None = Path("data/serving"),
    prepare_only: bool = False,
    progress: ProgressCallback | None = None,
) -> ReplayResult:
    selected = select_replay_matches(
        normalized_root=normalized_root,
        competition=competition,
        season=season,
        match_count=match_count,
        player_id=player_id,
    )
    supplied_valuation_inputs = valuations_path is not None or mapping_path is not None
    if supplied_valuation_inputs and not prepare_only:
        missing_inputs = [
            label
            for label, path in (
                ("valuations", valuations_path),
                ("mapping", mapping_path),
            )
            if path is None or not path.exists()
        ]
        if missing_inputs:
            raise ReplayError(
                "Replay valuation inputs are incomplete or missing: "
                + ", ".join(missing_inputs)
            )
    valuation_requested = supplied_valuation_inputs and not prepare_only
    if valuation_requested:
        try:
            resolve_model_directory(valuation_model_root, valuation_model_version)
        except FileNotFoundError as exc:
            raise ReplayError(
                f"Requested valuation model is unavailable: {exc}"
            ) from exc
    feature_runtime = None if prepare_only else load_feature_runtime(models_root)
    started = _utc_now()
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_directory = replay_root / f"competition={competition}" / f"season={season}" / f"run_id={run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = run_directory / "replay_manifest.parquet"
    _atomic_parquet(pl.DataFrame(selected, infer_schema_length=None), manifest_path)
    write_json(
        run_directory / "replay_plan.json",
        {
            "run_id": run_id,
            "competition": competition,
            "season": season,
            "player_id": player_id,
            "requested_matches": match_count,
            "selected_matches": len(selected),
            "prepare_only": prepare_only,
            "rating_artifact_directory": str(rating_artifact_directory),
            "valuations_path": str(valuations_path) if valuations_path else None,
            "mapping_path": str(mapping_path) if mapping_path else None,
            "valuation_model_root": str(valuation_model_root),
            "valuation_model_version": valuation_model_version,
            "created_at": started.isoformat(),
            "manifest": str(manifest_path),
            "method": "chronological historical replay",
        },
    )

    emitter = ProgressEmitter(run_id=run_id, log_path=run_directory / "progress.jsonl", callback=progress)
    emitter.emit(
        ProgressUpdate(
            stage="replay",
            state="started",
            description="Replaying historical matches",
            total=len(selected),
        )
    )
    results: list[dict[str, object]] = []
    completed = 0
    failed = 0
    replay_rating_frames: list[pl.DataFrame] = []
    previous_estimates: dict[int, float] = {}
    for index, row in enumerate(selected, start=1):
        match_id = int(row["match_id"])
        source_partition = Path(str(row["source_partition"]))
        state_path = run_directory / "replay_state.json"
        write_json(
            state_path,
            {
                "run_id": run_id,
                "status": "running",
                "current_sequence": index,
                "current_match_id": match_id,
                "completed_matches": completed,
                "failed_matches": failed,
                "updated_at": _utc_now().isoformat(),
            },
        )
        try:
            result = enrich_match(
                source_partition,
                competition=competition,
                competition_id=competition_id,
                season=season,
                output_root=run_directory / "enriched",
                models_root=models_root,
                prepare_only=prepare_only,
                force=True,
                runtime=feature_runtime,
            )
            rating_status = "pending_rating_model"
            rating_rows = 0
            valuation_status = "pending_valuation_model"
            valuation_rows = 0
            valuation_predictions_path: str | None = None
            valuation_error: str | None = None
            estimated_player_value_eur: float | None = None
            estimated_player_lower_90_eur: float | None = None
            estimated_player_upper_90_eur: float | None = None
            estimated_player_delta_eur: float | None = None
            probability_value_increase: float | None = None
            if not prepare_only:
                feature_path = result.output_directory / "player_match_features.parquet"
                scored = score_rating_features(
                    resolve_rating_positions(
                        pl.read_parquet(feature_path), rating_artifact_directory
                    ),
                    rating_artifact_directory,
                )
                replay_rating_frames.append(scored)
                all_ratings = pl.concat(
                    replay_rating_frames, how="diagonal_relaxed"
                )
                config = load_rating_config(rating_artifact_directory)
                all_ratings, form_state = add_form_history(
                    all_ratings, half_life_days=config.ewm_half_life_days
                )
                _atomic_parquet(
                    all_ratings, run_directory / "player_match_ratings.parquet"
                )
                _atomic_parquet(
                    form_state, run_directory / "player_form_state.parquet"
                )
                rating_rows = scored.height
                rating_status = "succeeded"
                valuation_enabled = valuation_requested
                if valuation_enabled:
                    valuation_directory = (
                        run_directory / "valuation" / f"sequence={index:02d}"
                    )
                    valuation_features_path = (
                        valuation_directory / "current_scoring_features.parquet"
                    )
                    predictions_path = (
                        valuation_directory / "valuation_predictions.parquet"
                    )
                    try:
                        build_current_scoring_dataset(
                            valuations_path=valuations_path,
                            mapping_path=mapping_path,
                            ratings_path=(
                                run_directory / "player_match_ratings.parquet"
                            ),
                            output_path=valuation_features_path,
                            as_of_date=(
                                _match_datetime(row["match_datetime"]).date()
                                + timedelta(days=1)
                            ),
                            config=ValuationFeatureConfig(
                                minimum_interval_minutes=1.0
                            ),
                        )
                        valuation_result = score_valuation_features(
                            features_path=valuation_features_path,
                            model_root=valuation_model_root,
                            model_version=valuation_model_version,
                            output_path=predictions_path,
                        )
                        valuation_status = "succeeded"
                        valuation_rows = valuation_result.rows
                        valuation_predictions_path = str(
                            valuation_result.output_path
                        )
                        if player_id is not None:
                            player_prediction = pl.read_parquet(
                                valuation_result.output_path
                            ).filter(
                                pl.col("whoscored_player_id") == player_id
                            )
                            if player_prediction.is_empty():
                                raise ReplayError(
                                    f"Selected player_id={player_id} has no valuation "
                                    "prediction; verify the approved entity mapping"
                                )
                            prediction = player_prediction.row(0, named=True)
                            estimated_player_value_eur = float(
                                prediction["predicted_market_value_eur"]
                            )
                            estimated_player_lower_90_eur = float(
                                prediction[
                                    "predicted_market_value_lower_90_eur"
                                ]
                            )
                            estimated_player_upper_90_eur = float(
                                prediction[
                                    "predicted_market_value_upper_90_eur"
                                ]
                            )
                            probability_value_increase = float(
                                prediction["probability_value_increase"]
                            )
                            previous = previous_estimates.get(player_id)
                            if previous is not None:
                                estimated_player_delta_eur = (
                                    estimated_player_value_eur - previous
                                )
                            previous_estimates[player_id] = (
                                estimated_player_value_eur
                            )
                    except Exception as exc:
                        valuation_status = "failed"
                        valuation_error = str(exc)
                elif not prepare_only:
                    valuation_status = "skipped_missing_inputs"
            match_failed = valuation_status == "failed"
            if match_failed:
                failed += 1
            else:
                completed += 1
            results.append(
                {
                    "replay_sequence": index,
                    "match_id": match_id,
                    "match_datetime": row["match_datetime"],
                    "status": "partial" if match_failed else result.status,
                    "affected_players": row["affected_players"],
                    "enriched_partition": str(result.output_directory),
                    "rating_update_status": rating_status,
                    "rating_rows": rating_rows,
                    "valuation_update_status": valuation_status,
                    "valuation_rows": valuation_rows,
                    "valuation_predictions_path": valuation_predictions_path,
                    "valuation_error": valuation_error,
                    "estimated_player_value_eur": estimated_player_value_eur,
                    "estimated_player_lower_90_eur": estimated_player_lower_90_eur,
                    "estimated_player_upper_90_eur": estimated_player_upper_90_eur,
                    "estimated_player_delta_eur": estimated_player_delta_eur,
                    "probability_value_increase": probability_value_increase,
                    "error": None,
                }
            )
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "replay_sequence": index,
                    "match_id": match_id,
                    "match_datetime": row["match_datetime"],
                    "status": "failed",
                    "affected_players": row["affected_players"],
                    "enriched_partition": None,
                    "rating_update_status": "not_run",
                    "rating_rows": 0,
                    "valuation_update_status": "not_run",
                    "valuation_rows": 0,
                    "valuation_predictions_path": None,
                    "valuation_error": None,
                    "estimated_player_value_eur": None,
                    "estimated_player_lower_90_eur": None,
                    "estimated_player_upper_90_eur": None,
                    "estimated_player_delta_eur": None,
                    "probability_value_increase": None,
                    "error": str(exc),
                }
            )
        emitter.emit(
            ProgressUpdate(
                stage="replay",
                state=("completed" if index == len(selected) else "advanced"),
                description="Replaying historical matches",
                completed=index,
                total=len(selected),
                current=f"match_id={match_id}",
                succeeded=completed,
                failed=failed,
            )
        )

    results_path = run_directory / "replay_results.parquet"
    _atomic_parquet(pl.DataFrame(results, infer_schema_length=None), results_path)
    published_impacts = _publish_replay_impacts(
        results=results,
        run_id=run_id,
        player_id=player_id,
        serving_root=serving_root,
    )
    final_status = "succeeded" if failed == 0 else ("partial" if completed else "failed")
    write_json(
        run_directory / "replay_state.json",
        {
            "run_id": run_id,
            "status": final_status,
            "completed_matches": completed,
            "failed_matches": failed,
            "completed_at": _utc_now().isoformat(),
            "results": str(results_path),
            "published_serving_impacts": published_impacts,
            "next_stage": (
                "copy feature artifacts, fit rating artifacts, then run full replay"
                if prepare_only
                else (
                    "inspect sequential valuation predictions"
                    if valuations_path is not None
                    and mapping_path is not None
                    and valuation_requested
                    else "provide valuations, mapping, and a valuation model"
                )
            ),
        },
    )
    return ReplayResult(
        run_id=run_id,
        run_directory=run_directory,
        manifest_path=manifest_path,
        results_path=results_path,
        selected_matches=len(selected),
        completed_matches=completed,
        failed_matches=failed,
    )
