from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from database.loader import DatabaseLoadResult, load_pipeline_to_postgres
from features.pipeline import EnrichmentRunResult, enrich_season
from entity_resolution.player_mapping import PlayerMappingResult, build_player_mapping
from ingestion.common import write_json
from ingestion.progress import ProgressCallback, ProgressEmitter, ProgressUpdate
from ratings.pipeline import (
    RatingPipelineResult,
    fit_and_score_rating_season,
    update_rating_season,
)
from serving.builder import ServingBuildResult, build_serving_tables
from valuation.features import (
    ValuationFeatureConfig,
    ValuationFeatureError,
    build_current_scoring_dataset,
)
from valuation.scoring import score_valuation_features


@dataclass(frozen=True)
class MaterializeResult:
    enrichment: EnrichmentRunResult
    ratings: RatingPipelineResult
    mapping: PlayerMappingResult
    serving: ServingBuildResult
    valuation_status: str
    predictions_path: Path | None
    database: DatabaseLoadResult | None
    manifest_path: Path


class MaterializeError(RuntimeError):
    """Raised when a season cannot be materialized consistently."""


def _materialize_season(
    *,
    competition: str,
    competition_id: int,
    season: str,
    as_of_date: date,
    transfermarkt_players_path: Path,
    valuations_path: Path,
    normalized_root: Path = Path("data/normalized/whoscored"),
    features_root: Path = Path("data/features/whoscored"),
    models_root: Path = Path("models/features"),
    rating_artifact_directory: Path = Path("models/ratings/post_match_v2"),
    ratings_root: Path = Path("data/features/ratings"),
    rating_state_root: Path = Path("data/state/ratings"),
    mapping_path: Path = Path(
        "data/normalized/entity_resolution/player_mapping_exact.parquet"
    ),
    manual_mapping_overrides: Path | None = None,
    valuation_model_root: Path = Path("data/modeling/valuation_model"),
    serving_root: Path = Path("data/serving"),
    database_url: str | None = None,
    progress: ProgressCallback | None = None,
) -> MaterializeResult:
    run_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    progress_path = serving_root / "_runs" / f"run_id={run_id}" / "progress.jsonl"
    emitter = ProgressEmitter(
        run_id=run_id, log_path=progress_path, callback=progress
    )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="started",
            description="Materializing season",
            total=6,
        )
    )
    enrichment = enrich_season(
        competition=competition,
        competition_id=competition_id,
        season=season,
        normalized_root=normalized_root,
        output_root=features_root,
        models_root=models_root,
        progress=progress,
    )
    if enrichment.failed:
        raise MaterializeError(
            f"Feature enrichment failed for {enrichment.failed} match(es); "
            f"see {enrichment.results_path}"
        )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="advanced",
            description="Materializing season",
            completed=1,
            total=6,
            current="enrichment complete",
            succeeded=1,
        )
    )

    rating_directory = ratings_root / f"competition={competition}" / f"season={season}"
    rating_state_directory = (
        rating_state_root / f"competition={competition}" / f"season={season}"
    )
    season_rating_artifact_directory = (
        rating_artifact_directory
        / f"competition={competition}"
        / f"season={season}"
    )
    has_rating_state = (
        (season_rating_artifact_directory / "rating_model_config.json").exists()
        and (
            season_rating_artifact_directory / "season_primary_positions.parquet"
        ).exists()
        and (
            season_rating_artifact_directory / "career_primary_positions.parquet"
        ).exists()
        and (rating_directory / "player_match_ratings.parquet").exists()
        and (rating_state_directory / "processed_matches.parquet").exists()
    )
    ratings = (
        update_rating_season(
            competition=competition,
            season=season,
            features_root=features_root,
            output_root=ratings_root,
            state_root=rating_state_root,
            artifact_directory=season_rating_artifact_directory,
        )
        if has_rating_state
        else fit_and_score_rating_season(
            competition=competition,
            season=season,
            features_root=features_root,
            output_root=ratings_root,
            state_root=rating_state_root,
            artifact_directory=season_rating_artifact_directory,
        )
    )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="advanced",
            description="Materializing season",
            completed=2,
            total=6,
            current=f"ratings {ratings.mode}",
            succeeded=2,
        )
    )

    mapping = build_player_mapping(
        transfermarkt_players_path=transfermarkt_players_path,
        whoscored_normalized_root=normalized_root,
        competition=competition,
        season=season,
        output_path=mapping_path,
        manual_overrides=manual_mapping_overrides,
    )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="advanced",
            description="Materializing season",
            completed=3,
            total=6,
            current=f"mapped={mapping.mapped_players} review={mapping.review_players}",
            succeeded=3,
        )
    )

    predictions_path: Path | None = None
    valuation_status = "skipped_no_active_model"
    if (valuation_model_root / "active.json").exists():
        current_features = valuation_model_root / "current_scoring_features.parquet"
        try:
            build_current_scoring_dataset(
                valuations_path=valuations_path,
                mapping_path=mapping_path,
                ratings_path=ratings_root / f"competition={competition}",
                output_path=current_features,
                as_of_date=as_of_date,
                config=ValuationFeatureConfig(minimum_interval_minutes=1.0),
            )
        except ValuationFeatureError as exc:
            if "No current scoring rows could be constructed" not in str(exc):
                raise
            valuation_status = "skipped_no_post_valuation_matches"
        else:
            predictions_path = serving_root / "player_valuation_predictions.parquet"
            score_valuation_features(
                features_path=current_features,
                model_root=valuation_model_root,
                model_version="active",
                output_path=predictions_path,
            )
            valuation_status = "scored_active_model"
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="advanced",
            description="Materializing season",
            completed=4,
            total=6,
            current=f"valuation={valuation_status}",
            succeeded=4,
        )
    )

    serving = build_serving_tables(
        ratings_path=ratings.output_path,
        valuations_path=valuations_path,
        mapping_path=mapping_path,
        output_root=serving_root,
        predictions_path=predictions_path,
    )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="advanced",
            description="Materializing season",
            completed=5,
            total=6,
            current=f"serving players={serving.players}",
            succeeded=5,
        )
    )
    database = (
        load_pipeline_to_postgres(
            database_url=database_url,
            competition=competition,
            season=season,
            normalized_root=normalized_root,
            features_root=features_root,
            ratings_path=ratings.output_path,
            form_state_path=ratings.form_state_path,
            serving_root=serving_root,
        )
        if database_url
        else None
    )
    manifest_path = serving_root / "materialize_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "succeeded",
            "run_id": run_id,
            "competition": competition,
            "season": season,
            "as_of_date": as_of_date.isoformat(),
            "enrichment": {
                "selected": enrichment.selected,
                "processed": enrichment.processed,
                "skipped": enrichment.skipped,
                "failed": enrichment.failed,
            },
            "ratings": {
                "mode": ratings.mode,
                "processed_matches": ratings.processed_matches,
                "skipped_matches": ratings.skipped_matches,
                "rating_rows": ratings.rating_rows,
            },
            "mapping": asdict(mapping),
            "valuation_status": valuation_status,
            "predictions_path": str(predictions_path) if predictions_path else None,
            "serving": asdict(serving),
            "database_run_id": database.run_id if database else None,
            "progress_log": str(progress_path),
        },
    )
    emitter.emit(
        ProgressUpdate(
            stage="materialize",
            state="completed",
            description="Materializing season",
            completed=6,
            total=6,
            current=(
                f"database={database.run_id}" if database else "database not requested"
            ),
            succeeded=6,
        )
    )
    return MaterializeResult(
        enrichment=enrichment,
        ratings=ratings,
        mapping=mapping,
        serving=serving,
        valuation_status=valuation_status,
        predictions_path=predictions_path,
        database=database,
        manifest_path=manifest_path,
    )


def materialize_season(**kwargs: object) -> MaterializeResult:
    """Run materialization and always leave a truthful terminal manifest."""
    try:
        return _materialize_season(**kwargs)
    except Exception as exc:
        serving_root = Path(kwargs.get("serving_root", Path("data/serving")))
        failure_path = serving_root / "materialize_manifest.json"
        write_json(
            failure_path,
            {
                "status": "failed",
                "competition": kwargs.get("competition"),
                "season": kwargs.get("season"),
                "completed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
