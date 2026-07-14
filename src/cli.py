from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import typer

from ingestion.progress import TerminalProgress

from ingestion.transfermarkt.normalize import (
    TransfermarktNormalizationError,
    normalize_transfermarkt_run,
)
from ingestion.transfermarkt.runner import (
    TransfermarktIngestionError,
    ingest_transfermarkt,
)
from ingestion.whoscored.discovery import discovered_matches_as_dicts
from ingestion.whoscored.competitions import load_competitions, resolve_competition
from ingestion.whoscored.runner import (
    WhoScoredIngestionError,
    ingest_whoscored,
    normalize_saved_match,
)
from ratings.model import RatingModelConfig
from ratings.pipeline import (
    RatingPipelineError,
    fit_and_score_rating_season,
    update_rating_season,
)
from valuation.features import (
    ValuationFeatureConfig,
    build_current_scoring_dataset,
    build_valuation_model_dataset,
)
from valuation.artifacts import resolve_model_directory
from valuation.scoring import score_valuation_features


app = typer.Typer(no_args_is_help=True, help="Market Value Pulse CLI.")
transfermarkt_app = typer.Typer(no_args_is_help=True, help="Transfermarkt data workflows.")
whoscored_app = typer.Typer(no_args_is_help=True, help="WhoScored data workflows.")
ratings_app = typer.Typer(no_args_is_help=True, help="Player match-rating workflows.")
model_app = typer.Typer(no_args_is_help=True, help="Valuation model workflows.")
pipeline_app = typer.Typer(no_args_is_help=True, help="End-to-end workflows.")
enrichment_app = typer.Typer(no_args_is_help=True, help="Match feature enrichment workflows.")
entities_app = typer.Typer(no_args_is_help=True, help="Cross-source entity resolution.")
serving_app = typer.Typer(no_args_is_help=True, help="API serving-table workflows.")
database_app = typer.Typer(no_args_is_help=True, help="PostgreSQL loading workflows.")

app.add_typer(transfermarkt_app, name="transfermarkt")
app.add_typer(whoscored_app, name="whoscored")
app.add_typer(ratings_app, name="ratings")
app.add_typer(model_app, name="model")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(enrichment_app, name="enrichment")
app.add_typer(entities_app, name="entities")
app.add_typer(serving_app, name="serving")
app.add_typer(database_app, name="database")


def _latest_valuation_path(root: Path = Path("data/normalized/transfermarkt")) -> Path:
    candidates = sorted(root.glob("competition=*/season=*/run_id=*/player_valuations.parquet"))
    if not candidates:
        raise typer.BadParameter(
            f"No normalized player_valuations.parquet found under {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_valuation_path(path: Path | None) -> Path:
    return path if path is not None else _latest_valuation_path()


def _latest_transfermarkt_players_path(
    root: Path = Path("data/normalized/transfermarkt"),
) -> Path:
    candidates = sorted(root.glob("competition=*/season=*/run_id=*/players.parquet"))
    if not candidates:
        raise typer.BadParameter(f"No normalized players.parquet found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _season_rating_artifact(root: Path, competition: str, season: str) -> Path:
    expected_competition = f"competition={competition}"
    expected_season = f"season={season}"
    if root.name == expected_season and root.parent.name == expected_competition:
        return root
    return root / expected_competition / expected_season


@whoscored_app.command("competitions")
def whoscored_competitions(
    registry: Path = typer.Option(Path("config/whoscored/competitions.json")),
) -> None:
    """List competition keys and source IDs accepted by automatic discovery."""
    try:
        rows = load_competitions(registry)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for row in rows:
        typer.echo(
            f"{row.key:<12} tournament={row.tournament_id:<3} "
            f"region={row.region_id:<3} {row.name}"
        )


@whoscored_app.command("discover-page")
def whoscored_discover_page(
    page: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    source_url: str = typer.Option(...),
    output: Path = typer.Option(Path("data/input/whoscored_matches.csv")),
) -> None:
    """Extract match URLs from a saved WhoScored fixture/season page."""
    import polars as pl

    rows = discovered_matches_as_dicts(page.read_text(encoding="utf-8"), source_url)
    if not rows:
        raise typer.BadParameter("No match links were found in the supplied page")
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output)
    typer.secho(f"Discovered {len(rows)} matches.", fg=typer.colors.GREEN)
    typer.echo(f"Manifest: {output}")


@whoscored_app.command("normalize-page")
def whoscored_normalize_page(
    page: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    source_url: str = typer.Option(...),
    competition: str = typer.Option(...),
    season: str = typer.Option(...),
    output_root: Path = typer.Option(Path("data/normalized/whoscored")),
) -> None:
    """Normalize one saved match page without making a network request."""
    try:
        success = normalize_saved_match(
            page_path=page,
            source_url=source_url,
            normalized_root=output_root,
            competition=competition,
            season=season,
        )
    except WhoScoredIngestionError as exc:
        typer.secho(f"WhoScored normalization failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("WhoScored normalization completed.", fg=typer.colors.GREEN)
    typer.echo(f"Success marker: {success}")


@whoscored_app.command("ingest")
def whoscored_ingest(
    competition: str = typer.Option(..., help="Configured key or alias, e.g. EPL."),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    manifest: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional targeted manifest; omitted for automatic discovery.",
    ),
    registry: Path = typer.Option(
        Path("config/whoscored/competitions.json"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    raw_root: Path = typer.Option(Path("data/raw/whoscored"), file_okay=False),
    normalized_root: Path = typer.Option(
        Path("data/normalized/whoscored"), file_okay=False
    ),
    workers: int = typer.Option(2, min=1, max=4),
    timeout_ms: int = typer.Option(45_000, min=5_000, max=180_000),
    delay_ms: int = typer.Option(1_000, min=0),
    max_retries: int = typer.Option(2, min=0, max=8),
    discovery_wait_seconds: float = typer.Option(1.0, min=0.0, max=10.0),
    max_previous_windows: int = typer.Option(60, min=0, max=120),
    max_next_windows: int = typer.Option(12, min=0, max=60),
    max_new_matches: int | None = typer.Option(
        None,
        "--max-new-matches",
        "--limit",
        min=1,
        help="Fetch at most the newest N unprocessed matches; --limit is an alias.",
    ),
    force: bool = typer.Option(False),
    headful: bool = typer.Option(False),
    no_progress: bool = typer.Option(False, help="Disable live terminal progress."),
) -> None:
    """Discover, fetch, validate, and incrementally normalize one league-season."""
    try:
        with TerminalProgress(enabled=not no_progress) as terminal_progress:
            result = ingest_whoscored(
                manifest_path=manifest,
                competition=competition,
                season=season,
                raw_root=raw_root,
                normalized_root=normalized_root,
                registry_path=registry,
                workers=workers,
                timeout_ms=timeout_ms,
                delay_ms=delay_ms,
                max_retries=max_retries,
                discovery_wait_seconds=discovery_wait_seconds,
                max_previous_windows=max_previous_windows,
                max_next_windows=max_next_windows,
                max_new_matches=max_new_matches,
                force=force,
                headful=headful,
                progress=terminal_progress,
            )
    except WhoScoredIngestionError as exc:
        typer.secho(f"WhoScored ingestion failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    color = typer.colors.GREEN if result.failed == 0 else typer.colors.YELLOW
    typer.secho("WhoScored ingestion completed.", fg=color)
    typer.echo(
        f"Discovered: {result.discovered}; selected: {result.requested}; "
        f"processed: {result.processed}; "
        f"skipped: {result.skipped}; deferred: {result.deferred}; failed: {result.failed}"
    )
    typer.echo(f"Run directory: {result.run_directory}")
    typer.echo(f"Manifest: {result.manifest_path}")


@transfermarkt_app.command("ingest")
def transfermarkt_ingest(
    league_config: Path = typer.Option(
        Path("config/leagues/GB1.json"), exists=True, file_okay=True, dir_okay=False
    ),
    season: int = typer.Option(..., min=1900, max=2200),
    output_root: Path = typer.Option(Path("data/raw/transfermarkt"), file_okay=False),
    concurrency: int = typer.Option(3, min=1, max=20),
    requests_per_minute: int = typer.Option(30, min=1, max=600),
    timeout_seconds: float = typer.Option(30.0, min=1.0, max=180.0),
    max_retries: int = typer.Option(4, min=0, max=10),
    skip_valuations: bool = typer.Option(False),
    no_progress: bool = typer.Option(False, help="Disable live terminal progress."),
) -> None:
    """Collect club rosters and raw valuation histories."""
    try:
        with TerminalProgress(enabled=not no_progress) as terminal_progress:
            result = ingest_transfermarkt(
                league_config_path=league_config,
                season=season,
                output_root=output_root,
                concurrency=concurrency,
                requests_per_minute=requests_per_minute,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                fetch_valuations=not skip_valuations,
                progress=terminal_progress,
            )
    except TransfermarktIngestionError as exc:
        typer.secho(f"Transfermarkt acquisition failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    color = typer.colors.GREEN if result.status == "succeeded" else typer.colors.YELLOW
    typer.secho(f"Transfermarkt acquisition {result.status}.", fg=color)
    typer.echo(f"Run directory: {result.run_directory}")
    typer.echo(f"Clubs discovered: {result.club_count}")
    typer.echo(f"Unique players discovered: {result.player_count}")
    typer.echo(f"Valuation responses: {result.valuation_count}")
    typer.echo(f"Failed requests: {result.failure_count}")
    typer.echo(f"Manifest: {result.manifest_path}")


@transfermarkt_app.command("normalize")
def transfermarkt_normalize(
    run_directory: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    output_root: Path = typer.Option(
        Path("data/normalized/transfermarkt"), file_okay=False
    ),
) -> None:
    """Flatten one completed raw run into normalized Parquet datasets."""
    try:
        result = normalize_transfermarkt_run(
            run_directory=run_directory,
            output_root=output_root,
        )
    except TransfermarktNormalizationError as exc:
        typer.secho(f"Transfermarkt normalization failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho("Transfermarkt normalization completed.", fg=typer.colors.GREEN)
    typer.echo(f"Output directory: {result.output_directory}")
    typer.echo(f"Players: {result.player_count}")
    typer.echo(f"Raw valuation records: {result.raw_valuation_count}")
    typer.echo(f"Normalized valuation records: {result.valuation_count}")
    typer.echo(f"Model-valid records: {result.model_valid_count}")
    typer.echo(f"Data-quality issues: {result.issue_count}")
    typer.echo(f"Summary: {result.summary_path}")


@entities_app.command("build-player-mapping")
def entities_build_player_mapping(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="WhoScored season label, e.g. 2025-2026."),
    transfermarkt_players: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    whoscored_normalized_root: Path = typer.Option(
        Path("data/normalized/whoscored"), file_okay=False
    ),
    manual_overrides: Path | None = typer.Option(None, exists=True, file_okay=True),
    output: Path = typer.Option(
        Path("data/normalized/entity_resolution/player_mapping_exact.parquet")
    ),
) -> None:
    """Build safe exact-name mappings and isolate ambiguous players for review."""
    from entity_resolution.player_mapping import PlayerMappingError, build_player_mapping

    try:
        result = build_player_mapping(
            transfermarkt_players_path=(
                transfermarkt_players or _latest_transfermarkt_players_path()
            ),
            whoscored_normalized_root=whoscored_normalized_root,
            competition=competition,
            season=season,
            output_path=output,
            manual_overrides=manual_overrides,
        )
    except (PlayerMappingError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Player mapping failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    color = typer.colors.GREEN if result.review_players == 0 else typer.colors.YELLOW
    typer.secho("Player mapping completed.", fg=color)
    typer.echo(f"Mapped: {result.mapped_players}; needs review: {result.review_players}")
    typer.echo(f"Mapping: {result.output_path}")
    typer.echo(f"Review queue: {result.review_path}")


@serving_app.command("build")
def serving_build(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    ratings: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    valuations: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    mapping: Path = typer.Option(
        Path("data/normalized/entity_resolution/player_mapping_exact.parquet"),
        exists=True,
        file_okay=True,
    ),
    predictions: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    output_root: Path = typer.Option(Path("data/serving"), file_okay=False),
) -> None:
    """Build the canonical Parquet tables consumed by the API/database loader."""
    from serving.builder import ServingBuildError, build_serving_tables

    ratings_path = ratings or (
        Path("data/features/ratings")
        / f"competition={competition}"
        / f"season={season}"
        / "player_match_ratings.parquet"
    )
    try:
        result = build_serving_tables(
            ratings_path=ratings_path,
            valuations_path=_resolve_valuation_path(valuations),
            mapping_path=mapping,
            output_root=output_root,
            predictions_path=predictions,
        )
    except (ServingBuildError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Serving build failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("Serving tables completed.", fg=typer.colors.GREEN)
    typer.echo(f"Players: {result.players}")
    typer.echo(f"Valuation rows: {result.valuation_rows}")
    typer.echo(f"Match impacts: {result.match_impact_rows}")
    typer.echo(f"Output: {result.output_root}")


@database_app.command("load")
def database_load(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    database_url: str = typer.Option(
        os.environ.get(
            "DATABASE_URL", "postgresql://mvp:mvp@localhost:5432/market_value_pulse"
        )
    ),
    normalized_root: Path = typer.Option(Path("data/normalized/whoscored"), file_okay=False),
    features_root: Path = typer.Option(Path("data/features/whoscored"), file_okay=False),
    ratings: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    form_state: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    serving_root: Path = typer.Option(Path("data/serving"), file_okay=False),
) -> None:
    """Idempotently upsert normalized matches, features, ratings, form, and values."""
    from database.loader import DatabaseLoadError, load_pipeline_to_postgres

    rating_directory = (
        Path("data/features/ratings")
        / f"competition={competition}"
        / f"season={season}"
    )
    state_directory = (
        Path("data/state/ratings")
        / f"competition={competition}"
        / f"season={season}"
    )
    try:
        result = load_pipeline_to_postgres(
            database_url=database_url,
            competition=competition,
            season=season,
            normalized_root=normalized_root,
            features_root=features_root,
            ratings_path=ratings or rating_directory / "player_match_ratings.parquet",
            form_state_path=form_state or state_directory / "player_form_state.parquet",
            serving_root=serving_root,
        )
    except (DatabaseLoadError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Database load failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("PostgreSQL load completed.", fg=typer.colors.GREEN)
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(
        f"Players: {result.players}; matches: {result.matches}; "
        f"features: {result.feature_rows}; ratings: {result.rating_rows}"
    )
    typer.echo(
        f"Valuations: {result.valuation_rows}; estimates: {result.estimate_rows}; "
        f"replay impacts: {result.impact_rows}; form states: {result.form_state_rows}"
    )
    typer.echo(f"Unchanged feature matches skipped: {result.skipped_feature_matches}")


@enrichment_app.command("score-season")
def enrichment_score_season(
    competition: str = typer.Option(..., help="Configured competition, e.g. EPL."),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    registry: Path = typer.Option(Path("config/whoscored/competitions.json"), exists=True),
    normalized_root: Path = typer.Option(Path("data/normalized/whoscored"), file_okay=False),
    output_root: Path = typer.Option(Path("data/features/whoscored"), file_okay=False),
    models_root: Path = typer.Option(Path("models/features"), file_okay=False),
    max_matches: int | None = typer.Option(None, min=1),
    prepare_only: bool = typer.Option(False, help="Build compatibility inputs without model inference."),
    force: bool = typer.Option(False),
    no_progress: bool = typer.Option(False),
) -> None:
    """Incrementally enrich normalized matches with versioned football features."""
    from features.pipeline import FeatureEnrichmentError, enrich_season

    try:
        configured = resolve_competition(competition, registry_path=registry)
        with TerminalProgress(enabled=not no_progress) as terminal_progress:
            result = enrich_season(
                competition=configured.key,
                competition_id=configured.tournament_id,
                season=season,
                normalized_root=normalized_root,
                output_root=output_root,
                models_root=models_root,
                max_matches=max_matches,
                prepare_only=prepare_only,
                force=force,
                progress=terminal_progress,
            )
    except (FeatureEnrichmentError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Enrichment failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    color = typer.colors.GREEN if result.failed == 0 else typer.colors.YELLOW
    typer.secho("Feature enrichment completed.", fg=color)
    typer.echo(
        f"Selected: {result.selected}; processed: {result.processed}; "
        f"skipped: {result.skipped}; failed: {result.failed}"
    )
    typer.echo(f"Run directory: {result.run_directory}")


@pipeline_app.command("replay")
def pipeline_replay(
    competition: str = typer.Option(..., help="Configured competition, e.g. EPL."),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    matches: int = typer.Option(8, min=1, max=50),
    player_id: int | None = typer.Option(None, help="Replay this player's latest appearances."),
    registry: Path = typer.Option(Path("config/whoscored/competitions.json"), exists=True),
    normalized_root: Path = typer.Option(Path("data/normalized/whoscored"), file_okay=False),
    replay_root: Path = typer.Option(Path("data/replays"), file_okay=False),
    models_root: Path = typer.Option(Path("models/features"), file_okay=False),
    rating_artifact_directory: Path = typer.Option(
        Path("models/ratings/post_match_v2"), file_okay=False
    ),
    valuations: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    mapping: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    valuation_model_root: Path = typer.Option(
        Path("data/modeling/valuation_model"), file_okay=False
    ),
    valuation_model_version: str = typer.Option("active"),
    serving_root: Path = typer.Option(Path("data/serving"), file_okay=False),
    prepare_only: bool = typer.Option(False, help="Replay the real data contract without model artifacts."),
    no_progress: bool = typer.Option(False),
) -> None:
    """Replay real completed matches chronologically without changing source data."""
    from pipelines.replay import ReplayError, run_historical_replay

    try:
        configured = resolve_competition(competition, registry_path=registry)
        with TerminalProgress(enabled=not no_progress) as terminal_progress:
            result = run_historical_replay(
                competition=configured.key,
                competition_id=configured.tournament_id,
                season=season,
                match_count=matches,
                player_id=player_id,
                normalized_root=normalized_root,
                replay_root=replay_root,
                models_root=models_root,
                rating_artifact_directory=_season_rating_artifact(
                    rating_artifact_directory, configured.key, season
                ),
                valuations_path=valuations,
                mapping_path=mapping,
                valuation_model_root=valuation_model_root,
                valuation_model_version=valuation_model_version,
                serving_root=serving_root,
                prepare_only=prepare_only,
                progress=terminal_progress,
            )
    except (ReplayError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Replay failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    color = typer.colors.GREEN if result.failed_matches == 0 else typer.colors.YELLOW
    typer.secho("Historical replay completed.", fg=color)
    typer.echo(
        f"Selected: {result.selected_matches}; completed: {result.completed_matches}; "
        f"failed: {result.failed_matches}"
    )
    typer.echo(f"Run directory: {result.run_directory}")


@pipeline_app.command("materialize")
def pipeline_materialize(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    as_of_date: str = typer.Option(date.today().isoformat()),
    registry: Path = typer.Option(Path("config/whoscored/competitions.json"), exists=True),
    transfermarkt_players: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    valuations: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    normalized_root: Path = typer.Option(Path("data/normalized/whoscored"), file_okay=False),
    features_root: Path = typer.Option(Path("data/features/whoscored"), file_okay=False),
    models_root: Path = typer.Option(Path("models/features"), file_okay=False),
    rating_artifact_directory: Path = typer.Option(
        Path("models/ratings/post_match_v2"), file_okay=False
    ),
    manual_mapping_overrides: Path | None = typer.Option(None, exists=True, file_okay=True),
    valuation_model_root: Path = typer.Option(
        Path("data/modeling/valuation_model"), file_okay=False
    ),
    serving_root: Path = typer.Option(Path("data/serving"), file_okay=False),
    load_database: bool = typer.Option(False, help="Upsert outputs into PostgreSQL."),
    database_url: str = typer.Option(
        os.environ.get(
            "DATABASE_URL", "postgresql://mvp:mvp@localhost:5432/market_value_pulse"
        )
    ),
    no_progress: bool = typer.Option(False),
) -> None:
    """Incrementally enrich, rate, map, score, serve, and optionally load PostgreSQL."""
    from pipelines.materialize import materialize_season

    configured = resolve_competition(competition, registry_path=registry)
    try:
        with TerminalProgress(enabled=not no_progress) as terminal_progress:
            result = materialize_season(
                competition=configured.key,
                competition_id=configured.tournament_id,
                season=season,
                as_of_date=date.fromisoformat(as_of_date),
                transfermarkt_players_path=(
                    transfermarkt_players or _latest_transfermarkt_players_path()
                ),
                valuations_path=_resolve_valuation_path(valuations),
                normalized_root=normalized_root,
                features_root=features_root,
                models_root=models_root,
                rating_artifact_directory=rating_artifact_directory,
                manual_mapping_overrides=manual_mapping_overrides,
                valuation_model_root=valuation_model_root,
                serving_root=serving_root,
                database_url=database_url if load_database else None,
                progress=terminal_progress,
            )
    except Exception as exc:
        typer.secho(f"Season materialization failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    color = typer.colors.GREEN if result.enrichment.failed == 0 else typer.colors.YELLOW
    typer.secho("Season materialization completed.", fg=color)
    typer.echo(
        f"Enrichment processed: {result.enrichment.processed}; "
        f"rating matches processed: {result.ratings.processed_matches}"
    )
    typer.echo(
        f"Mapped players: {result.mapping.mapped_players}; "
        f"review queue: {result.mapping.review_players}"
    )
    typer.echo(f"Valuation: {result.valuation_status}")
    typer.echo(f"Serving players: {result.serving.players}")
    typer.echo(f"Database run: {result.database.run_id if result.database else 'not requested'}")
    typer.echo(f"Manifest: {result.manifest_path}")


@ratings_app.command("build")
@ratings_app.command("fit")
def ratings_fit(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    features_root: Path = typer.Option(Path("data/features/whoscored"), file_okay=False),
    output_root: Path = typer.Option(Path("data/features/ratings"), file_okay=False),
    state_root: Path = typer.Option(Path("data/state/ratings"), file_okay=False),
    artifact_directory: Path = typer.Option(
        Path("models/ratings/post_match_v2"), file_okay=False
    ),
    half_life_days: float = typer.Option(90.0, min=1.0),
) -> None:
    """Fit versioned rating priors/statistics and score the historical season."""
    resolved_artifact = _season_rating_artifact(
        artifact_directory, competition, season
    )
    try:
        result = fit_and_score_rating_season(
            competition=competition,
            season=season,
            features_root=features_root,
            output_root=output_root,
            state_root=state_root,
            artifact_directory=resolved_artifact,
            config=RatingModelConfig(ewm_half_life_days=half_life_days),
        )
    except (RatingPipelineError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Rating fit failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("Historical player-match ratings completed.", fg=typer.colors.GREEN)
    typer.echo(f"Rating version: {result.rating_version}")
    typer.echo(f"Matches: {result.processed_matches}; rating rows: {result.rating_rows}")
    typer.echo(f"Ratings: {result.output_path}")
    typer.echo(f"Form state: {result.form_state_path}")
    typer.echo(f"Artifacts: {result.artifact_directory}")


@ratings_app.command("update")
def ratings_update(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    features_root: Path = typer.Option(Path("data/features/whoscored"), file_okay=False),
    output_root: Path = typer.Option(Path("data/features/ratings"), file_okay=False),
    state_root: Path = typer.Option(Path("data/state/ratings"), file_okay=False),
    artifact_directory: Path = typer.Option(
        Path("models/ratings/post_match_v2"), file_okay=False
    ),
) -> None:
    """Score only new or changed match partitions and refresh player form state."""
    resolved_artifact = _season_rating_artifact(
        artifact_directory, competition, season
    )
    try:
        result = update_rating_season(
            competition=competition,
            season=season,
            features_root=features_root,
            output_root=output_root,
            state_root=state_root,
            artifact_directory=resolved_artifact,
        )
    except (RatingPipelineError, ValueError, FileNotFoundError) as exc:
        typer.secho(f"Rating update failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho("Incremental player-match rating update completed.", fg=typer.colors.GREEN)
    typer.echo(
        f"Selected: {result.selected_matches}; processed: {result.processed_matches}; "
        f"skipped: {result.skipped_matches}"
    )
    typer.echo(f"Rating rows: {result.rating_rows}")
    typer.echo(f"Ratings: {result.output_path}")
    typer.echo(f"Form state: {result.form_state_path}")


@model_app.command("build-features")
def model_build_features(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    valuations: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    mapping: Path = typer.Option(
        Path("data/normalized/entity_resolution/player_mapping_exact.parquet"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    ratings: Path | None = typer.Option(
        None,
        file_okay=True,
        dir_okay=True,
        help=(
            "One rating parquet or a competition directory containing "
            "season=*/player_match_ratings.parquet partitions."
        ),
    ),
    output: Path = typer.Option(
        Path("data/modeling/valuation_model/valuation_model_dataset.parquet"),
        file_okay=True,
        dir_okay=False,
    ),
    minimum_minutes: float = typer.Option(180.0, min=1.0),
    ewm_half_life_days: float = typer.Option(90.0, min=1.0),
    rolling_short: int = typer.Option(3, min=1, max=10),
    rolling_long: int = typer.Option(20, min=5, max=100),
) -> None:
    """Build leakage-safe valuation intervals and rolling-form features."""
    result = build_valuation_model_dataset(
        valuations_path=_resolve_valuation_path(valuations),
        mapping_path=mapping,
        ratings_path=(
            ratings
            or Path("data/features/ratings") / f"competition={competition}"
        ),
        output_path=output,
        config=ValuationFeatureConfig(
            minimum_interval_minutes=minimum_minutes,
            ewm_half_life_days=ewm_half_life_days,
            rolling_short_matches=rolling_short,
            rolling_long_matches=rolling_long,
        ),
    )
    typer.secho("Valuation features completed.", fg=typer.colors.GREEN)
    typer.echo(f"Observations: {result.observations}")
    typer.echo(f"Players: {result.players}")
    typer.echo(f"Date range: {result.first_valuation_date} to {result.last_valuation_date}")
    typer.echo(f"Output: {result.output_path}")


@model_app.command("train")
def model_train(
    dataset: Path = typer.Option(
        Path("data/modeling/valuation_model/valuation_model_dataset.parquet"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_root: Path = typer.Option(
        Path("data/modeling/valuation_model"), file_okay=False
    ),
    test_fraction: float = typer.Option(0.20, min=0.05, max=0.50),
    num_warmup: int = typer.Option(1_000, min=100),
    num_samples: int = typer.Option(1_000, min=100),
    num_chains: int = typer.Option(2, min=1, max=4),
    target_accept: float = typer.Option(0.93, min=0.70, max=0.999),
    seed: int = typer.Option(42),
    promote: bool = typer.Option(True, help="Update active.json after a successful fit."),
) -> None:
    """Fit the hierarchical Bayesian model and OLS benchmark."""
    from pipelines.train import train_valuation_model
    from valuation.bayesian import BayesianFitConfig

    result = train_valuation_model(
        dataset_path=dataset,
        model_root=model_root,
        test_fraction=test_fraction,
        bayesian_config=BayesianFitConfig(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_probability=target_accept,
            random_seed=seed,
        ),
        promote=promote,
    )
    typer.secho("Valuation model training completed.", fg=typer.colors.GREEN)
    typer.echo(f"Model version: {result.model_version}")
    typer.echo(f"Selected prediction: {result.selected_prediction}")
    typer.echo(f"Holdout MAE: {result.selected_mae:.6f}")
    typer.echo(f"Artifacts: {result.artifact_directory}")
    typer.echo(f"Promoted: {result.promoted}")
    if promote and not result.promoted:
        checks_path = result.artifact_directory / "promotion_checks.json"
        failed_checks: list[str] = []
        if checks_path.exists():
            checks = json.loads(checks_path.read_text(encoding="utf-8")).get(
                "checks", {}
            )
            failed_checks = [name for name, passed in checks.items() if not passed]
        typer.secho(
            "Candidate was not promoted because one or more quality gates failed. "
            f"Inspect {result.artifact_directory / 'promotion_checks.json'}.",
            fg=typer.colors.YELLOW,
        )
        if failed_checks:
            typer.echo("Failed gates: " + ", ".join(failed_checks))
        typer.echo(
            "Candidate summary: uv run mvp model summary --model-version latest"
        )


@model_app.command("build-current-features")
def model_build_current_features(
    competition: str = typer.Option("EPL"),
    season: str = typer.Option(..., help="Season label, e.g. 2025-2026."),
    as_of_date: str = typer.Option(..., help="Scoring date in YYYY-MM-DD format."),
    valuations: Path | None = typer.Option(None, file_okay=True, dir_okay=False),
    mapping: Path = typer.Option(
        Path("data/normalized/entity_resolution/player_mapping_exact.parquet"),
        exists=True,
    ),
    ratings: Path | None = typer.Option(None, file_okay=True, dir_okay=True),
    minimum_minutes: float = typer.Option(
        1.0,
        min=1.0,
        help="Minimum post-valuation minutes required for live scoring.",
    ),
    output: Path = typer.Option(
        Path("data/modeling/valuation_model/current_scoring_features.parquet")
    ),
) -> None:
    """Build current player features from the latest valuation to an as-of date."""
    result = build_current_scoring_dataset(
        valuations_path=_resolve_valuation_path(valuations),
        mapping_path=mapping,
        ratings_path=(
            ratings
            or Path("data/features/ratings") / f"competition={competition}"
        ),
        output_path=output,
        as_of_date=date.fromisoformat(as_of_date),
        config=ValuationFeatureConfig(
            minimum_interval_minutes=minimum_minutes
        ),
    )
    typer.secho("Current scoring features completed.", fg=typer.colors.GREEN)
    typer.echo(f"Rows: {result.observations}")
    typer.echo(f"Output: {result.output_path}")


@model_app.command("score")
def model_score(
    features: Path = typer.Option(
        Path("data/modeling/valuation_model/current_scoring_features.parquet"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_root: Path = typer.Option(
        Path("data/modeling/valuation_model"), file_okay=False
    ),
    model_version: str = typer.Option("active"),
    output: Path = typer.Option(
        Path("data/serving/player_valuation_predictions.parquet")
    ),
) -> None:
    """Score current player features with a saved posterior."""
    result = score_valuation_features(
        features_path=features,
        model_root=model_root,
        model_version=model_version,
        output_path=output,
    )
    typer.secho("Player valuation scoring completed.", fg=typer.colors.GREEN)
    typer.echo(f"Rows: {result.rows}")
    typer.echo(f"Model: {result.model_directory}")
    typer.echo(f"Output: {result.output_path}")


@model_app.command("summary")
def model_summary(
    model_root: Path = typer.Option(
        Path("data/modeling/valuation_model"), file_okay=False
    ),
    model_version: str = typer.Option("active"),
) -> None:
    """Print the saved calibration and model-selection summary."""
    directory = resolve_model_directory(model_root, model_version)
    summary_path = directory / "model_summary.json"
    metrics_path = directory / "model_metrics.parquet"
    if not summary_path.exists():
        raise typer.BadParameter(f"Missing model summary: {summary_path}")
    typer.echo(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
    if metrics_path.exists():
        import polars as pl

        typer.echo("\nModel comparison:")
        typer.echo(str(pl.read_parquet(metrics_path)))

if __name__ == "__main__":
    app()
