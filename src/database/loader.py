from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import polars as pl

from ingestion.common import sha256_file


class DatabaseLoadError(RuntimeError):
    """Raised when an idempotent database load cannot complete."""


@dataclass(frozen=True)
class DatabaseLoadResult:
    run_id: str
    players: int
    matches: int
    feature_rows: int
    rating_rows: int
    valuation_rows: int
    estimate_rows: int
    impact_rows: int
    form_state_rows: int
    skipped_feature_matches: int


def _json(value: object) -> str:
    def clean(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        return item

    return json.dumps(clean(value), default=str, allow_nan=False)


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _chunks(rows: list[tuple[Any, ...]], size: int = 1_000) -> Iterable[list[tuple[Any, ...]]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def _executemany(connection: Any, query: str, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    with connection.cursor() as cursor:
        for chunk in _chunks(rows):
            cursor.executemany(query, chunk)


def _match_rows(normalized_root: Path, competition: str, season: str) -> list[dict[str, Any]]:
    source = normalized_root / f"competition={competition}" / f"season={season}" / "matches"
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("match_id=*/matches.parquet")):
        frame = pl.read_parquet(path)
        if not frame.is_empty():
            rows.append(frame.row(0, named=True))
    return rows


def _feature_records(features_root: Path, competition: str, season: str) -> list[dict[str, Any]]:
    source = features_root / f"competition={competition}" / f"season={season}" / "matches"
    records = []
    for partition in sorted(source.glob("match_id=*")):
        feature_path = partition / "player_match_features.parquet"
        success_path = partition / "_SUCCESS.json"
        if success_path.exists() and feature_path.exists():
            marker = json.loads(success_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "match_id": int(partition.name.split("=", 1)[1]),
                    "path": feature_path,
                    "source_hash": sha256_file(feature_path),
                    "feature_version": str(
                        marker.get("feature_version", "match_features_unknown")
                    ),
                }
            )
    return records


def _existing_partition_hashes(
    connection: Any, competition: str, season: str
) -> dict[int, str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT match_id, source_hash
            FROM processed_match_partitions
            WHERE competition = %s AND season = %s AND stage = 'features_to_postgres'
            """,
            (competition, season),
        )
        return {int(row[0]): str(row[1]) for row in cursor.fetchall()}


def load_pipeline_to_postgres(
    *,
    database_url: str,
    competition: str,
    season: str,
    normalized_root: Path = Path("data/normalized/whoscored"),
    features_root: Path = Path("data/features/whoscored"),
    ratings_path: Path,
    form_state_path: Path,
    serving_root: Path = Path("data/serving"),
) -> DatabaseLoadResult:
    try:
        import psycopg
    except ImportError as exc:
        raise DatabaseLoadError(
            "PostgreSQL loading requires psycopg. Run `uv sync` with the project dependencies."
        ) from exc

    run_id = f"db-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    started = datetime.now(UTC)
    connection = psycopg.connect(database_url)
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO pipeline_runs (run_id, pipeline, status, started_at, counts)
                    VALUES (%s, 'postgres_incremental_load', 'running', %s, '{}'::jsonb)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (run_id, started),
                )
    except Exception as exc:
        connection.close()
        raise DatabaseLoadError(
            "Database schema is unavailable; run the schema initializer first: "
            f"{exc}"
        ) from exc
    try:
        with connection.transaction():
            players_frame = pl.read_parquet(serving_root / "players.parquet")
            ratings = pl.read_parquet(ratings_path)
            player_records: dict[int, tuple[Any, ...]] = {
                int(row["whoscored_player_id"]): (
                    int(row["whoscored_player_id"]),
                    str(row.get("player_name") or f"Player {row['whoscored_player_id']}"),
                    str(row.get("position_group") or "Unknown"),
                )
                for row in ratings.sort("match_datetime").to_dicts()
            }
            for row in players_frame.to_dicts():
                player_records[int(row["player_id"])] = (
                    int(row["player_id"]),
                    str(row["display_name"]),
                    str(row.get("position") or "Unknown"),
                )
            player_rows = list(player_records.values())
            _executemany(
                connection,
                """
                INSERT INTO players (player_id, display_name, position_group)
                VALUES (%s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET
                  display_name = EXCLUDED.display_name,
                  position_group = EXCLUDED.position_group
                """,
                player_rows,
            )
            source_rows: list[tuple[Any, ...]] = [
                (player_id, "whoscored", str(player_id), "source_id", 1.0)
                for player_id in sorted(player_records)
            ]
            for row in players_frame.to_dicts():
                player_id = int(row["player_id"])
                source_rows.append(
                    (
                        player_id,
                        "transfermarkt",
                        str(row["transfermarkt_player_id"]),
                        str(row.get("mapping_method") or "entity_resolution"),
                        float(row.get("mapping_confidence") or 0.95),
                    )
                )
            _executemany(
                connection,
                """
                INSERT INTO player_source_ids
                  (player_id, source, source_player_id, match_method, confidence)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source, source_player_id) DO UPDATE SET
                  player_id = EXCLUDED.player_id,
                  match_method = EXCLUDED.match_method,
                  confidence = EXCLUDED.confidence
                """,
                source_rows,
            )

            matches = _match_rows(normalized_root, competition, season)
            match_rows = [
                (
                    int(row["match_id"]),
                    competition,
                    season,
                    row.get("start_date"),
                    row.get("home_team_id"),
                    row.get("away_team_id"),
                    row.get("source_url"),
                )
                for row in matches
            ]
            _executemany(
                connection,
                """
                INSERT INTO matches
                  (match_id, competition, season, kickoff_at, home_team_id, away_team_id, source_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE SET
                  kickoff_at = EXCLUDED.kickoff_at,
                  home_team_id = EXCLUDED.home_team_id,
                  away_team_id = EXCLUDED.away_team_id,
                  source_url = EXCLUDED.source_url
                """,
                match_rows,
            )

            previous_hashes = _existing_partition_hashes(connection, competition, season)
            feature_records = _feature_records(features_root, competition, season)
            changed_records = [
                record
                for record in feature_records
                if previous_hashes.get(int(record["match_id"])) != record["source_hash"]
            ]
            feature_rows: list[tuple[Any, ...]] = []
            partition_rows: list[tuple[Any, ...]] = []
            processed_at = datetime.now(UTC)
            for record in changed_records:
                frame = pl.read_parquet(record["path"])
                feature_version = str(record["feature_version"])
                for row in frame.to_dicts():
                    feature_rows.append(
                        (
                            int(row["whoscored_player_id"]),
                            int(row["match_id"]),
                            feature_version,
                            row.get("match_datetime"),
                            row.get("position_group"),
                            row.get("minutes"),
                            _json(row),
                            record["source_hash"],
                        )
                    )
                partition_rows.append(
                    (
                        competition,
                        season,
                        int(record["match_id"]),
                        "features_to_postgres",
                        record["source_hash"],
                        feature_version,
                        processed_at,
                    )
                )
            _executemany(
                connection,
                """
                INSERT INTO player_match_features
                  (player_id, match_id, feature_version, match_datetime, position_group,
                   minutes, features, source_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (player_id, match_id, feature_version) DO UPDATE SET
                  match_datetime = EXCLUDED.match_datetime,
                  position_group = EXCLUDED.position_group,
                  minutes = EXCLUDED.minutes,
                  features = EXCLUDED.features,
                  source_hash = EXCLUDED.source_hash,
                  updated_at = NOW()
                """,
                feature_rows,
            )
            _executemany(
                connection,
                """
                INSERT INTO processed_match_partitions
                  (competition, season, match_id, stage, source_hash, model_version, processed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (competition, season, match_id, stage) DO UPDATE SET
                  source_hash = EXCLUDED.source_hash,
                  model_version = EXCLUDED.model_version,
                  processed_at = EXCLUDED.processed_at
                """,
                partition_rows,
            )

            rating_rows = [
                (
                    int(row["whoscored_player_id"]),
                    int(row["match_id"]),
                    str(row["rating_version"]),
                    row.get("post_match_rating"),
                    row.get("minutes"),
                    _json(row),
                )
                for row in ratings.to_dicts()
            ]
            _executemany(
                connection,
                """
                INSERT INTO player_match_ratings
                  (player_id, match_id, rating_version, rating, minutes, features)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (player_id, match_id, rating_version) DO UPDATE SET
                  rating = EXCLUDED.rating,
                  minutes = EXCLUDED.minutes,
                  features = EXCLUDED.features
                """,
                rating_rows,
            )

            valuations = pl.read_parquet(serving_root / "valuation_history.parquet")
            valuation_rows = [
                (
                    int(row["player_id"]),
                    row["valuation_date"],
                    int(row["value_eur"]),
                    str(row["source"]),
                )
                for row in valuations.to_dicts()
            ]
            _executemany(
                connection,
                """
                INSERT INTO market_valuations (player_id, valuation_date, value_eur, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (player_id, valuation_date, source) DO UPDATE SET
                  value_eur = EXCLUDED.value_eur
                """,
                valuation_rows,
            )

            estimate_rows = []
            for row in players_frame.to_dicts():
                if row.get("estimated_value_eur") is None:
                    continue
                estimate_rows.append(
                    (
                        int(row["player_id"]),
                        row.get("refreshed_at") or processed_at,
                        str(row.get("valuation_model_version") or "active"),
                        int(row["estimated_value_eur"]),
                        _optional_int(row.get("estimated_lower_eur")),
                        _optional_int(row.get("estimated_upper_eur")),
                        row.get("direction"),
                        row.get("confidence"),
                        row.get("predicted_pct_change"),
                        row.get("probability_value_increase"),
                    )
                )
            _executemany(
                connection,
                """
                INSERT INTO valuation_estimates
                  (player_id, scored_at, model_version, estimate_eur, lower_eur, upper_eur,
                   direction, confidence, predicted_pct_change, probability_value_increase)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, model_version) DO UPDATE SET
                  scored_at = EXCLUDED.scored_at,
                  estimate_eur = EXCLUDED.estimate_eur,
                  lower_eur = EXCLUDED.lower_eur,
                  upper_eur = EXCLUDED.upper_eur,
                  direction = EXCLUDED.direction,
                  confidence = EXCLUDED.confidence,
                  predicted_pct_change = EXCLUDED.predicted_pct_change,
                  probability_value_increase = EXCLUDED.probability_value_increase
                """,
                estimate_rows,
            )

            impacts_path = serving_root / "match_impacts.parquet"
            impact_rows = []
            if impacts_path.exists():
                for row in pl.read_parquet(impacts_path).to_dicts():
                    if not row.get("replay_run_id"):
                        continue
                    impact_rows.append(
                        (
                            int(row["player_id"]),
                            int(row["match_id"]),
                            str(row["replay_run_id"]),
                            row.get("replay_sequence"),
                            _optional_int(row.get("valuation_estimate_eur")),
                            _optional_int(row.get("valuation_lower_90_eur")),
                            _optional_int(row.get("valuation_upper_90_eur")),
                            _optional_int(row.get("estimated_value_delta_eur")),
                            row.get("probability_value_increase"),
                            str(row.get("valuation_update_status") or "unknown"),
                        )
                    )
            _executemany(
                connection,
                """
                INSERT INTO valuation_match_impacts
                  (player_id, match_id, replay_run_id, replay_sequence, estimate_eur,
                   lower_eur, upper_eur, estimated_value_delta_eur,
                   probability_value_increase, valuation_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, match_id, replay_run_id) DO UPDATE SET
                  replay_sequence = EXCLUDED.replay_sequence,
                  estimate_eur = EXCLUDED.estimate_eur,
                  lower_eur = EXCLUDED.lower_eur,
                  upper_eur = EXCLUDED.upper_eur,
                  estimated_value_delta_eur = EXCLUDED.estimated_value_delta_eur,
                  probability_value_increase = EXCLUDED.probability_value_increase,
                  valuation_status = EXCLUDED.valuation_status,
                  scored_at = NOW()
                """,
                impact_rows,
            )

            form_state = pl.read_parquet(form_state_path)
            rating_version = str(ratings["rating_version"][0]) if ratings.height else "unknown"
            state_rows = [
                (
                    int(row["whoscored_player_id"]),
                    rating_version,
                    row.get("last_match_datetime"),
                    row["ewm_numerator"],
                    row["ewm_denominator"],
                    row.get("form_rating_ewm"),
                    row.get("rolling_3_match_rating"),
                    row.get("rolling_20_match_rating"),
                    row["rolling_3_history_json"],
                    row["rolling_20_history_json"],
                    row["updated_at"],
                )
                for row in form_state.to_dicts()
            ]
            _executemany(
                connection,
                """
                INSERT INTO player_form_state
                  (player_id, rating_version, last_match_datetime, ewm_numerator,
                   ewm_denominator, form_rating_ewm, rolling_3_match_rating,
                   rolling_20_match_rating, rolling_3_history, rolling_20_history, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (player_id, rating_version) DO UPDATE SET
                  last_match_datetime = EXCLUDED.last_match_datetime,
                  ewm_numerator = EXCLUDED.ewm_numerator,
                  ewm_denominator = EXCLUDED.ewm_denominator,
                  form_rating_ewm = EXCLUDED.form_rating_ewm,
                  rolling_3_match_rating = EXCLUDED.rolling_3_match_rating,
                  rolling_20_match_rating = EXCLUDED.rolling_20_match_rating,
                  rolling_3_history = EXCLUDED.rolling_3_history,
                  rolling_20_history = EXCLUDED.rolling_20_history,
                  updated_at = EXCLUDED.updated_at
                """,
                state_rows,
            )

            counts = {
                "players": len(player_rows),
                "matches": len(match_rows),
                "feature_rows": len(feature_rows),
                "rating_rows": len(rating_rows),
                "valuation_rows": len(valuation_rows),
                "estimate_rows": len(estimate_rows),
                "impact_rows": len(impact_rows),
                "form_state_rows": len(state_rows),
                "skipped_feature_matches": len(feature_records) - len(changed_records),
            }
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'succeeded', completed_at = %s, counts = %s::jsonb
                    WHERE run_id = %s
                    """,
                    (datetime.now(UTC), _json(counts), run_id),
                )
        return DatabaseLoadResult(run_id=run_id, **counts)
    except Exception as exc:
        connection.rollback()
        try:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE pipeline_runs
                        SET status = 'failed', completed_at = %s, error = %s
                        WHERE run_id = %s
                        """,
                        (datetime.now(UTC), f"{type(exc).__name__}: {exc}", run_id),
                    )
        except Exception:
            connection.rollback()
        raise DatabaseLoadError(str(exc)) from exc
    finally:
        connection.close()
