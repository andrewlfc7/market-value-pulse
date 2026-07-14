from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ingestion.common import sha256_file, write_json
from ratings.model import (
    RatingModelConfig,
    RatingModelError,
    fit_rating_artifacts,
    load_rating_config,
    score_rating_features,
)


class RatingPipelineError(RuntimeError):
    """Raised when historical or incremental rating scoring cannot complete."""


@dataclass(frozen=True)
class RatingPipelineResult:
    mode: str
    rating_version: str
    selected_matches: int
    processed_matches: int
    skipped_matches: int
    rating_rows: int
    output_path: Path
    processed_matches_path: Path
    form_state_path: Path
    artifact_directory: Path


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _rating_directory(root: Path, competition: str, season: str) -> Path:
    return root / f"competition={competition}" / f"season={season}"


def _state_directory(root: Path, competition: str, season: str) -> Path:
    return root / f"competition={competition}" / f"season={season}"


def _feature_partitions(root: Path, competition: str, season: str) -> list[Path]:
    source = root / f"competition={competition}" / f"season={season}" / "matches"
    if not source.exists():
        raise RatingPipelineError(f"Feature season does not exist: {source}")
    partitions = [
        path
        for path in source.glob("match_id=*")
        if (path / "_SUCCESS.json").exists()
        and (path / "player_match_features.parquet").exists()
    ]
    if not partitions:
        raise RatingPipelineError(f"No completed player-match feature partitions under {source}")
    return sorted(partitions, key=lambda path: int(path.name.split("=", 1)[1]))


def _partition_records(partitions: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "match_id": int(path.name.split("=", 1)[1]),
            "feature_path": str(path / "player_match_features.parquet"),
            "feature_sha256": sha256_file(path / "player_match_features.parquet"),
        }
        for path in partitions
    ]


def _starter_position_priors(features: pl.DataFrame) -> pl.DataFrame:
    if "started" not in features.columns:
        return pl.DataFrame()
    return (
        features.filter(
            pl.col("started").cast(pl.Boolean, strict=False).fill_null(False)
            & pl.col("position_group").is_not_null()
            & (pl.col("position_group") != "Unknown")
        )
        .group_by(["season", "whoscored_player_id", "position_group"])
        .len(name="position_starts")
        .sort(
            ["season", "whoscored_player_id", "position_starts", "position_group"],
            descending=[False, False, True, False],
        )
        .group_by(["season", "whoscored_player_id"], maintain_order=True)
        .first()
        .select(
            "season",
            "whoscored_player_id",
            pl.col("position_group").alias("season_primary_position_group"),
            "position_starts",
        )
    )


def _career_position_priors(features: pl.DataFrame) -> pl.DataFrame:
    if "started" not in features.columns:
        return pl.DataFrame()
    return (
        features.filter(
            pl.col("started").cast(pl.Boolean, strict=False).fill_null(False)
            & pl.col("position_group").is_not_null()
            & (pl.col("position_group") != "Unknown")
        )
        .group_by(["whoscored_player_id", "position_group"])
        .len(name="career_position_starts")
        .sort(
            ["whoscored_player_id", "career_position_starts", "position_group"],
            descending=[False, True, False],
        )
        .group_by("whoscored_player_id", maintain_order=True)
        .first()
        .select(
            "whoscored_player_id",
            pl.col("position_group").alias("career_primary_position_group"),
            "career_position_starts",
        )
    )


def _competition_position_history(
    features_root: Path, competition: str
) -> pl.DataFrame:
    source = features_root / f"competition={competition}"
    paths = sorted(
        path
        for path in source.glob(
            "season=*/matches/match_id=*/player_match_features.parquet"
        )
        if (path.parent / "_SUCCESS.json").exists()
    )
    if not paths:
        return pl.DataFrame()
    columns = [
        "season",
        "whoscored_player_id",
        "position_group",
        "started",
    ]
    return pl.concat(
        [pl.read_parquet(path, columns=columns) for path in paths],
        how="diagonal_relaxed",
    )


def _apply_primary_positions(
    features: pl.DataFrame,
    primary_positions: pl.DataFrame,
    career_positions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if "started" not in features.columns:
        return features
    output = features
    if not primary_positions.is_empty():
        output = output.join(
            primary_positions.select(
                "season", "whoscored_player_id", "season_primary_position_group"
            ),
            on=["season", "whoscored_player_id"],
            how="left",
        )
    else:
        output = output.with_columns(
            pl.lit(None, dtype=pl.String).alias("season_primary_position_group")
        )
    if career_positions is not None and not career_positions.is_empty():
        output = output.join(
            career_positions.select(
                "whoscored_player_id", "career_primary_position_group"
            ),
            on="whoscored_player_id",
            how="left",
        )
    else:
        output = output.with_columns(
            pl.lit(None, dtype=pl.String).alias("career_primary_position_group")
        )
    return (
        output.with_columns(
            pl.when(
                ~pl.col("started")
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
            )
            .then(
                pl.coalesce(
                    "season_primary_position_group",
                    "career_primary_position_group",
                )
            )
            .otherwise(pl.col("position_group"))
            .alias("position_group")
        )
        .drop("season_primary_position_group", "career_primary_position_group")
    )


def _load_features(
    records: list[dict[str, Any]],
    *,
    primary_positions: pl.DataFrame | None = None,
    career_positions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    frames = [pl.read_parquet(str(record["feature_path"])) for record in records]
    if not frames:
        raise RatingPipelineError("No player-match features were selected")
    features = pl.concat(frames, how="diagonal_relaxed")
    if "started" not in features.columns:
        return features

    # WhoScored labels bench players as SUB. Match-level substitution links are
    # useful, but a like-for-like assumption is not always tactically correct.
    # Match the historical rating build by assigning used substitutes their
    # most frequent starting position for that season when one is available.
    resolved_positions = (
        primary_positions
        if primary_positions is not None
        else _starter_position_priors(features)
    )
    resolved_career = (
        career_positions
        if career_positions is not None
        else _career_position_priors(features)
    )
    return _apply_primary_positions(
        features, resolved_positions, resolved_career
    )


def resolve_rating_positions(
    features: pl.DataFrame, artifact_directory: Path
) -> pl.DataFrame:
    """Apply the fitted season/career substitute-position contract."""
    season_path = artifact_directory / "season_primary_positions.parquet"
    career_path = artifact_directory / "career_primary_positions.parquet"
    if not season_path.exists() or not career_path.exists():
        raise RatingPipelineError(
            "Rating position artifacts are missing. Run `mvp ratings fit` first."
        )
    return _apply_primary_positions(
        features,
        pl.read_parquet(season_path),
        pl.read_parquet(career_path),
    )


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RatingPipelineError(f"Invalid match_datetime: {value!r}") from exc
    return parsed.replace(tzinfo=None)


def _weighted_rating(history: list[tuple[float, float]]) -> float | None:
    denominator = sum(minutes for minutes, _ in history)
    if denominator <= 0:
        return None
    return sum(minutes * rating for minutes, rating in history) / denominator


def add_form_history(
    ratings: pl.DataFrame,
    *,
    half_life_days: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if ratings.is_empty():
        return ratings, pl.DataFrame()
    rows = sorted(
        ratings.to_dicts(),
        key=lambda row: (
            int(row["whoscored_player_id"]),
            _datetime(row["match_datetime"]),
            int(row["match_id"]),
        ),
    )
    state: dict[int, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for row in rows:
        player_id = int(row["whoscored_player_id"])
        current_date = _datetime(row["match_datetime"])
        player = state.setdefault(
            player_id,
            {
                "ewm_numerator": 0.0,
                "ewm_denominator": 0.0,
                "last_match_datetime": None,
                "recent": [],
                "player_name": row.get("player_name"),
                "position_group": row.get("position_group"),
            },
        )
        previous_date = player["last_match_datetime"]
        if previous_date is not None:
            elapsed_days = max((current_date - previous_date).total_seconds() / 86_400.0, 0.0)
            decay = math.exp(-math.log(2.0) * elapsed_days / half_life_days)
            player["ewm_numerator"] *= decay
            player["ewm_denominator"] *= decay

        rating_value = row.get("post_match_rating")
        minutes = max(float(row.get("minutes") or 0.0), 0.0)
        if rating_value is not None and minutes > 0:
            rating = float(rating_value)
            player["ewm_numerator"] += minutes * rating
            player["ewm_denominator"] += minutes
            player["recent"].append((minutes, rating, int(row["match_id"]), current_date.isoformat()))
            player["recent"] = player["recent"][-20:]
        player["last_match_datetime"] = current_date
        recent = player["recent"]
        row["form_rating_ewm"] = (
            player["ewm_numerator"] / player["ewm_denominator"]
            if player["ewm_denominator"] > 0
            else None
        )
        row["rolling_3_match_rating"] = _weighted_rating(
            [(item[0], item[1]) for item in recent[-3:]]
        )
        row["rolling_20_match_rating"] = _weighted_rating(
            [(item[0], item[1]) for item in recent]
        )
        output.append(row)

    state_rows = []
    for player_id, player in sorted(state.items()):
        recent = player["recent"]
        state_rows.append(
            {
                "whoscored_player_id": player_id,
                "player_name": player["player_name"],
                "position_group": player["position_group"],
                "last_match_datetime": player["last_match_datetime"],
                "ewm_numerator": player["ewm_numerator"],
                "ewm_denominator": player["ewm_denominator"],
                "form_rating_ewm": (
                    player["ewm_numerator"] / player["ewm_denominator"]
                    if player["ewm_denominator"] > 0
                    else None
                ),
                "rolling_3_match_rating": _weighted_rating(
                    [(item[0], item[1]) for item in recent[-3:]]
                ),
                "rolling_20_match_rating": _weighted_rating(
                    [(item[0], item[1]) for item in recent]
                ),
                "rolling_3_history_json": json.dumps(recent[-3:]),
                "rolling_20_history_json": json.dumps(recent),
                "updated_at": _utc_now().replace(tzinfo=None),
            }
        )
    output_frame = pl.DataFrame(output, infer_schema_length=None).sort(
        ["match_datetime", "match_id", "whoscored_player_id"]
    )
    state_frame = pl.DataFrame(state_rows, infer_schema_length=None).sort(
        "whoscored_player_id"
    )
    return output_frame, state_frame


def append_form_history(
    ratings: pl.DataFrame,
    existing_state: pl.DataFrame,
    *,
    half_life_days: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Advance persisted form state for chronological, append-only rating rows."""
    state_by_player = {
        int(row["whoscored_player_id"]): dict(row)
        for row in existing_state.to_dicts()
    }
    output: list[dict[str, Any]] = []
    rows = sorted(
        ratings.to_dicts(),
        key=lambda row: (
            _datetime(row["match_datetime"]),
            int(row["match_id"]),
            int(row["whoscored_player_id"]),
        ),
    )
    for row in rows:
        player_id = int(row["whoscored_player_id"])
        current_date = _datetime(row["match_datetime"])
        player = state_by_player.get(player_id)
        if player is None:
            player = {
                "whoscored_player_id": player_id,
                "player_name": row.get("player_name"),
                "position_group": row.get("position_group"),
                "last_match_datetime": None,
                "ewm_numerator": 0.0,
                "ewm_denominator": 0.0,
                "rolling_20_history_json": "[]",
            }
        previous_date = (
            _datetime(player["last_match_datetime"])
            if player.get("last_match_datetime") is not None
            else None
        )
        if previous_date is not None and current_date < previous_date:
            raise RatingPipelineError(
                "Append-only form update received an out-of-order rating for "
                f"player_id={player_id}"
            )
        numerator = float(player.get("ewm_numerator") or 0.0)
        denominator = float(player.get("ewm_denominator") or 0.0)
        if previous_date is not None:
            elapsed_days = max(
                (current_date - previous_date).total_seconds() / 86_400.0, 0.0
            )
            decay = math.exp(-math.log(2.0) * elapsed_days / half_life_days)
            numerator *= decay
            denominator *= decay
        history_value = player.get("rolling_20_history_json") or "[]"
        recent = (
            json.loads(history_value)
            if isinstance(history_value, str)
            else list(history_value)
        )
        rating_value = row.get("post_match_rating")
        minutes = max(float(row.get("minutes") or 0.0), 0.0)
        if rating_value is not None and minutes > 0:
            rating = float(rating_value)
            numerator += minutes * rating
            denominator += minutes
            recent.append(
                [minutes, rating, int(row["match_id"]), current_date.isoformat()]
            )
            recent = recent[-20:]
        form_rating = numerator / denominator if denominator > 0 else None
        rolling_3 = _weighted_rating(
            [(float(item[0]), float(item[1])) for item in recent[-3:]]
        )
        rolling_20 = _weighted_rating(
            [(float(item[0]), float(item[1])) for item in recent]
        )
        row.update(
            {
                "form_rating_ewm": form_rating,
                "rolling_3_match_rating": rolling_3,
                "rolling_20_match_rating": rolling_20,
            }
        )
        output.append(row)
        state_by_player[player_id] = {
            "whoscored_player_id": player_id,
            "player_name": row.get("player_name") or player.get("player_name"),
            "position_group": row.get("position_group") or player.get("position_group"),
            "last_match_datetime": current_date,
            "ewm_numerator": numerator,
            "ewm_denominator": denominator,
            "form_rating_ewm": form_rating,
            "rolling_3_match_rating": rolling_3,
            "rolling_20_match_rating": rolling_20,
            "rolling_3_history_json": json.dumps(recent[-3:]),
            "rolling_20_history_json": json.dumps(recent),
            "updated_at": _utc_now().replace(tzinfo=None),
        }
    output_frame = pl.DataFrame(output, infer_schema_length=None).sort(
        ["match_datetime", "match_id", "whoscored_player_id"]
    )
    state_frame = pl.DataFrame(
        list(state_by_player.values()), infer_schema_length=None
    ).sort("whoscored_player_id")
    return output_frame, state_frame


def _processed_frame(
    records: list[dict[str, Any]], rating_version: str
) -> pl.DataFrame:
    processed_at = _utc_now().replace(tzinfo=None)
    return pl.DataFrame(
        [
            {
                **record,
                "rating_version": rating_version,
                "processed_at": processed_at,
            }
            for record in records
        ],
        infer_schema_length=None,
    ).sort("match_id")


def _write_result(
    *,
    ratings: pl.DataFrame,
    form_state: pl.DataFrame,
    processed: pl.DataFrame,
    output_directory: Path,
    state_directory: Path,
    artifact_directory: Path,
    mode: str,
    selected_matches: int,
    processed_count: int,
    skipped_count: int,
) -> RatingPipelineResult:
    output_path = output_directory / "player_match_ratings.parquet"
    processed_path = state_directory / "processed_matches.parquet"
    form_state_path = state_directory / "player_form_state.parquet"
    _atomic_parquet(ratings, output_path)
    _atomic_parquet(processed, processed_path)
    _atomic_parquet(form_state, form_state_path)
    config = load_rating_config(artifact_directory)
    write_json(
        output_directory / "rating_run_summary.json",
        {
            "mode": mode,
            "rating_version": config.version,
            "selected_matches": selected_matches,
            "processed_matches": processed_count,
            "skipped_matches": skipped_count,
            "rating_rows": ratings.height,
            "rated_rows": int(ratings["post_match_rating"].is_not_null().sum()),
            "artifact_directory": str(artifact_directory),
            "output_path": str(output_path),
            "processed_matches_path": str(processed_path),
            "form_state_path": str(form_state_path),
            "completed_at": _utc_now().isoformat(),
        },
    )
    return RatingPipelineResult(
        mode=mode,
        rating_version=config.version,
        selected_matches=selected_matches,
        processed_matches=processed_count,
        skipped_matches=skipped_count,
        rating_rows=ratings.height,
        output_path=output_path,
        processed_matches_path=processed_path,
        form_state_path=form_state_path,
        artifact_directory=artifact_directory,
    )


def fit_and_score_rating_season(
    *,
    competition: str,
    season: str,
    features_root: Path = Path("data/features/whoscored"),
    output_root: Path = Path("data/features/ratings"),
    state_root: Path = Path("data/state/ratings"),
    artifact_directory: Path = Path("models/ratings/post_match_v2"),
    config: RatingModelConfig | None = None,
) -> RatingPipelineResult:
    config = config or RatingModelConfig()
    partitions = _feature_partitions(features_root, competition, season)
    records = _partition_records(partitions)
    position_history = _competition_position_history(features_root, competition)
    primary_positions = _starter_position_priors(position_history)
    career_positions = _career_position_priors(position_history)
    features = _load_features(
        records,
        primary_positions=primary_positions,
        career_positions=career_positions,
    )
    try:
        fit_rating_artifacts(features, artifact_directory, config=config)
        if not primary_positions.is_empty():
            _atomic_parquet(
                primary_positions,
                artifact_directory / "season_primary_positions.parquet",
            )
        if not career_positions.is_empty():
            _atomic_parquet(
                career_positions,
                artifact_directory / "career_primary_positions.parquet",
            )
        ratings = score_rating_features(features, artifact_directory)
    except RatingModelError as exc:
        raise RatingPipelineError(str(exc)) from exc
    ratings, form_state = add_form_history(
        ratings, half_life_days=config.ewm_half_life_days
    )
    return _write_result(
        ratings=ratings,
        form_state=form_state,
        processed=_processed_frame(records, config.version),
        output_directory=_rating_directory(output_root, competition, season),
        state_directory=_state_directory(state_root, competition, season),
        artifact_directory=artifact_directory,
        mode="fit_full_history",
        selected_matches=len(records),
        processed_count=len(records),
        skipped_count=0,
    )


def update_rating_season(
    *,
    competition: str,
    season: str,
    features_root: Path = Path("data/features/whoscored"),
    output_root: Path = Path("data/features/ratings"),
    state_root: Path = Path("data/state/ratings"),
    artifact_directory: Path = Path("models/ratings/post_match_v2"),
) -> RatingPipelineResult:
    config = load_rating_config(artifact_directory)
    output_directory = _rating_directory(output_root, competition, season)
    state_directory = _state_directory(state_root, competition, season)
    output_path = output_directory / "player_match_ratings.parquet"
    processed_path = state_directory / "processed_matches.parquet"
    form_state_path = state_directory / "player_form_state.parquet"
    if (
        not output_path.exists()
        or not processed_path.exists()
        or not form_state_path.exists()
    ):
        raise RatingPipelineError(
            "Incremental rating state is missing. Run `mvp ratings fit` first."
        )
    records = _partition_records(_feature_partitions(features_root, competition, season))
    existing_processed = pl.read_parquet(processed_path)
    previous_hashes = {
        int(row["match_id"]): str(row["feature_sha256"])
        for row in existing_processed.to_dicts()
    }
    changed = [
        record
        for record in records
        if previous_hashes.get(int(record["match_id"])) != record["feature_sha256"]
    ]
    if changed:
        changed_ids = {int(record["match_id"]) for record in changed}
        # A corrected/rebuilt historical partition changes the population used
        # to fit season-position priors and z-score statistics. Refit the whole
        # season in that case. A genuinely new match remains append-only and is
        # scored with the frozen artifacts, which is the continuous-operation
        # contract.
        if any(match_id in previous_hashes for match_id in changed_ids):
            return fit_and_score_rating_season(
                competition=competition,
                season=season,
                features_root=features_root,
                output_root=output_root,
                state_root=state_root,
                artifact_directory=artifact_directory,
                config=config,
            )
        position_path = artifact_directory / "season_primary_positions.parquet"
        career_position_path = (
            artifact_directory / "career_primary_positions.parquet"
        )
        if not position_path.exists() or not career_position_path.exists():
            raise RatingPipelineError(
                "Rating position artifacts are missing. Run `mvp ratings fit` once "
                "with the corrected rating pipeline."
            )
        scored = score_rating_features(
            _load_features(
                changed,
                primary_positions=pl.read_parquet(position_path),
                career_positions=pl.read_parquet(career_position_path),
            ),
            artifact_directory,
        )
        previous_ratings = pl.read_parquet(output_path)
        replaced_ratings = previous_ratings.filter(
            pl.col("match_id").is_in(changed_ids)
        )
        existing_ratings = previous_ratings.filter(
            ~pl.col("match_id").is_in(changed_ids)
        )
        existing_state = pl.read_parquet(form_state_path)
        affected_players = set(
            replaced_ratings["whoscored_player_id"].drop_nulls().to_list()
        ) | set(scored["whoscored_player_id"].drop_nulls().to_list())
        is_append_only = replaced_ratings.is_empty()
        if is_append_only:
            last_by_player = {
                int(row["whoscored_player_id"]): _datetime(
                    row["last_match_datetime"]
                )
                for row in existing_state.to_dicts()
                if row.get("last_match_datetime") is not None
            }
            is_append_only = all(
                _datetime(row["match_datetime"])
                >= last_by_player.get(
                    int(row["whoscored_player_id"]), datetime.min
                )
                for row in scored.to_dicts()
            )
        if is_append_only:
            scored_with_form, form_state = append_form_history(
                scored,
                existing_state,
                half_life_days=config.ewm_half_life_days,
            )
            ratings = pl.concat(
                [existing_ratings, scored_with_form], how="diagonal_relaxed"
            ).sort(["match_datetime", "match_id", "whoscored_player_id"])
        else:
            combined = pl.concat(
                [existing_ratings, scored], how="diagonal_relaxed"
            )
            affected = combined.filter(
                pl.col("whoscored_player_id").is_in(affected_players)
            )
            recomputed, affected_state = add_form_history(
                affected, half_life_days=config.ewm_half_life_days
            )
            ratings = pl.concat(
                [
                    combined.filter(
                        ~pl.col("whoscored_player_id").is_in(affected_players)
                    ),
                    recomputed,
                ],
                how="diagonal_relaxed",
            ).sort(["match_datetime", "match_id", "whoscored_player_id"])
            form_state = pl.concat(
                [
                    existing_state.filter(
                        ~pl.col("whoscored_player_id").is_in(affected_players)
                    ),
                    affected_state,
                ],
                how="diagonal_relaxed",
            ).sort("whoscored_player_id")
        processed_by_id = {
            int(row["match_id"]): row for row in existing_processed.to_dicts()
        }
        for row in _processed_frame(changed, config.version).to_dicts():
            processed_by_id[int(row["match_id"])] = row
        processed = pl.DataFrame(
            list(processed_by_id.values()), infer_schema_length=None
        ).sort("match_id")
    else:
        ratings = pl.read_parquet(output_path)
        form_state = pl.read_parquet(form_state_path)
        processed = existing_processed

    return _write_result(
        ratings=ratings,
        form_state=form_state,
        processed=processed,
        output_directory=output_directory,
        state_directory=state_directory,
        artifact_directory=artifact_directory,
        mode="incremental_update",
        selected_matches=len(records),
        processed_count=len(changed),
        skipped_count=len(records) - len(changed),
    )
