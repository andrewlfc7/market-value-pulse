from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from uuid import uuid4

import numpy as np
import polars as pl

from . import carry, defensive_xpv, shot_models, spatial, xa
from .adapter import CompatibilityBundle, build_compatibility_bundle
from .artifacts import ResolvedArtifact, load_artifact
from ingestion.common import sha256_file, write_json
from ingestion.progress import ProgressCallback, ProgressEmitter, ProgressUpdate


class FeatureEnrichmentError(RuntimeError):
    """Raised when a match cannot complete the enrichment contract."""


@dataclass(frozen=True)
class MatchEnrichmentResult:
    match_id: int
    output_directory: Path
    status: str
    player_rows: int


@dataclass(frozen=True)
class EnrichmentRunResult:
    run_id: str
    run_directory: Path
    selected: int
    processed: int
    skipped: int
    failed: int
    results_path: Path


StageCallback = Callable[[str, str], None]
# Bump the contract whenever derived feature semantics change. The version is
# included in each input signature, so existing partitions are recomputed once
# instead of silently reusing values produced by the previous implementation.
FEATURE_VERSION = "match_features_v2"


@dataclass(frozen=True)
class FeatureConfig:
    penalty_xg: float = 0.776
    own_goal_xg: float = 0.0
    big_chance_xg: float = 0.30
    min_carry_m: float = 5.0
    max_carry_m: float = 60.0
    min_carry_seconds: float = 1.0
    max_carry_seconds: float = 10.0
    spatial_grid_x: int = 12
    spatial_grid_y: int = 8
    defensive_previous_gap_seconds: float = 15.0
    defensive_next_gap_seconds: float = 12.0


@dataclass(frozen=True)
class FeatureRuntime:
    artifacts: dict[str, ResolvedArtifact]
    xg_model: object
    xgot_model: object
    xg_metadata: dict[str, object]
    xgot_metadata: dict[str, object]
    xa_model: object
    xt_grid: np.ndarray
    xpv_grid: np.ndarray
    artifact_signature: str


def load_feature_config(path: Path | None = Path("config/features/scoring.json")) -> FeatureConfig:
    if path is None or not path.exists():
        return FeatureConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = FeatureConfig.__dataclass_fields__
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise FeatureEnrichmentError(f"Unknown feature configuration keys: {unknown}")
    config = FeatureConfig(**payload)
    if not 0.0 <= config.penalty_xg <= 1.0:
        raise FeatureEnrichmentError("penalty_xg must be between 0 and 1")
    if not 0.0 < config.big_chance_xg <= 1.0:
        raise FeatureEnrichmentError("big_chance_xg must be between 0 and 1")
    if not 0.0 < config.min_carry_m < config.max_carry_m:
        raise FeatureEnrichmentError("carry distance limits are invalid")
    return config


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    os.replace(temporary, path)


def _output_directory(
    root: Path, *, competition: str, season: str, match_id: int
) -> Path:
    return root / f"competition={competition}" / f"season={season}" / "matches" / f"match_id={match_id}"


def _stamp(
    frame: pl.DataFrame,
    *,
    competition_id: int,
    competition: str,
    season: str,
    family: str,
    version: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(
        pl.lit("whoscored").alias("provider"),
        pl.lit(competition_id).cast(pl.Int64).alias("competition_id"),
        pl.lit(competition).alias("competition_slug"),
        pl.lit(season).alias("season"),
        pl.lit(family).alias("model_family"),
        pl.lit(version).alias("model_version"),
        pl.lit(_utc_now().replace(tzinfo=None)).alias("scored_at"),
    )


def _stage(
    status: dict[str, object],
    path: Path,
    name: str,
    state: str,
    callback: StageCallback | None,
    **details: object,
) -> None:
    stages = status.setdefault("stages", {})
    assert isinstance(stages, dict)
    record = stages.setdefault(name, {})
    assert isinstance(record, dict)
    record.update({"status": state, **details})
    timestamp_key = "started_at" if state == "running" else "completed_at"
    record[timestamp_key] = _utc_now().isoformat()
    write_json(path, status)
    if callback is not None:
        callback(name, state)


def _artifact_signature(artifacts: dict[str, ResolvedArtifact]) -> str:
    objects: list[tuple[str, str]] = []
    for family, artifact in sorted(artifacts.items()):
        for path in sorted(item for item in artifact.directory.rglob("*") if item.is_file()):
            objects.append((f"{family}/{path.relative_to(artifact.directory)}", sha256_file(path)))
    return hashlib.sha256(
        json.dumps(objects, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_feature_runtime(models_root: Path) -> FeatureRuntime:
    """Validate and load every model once for an enrichment process."""
    artifacts = {
        family: load_artifact(family, models_root=models_root, log=False)
        for family in ("xg", "xgot", "xa", "xthreat", "goal_probability")
    }
    xg = artifacts["xg"]
    xgot = artifacts["xgot"]
    xa_artifact = artifacts["xa"]
    xt_payload = artifacts["xthreat"].load_model()
    xpv_payload = artifacts["goal_probability"].load_model()
    return FeatureRuntime(
        artifacts=artifacts,
        xg_model=xg.load_model(),
        xgot_model=xgot.load_model(),
        xg_metadata=xg.load_extra_joblib("metadata.joblib"),
        xgot_metadata=xgot.load_extra_joblib("metadata.joblib"),
        xa_model=xa_artifact.load_model(),
        xt_grid=np.asarray(xt_payload["values"], dtype=float),
        xpv_grid=np.asarray(xpv_payload["values"], dtype=float),
        artifact_signature=_artifact_signature(artifacts),
    )


def _input_signature(
    match_directory: Path,
    *,
    config: FeatureConfig,
    runtime: FeatureRuntime | None,
    prepare_only: bool,
) -> tuple[str, dict[str, str]]:
    source_objects = {
        name: sha256_file(match_directory / name)
        for name in (
            "matches.parquet",
            "player_matches.parquet",
            "events.parquet",
            "shots.parquet",
            "_SUCCESS.json",
        )
        if (match_directory / name).exists()
    }
    payload = {
        "feature_version": FEATURE_VERSION,
        "prepare_only": prepare_only,
        "source_objects": source_objects,
        "configuration": {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
        "artifact_signature": runtime.artifact_signature if runtime else None,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return signature, source_objects


def _marker_matches(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") in {"succeeded", "prepared"} and payload.get(
        "input_signature"
    ) == signature


def _score_shots(
    bundle: CompatibilityBundle,
    runtime: FeatureRuntime,
    config: FeatureConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    xg_art = runtime.artifacts["xg"]
    xg_metadata = runtime.xg_metadata
    xgot_metadata = runtime.xgot_metadata
    version = xg_art.artifact_version
    predictions = shot_models.score_shots_frame(
        bundle.shots,
        xg_model=runtime.xg_model,
        xgot_model=runtime.xgot_model,
        xg_metadata=xg_metadata,
        xgot_metadata=xgot_metadata,
        xg_model_name=xg_metadata.get("model_name", "xg_lgb"),
        xgot_model_name=xgot_metadata.get("model_name", "xgot_lgb"),
        model_version=version,
        penalty_xg=config.penalty_xg,
        own_goal_xg=config.own_goal_xg,
    )
    shots_with_uid = shot_models.add_shot_uid(bundle.shots)
    if not predictions.is_empty():
        modeled = shots_with_uid.join(
            predictions.select("shot_uid", "xg", "xgot", "xgot_available"),
            on="shot_uid",
            how="left",
        )
    else:
        modeled = shots_with_uid
    for column, dtype in (
        ("xg", pl.Float64),
        ("xgot", pl.Float64),
        ("xgot_available", pl.Int64),
    ):
        if column not in modeled.columns:
            modeled = modeled.with_columns(pl.lit(None).cast(dtype).alias(column))
    return predictions, modeled


def _score_carries(bundle: CompatibilityBundle, config: FeatureConfig) -> pl.DataFrame:
    events = carry.prepare_events_frame(bundle.events)
    return carry.infer_carries(
        events=events,
        min_carry_m=config.min_carry_m,
        max_carry_m=config.max_carry_m,
        min_duration_s=config.min_carry_seconds,
        max_duration_s=config.max_carry_seconds,
        nx=config.spatial_grid_x,
        ny=config.spatial_grid_y,
    )


def _score_xa(
    bundle: CompatibilityBundle,
    shots_with_models: pl.DataFrame,
    runtime: FeatureRuntime,
) -> pl.DataFrame:
    artifact = runtime.artifacts["xa"]
    targets = xa.build_actual_targets_from_shots(shots_with_models)
    return xa.score_passes(
        bundle.passes,
        targets,
        runtime.xa_model,
        SimpleNamespace(
            model_name=artifact.metadata.get("source_model_name", artifact.artifact_version),
            model_version=artifact.artifact_version,
        ),
    )


def _score_spatial_actions(
    bundle: CompatibilityBundle,
    carries: pl.DataFrame,
    *,
    runtime: FeatureRuntime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    xt_artifact = runtime.artifacts["xthreat"]
    xpv_artifact = runtime.artifacts["goal_probability"]
    actions = spatial.build_actions(bundle.passes, carries)
    xt_frame = spatial.score_actions(
        actions,
        runtime.xt_grid,
        metric="xT",
        model_name=xt_artifact.metadata.get(
            "source_model_name", xt_artifact.artifact_version
        ),
        model_version=xt_artifact.artifact_version,
    )
    xpv_frame = spatial.score_actions(
        actions,
        runtime.xpv_grid,
        metric="xPV",
        model_name=xpv_artifact.metadata.get(
            "source_model_name", xpv_artifact.artifact_version
        ),
        model_version=xpv_artifact.artifact_version,
    )
    return xt_frame, xpv_frame


def _score_defensive(
    bundle: CompatibilityBundle,
    runtime: FeatureRuntime,
    config: FeatureConfig,
) -> pl.DataFrame:
    artifact = runtime.artifacts["goal_probability"]
    events = defensive_xpv.prepare_events_frame(bundle.events)
    if events.is_empty():
        return pl.DataFrame()
    return defensive_xpv.score_all_matches(
        events=events,
        grid=runtime.xpv_grid,
        match_team_map=defensive_xpv.build_match_team_map(events),
        max_prev_gap_s=config.defensive_previous_gap_seconds,
        max_next_gap_s=config.defensive_next_gap_seconds,
        model_name=artifact.metadata.get("source_model_name", artifact.artifact_version),
        model_version=artifact.artifact_version,
    )


def _position_group(position: pl.Expr) -> pl.Expr:
    normalized = position.cast(pl.String).fill_null("").str.to_uppercase()
    return (
        pl.when(normalized.str.starts_with("GK")).then(pl.lit("Goalkeeper"))
        .when(
            normalized.is_in(["DC", "DL", "DR", "DCL", "DCR", "WB", "WBL", "WBR"])
        ).then(pl.lit("Defender"))
        .when(
            normalized.is_in(["DMC", "MC", "ML", "MR", "MCL", "MCR"])
        ).then(pl.lit("Midfielder"))
        .when(
            normalized.is_in(["AMC", "AML", "AMR", "FW", "FWL", "FWR", "ST", "CF"])
        ).then(pl.lit("Forward"))
        .otherwise(pl.lit("Unknown"))
    )


def _join_aggregate(
    base: pl.DataFrame,
    frame: pl.DataFrame,
    expressions: list[pl.Expr],
) -> pl.DataFrame:
    if frame.is_empty() or "player_id" not in frame.columns:
        return base
    aggregate = frame.filter(pl.col("player_id").is_not_null()).group_by("player_id").agg(expressions)
    return base.join(aggregate, on="player_id", how="left")


def _pass_features(passes: pl.DataFrame) -> pl.DataFrame:
    if passes.is_empty():
        return pl.DataFrame()
    all_pass_assists = (
        passes.filter(pl.col("player_id").is_not_null())
        .group_by("player_id")
        .agg(pl.col("is_assist").sum().alias("assists"))
    )
    open_play = passes.filter(
        ~pl.col("is_corner").cast(pl.Boolean).fill_null(False)
        & ~pl.col("is_free_kick").cast(pl.Boolean).fill_null(False)
        & ~pl.col("is_throw_in").cast(pl.Boolean).fill_null(False)
        & ~pl.col("is_goal_kick").cast(pl.Boolean).fill_null(False)
    )
    forward_progress = pl.col("end_x") - pl.col("x")
    prepared = open_play.with_columns(forward_progress.alias("forward_progress")).with_columns(
        (
            (
                (pl.col("x") < 50.0)
                & (pl.col("end_x") < 50.0)
                & (pl.col("forward_progress") >= 28.6)
            )
            | (
                (pl.col("x") < 50.0)
                & (pl.col("end_x") >= 50.0)
                & (pl.col("forward_progress") >= 14.3)
            )
            | (
                (pl.col("x") >= 50.0)
                & (pl.col("forward_progress") >= 9.5)
            )
        )
        .fill_null(False)
        .alias("is_progressive_pass")
    )
    open_play_features = (
        prepared.filter(pl.col("player_id").is_not_null())
        .group_by("player_id")
        .agg(
            pl.len().alias("passes"),
            pl.col("success").sum().alias("completed_passes"),
            pl.col("is_key_pass").sum().alias("key_passes"),
            (
                pl.col("is_progressive_pass")
                & pl.col("success").cast(pl.Boolean)
            )
            .sum()
            .alias("progressive_passes"),
        )
    )
    return open_play_features.join(all_pass_assists, on="player_id", how="full", coalesce=True)


def _shot_creation_features(
    passes: pl.DataFrame, shots: pl.DataFrame, *, xg_threshold: float = 0.30
) -> pl.DataFrame:
    if passes.is_empty() or shots.is_empty() or "xg" not in shots.columns:
        return pl.DataFrame()
    qualifying = shots.filter(pl.col("xg").fill_null(0.0) >= xg_threshold)
    if qualifying.is_empty():
        return pl.DataFrame()
    pass_players = passes.select(
        pl.col("match_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64, strict=False),
        pl.col("event_id").cast(pl.Int64, strict=False).alias("assist_event_id"),
        pl.col("player_id").cast(pl.Int64, strict=False).alias("pass_player_id"),
    ).drop_nulls(["team_id", "assist_event_id", "pass_player_id"])
    created = (
        qualifying.select(
            pl.col("match_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("related_event_id")
            .cast(pl.Int64, strict=False)
            .alias("assist_event_id")
        )
        .drop_nulls(["team_id", "assist_event_id"])
        .join(
            pass_players,
            on=["match_id", "team_id", "assist_event_id"],
            how="inner",
        )
        .with_columns(pl.col("pass_player_id").alias("player_id"))
        .drop_nulls("player_id")
    )
    return created.group_by("player_id").agg(
        pl.len().alias("big_chances_created")
    )


def _discipline_features(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty() or "card_type_display_name" not in events.columns:
        return pl.DataFrame()
    card = (
        pl.col("card_type_display_name")
        .cast(pl.String)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z]", "")
    )
    return (
        events.filter(pl.col("player_id").is_not_null())
        .with_columns(card.alias("card_name"))
        .group_by("player_id")
        .agg(
            (
                pl.col("card_name").str.contains("yellow")
                & ~pl.col("card_name").str.contains("secondyellow")
            )
            .cast(pl.Int64)
            .sum()
            .alias("yellow_cards"),
            pl.col("card_name")
            .str.contains("red|secondyellow")
            .cast(pl.Int64)
            .sum()
            .alias("red_cards"),
        )
    )


def _shot_adjustment_features(
    shots: pl.DataFrame, *, xg_threshold: float
) -> pl.DataFrame:
    if shots.is_empty():
        return pl.DataFrame()
    is_goal = pl.col("is_goal").fill_null(0).cast(pl.Boolean)
    is_own_goal = (
        pl.col("is_own_goal").fill_null(0).cast(pl.Boolean)
        if "is_own_goal" in shots.columns
        else pl.lit(False)
    )
    return (
        shots.filter(pl.col("player_id").is_not_null())
        .with_columns(
            ((pl.col("xg").fill_null(0.0) >= xg_threshold) & ~is_goal)
            .cast(pl.Int64)
            .alias("is_big_chance_missed"),
            pl.when((pl.col("xg").fill_null(0.0) >= xg_threshold) & ~is_goal)
            .then(pl.col("xg").fill_null(0.0))
            .otherwise(0.0)
            .alias("big_chance_xg_missed_value"),
            is_own_goal.cast(pl.Int64).alias("own_goal_value"),
        )
        .group_by("player_id")
        .agg(
            pl.col("is_big_chance_missed").sum().alias("big_chances_missed"),
            pl.col("big_chance_xg_missed_value").sum().alias("big_chance_xg_missed"),
            pl.col("own_goal_value").sum().alias("own_goals"),
        )
    )


def _goalkeeper_features(
    bundle: CompatibilityBundle, shots: pl.DataFrame
) -> pl.DataFrame:
    if shots.is_empty() or bundle.player_matches.is_empty():
        return pl.DataFrame()
    goalkeepers = [
        row
        for row in bundle.player_matches.to_dicts()
        if str(row.get("position") or "").upper().startswith("GK")
        and float(row.get("minutes") or 0.0) > 0.0
    ]
    if not goalkeepers:
        return pl.DataFrame()
    events = bundle.events.to_dicts()
    maximum_minute = max(
        [float(row.get("expanded_minute") or row.get("minute") or 0.0) for row in events]
        + [90.0]
    )
    on_minutes: dict[int, float] = {}
    off_minutes: dict[int, float] = {}
    for event in events:
        name = str(event.get("type_name") or "").casefold()
        player_id = event.get("player_id")
        if player_id is None:
            continue
        minute = float(event.get("expanded_minute") or event.get("minute") or 0.0)
        if "substitutionon" in name or "substitution on" in name:
            on_minutes[int(player_id)] = minute
        if "substitutionoff" in name or "substitution off" in name:
            off_minutes[int(player_id)] = minute
    intervals: list[dict[str, object]] = []
    for goalkeeper in goalkeepers:
        player_id = int(goalkeeper["player_id"])
        minutes = float(goalkeeper.get("minutes") or 0.0)
        start = 0.0 if goalkeeper.get("started") else on_minutes.get(
            player_id, max(maximum_minute - minutes, 0.0)
        )
        end = off_minutes.get(player_id, maximum_minute)
        intervals.append(
            {
                "player_id": player_id,
                "team_id": int(goalkeeper["team_id"]),
                "start": start,
                "end": max(end, start),
            }
        )
    team_ids = {int(row["team_id"]) for row in bundle.player_matches.to_dicts() if row.get("team_id") is not None}
    totals: dict[int, dict[str, float]] = {
        int(row["player_id"]): {
            "shots_on_target_faced": 0.0,
            "goals_conceded_xgot_sample": 0.0,
            "xgot_faced": 0.0,
            "goals_conceded": 0.0,
        }
        for row in goalkeepers
    }
    for shot in shots.to_dicts():
        shooting_team = shot.get("team_id")
        if shooting_team is None:
            continue
        own_goal = bool(shot.get("is_own_goal") or False)
        if own_goal:
            # Own goals are explicit player penalties, not shots faced or
            # shot-stopping opportunities for the goalkeeper.
            continue
        defending_team = next(
            (team for team in team_ids if team != int(shooting_team)), None
        )
        if defending_team is None:
            continue
        minute = float(shot.get("expanded_minute") or shot.get("minute") or 0.0)
        candidates = [
            interval
            for interval in intervals
            if interval["team_id"] == defending_team
            and float(interval["start"]) <= minute <= float(interval["end"])
        ]
        if not candidates:
            continue
        goalkeeper_id = int(candidates[0]["player_id"])
        target = totals[goalkeeper_id]
        type_name = str(shot.get("type_name") or "").casefold()
        is_goal = bool(shot.get("is_goal") or False)
        xgot = shot.get("xgot")
        xgot_available = bool(shot.get("xgot_available") or xgot is not None)
        on_target = is_goal or "savedshot" in type_name or "saved shot" in type_name or xgot_available
        if on_target:
            target["shots_on_target_faced"] += 1.0
        if is_goal:
            target["goals_conceded"] += 1.0
        if xgot_available:
            target["xgot_faced"] += float(xgot or 0.0)
            if is_goal:
                target["goals_conceded_xgot_sample"] += 1.0
    return pl.DataFrame(
        [{"player_id": player_id, **values} for player_id, values in totals.items()],
        infer_schema_length=None,
    )


def _player_match_features(
    bundle: CompatibilityBundle,
    *,
    shots: pl.DataFrame,
    xa_frame: pl.DataFrame,
    carries: pl.DataFrame,
    xt_frame: pl.DataFrame,
    xpv_frame: pl.DataFrame,
    defensive: pl.DataFrame,
    season: str,
    config: FeatureConfig,
) -> pl.DataFrame:
    base = bundle.player_matches
    if base.is_empty():
        return base
    event_players = set(
        bundle.events.filter(pl.col("type_name") != "Card")["player_id"]
        .drop_nulls()
        .to_list()
    )
    suspect_appearances = [
        row
        for row in base.to_dicts()
        if float(row.get("minutes") or 0.0) <= 0.0
        and (bool(row.get("started")) or row.get("player_id") in event_players)
    ]
    if suspect_appearances:
        player_ids = sorted(int(row["player_id"]) for row in suspect_appearances)
        raise FeatureEnrichmentError(
            f"Appearance minutes are missing for match {bundle.match_id}: {player_ids}"
        )
    base = base.filter(pl.col("minutes").fill_null(0.0) > 0.0)
    if base.is_empty():
        raise FeatureEnrichmentError(
            f"No player appearances with positive minutes in match {bundle.match_id}"
        )
    position_group = (
        pl.col("position_group")
        if "position_group" in base.columns
        else _position_group(pl.col("position"))
    )
    base = base.with_columns(
        pl.col("player_id").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64).alias("whoscored_player_id"),
        pl.lit(season).alias("season"),
        pl.lit(bundle.matches["start_date"][0]).cast(pl.String).str.to_datetime(
            strict=False
        ).alias("match_datetime"),
        position_group.alias("position_group"),
    )
    base = _join_aggregate(
        base,
        shots,
        [
            pl.len().alias("shots"),
            pl.col("is_goal").cast(pl.Int64).sum().alias("goals"),
            pl.col("xg").fill_null(0.0).sum().alias("xg"),
            pl.col("xgot").fill_null(0.0).sum().alias("xgot"),
        ],
    )
    pass_features = _pass_features(bundle.passes)
    if not pass_features.is_empty():
        base = base.join(pass_features, on="player_id", how="left")
    creation = _shot_creation_features(
        bundle.passes, shots, xg_threshold=config.big_chance_xg
    )
    if not creation.is_empty():
        base = base.join(creation, on="player_id", how="left")
    base = _join_aggregate(base, xa_frame, [pl.col("xa_model").fill_null(0.0).sum().alias("xa")])
    base = _join_aggregate(
        base,
        carries,
        [
            pl.len().alias("carries"),
            pl.col("is_progressive_carry").sum().alias("progressive_carries"),
            pl.col("is_final_third_carry").sum().alias("final_third_carries"),
            pl.col("is_carry_into_box").sum().alias("carries_into_box"),
        ],
    )
    base = _join_aggregate(base, xt_frame, [pl.col("xT_added").sum().alias("xt_added")])
    base = _join_aggregate(base, xpv_frame, [pl.col("xPV_added").sum().alias("xpv_added")])
    base = _join_aggregate(
        base,
        defensive,
        [
            pl.col("opponent_threat_prevented").sum().alias("opponent_threat_prevented"),
            pl.col("net_threat_reduction").sum().alias("defensive_net_threat_reduction"),
        ],
    )
    for extra in (
        _discipline_features(bundle.events),
        _shot_adjustment_features(shots, xg_threshold=config.big_chance_xg),
        _goalkeeper_features(bundle, shots),
    ):
        if not extra.is_empty():
            base = base.join(extra, on="player_id", how="left")
    numeric = [
        "shots", "goals", "xg", "xgot", "passes", "completed_passes",
        "key_passes", "assists", "xa", "big_chances_created", "progressive_passes",
        "carries", "progressive_carries", "final_third_carries", "carries_into_box",
        "xt_added", "xpv_added", "opponent_threat_prevented",
        "defensive_net_threat_reduction", "yellow_cards", "red_cards",
        "big_chances_missed", "big_chance_xg_missed", "own_goals",
        "shots_on_target_faced", "goals_conceded_xgot_sample", "xgot_faced",
        "goals_conceded",
    ]
    for column in numeric:
        if column not in base.columns:
            base = base.with_columns(pl.lit(0.0).alias(column))
    output = base.with_columns([pl.col(column).fill_null(0) for column in numeric])
    unknown_positions = output.filter(pl.col("position_group") == "Unknown")
    if not unknown_positions.is_empty():
        observed = sorted(
            str(value)
            for value in unknown_positions["position"].drop_nulls().unique().to_list()
        )
        raise FeatureEnrichmentError(
            f"Unknown position codes in match {bundle.match_id}: {observed}"
        )
    invalid_minutes = output.filter(
        pl.col("minutes").is_null() | (pl.col("minutes") < 0) | (pl.col("minutes") > 130)
    ).height
    if invalid_minutes:
        raise FeatureEnrichmentError(
            f"Invalid minutes for {invalid_minutes} player rows in match {bundle.match_id}"
        )
    return output


def enrich_match(
    match_directory: Path,
    *,
    competition: str,
    competition_id: int,
    season: str,
    output_root: Path = Path("data/features/whoscored"),
    models_root: Path = Path("models/features"),
    season_id: int | None = None,
    prepare_only: bool = False,
    force: bool = False,
    stage_callback: StageCallback | None = None,
    runtime: FeatureRuntime | None = None,
    feature_config: FeatureConfig | None = None,
) -> MatchEnrichmentResult:
    config = feature_config or load_feature_config()
    if not prepare_only and runtime is None:
        runtime = load_feature_runtime(models_root)
    match_id = int(match_directory.name.split("=", 1)[1])
    output = _output_directory(
        output_root, competition=competition, season=season, match_id=match_id
    )
    success_path = output / "_SUCCESS.json"
    prepared_path = output / "_PREPARED.json"
    completion_marker = prepared_path if prepare_only else success_path
    input_signature, source_objects = _input_signature(
        match_directory,
        config=config,
        runtime=runtime,
        prepare_only=prepare_only,
    )
    if not force and _marker_matches(completion_marker, input_signature):
        return MatchEnrichmentResult(match_id, output, "skipped_existing", 0)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "enrichment_status.json"
    status: dict[str, object] = {
        "match_id": match_id,
        "competition": competition,
        "season": season,
        "status": "running",
        "started_at": _utc_now().isoformat(),
        "source_partition": str(match_directory),
        "feature_version": FEATURE_VERSION,
        "input_signature": input_signature,
        "stages": {},
    }
    write_json(status_path, status)
    try:
        _stage(status, status_path, "compatibility", "running", stage_callback)
        bundle = build_compatibility_bundle(
            match_directory,
            competition_id=competition_id,
            season=season,
            season_id=season_id,
        )
        _atomic_parquet(bundle.events, output / "adapted_events.parquet")
        _atomic_parquet(bundle.passes, output / "passes.parquet")
        _atomic_parquet(bundle.shots, output / "adapted_shots.parquet")
        write_json(output / "data_quality.json", {"checks": list(bundle.checks)})
        _stage(
            status,
            status_path,
            "compatibility",
            "completed",
            stage_callback,
            rows=bundle.passes.height,
        )
        if prepare_only:
            status.update({"status": "prepared", "completed_at": _utc_now().isoformat()})
            write_json(status_path, status)
            write_json(
                prepared_path,
                {
                    "status": "prepared",
                    "match_id": match_id,
                    "prepared_at": _utc_now().isoformat(),
                    "source_partition": str(match_directory),
                    "feature_version": FEATURE_VERSION,
                    "input_signature": input_signature,
                    "source_objects": source_objects,
                    "objects": {
                        path.name: {"path": str(path), "sha256": sha256_file(path)}
                        for path in sorted(output.glob("*.parquet"))
                    },
                },
            )
            return MatchEnrichmentResult(match_id, output, "prepared", bundle.player_matches.height)

        assert runtime is not None
        artifacts = runtime.artifacts
        _stage(status, status_path, "models", "running", stage_callback)
        _stage(
            status,
            status_path,
            "models",
            "completed",
            stage_callback,
            artifact_signature=runtime.artifact_signature,
        )

        _stage(status, status_path, "xg_xgot", "running", stage_callback)
        shot_predictions, shots_with_models = _score_shots(bundle, runtime, config)
        shot_predictions = _stamp(
            shot_predictions,
            competition_id=competition_id,
            competition=competition,
            season=season,
            family="xg_xgot",
            version=artifacts["xg"].artifact_version,
        )
        _atomic_parquet(shot_predictions, output / "shot_model_predictions.parquet")
        _atomic_parquet(shots_with_models, output / "shots_with_models.parquet")
        _stage(status, status_path, "xg_xgot", "completed", stage_callback, rows=shot_predictions.height)

        _stage(status, status_path, "carries", "running", stage_callback)
        carries = _score_carries(bundle, config)
        carries = _stamp(carries, competition_id=competition_id, competition=competition, season=season, family="carry", version=carry.MODEL_VERSION)
        _atomic_parquet(carries, output / "carries.parquet")
        _stage(status, status_path, "carries", "completed", stage_callback, rows=carries.height)

        _stage(status, status_path, "xa", "running", stage_callback)
        xa_frame = _score_xa(bundle, shots_with_models, runtime)
        xa_frame = _stamp(xa_frame, competition_id=competition_id, competition=competition, season=season, family="xa", version=artifacts["xa"].artifact_version)
        _atomic_parquet(xa_frame, output / "passes_with_xa.parquet")
        _stage(status, status_path, "xa", "completed", stage_callback, rows=xa_frame.height)

        _stage(status, status_path, "spatial_value", "running", stage_callback)
        xt_frame, xpv_frame = _score_spatial_actions(
            bundle,
            carries,
            runtime=runtime,
        )
        xt_frame = _stamp(xt_frame, competition_id=competition_id, competition=competition, season=season, family="xthreat", version=artifacts["xthreat"].artifact_version)
        xpv_frame = _stamp(xpv_frame, competition_id=competition_id, competition=competition, season=season, family="goal_probability", version=artifacts["goal_probability"].artifact_version)
        _atomic_parquet(xt_frame, output / "xthreat_actions.parquet")
        _atomic_parquet(xpv_frame, output / "xpv_actions.parquet")
        _stage(status, status_path, "spatial_value", "completed", stage_callback, rows=xt_frame.height + xpv_frame.height)

        _stage(status, status_path, "defensive_xpv", "running", stage_callback)
        defensive = _score_defensive(bundle, runtime, config)
        defensive = _stamp(defensive, competition_id=competition_id, competition=competition, season=season, family="defensive_xpv", version=artifacts["goal_probability"].artifact_version)
        _atomic_parquet(defensive, output / "defensive_xpv_actions.parquet")
        _stage(status, status_path, "defensive_xpv", "completed", stage_callback, rows=defensive.height)

        _stage(status, status_path, "player_aggregate", "running", stage_callback)
        player_features = _player_match_features(
            bundle,
            shots=shots_with_models,
            xa_frame=xa_frame,
            carries=carries,
            xt_frame=xt_frame,
            xpv_frame=xpv_frame,
            defensive=defensive,
            season=season,
            config=config,
        )
        _atomic_parquet(player_features, output / "player_match_features.parquet")
        _stage(status, status_path, "player_aggregate", "completed", stage_callback, rows=player_features.height)

        objects = {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(output.glob("*.parquet"))
        }
        write_json(
            success_path,
            {
                "status": "succeeded",
                "match_id": match_id,
                "competition": competition,
                "season": season,
                "completed_at": _utc_now().isoformat(),
                "source_partition": str(match_directory),
                "models_root": str(models_root),
                "feature_version": FEATURE_VERSION,
                "input_signature": input_signature,
                "artifact_signature": runtime.artifact_signature,
                "source_objects": source_objects,
                "objects": objects,
            },
        )
        status.update({"status": "completed", "completed_at": _utc_now().isoformat()})
        write_json(status_path, status)
        return MatchEnrichmentResult(match_id, output, "succeeded", player_features.height)
    except Exception as exc:
        stages = status.get("stages")
        if isinstance(stages, dict):
            for record in stages.values():
                if isinstance(record, dict) and record.get("status") == "running":
                    record.update(
                        {"status": "failed", "completed_at": _utc_now().isoformat()}
                    )
        status.update(
            {
                "status": "failed",
                "completed_at": _utc_now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(status_path, status)
        raise FeatureEnrichmentError(f"match_id={match_id}: {exc}") from exc


def _match_sort_key(path: Path) -> tuple[str, int]:
    match_id = int(path.name.split("=", 1)[1])
    matches_path = path / "matches.parquet"
    if not matches_path.exists():
        return "", match_id
    frame = pl.read_parquet(matches_path, columns=["start_date"])
    value = frame["start_date"][0] if frame.height else None
    return str(value or ""), match_id


def enrich_season(
    *,
    competition: str,
    competition_id: int,
    season: str,
    normalized_root: Path = Path("data/normalized/whoscored"),
    output_root: Path = Path("data/features/whoscored"),
    models_root: Path = Path("models/features"),
    max_matches: int | None = None,
    prepare_only: bool = False,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> EnrichmentRunResult:
    source = normalized_root / f"competition={competition}" / f"season={season}" / "matches"
    if not source.exists():
        raise FeatureEnrichmentError(f"Normalized season does not exist: {source}")
    config = load_feature_config()
    runtime = None if prepare_only else load_feature_runtime(models_root)
    partitions = [path for path in source.glob("match_id=*") if (path / "_SUCCESS.json").exists()]
    partitions.sort(key=_match_sort_key, reverse=True)
    skipped_existing = 0
    if not force:
        pending = []
        for partition in partitions:
            match_id = int(partition.name.split("=", 1)[1])
            marker_name = "_PREPARED.json" if prepare_only else "_SUCCESS.json"
            marker = (
                _output_directory(
                    output_root,
                    competition=competition,
                    season=season,
                    match_id=match_id,
                )
                / marker_name
            )
            signature, _ = _input_signature(
                partition,
                config=config,
                runtime=runtime,
                prepare_only=prepare_only,
            )
            if _marker_matches(marker, signature):
                skipped_existing += 1
            else:
                pending.append(partition)
        partitions = pending
    if max_matches is not None:
        partitions = partitions[:max_matches]

    started = _utc_now()
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_directory = output_root / f"competition={competition}" / f"season={season}" / "_runs" / f"run_id={run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)
    emitter = ProgressEmitter(run_id=run_id, log_path=run_directory / "progress.jsonl", callback=progress)
    emitter.emit(ProgressUpdate(stage="enrichment", state="started", description="Enriching WhoScored matches", total=len(partitions), skipped=skipped_existing))

    if not partitions:
        emitter.emit(
            ProgressUpdate(
                stage="enrichment",
                state="completed",
                description="No unprocessed WhoScored matches",
                completed=0,
                total=0,
                skipped=skipped_existing,
            )
        )

    rows = []
    processed = 0
    failed = 0
    current_stage = "starting"
    for index, partition in enumerate(partitions, start=1):
        match_id = int(partition.name.split("=", 1)[1])

        def on_stage(stage: str, state: str) -> None:
            nonlocal current_stage
            current_stage = stage
            if state == "running":
                emitter.emit(
                    ProgressUpdate(
                        stage="enrichment",
                        state="advanced",
                        description="Enriching WhoScored matches",
                        completed=index - 1,
                        total=len(partitions),
                        current=f"match_id={match_id} stage={stage}",
                        succeeded=processed,
                        skipped=skipped_existing,
                        failed=failed,
                    )
                )

        try:
            result = enrich_match(
                partition,
                competition=competition,
                competition_id=competition_id,
                season=season,
                output_root=output_root,
                models_root=models_root,
                prepare_only=prepare_only,
                force=force,
                stage_callback=on_stage,
                runtime=runtime,
                feature_config=config,
            )
            processed += int(result.status in {"succeeded", "prepared"})
            rows.append({"match_id": match_id, "status": result.status, "output_directory": str(result.output_directory), "error": None})
        except FeatureEnrichmentError as exc:
            failed += 1
            rows.append({"match_id": match_id, "status": "failed", "output_directory": None, "error": str(exc)})
        emitter.emit(
            ProgressUpdate(
                stage="enrichment",
                state=("completed" if index == len(partitions) else "advanced"),
                description="Enriching WhoScored matches",
                completed=index,
                total=len(partitions),
                current=f"match_id={match_id} stage={current_stage}",
                succeeded=processed,
                skipped=skipped_existing,
                failed=failed,
            )
        )

    results_path = run_directory / "match_results.parquet"
    results = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
        schema={"match_id": pl.Int64, "status": pl.String, "output_directory": pl.String, "error": pl.String}
    )
    _atomic_parquet(results, results_path)
    write_json(
        run_directory / "manifest.json",
        {
            "run_id": run_id,
            "status": "succeeded" if failed == 0 else ("partial" if processed else "failed"),
            "competition": competition,
            "season": season,
            "started_at": started.isoformat(),
            "completed_at": _utc_now().isoformat(),
            "prepare_only": prepare_only,
            "counts": {"selected": len(partitions), "processed": processed, "skipped_existing": skipped_existing, "failed": failed},
            "results": str(results_path),
        },
    )
    return EnrichmentRunResult(run_id, run_directory, len(partitions), processed, skipped_existing, failed, results_path)
