from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ingestion.common import write_json


class RatingModelError(RuntimeError):
    """Raised when rating artifacts or feature rows violate the model contract."""


@dataclass(frozen=True)
class RatingModelConfig:
    version: str = "post_match_v2"
    minutes_floor: float = 30.0
    minutes_cap: float = 90.0
    feature_z_clip: float = 4.5
    composite_z_clip: float = 5.0
    big_chance_xg_threshold: float = 0.30
    big_chance_miss_penalty: float = 0.10
    big_chance_miss_penalty_cap: float = 0.30
    own_goal_penalty: float = 0.70
    own_goal_penalty_cap: float = 1.40
    outfield_rating_span: float = 3.30
    outfield_rating_temperature: float = 3.30
    goal_bonus: float = 0.10
    assist_bonus: float = 0.05
    assist_overperformance_bonus: float = 0.05
    decisive_action_bonus_cap: float = 0.50
    ewm_half_life_days: float = 90.0


OUTFIELD_POSITIONS = {"Defender", "Midfielder", "Forward"}
POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "Forward": {
        "threat_component": 0.28,
        "creation_component": 0.15,
        "progression_component": 0.10,
        "retention_component": 0.05,
        "attacking_xpv_component": 0.12,
        "defensive_component": 0.03,
        "finishing_component": 0.27,
    },
    "Midfielder": {
        "threat_component": 0.08,
        "creation_component": 0.25,
        "progression_component": 0.22,
        "retention_component": 0.12,
        "attacking_xpv_component": 0.13,
        "defensive_component": 0.08,
        "finishing_component": 0.12,
    },
    "Defender": {
        "threat_component": 0.03,
        "creation_component": 0.08,
        "progression_component": 0.17,
        "retention_component": 0.15,
        "attacking_xpv_component": 0.10,
        "defensive_component": 0.40,
        "finishing_component": 0.07,
    },
}

STANDARDIZED_FEATURES = (
    "log_xg_90",
    "log_xgot_90",
    "log_shots_90",
    "log_xa_90",
    "log_key_passes_90",
    "log_big_chances_created_90",
    "log_progressive_passes_90",
    "log_progressive_carries_90",
    "log_final_third_carries_90",
    "pass_completion_above_expected",
    "xpv_added_90",
    "opponent_threat_prevented_90",
    "defensive_net_threat_reduction_90",
    "finishing_above_expected",
    "shot_placement_above_expected",
)

COUNT_FEATURES = {
    "shots": "log_shots_90",
    "xg": "log_xg_90",
    "xgot": "log_xgot_90",
    "xa": "log_xa_90",
    "key_passes": "log_key_passes_90",
    "big_chances_created": "log_big_chances_created_90",
    "progressive_passes": "log_progressive_passes_90",
    "progressive_carries": "log_progressive_carries_90",
    "final_third_carries": "log_final_third_carries_90",
}

REQUIRED_FEATURE_COLUMNS = {
    "season",
    "match_id",
    "match_datetime",
    "whoscored_player_id",
    "position_group",
    "minutes",
}

NUMERIC_DEFAULTS = {
    "shots": 0.0,
    "goals": 0.0,
    "xg": 0.0,
    "xgot": 0.0,
    "xa": 0.0,
    "key_passes": 0.0,
    "big_chances_created": 0.0,
    "progressive_passes": 0.0,
    "progressive_carries": 0.0,
    "final_third_carries": 0.0,
    "passes": 0.0,
    "completed_passes": 0.0,
    "assists": 0.0,
    "xpv_added": 0.0,
    "opponent_threat_prevented": 0.0,
    "defensive_net_threat_reduction": 0.0,
    "yellow_cards": 0.0,
    "red_cards": 0.0,
    "big_chances_missed": 0.0,
    "big_chance_xg_missed": 0.0,
    "own_goals": 0.0,
    "shots_on_target_faced": 0.0,
    "goals_conceded_xgot_sample": 0.0,
    "xgot_faced": 0.0,
    "goals_conceded": 0.0,
}


def _position(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"goalkeeper", "gk"}:
        return "Goalkeeper"
    if normalized in {"defender", "def", "d"}:
        return "Defender"
    if normalized in {"midfielder", "mid", "m"}:
        return "Midfielder"
    if normalized in {"forward", "fwd", "fw", "f"}:
        return "Forward"
    return "Unknown"


def _number(row: dict[str, Any], name: str) -> float:
    value = row.get(name, NUMERIC_DEFAULTS.get(name, 0.0))
    try:
        result = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["season"]), _position(row.get("position_group"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std if std > 1e-9 else 1.0


def _fit_pass_priors(rows: list[dict[str, Any]]) -> pl.DataFrame:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        position = _position(row.get("position_group"))
        if position not in OUTFIELD_POSITIONS:
            continue
        grouped.setdefault((str(row["season"]), position), []).append(row)

    priors: list[dict[str, Any]] = []
    for (season, position), group in sorted(grouped.items()):
        eligible = [row for row in group if _number(row, "passes") >= 5.0]
        if not eligible:
            eligible = group
        attempts = [_number(row, "passes") for row in eligible]
        completed = [_number(row, "completed_passes") for row in eligible]
        total_attempts = sum(attempts)
        total_completed = min(sum(completed), total_attempts)
        p0 = total_completed / total_attempts if total_attempts > 0 else 0.75
        rates = [
            min(max(done / tried, 0.0), 1.0)
            for done, tried in zip(completed, attempts, strict=True)
            if tried > 0
        ]
        observed_variance = float(np.var(rates, ddof=1)) if len(rates) > 1 else 0.0
        inverse_attempts = [1.0 / tried for tried in attempts if tried > 0]
        sampling_variance = (
            p0 * (1.0 - p0) * float(np.mean(inverse_attempts))
            if inverse_attempts
            else 0.0
        )
        between_match_variance = min(
            max(observed_variance - sampling_variance, 0.0025), 0.05
        )
        prior_strength = min(
            max((p0 * (1.0 - p0) / between_match_variance) - 1.0, 5.0),
            60.0,
        )
        priors.append(
            {
                "season": season,
                "position_group": position,
                "pass_completion_prior": p0,
                "prior_strength": prior_strength,
                "observed_variance": observed_variance,
                "sampling_variance": sampling_variance,
                "between_match_variance": between_match_variance,
                "match_rows": len(eligible),
                "attempted_passes": total_attempts,
                "completed_passes": total_completed,
            }
        )
    if not priors:
        raise RatingModelError("No outfield rows were available to fit pass priors")
    return pl.DataFrame(priors, infer_schema_length=None)


def _prior_lookup(frame: pl.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    return {
        (str(row["season"]), str(row["position_group"])): (
            float(row["pass_completion_prior"]),
            float(row["prior_strength"]),
        )
        for row in frame.to_dicts()
    }


def _transform_rows(
    features: pl.DataFrame,
    priors: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, Any]]:
    missing = sorted(REQUIRED_FEATURE_COLUMNS.difference(features.columns))
    if missing:
        raise RatingModelError(f"Player-match features are missing columns: {missing}")

    transformed: list[dict[str, Any]] = []
    for source in features.to_dicts():
        row = dict(source)
        row["position_group"] = _position(row.get("position_group"))
        minutes = max(_number(row, "minutes"), 0.0)
        adjusted_minutes = min(90.0, max(30.0, minutes))
        for source_name, output_name in COUNT_FEATURES.items():
            per_90 = 90.0 * max(_number(row, source_name), 0.0) / adjusted_minutes
            row[f"{source_name}_90"] = per_90
            row[output_name] = math.log1p(per_90)

        for source_name, output_name in (
            ("xpv_added", "xpv_added_90"),
            ("opponent_threat_prevented", "opponent_threat_prevented_90"),
            (
                "defensive_net_threat_reduction",
                "defensive_net_threat_reduction_90",
            ),
        ):
            row[output_name] = 90.0 * _number(row, source_name) / adjusted_minutes

        p0, strength = priors.get(
            (str(row["season"]), row["position_group"]), (0.75, 20.0)
        )
        attempts = max(_number(row, "passes"), 0.0)
        completed = min(max(_number(row, "completed_passes"), 0.0), attempts)
        smoothed = (completed + strength * p0) / (attempts + strength)
        row["pass_completion_prior"] = p0
        row["pass_prior_strength"] = strength
        row["smoothed_pass_completion"] = smoothed
        row["pass_completion_above_expected"] = smoothed - p0
        row["finishing_above_expected"] = min(
            max(_number(row, "goals") - _number(row, "xg"), -2.0), 3.0
        )
        row["shot_placement_above_expected"] = min(
            max(_number(row, "xgot") - _number(row, "xg"), -2.0), 3.0
        )
        shots_on_target = max(_number(row, "shots_on_target_faced"), 0.0)
        goals_in_sample = max(_number(row, "goals_conceded_xgot_sample"), 0.0)
        row["saves"] = max(shots_on_target - goals_in_sample, 0.0)
        row["save_percentage"] = (
            row["saves"] / shots_on_target if shots_on_target > 0 else 0.0
        )
        row["goals_prevented"] = _number(row, "xgot_faced") - goals_in_sample
        transformed.append(row)
    return transformed


def _fit_zscore_rows(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for (season, position), group in sorted(grouped.items()):
        wanted = list(features)
        if position == "Goalkeeper":
            wanted = ["goals_prevented", "save_percentage"]
            sample = [row for row in group if _number(row, "shots_on_target_faced") > 0]
            if sample:
                group = sample
        elif position not in OUTFIELD_POSITIONS:
            continue
        for feature in wanted:
            mean, std = _mean_std([_number(row, feature) for row in group])
            output.append(
                {
                    "season": season,
                    "position_group": position,
                    "feature": feature,
                    "mean": mean,
                    "std": std,
                    "sample_size": len(group),
                }
            )
    return output


def _stats_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    return {
        (str(row["season"]), str(row["position_group"]), str(row["feature"])): (
            float(row["mean"]),
            float(row["std"]),
        )
        for row in rows
    }


def _z(
    row: dict[str, Any],
    feature: str,
    stats: dict[tuple[str, str, str], tuple[float, float]],
    clip: float,
) -> float:
    key = (str(row["season"]), str(row["position_group"]), feature)
    if key not in stats:
        raise RatingModelError(f"Missing normalization statistic for {key}")
    mean, std = stats[key]
    return min(max((_number(row, feature) - mean) / std, -clip), clip)


def _outfield_components(
    row: dict[str, Any],
    stats: dict[tuple[str, str, str], tuple[float, float]],
    clip: float,
) -> dict[str, float]:
    z = lambda feature: _z(row, feature, stats, clip)
    return {
        "threat_component": (
            0.40 * z("log_xg_90")
            + 0.35 * z("log_xgot_90")
            + 0.25 * z("log_shots_90")
        ),
        "creation_component": (
            0.55 * z("log_xa_90")
            + 0.25 * z("log_key_passes_90")
            + 0.20 * z("log_big_chances_created_90")
        ),
        "progression_component": (
            0.45 * z("log_progressive_passes_90")
            + 0.35 * z("log_progressive_carries_90")
            + 0.20 * z("log_final_third_carries_90")
        ),
        "retention_component": z("pass_completion_above_expected"),
        "attacking_xpv_component": z("xpv_added_90"),
        "defensive_component": (
            0.60 * z("opponent_threat_prevented_90")
            + 0.40 * z("defensive_net_threat_reduction_90")
        ),
        "finishing_component": (
            0.60 * z("finishing_above_expected")
            + 0.40 * z("shot_placement_above_expected")
        ),
    }


def _reliable_composite(
    row: dict[str, Any],
    stats: dict[tuple[str, str, str], tuple[float, float]],
    config: RatingModelConfig,
) -> tuple[dict[str, float], float, float]:
    components = _outfield_components(row, stats, config.feature_z_clip)
    weights = POSITION_WEIGHTS[str(row["position_group"])]
    raw = sum(components[name] * weight for name, weight in weights.items())
    minutes_reliability = math.sqrt(min(max(_number(row, "minutes") / 90.0, 0.0), 1.0))
    reliability = 0.25 + 0.75 * minutes_reliability
    return components, raw, reliability * raw


def _outfield_base_rating(
    standardized_composite: float,
    config: RatingModelConfig,
) -> float:
    """Map a standardized performance to a conservative, bounded rating.

    V1's 3.70*tanh(z/2) curve put an ordinary top-decile performance close to
    9 before any match-outcome adjustment.  The v2 curve keeps 6 as the
    neutral point, maps z=2.5 to about 8.1, and reserves 9+ for genuinely
    exceptional performances with decisive actions.
    """
    return 6.0 + config.outfield_rating_span * math.tanh(
        standardized_composite / config.outfield_rating_temperature
    )


def _decisive_action_bonus(
    row: dict[str, Any],
    assist_overperformance: float,
    config: RatingModelConfig,
) -> float:
    """Apply a deliberately small outcome bonus after chance-quality scoring.

    Goals and assists already influence finishing, creation and xPV.  This is
    only residual credit for the scoreboard outcome, not a second full feature
    component.
    """
    return min(
        max(
            config.goal_bonus * min(_number(row, "goals"), 3.0)
            + config.assist_bonus * min(_number(row, "assists"), 2.0)
            + config.assist_overperformance_bonus * assist_overperformance,
            0.0,
        ),
        config.decisive_action_bonus_cap,
    )


def fit_rating_artifacts(
    features: pl.DataFrame,
    artifact_directory: Path,
    *,
    config: RatingModelConfig | None = None,
) -> Path:
    config = config or RatingModelConfig()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    source_rows = features.to_dicts()
    priors = _fit_pass_priors(source_rows)
    transformed = _transform_rows(features, _prior_lookup(priors))
    zscore_rows = _fit_zscore_rows(transformed, STANDARDIZED_FEATURES)
    stats = _stats_lookup(zscore_rows)

    composite_groups: dict[tuple[str, str], list[float]] = {}
    for row in transformed:
        if row["position_group"] not in OUTFIELD_POSITIONS:
            continue
        _, _, reliable = _reliable_composite(row, stats, config)
        composite_groups.setdefault(_group_key(row), []).append(reliable)
    for (season, position), values in sorted(composite_groups.items()):
        mean, std = _mean_std(values)
        zscore_rows.append(
            {
                "season": season,
                "position_group": position,
                "feature": "reliability_adjusted_composite",
                "mean": mean,
                "std": std,
                "sample_size": len(values),
            }
        )

    zscore_stats = pl.DataFrame(zscore_rows, infer_schema_length=None).sort(
        ["season", "position_group", "feature"]
    )
    zscore_stats.write_parquet(
        artifact_directory / "zscore_stats.parquet", compression="zstd"
    )
    priors.write_parquet(
        artifact_directory / "pass_completion_priors.parquet", compression="zstd"
    )
    write_json(artifact_directory / "rating_model_config.json", asdict(config))
    write_json(
        artifact_directory / "feature_schema.json",
        {
            "version": config.version,
            "required_columns": sorted(REQUIRED_FEATURE_COLUMNS),
            "numeric_defaults": NUMERIC_DEFAULTS,
            "standardized_features": list(STANDARDIZED_FEATURES),
            "position_weights": POSITION_WEIGHTS,
            "rating_column": "post_match_rating",
        },
    )
    return artifact_directory


def load_rating_config(artifact_directory: Path) -> RatingModelConfig:
    path = artifact_directory / "rating_model_config.json"
    if not path.exists():
        raise RatingModelError(f"Missing rating model config: {path}")
    return RatingModelConfig(**json.loads(path.read_text(encoding="utf-8")))


def score_rating_features(
    features: pl.DataFrame,
    artifact_directory: Path,
) -> pl.DataFrame:
    config = load_rating_config(artifact_directory)
    priors_path = artifact_directory / "pass_completion_priors.parquet"
    stats_path = artifact_directory / "zscore_stats.parquet"
    if not priors_path.exists() or not stats_path.exists():
        raise RatingModelError(f"Incomplete rating artifacts in {artifact_directory}")
    transformed = _transform_rows(
        features, _prior_lookup(pl.read_parquet(priors_path))
    )
    stats = _stats_lookup(pl.read_parquet(stats_path).to_dicts())
    output: list[dict[str, Any]] = []
    for row in transformed:
        position = str(row["position_group"])
        minutes = max(_number(row, "minutes"), 0.0)
        assist_overperformance = min(
            max(max(_number(row, "assists") - _number(row, "xa"), 0.0), 0.0),
            2.0,
        )
        bonus = _decisive_action_bonus(row, assist_overperformance, config)
        card_penalty = (
            0.15 * _number(row, "yellow_cards")
            + 1.25 * _number(row, "red_cards")
        )
        big_chance_penalty = min(
            config.big_chance_miss_penalty * _number(row, "big_chances_missed"),
            config.big_chance_miss_penalty_cap,
        )
        own_goal_penalty = min(
            config.own_goal_penalty * _number(row, "own_goals"),
            config.own_goal_penalty_cap,
        )

        components: dict[str, float] = {}
        raw_composite: float | None = None
        adjusted_composite: float | None = None
        standardized_composite: float | None = None
        reliability: float | None = None
        clean_sheet_bonus = 0.0
        goalkeeper_component: float | None = None
        base_rating: float | None = None
        if position in OUTFIELD_POSITIONS and minutes > 0:
            components, raw_composite, adjusted_composite = _reliable_composite(
                row, stats, config
            )
            reliability = (
                adjusted_composite / raw_composite if abs(raw_composite) > 1e-12 else
                0.25 + 0.75 * math.sqrt(min(max(minutes / 90.0, 0.0), 1.0))
            )
            standardized_composite = _z(
                {**row, "reliability_adjusted_composite": adjusted_composite},
                "reliability_adjusted_composite",
                stats,
                config.composite_z_clip,
            )
            base_rating = _outfield_base_rating(standardized_composite, config)
        elif position == "Goalkeeper" and minutes > 0:
            goalkeeper_component = (
                0.80 * _z(row, "goals_prevented", stats, config.feature_z_clip)
                + 0.20 * _z(row, "save_percentage", stats, config.feature_z_clip)
            )
            reliability = math.sqrt(
                min(max(_number(row, "shots_on_target_faced") / 5.0, 0.0), 1.0)
            )
            clean_sheet_bonus = (
                0.15
                if _number(row, "goals_conceded") == 0
                and _number(row, "shots_on_target_faced") >= 3
                else 0.0
            )
            base_rating = 6.0 + reliability * goalkeeper_component + clean_sheet_bonus

        rating = (
            min(
                max(
                    base_rating
                    + bonus
                    - card_penalty
                    - big_chance_penalty
                    - own_goal_penalty,
                    1.0,
                ),
                10.0,
            )
            if base_rating is not None
            else None
        )
        output.append(
            {
                **row,
                **components,
                "raw_position_composite": raw_composite,
                "appearance_reliability": reliability,
                "reliability_adjusted_composite": adjusted_composite,
                "standardized_composite": standardized_composite,
                "goalkeeper_component": goalkeeper_component,
                "clean_sheet_bonus": clean_sheet_bonus,
                "base_rating": base_rating,
                "assist_overperformance": assist_overperformance,
                "decisive_action_bonus": bonus,
                "card_penalty": card_penalty,
                "big_chance_miss_penalty": big_chance_penalty,
                "own_goal_penalty": own_goal_penalty,
                "post_match_rating": rating,
                "rating_version": config.version,
            }
        )
    frame = pl.DataFrame(output, infer_schema_length=None)
    duplicates = (
        frame.group_by(["season", "match_id", "whoscored_player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicates:
        raise RatingModelError(f"Found {duplicates} duplicate player-match rating keys")
    invalid = frame.filter(
        pl.col("post_match_rating").is_not_null()
        & ~pl.col("post_match_rating").is_between(1.0, 10.0, closed="both")
    ).height
    if invalid:
        raise RatingModelError(f"Found {invalid} ratings outside [1, 10]")
    return frame
