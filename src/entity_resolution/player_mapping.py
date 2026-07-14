from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ingestion.common import write_json
from ingestion.transfermarkt.normalize import normalize_name


class PlayerMappingError(RuntimeError):
    """Raised when a safe one-to-one player crosswalk cannot be produced."""


@dataclass(frozen=True)
class PlayerMappingResult:
    output_path: Path
    review_path: Path
    mapped_players: int
    review_players: int


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def collect_whoscored_players(
    normalized_root: Path,
    *,
    competition: str,
    season: str,
) -> pl.DataFrame:
    source = normalized_root / f"competition={competition}" / f"season={season}" / "matches"
    paths = sorted(source.glob("match_id=*/player_matches.parquet"))
    if not paths:
        raise PlayerMappingError(f"No WhoScored player-match files found under {source}")
    frames = [
        pl.read_parquet(path).select(
            pl.col("player_id").cast(pl.Int64).alias("whoscored_player_id"),
            pl.col("player_name").cast(pl.String),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("position").cast(pl.String),
        )
        for path in paths
    ]
    rows = pl.concat(frames, how="diagonal_relaxed").drop_nulls(
        ["whoscored_player_id", "player_name"]
    )
    return (
        rows.group_by("whoscored_player_id")
        .agg(
            pl.col("player_name").mode().first().alias("whoscored_player_name"),
            pl.col("team_id").drop_nulls().last().alias("whoscored_team_id"),
            pl.col("position").drop_nulls().last().alias("whoscored_position"),
            pl.len().alias("appearances_observed"),
        )
        .with_columns(
            pl.col("whoscored_player_name")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("normalized_player_name")
        )
        .sort("whoscored_player_id")
    )


def _manual_rows(
    manual_overrides: Path | None,
    whoscored: pl.DataFrame,
    transfermarkt: pl.DataFrame,
) -> list[dict[str, Any]]:
    if manual_overrides is None:
        return []
    frame = (
        pl.read_parquet(manual_overrides)
        if manual_overrides.suffix.casefold() == ".parquet"
        else pl.read_csv(manual_overrides)
    )
    required = {"whoscored_player_id", "transfermarkt_player_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PlayerMappingError(f"Manual override file is missing columns: {missing}")
    ws = {
        int(row["whoscored_player_id"]): row for row in whoscored.to_dicts()
    }
    tm = {
        int(row["transfermarkt_player_id"]): row for row in transfermarkt.to_dicts()
    }
    output: list[dict[str, Any]] = []
    for override in frame.to_dicts():
        ws_id = int(override["whoscored_player_id"])
        tm_id = int(override["transfermarkt_player_id"])
        if ws_id not in ws or tm_id not in tm:
            raise PlayerMappingError(
                f"Manual override references an unknown player: WhoScored={ws_id}, Transfermarkt={tm_id}"
            )
        output.append(
            {
                "whoscored_player_id": ws_id,
                "transfermarkt_player_id": tm_id,
                "whoscored_player_name": ws[ws_id]["whoscored_player_name"],
                "transfermarkt_player_name": tm[tm_id]["player_name"],
                "normalized_player_name": ws[ws_id]["normalized_player_name"],
                "match_method": "manual_override",
                "confidence": 1.0,
                "review_status": "approved",
            }
        )
    return output


def build_player_mapping(
    *,
    transfermarkt_players_path: Path,
    whoscored_normalized_root: Path,
    competition: str,
    season: str,
    output_path: Path = Path(
        "data/normalized/entity_resolution/player_mapping_exact.parquet"
    ),
    manual_overrides: Path | None = None,
) -> PlayerMappingResult:
    transfermarkt = pl.read_parquet(transfermarkt_players_path).select(
        "transfermarkt_player_id",
        "player_name",
        "normalized_player_name",
        "date_of_birth",
        "club_name",
        "position",
    ).unique("transfermarkt_player_id", keep="last")
    whoscored = collect_whoscored_players(
        whoscored_normalized_root, competition=competition, season=season
    )
    manual = _manual_rows(manual_overrides, whoscored, transfermarkt)
    used_ws = {int(row["whoscored_player_id"]) for row in manual}
    used_tm = {int(row["transfermarkt_player_id"]) for row in manual}

    ws_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in whoscored.to_dicts():
        ws_by_name.setdefault(str(row["normalized_player_name"]), []).append(row)
    tm_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in transfermarkt.to_dicts():
        tm_by_name.setdefault(str(row["normalized_player_name"]), []).append(row)

    mapped = list(manual)
    review: list[dict[str, Any]] = []
    for ws_row in whoscored.to_dicts():
        ws_id = int(ws_row["whoscored_player_id"])
        if ws_id in used_ws:
            continue
        name = str(ws_row["normalized_player_name"])
        candidates = [
            row
            for row in tm_by_name.get(name, [])
            if int(row["transfermarkt_player_id"]) not in used_tm
        ]
        if len(ws_by_name.get(name, [])) == 1 and len(candidates) == 1:
            tm_row = candidates[0]
            tm_id = int(tm_row["transfermarkt_player_id"])
            used_tm.add(tm_id)
            mapped.append(
                {
                    "whoscored_player_id": ws_id,
                    "transfermarkt_player_id": tm_id,
                    "whoscored_player_name": ws_row["whoscored_player_name"],
                    "transfermarkt_player_name": tm_row["player_name"],
                    "normalized_player_name": name,
                    "match_method": "exact_unique_normalized_name",
                    "confidence": 0.95,
                    "review_status": "approved",
                }
            )
        else:
            review.append(
                {
                    "whoscored_player_id": ws_id,
                    "whoscored_player_name": ws_row["whoscored_player_name"],
                    "normalized_player_name": name,
                    "candidate_count": len(candidates),
                    "candidate_transfermarkt_ids": [
                        int(row["transfermarkt_player_id"]) for row in candidates
                    ],
                    "candidate_names": [str(row["player_name"]) for row in candidates],
                    "review_reason": "no_exact_candidate" if not candidates else "ambiguous_exact_name",
                }
            )

    mapping_frame = pl.DataFrame(mapped, infer_schema_length=None) if mapped else pl.DataFrame(
        schema={
            "whoscored_player_id": pl.Int64,
            "transfermarkt_player_id": pl.Int64,
            "whoscored_player_name": pl.String,
            "transfermarkt_player_name": pl.String,
            "normalized_player_name": pl.String,
            "match_method": pl.String,
            "confidence": pl.Float64,
            "review_status": pl.String,
        }
    )
    duplicate_ws = mapping_frame.group_by("whoscored_player_id").len().filter(pl.col("len") > 1).height
    duplicate_tm = mapping_frame.group_by("transfermarkt_player_id").len().filter(pl.col("len") > 1).height
    if duplicate_ws or duplicate_tm:
        raise PlayerMappingError("Player mapping must be one-to-one across both sources")
    mapping_frame = mapping_frame.sort("whoscored_player_id")
    review_frame = pl.DataFrame(review, infer_schema_length=None) if review else pl.DataFrame(
        schema={
            "whoscored_player_id": pl.Int64,
            "whoscored_player_name": pl.String,
            "normalized_player_name": pl.String,
            "candidate_count": pl.Int64,
            "candidate_transfermarkt_ids": pl.List(pl.Int64),
            "candidate_names": pl.List(pl.String),
            "review_reason": pl.String,
        }
    )
    review_path = output_path.with_name("player_mapping_review.parquet")
    _atomic_parquet(mapping_frame, output_path)
    _atomic_parquet(review_frame, review_path)
    write_json(
        output_path.with_name("player_mapping_summary.json"),
        {
            "competition": competition,
            "season": season,
            "transfermarkt_players_path": str(transfermarkt_players_path),
            "whoscored_normalized_root": str(whoscored_normalized_root),
            "whoscored_players": whoscored.height,
            "transfermarkt_players": transfermarkt.height,
            "mapped_players": mapping_frame.height,
            "review_players": review_frame.height,
            "manual_overrides": str(manual_overrides) if manual_overrides else None,
            "mapping_output": str(output_path),
            "review_output": str(review_path),
        },
    )
    return PlayerMappingResult(
        output_path=output_path,
        review_path=review_path,
        mapped_players=mapping_frame.height,
        review_players=review_frame.height,
    )
