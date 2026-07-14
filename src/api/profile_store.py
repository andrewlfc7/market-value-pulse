from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl


class ProfileStore:
    """Read player-profile serving artifacts independently of the main repository."""

    def __init__(self, root: Path = Path("data/serving")) -> None:
        self.root = root

    @property
    def profiles_path(self) -> Path:
        return self.root / "player_profiles.parquet"

    @property
    def similarities_path(self) -> Path:
        return self.root / "player_similarities.parquet"

    @staticmethod
    def _plain(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @classmethod
    def _plain_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        return {key: cls._plain(value) for key, value in row.items()}

    @staticmethod
    def _player_id_column(frame: pl.DataFrame) -> str | None:
        if "player_id" in frame.columns:
            return "player_id"
        if "whoscored_player_id" in frame.columns:
            return "whoscored_player_id"
        return None

    def player_profile(self, player_id: str) -> dict[str, Any] | None:
        if not self.profiles_path.exists():
            return None

        frame = pl.read_parquet(self.profiles_path)
        id_column = self._player_id_column(frame)
        if id_column is None:
            return None

        selected = frame.filter(pl.col(id_column).cast(pl.String) == str(player_id))
        if selected.is_empty():
            return None

        if "display_order" in selected.columns:
            selected = selected.sort("display_order")

        rows = [self._plain_row(row) for row in selected.to_dicts()]
        first = rows[0]

        metrics = []
        for index, row in enumerate(rows):
            metrics.append(
                {
                    "key": row.get("metric_key"),
                    "label": row.get("metric_label"),
                    "phase": row.get("phase"),
                    "value": row.get("metric_value"),
                    "percentile": row.get("percentile"),
                    "higher_is_better": row.get("higher_is_better", True),
                    "display_order": row.get("display_order", index),
                }
            )

        return {
            "player_id": int(first.get(id_column) or player_id),
            "whoscored_player_id": int(
                first.get("whoscored_player_id")
                or first.get(id_column)
                or player_id
            ),
            "player_name": first.get("player_name"),
            "season": first.get("season"),
            "primary_role": first.get("primary_role"),
            "secondary_role": first.get("secondary_role"),
            "primary_role_share": first.get("primary_role_share"),
            "is_hybrid_role": first.get("is_hybrid_role"),
            "minutes": first.get("minutes"),
            "appearances": first.get("appearances"),
            "sample_status": first.get("sample_status"),
            "benchmark": {
                "competition": "EPL",
                "role": first.get("primary_role"),
                "minimum_minutes": first.get("benchmark_minutes", 900),
            },
            "metrics": metrics,
        }

    def similar_players(
        self,
        player_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.similarities_path.exists():
            return []

        frame = pl.read_parquet(self.similarities_path)
        id_column = self._player_id_column(frame)
        if id_column is None:
            return []

        selected = frame.filter(pl.col(id_column).cast(pl.String) == str(player_id))
        if selected.is_empty():
            return []

        if "rank" in selected.columns:
            selected = selected.sort("rank")
        elif "profile_similarity" in selected.columns:
            selected = selected.sort("profile_similarity", descending=True)
        elif "similarity" in selected.columns:
            selected = selected.sort("similarity", descending=True)

        return [
            self._plain_row(row)
            for row in selected.head(max(1, min(limit, 25))).to_dicts()
        ]
