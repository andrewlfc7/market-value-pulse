from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl


class ParquetRepository:
    """Small file-backed serving layer; replaceable by Postgres without API changes."""

    def __init__(self, root: Path = Path("data/serving")) -> None:
        self.root = root

    def _rows(self, filename: str) -> list[dict[str, Any]]:
        path = self.root / filename
        return pl.read_parquet(path).to_dicts() if path.exists() else []

    def players(self) -> list[dict[str, Any]]:
        return self._rows("players.parquet")

    def player(self, player_id: str) -> dict[str, Any] | None:
        player = next(
            (row for row in self.players() if str(row.get("player_id")) == player_id),
            None,
        )
        if player is None:
            return None
        player["valuation_history"] = [
            row for row in self._rows("valuation_history.parquet")
            if str(row.get("player_id")) == player_id
        ]
        player["match_impacts"] = [
            row for row in self._rows("match_impacts.parquet")
            if str(row.get("player_id")) == player_id
        ]
        player["match_impacts"].sort(
            key=lambda row: str(row.get("match_datetime") or ""), reverse=True
        )
        return player

    def pipeline_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self._rows("pipeline_runs.parquet")
        return sorted(
            rows, key=lambda row: str(row.get("started_at") or ""), reverse=True
        )[:limit]


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _plain_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _plain(value) for key, value in row.items()}


class PostgresRepository:
    """Read model-serving state from the idempotently loaded PostgreSQL tables."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.database_url, row_factory=dict_row)

    def players(self) -> list[dict[str, Any]]:
        query = """
            SELECT p.player_id, p.display_name, p.position_group AS position,
                   fs.form_rating_ewm AS current_form_rating,
                   fs.rolling_3_match_rating, fs.rolling_20_match_rating,
                   mv.valuation_date AS latest_valuation_date,
                   mv.value_eur AS current_market_value_eur,
                   ve.estimate_eur AS estimated_value_eur,
                   ve.lower_eur AS estimated_lower_eur,
                   ve.upper_eur AS estimated_upper_eur,
                   ve.predicted_pct_change,
                   ve.probability_value_increase,
                   ve.model_version AS valuation_model_version,
                   ve.direction, ve.confidence, ve.scored_at AS refreshed_at
            FROM players p
            LEFT JOIN LATERAL (
              SELECT * FROM player_form_state s
              WHERE s.player_id = p.player_id ORDER BY updated_at DESC LIMIT 1
            ) fs ON TRUE
            LEFT JOIN LATERAL (
              SELECT * FROM market_valuations v
              WHERE v.player_id = p.player_id ORDER BY valuation_date DESC LIMIT 1
            ) mv ON TRUE
            LEFT JOIN LATERAL (
              SELECT * FROM valuation_estimates e
              WHERE e.player_id = p.player_id ORDER BY scored_at DESC LIMIT 1
            ) ve ON TRUE
            ORDER BY p.display_name
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query)
            return [_plain_row(dict(row)) for row in cursor.fetchall()]

    def player(self, player_id: str) -> dict[str, Any] | None:
        player = next(
            (row for row in self.players() if str(row["player_id"]) == player_id),
            None,
        )
        if player is None:
            return None
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT valuation_date, value_eur, source
                FROM market_valuations
                WHERE player_id = %s ORDER BY valuation_date
                """,
                (int(player_id),),
            )
            player["valuation_history"] = [
                _plain_row(dict(row)) for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                WITH selected_rating_version AS (
                  SELECT rating_version
                  FROM player_form_state
                  WHERE player_id = %s
                  ORDER BY updated_at DESC
                  LIMIT 1
                )
                SELECT r.match_id, m.kickoff_at AS match_datetime, r.rating,
                       r.minutes, r.rating_version, r.features,
                       vi.estimated_value_delta_eur,
                       vi.estimate_eur AS valuation_estimate_eur,
                       vi.lower_eur AS valuation_lower_90_eur,
                       vi.upper_eur AS valuation_upper_90_eur,
                       vi.probability_value_increase,
                       vi.valuation_status AS valuation_update_status,
                       vi.replay_run_id
                FROM player_match_ratings r
                JOIN matches m ON m.match_id = r.match_id
                LEFT JOIN LATERAL (
                  SELECT * FROM valuation_match_impacts i
                  WHERE i.player_id = r.player_id AND i.match_id = r.match_id
                  ORDER BY i.scored_at DESC LIMIT 1
                ) vi ON TRUE
                WHERE r.player_id = %s
                  AND r.rating_version = (
                    SELECT rating_version FROM selected_rating_version
                  )
                ORDER BY m.kickoff_at DESC LIMIT 20
                """,
                (int(player_id), int(player_id)),
            )
            impacts = []
            for record in cursor.fetchall():
                row = _plain_row(dict(record))
                features = row.pop("features") or {}
                rating = row.get("rating")
                components = [
                    ("Shot threat", features.get("threat_component")),
                    ("Chance creation", features.get("creation_component")),
                    ("Ball progression", features.get("progression_component")),
                    ("Defensive threat prevention", features.get("defensive_component")),
                    ("Finishing", features.get("finishing_component")),
                    ("Shot stopping", features.get("goalkeeper_component")),
                ]
                available = [
                    (label, abs(float(value)))
                    for label, value in components
                    if value is not None
                ]
                available.sort(key=lambda item: item[1], reverse=True)
                row.update(
                    {
                        "player_id": int(player_id),
                        "performance_impact_score": (
                            float(rating) - 6.0 if rating is not None else None
                        ),
                        "impact_direction": (
                            "positive"
                            if rating is not None and float(rating) > 6.25
                            else "negative"
                            if rating is not None and float(rating) < 5.75
                            else "neutral"
                        ),
                        "explanation": " · ".join(
                            label for label, _ in available[:2]
                        )
                        or "Position-adjusted match performance",
                        "estimated_value_delta_eur": row.get(
                            "estimated_value_delta_eur"
                        ),
                    }
                )
                impacts.append(row)
            player["match_impacts"] = impacts
        return player

    def pipeline_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, pipeline, status, started_at, completed_at, counts, error
                FROM pipeline_runs ORDER BY started_at DESC LIMIT %s
                """,
                (limit,),
            )
            return [_plain_row(dict(row)) for row in cursor.fetchall()]


def build_repository() -> ParquetRepository | PostgresRepository:
    database_url = os.environ.get("DATABASE_URL")
    return PostgresRepository(database_url) if database_url else ParquetRepository()


def read_metadata(name: str, root: Path = Path("metadata")) -> dict[str, Any]:
    path = root / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
