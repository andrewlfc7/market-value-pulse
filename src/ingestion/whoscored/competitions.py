from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Competition:
    key: str
    name: str
    aliases: tuple[str, ...]
    region_id: int
    tournament_id: int
    slug: str
    expected_matches: int | None
    stage_overrides: dict[str, int]

    @property
    def tournament_url(self) -> str:
        return (
            "https://www.whoscored.com/regions/"
            f"{self.region_id}/tournaments/{self.tournament_id}/{self.slug}"
        )


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_competitions(
    path: Path = Path("config/whoscored/competitions.json"),
) -> list[Competition]:
    if not path.exists():
        raise FileNotFoundError(f"Missing WhoScored competition registry: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return [
        Competition(
            key=str(row["key"]),
            name=str(row["name"]),
            aliases=tuple(str(alias) for alias in row.get("aliases", [])),
            region_id=int(row["region_id"]),
            tournament_id=int(row["tournament_id"]),
            slug=str(row["slug"]),
            expected_matches=(
                int(row["expected_matches"])
                if row.get("expected_matches") is not None
                else None
            ),
            stage_overrides={
                str(season): int(stage)
                for season, stage in row.get("stage_overrides", {}).items()
            },
        )
        for row in payload.get("competitions", [])
    ]


def resolve_competition(
    value: str,
    *,
    registry_path: Path = Path("config/whoscored/competitions.json"),
) -> Competition:
    competitions = load_competitions(registry_path)
    target = _key(value)
    for competition in competitions:
        candidates = {competition.key, competition.name, *competition.aliases}
        if target in {_key(candidate) for candidate in candidates}:
            return competition
    available = ", ".join(row.key for row in competitions)
    raise ValueError(f"Unknown WhoScored competition {value!r}; configured: {available}")
