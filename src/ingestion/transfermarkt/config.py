from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeagueConfig:
    competition_id: str
    competition_name: str
    base_url: str
    competition_path_template: str

    def competition_url(self, season: int) -> str:
        path = self.competition_path_template.format(season=season)
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def load_league_config(path: Path) -> LeagueConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "competition_id",
        "competition_name",
        "base_url",
        "competition_path_template",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"League config is missing fields: {', '.join(missing)}")

    return LeagueConfig(
        competition_id=str(payload["competition_id"]),
        competition_name=str(payload["competition_name"]),
        base_url=str(payload["base_url"]),
        competition_path_template=str(payload["competition_path_template"]),
    )
