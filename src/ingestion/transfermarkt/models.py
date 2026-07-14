from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClubRecord:
    club_id: int
    club_name: str
    profile_path: str
    roster_url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "club_id": self.club_id,
            "club_name": self.club_name,
            "profile_path": self.profile_path,
            "roster_url": self.roster_url,
        }


@dataclass(frozen=True)
class PlayerRecord:
    player_id: int
    player_name: str
    profile_path: str
    club_id: int
    club_name: str
    season: int
    position: str | None = None
    date_of_birth: str | None = None
    nationalities: tuple[str, ...] = ()
    roster_status: str = "current_squad"
    source_section: str = "squad_table"
    destination_club_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "profile_path": self.profile_path,
            "club_id": self.club_id,
            "club_name": self.club_name,
            "season": self.season,
            "position": self.position,
            "date_of_birth": self.date_of_birth,
            "nationalities": list(self.nationalities),
            "roster_status": self.roster_status,
            "source_section": self.source_section,
            "destination_club_name": self.destination_club_name,
        }


@dataclass(frozen=True)
class FetchFailure:
    stage: str
    identifier: str
    url: str
    attempts: int
    error: str
    status_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "identifier": self.identifier,
            "url": self.url,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class TransfermarktIngestionResult:
    status: str
    run_directory: Path
    manifest_path: Path
    club_count: int
    player_count: int
    valuation_count: int
    failure_count: int
