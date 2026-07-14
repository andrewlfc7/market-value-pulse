from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from ingestion.whoscored.normalize import NormalizedMatch


class WhoScoredValidationError(RuntimeError):
    """Raised when a normalized match violates a critical contract."""


@dataclass(frozen=True)
class DataQualityCheck:
    name: str
    passed: bool
    severity: str
    details: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _check(
    checks: list[DataQualityCheck],
    name: str,
    passed: bool,
    details: str,
    *,
    severity: str = "error",
) -> None:
    checks.append(
        DataQualityCheck(
            name=name,
            passed=bool(passed),
            severity=severity,
            details=details,
        )
    )


def validate_normalized_match(bundle: NormalizedMatch) -> list[DataQualityCheck]:
    checks: list[DataQualityCheck] = []
    matches = bundle.matches
    events = bundle.events
    shots = bundle.shots
    player_matches = bundle.player_matches

    _check(checks, "one_match_row", matches.height == 1, f"rows={matches.height}")
    if matches.height == 1:
        row = matches.row(0, named=True)
        home = row.get("home_team_id")
        away = row.get("away_team_id")
        _check(
            checks,
            "distinct_teams",
            home is not None and away is not None and home != away,
            f"home_team_id={home}, away_team_id={away}",
        )

    _check(checks, "events_present", not events.is_empty(), f"rows={events.height}")
    required_event_columns = {
        "match_id",
        "persistent_id",
        "event_id",
        "team_id",
        "minute",
        "type_display_name",
    }
    missing = sorted(required_event_columns.difference(events.columns))
    _check(checks, "event_schema", not missing, f"missing={missing}")
    if not events.is_empty() and not missing:
        duplicates = (
            events.group_by(["match_id", "persistent_id"])
            .len()
            .filter(pl.col("persistent_id").is_not_null() & (pl.col("len") > 1))
            .height
        )
        _check(
            checks,
            "unique_persistent_event_id",
            duplicates == 0,
            f"duplicate_keys={duplicates}",
        )
        for coordinate in ("x", "y", "end_x", "end_y"):
            if coordinate not in events.columns:
                continue
            invalid = events.filter(
                pl.col(coordinate).is_not_null()
                & ~pl.col(coordinate).cast(pl.Float64, strict=False).is_between(
                    0.0, 100.0, closed="both"
                )
            ).height
            _check(
                checks,
                f"{coordinate}_range",
                invalid == 0,
                f"invalid_rows={invalid}",
            )

    if not shots.is_empty():
        non_shots = shots.filter(~pl.col("is_shot")).height
        _check(checks, "shots_are_shots", non_shots == 0, f"invalid_rows={non_shots}")

    if not player_matches.is_empty():
        duplicates = (
            player_matches.group_by(["match_id", "player_id"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        _check(
            checks,
            "unique_player_match",
            duplicates == 0,
            f"duplicate_keys={duplicates}",
        )
    else:
        _check(
            checks,
            "player_matches_present",
            False,
            "The source payload did not expose lineup players",
            severity="warning",
        )

    failed_errors = [
        check.name
        for check in checks
        if not check.passed and check.severity == "error"
    ]
    if failed_errors:
        raise WhoScoredValidationError(
            "Critical WhoScored checks failed: " + ", ".join(failed_errors)
        )
    return checks
