from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ingestion.common import (
    atomic_write_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from ingestion.progress import ProgressCallback, ProgressEmitter, ProgressUpdate
from ingestion.transfermarkt.config import (
    LeagueConfig,
    load_league_config,
)
from ingestion.transfermarkt.http import (
    FetchError,
    TransfermarktHttpClient,
)
from ingestion.transfermarkt.models import (
    ClubRecord,
    FetchFailure,
    PlayerRecord,
    TransfermarktIngestionResult,
)
from ingestion.transfermarkt.parse import (
    parse_clubs,
    parse_players,
)


class TransfermarktIngestionError(RuntimeError):
    """Raised when the source cannot provide the minimum required raw dataset."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _fetch_club_roster(
    *,
    client: TransfermarktHttpClient,
    club: ClubRecord,
    season: int,
    clubs_html_directory: Path,
) -> tuple[int, list[PlayerRecord], FetchFailure | None]:
    try:
        response = await client.get(club.roster_url)
        html_path = clubs_html_directory / f"{club.club_id}.html"
        atomic_write_bytes(html_path, response.content)
        players = parse_players(
            html=response.content.decode("utf-8", errors="replace"),
            club=club,
            season=season,
        )
        if not players:
            return club.club_id, [], FetchFailure(
                stage="club_roster_parse",
                identifier=str(club.club_id),
                url=club.roster_url,
                attempts=response.attempts,
                error="No player profile links were found.",
                status_code=response.status_code,
            )
        return club.club_id, players, None
    except FetchError as exc:
        return club.club_id, [], FetchFailure(
            stage="club_roster_fetch",
            identifier=str(club.club_id),
            url=club.roster_url,
            attempts=exc.attempts,
            status_code=exc.status_code,
            error=str(exc),
        )


async def _fetch_player_valuation(
    *,
    client: TransfermarktHttpClient,
    base_url: str,
    player: PlayerRecord,
    valuations_directory: Path,
) -> tuple[int, bool, FetchFailure | None]:
    url = (
        f"{base_url.rstrip('/')}/ceapi/marketValueDevelopment/graph/"
        f"{player.player_id}"
    )
    try:
        response = await client.get(url, accept_json=True)
        payload = json.loads(response.content)
        records = payload.get("list")
        if not isinstance(records, list):
            return player.player_id, False, FetchFailure(
                stage="valuation_validate",
                identifier=str(player.player_id),
                url=url,
                attempts=response.attempts,
                status_code=response.status_code,
                error="Response does not contain a list field.",
            )

        output_path = valuations_directory / f"{player.player_id}.json"
        atomic_write_bytes(
            output_path,
            (
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        return player.player_id, True, None
    except json.JSONDecodeError as exc:
        return player.player_id, False, FetchFailure(
            stage="valuation_parse",
            identifier=str(player.player_id),
            url=url,
            attempts=1,
            error=f"Invalid JSON response: {exc}",
        )
    except FetchError as exc:
        return player.player_id, False, FetchFailure(
            stage="valuation_fetch",
            identifier=str(player.player_id),
            url=url,
            attempts=exc.attempts,
            status_code=exc.status_code,
            error=str(exc),
        )


async def _run(
    *,
    league_config: LeagueConfig,
    league_config_path: Path,
    season: int,
    output_root: Path,
    concurrency: int,
    requests_per_minute: int,
    timeout_seconds: float,
    max_retries: int,
    fetch_valuations: bool,
    progress: ProgressCallback | None,
) -> TransfermarktIngestionResult:
    started_at = _utc_now()
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_directory = (
        output_root
        / f"run_date={started_at.date().isoformat()}"
        / f"run_id={run_id}"
        / f"competition={league_config.competition_id}"
        / f"season={season}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    manifest_path = run_directory / "manifest.json"
    competition_html_path = run_directory / "competition.html"
    clubs_jsonl_path = run_directory / "clubs.jsonl"
    players_jsonl_path = run_directory / "players.jsonl"
    failures_jsonl_path = run_directory / "failed_requests.jsonl"
    clubs_html_directory = run_directory / "clubs"
    valuations_directory = run_directory / "valuations"
    progress_emitter = ProgressEmitter(
        run_id=run_id,
        log_path=run_directory / "progress.jsonl",
        callback=progress,
    )

    manifest: dict[str, object] = {
        "run_id": run_id,
        "pipeline": "transfermarkt_raw_acquisition",
        "status": "running",
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "league_config_path": str(league_config_path),
        "scope": {
            "competition_id": league_config.competition_id,
            "competition_name": league_config.competition_name,
            "season": season,
        },
        "http": {
            "concurrency": concurrency,
            "requests_per_minute": requests_per_minute,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
        },
        "counts": {},
        "objects": {},
        "error": None,
    }
    write_json(manifest_path, manifest)

    failures: list[FetchFailure] = []
    valuation_count = 0

    try:
        async with TransfermarktHttpClient(
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ) as client:
            progress_emitter.emit(
                ProgressUpdate(
                    stage="competition",
                    state="started",
                    description="Fetching Transfermarkt competition",
                    total=1,
                )
            )
            competition_url = league_config.competition_url(season)
            try:
                competition_response = await client.get(competition_url)
            except FetchError as exc:
                raise TransfermarktIngestionError(
                    f"Could not fetch competition page {competition_url}: {exc}"
                ) from exc

            atomic_write_bytes(competition_html_path, competition_response.content)
            clubs = parse_clubs(
                html=competition_response.content.decode(
                    "utf-8",
                    errors="replace",
                ),
                base_url=league_config.base_url,
                season=season,
            )
            if not clubs:
                raise TransfermarktIngestionError(
                    "Competition page contained no recognizable club links."
                )
            progress_emitter.emit(
                ProgressUpdate(
                    stage="competition",
                    state="completed",
                    description="Fetched Transfermarkt competition",
                    completed=1,
                    total=1,
                    succeeded=1,
                    current=f"clubs={len(clubs)}",
                )
            )
            write_jsonl(clubs_jsonl_path, (club.as_dict() for club in clubs))

            progress_emitter.emit(
                ProgressUpdate(
                    stage="rosters",
                    state="started",
                    description="Fetching Transfermarkt rosters",
                    total=len(clubs),
                )
            )
            roster_tasks = [
                _fetch_club_roster(
                    client=client,
                    club=club,
                    season=season,
                    clubs_html_directory=clubs_html_directory,
                )
                for club in clubs
            ]
            roster_results = []
            roster_failures = 0
            for index, task in enumerate(asyncio.as_completed(roster_tasks), start=1):
                club_id, players_for_club, failure = await task
                roster_results.append((players_for_club, failure))
                roster_failures += int(failure is not None)
                progress_emitter.emit(
                    ProgressUpdate(
                        stage="rosters",
                        state=("completed" if index == len(clubs) else "advanced"),
                        description="Fetching Transfermarkt rosters",
                        completed=index,
                        total=len(clubs),
                        succeeded=index - roster_failures,
                        failed=roster_failures,
                        current=f"club_id={club_id}",
                    )
                )

            unique_players: dict[int, PlayerRecord] = {}
            for players, failure in roster_results:
                if failure:
                    failures.append(failure)
                for player in players:
                    unique_players.setdefault(player.player_id, player)

            players = sorted(
                unique_players.values(),
                key=lambda item: item.player_id,
            )
            if not players:
                raise TransfermarktIngestionError(
                    "No players were discovered from the fetched club rosters."
                )
            write_jsonl(
                players_jsonl_path,
                (player.as_dict() for player in players),
            )

            if fetch_valuations:
                progress_emitter.emit(
                    ProgressUpdate(
                        stage="valuations",
                        state="started",
                        description="Fetching player valuation histories",
                        total=len(players),
                    )
                )
                valuation_tasks = [
                    _fetch_player_valuation(
                        client=client,
                        base_url=league_config.base_url,
                        player=player,
                        valuations_directory=valuations_directory,
                    )
                    for player in players
                ]
                valuation_failures = 0
                for index, task in enumerate(
                    asyncio.as_completed(valuation_tasks), start=1
                ):
                    player_id, success, failure = await task
                    valuation_count += int(success)
                    if failure:
                        failures.append(failure)
                        valuation_failures += 1
                    progress_emitter.emit(
                        ProgressUpdate(
                            stage="valuations",
                            state=(
                                "completed" if index == len(players) else "advanced"
                            ),
                            description="Fetching player valuation histories",
                            completed=index,
                            total=len(players),
                            succeeded=valuation_count,
                            failed=valuation_failures,
                            current=f"player_id={player_id}",
                        )
                    )

        write_jsonl(
            failures_jsonl_path,
            (failure.as_dict() for failure in failures),
        )

        status = "succeeded" if not failures else "partial"
        manifest.update(
            {
                "status": status,
                "completed_at": _utc_now().isoformat(),
                "counts": {
                    "clubs": len(clubs),
                    "players": len(players),
                    "valuation_responses": valuation_count,
                    "failed_requests": len(failures),
                },
                "objects": {
                    "competition_html": {
                        "path": str(competition_html_path),
                        "sha256": sha256_file(competition_html_path),
                    },
                    "clubs_jsonl": {
                        "path": str(clubs_jsonl_path),
                        "sha256": sha256_file(clubs_jsonl_path),
                    },
                    "players_jsonl": {
                        "path": str(players_jsonl_path),
                        "sha256": sha256_file(players_jsonl_path),
                    },
                    "failed_requests_jsonl": {
                        "path": str(failures_jsonl_path),
                        "sha256": sha256_file(failures_jsonl_path),
                    },
                    "valuations_directory": {
                        "path": str(valuations_directory),
                    },
                    "progress_log": {
                        "path": str(run_directory / "progress.jsonl"),
                    },
                },
            }
        )
        write_json(manifest_path, manifest)

        return TransfermarktIngestionResult(
            status=status,
            run_directory=run_directory,
            manifest_path=manifest_path,
            club_count=len(clubs),
            player_count=len(players),
            valuation_count=valuation_count,
            failure_count=len(failures),
        )
    except Exception as exc:
        progress_emitter.emit(
            ProgressUpdate(
                stage="pipeline",
                state="failed",
                description="Transfermarkt ingestion failed",
                failed=1,
                current=f"{type(exc).__name__}: {exc}",
            )
        )
        manifest.update(
            {
                "status": "failed",
                "completed_at": _utc_now().isoformat(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        write_json(manifest_path, manifest)
        if isinstance(exc, TransfermarktIngestionError):
            raise
        raise TransfermarktIngestionError(str(exc)) from exc


def ingest_transfermarkt(
    *,
    league_config_path: Path,
    season: int,
    output_root: Path,
    concurrency: int,
    requests_per_minute: int,
    timeout_seconds: float,
    max_retries: int,
    fetch_valuations: bool,
    progress: ProgressCallback | None = None,
) -> TransfermarktIngestionResult:
    league_config = load_league_config(league_config_path)
    return asyncio.run(
        _run(
            league_config=league_config,
            league_config_path=league_config_path,
            season=season,
            output_root=output_root,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fetch_valuations=fetch_valuations,
            progress=progress,
        )
    )
