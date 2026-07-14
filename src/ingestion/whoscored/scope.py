from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from ingestion.common import atomic_write_bytes, write_json
from ingestion.whoscored.calendar import configure_page, discover_fixture_calendar
from ingestion.whoscored.catalog import (
    catalog_payload,
    normalize_season_label,
    parse_seasons,
    parse_stages,
    resolve_fixture_stage,
    resolve_season,
)
from ingestion.whoscored.competitions import Competition, resolve_competition
from ingestion.whoscored.fetch import USER_AGENT


@dataclass(frozen=True)
class ScopeDiscoveryResult:
    competition: Competition
    requested_season: str
    canonical_season: str
    season_id: int
    stage_id: int
    fixture_url: str
    matches: list[dict[str, Any]]
    pages_collected: int
    warnings: tuple[str, ...]


def _load_page(page, url: str, timeout_ms: int) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_selector("select#seasons, a[href*='/stages/']", timeout=15_000)
    except Exception:
        pass
    return page.content()


def discover_competition_season(
    *,
    competition_name: str,
    season: str,
    raw_directory: Path,
    registry_path: Path = Path("config/whoscored/competitions.json"),
    max_previous: int = 60,
    max_next: int = 12,
    timeout_ms: int = 60_000,
    wait_seconds: float = 1.0,
    headful: bool = False,
) -> ScopeDiscoveryResult:
    competition = resolve_competition(competition_name, registry_path=registry_path)
    raw_directory.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headful)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 1200},
            extra_http_headers={
                "accept-language": "en-US,en;q=0.9",
                "referer": "https://www.whoscored.com/",
            },
        )
        page = context.new_page()
        configure_page(page)
        tournament_html = _load_page(page, competition.tournament_url, timeout_ms)
        atomic_write_bytes(
            raw_directory / "tournament_page.html", tournament_html.encode("utf-8")
        )
        seasons = parse_seasons(tournament_html)
        selected_season = resolve_season(seasons, season)

        season_html = _load_page(page, selected_season.url, timeout_ms)
        atomic_write_bytes(
            raw_directory / "season_page.html", season_html.encode("utf-8")
        )
        stages = parse_stages(season_html)
        canonical_season = normalize_season_label(selected_season.name)
        stage = resolve_fixture_stage(
            stages,
            stage_override=competition.stage_overrides.get(canonical_season),
        )
        write_json(raw_directory / "catalog.json", catalog_payload(seasons, stages))
        page.close()

        calendar = discover_fixture_calendar(
            context,
            fixture_url=stage.url,
            raw_directory=raw_directory / "fixture_calendar_pages",
            max_previous=max_previous,
            max_next=max_next,
            timeout_ms=timeout_ms,
            wait_seconds=wait_seconds,
        )
        context.close()
        browser.close()

    matches = [
        {
            **row,
            "competition": competition.key,
            "competition_name": competition.name,
            "region_id": competition.region_id,
            "tournament_id": competition.tournament_id,
            "season": canonical_season,
            "season_id": selected_season.season_id,
            "stage_id": stage.stage_id,
        }
        for row in calendar.rows
        if row.get("page_kind") != "preview"
    ]
    if not matches:
        raise ValueError(f"No matches were discovered from {stage.url}")
    warnings: list[str] = []
    if competition.expected_matches and len(matches) < competition.expected_matches:
        warnings.append(
            f"Discovered {len(matches)} matches; a completed {competition.name} season "
            f"usually has {competition.expected_matches}. This is expected for an in-progress season."
        )
    write_json(
        raw_directory / "discovery_summary.json",
        {
            "competition": asdict(competition),
            "requested_season": season,
            "canonical_season": canonical_season,
            "season_id": selected_season.season_id,
            "stage_id": stage.stage_id,
            "fixture_url": stage.url,
            "matches_discovered": len(matches),
            "calendar_pages_collected": calendar.pages_collected,
            "calendar_range": {
                "first": calendar.first_calendar_label,
                "last": calendar.last_calendar_label,
            },
            "warnings": warnings,
        },
    )
    return ScopeDiscoveryResult(
        competition=competition,
        requested_season=season,
        canonical_season=canonical_season,
        season_id=selected_season.season_id,
        stage_id=stage.stage_id,
        fixture_url=stage.url,
        matches=matches,
        pages_collected=calendar.pages_collected,
        warnings=tuple(warnings),
    )
