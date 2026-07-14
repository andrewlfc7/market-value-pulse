from __future__ import annotations

import html as html_module
import re
from dataclasses import asdict, dataclass
from typing import Any

from bs4 import BeautifulSoup

BASE_URL = "https://www.whoscored.com"
SEASON_RE = re.compile(
    r"/regions/(?P<region_id>\d+)/tournaments/(?P<tournament_id>\d+)/seasons/(?P<season_id>\d+)/(?P<slug>[^/?#]+)",
    re.IGNORECASE,
)
STAGE_RE = re.compile(
    r"/regions/(?P<region_id>\d+)/tournaments/(?P<tournament_id>\d+)/seasons/(?P<season_id>\d+)/stages/(?P<stage_id>\d+)/(?P<section>[^/?#]+)/(?P<slug>[^/?#]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SeasonOption:
    region_id: int
    tournament_id: int
    season_id: int
    name: str
    slug: str
    url: str
    selected: bool


@dataclass(frozen=True)
class StageOption:
    region_id: int
    tournament_id: int
    season_id: int
    stage_id: int
    section: str
    slug: str
    url: str
    selected: bool


def normalize_path(value: str) -> str:
    path = html_module.unescape(value).strip()
    path = re.sub(r"^https?://(?:www\.)?whoscored\.com", "", path, flags=re.I)
    return path if path.startswith("/") else f"/{path}"


def absolute_url(value: str) -> str:
    return f"{BASE_URL}{normalize_path(value)}"


def normalize_season_label(value: str) -> str:
    cleaned = re.sub(r"\s+", "", str(value)).replace("/", "-").replace("_", "-")
    match = re.fullmatch(r"(?P<start>\d{4})(?:-(?P<end>\d{2}|\d{4}))?", cleaned)
    if not match:
        return cleaned.lower()
    start = int(match.group("start"))
    end_text = match.group("end")
    if end_text is None:
        end = start + 1
    elif len(end_text) == 2:
        end = (start // 100) * 100 + int(end_text)
        if end <= start:
            end += 100
    else:
        end = int(end_text)
    return f"{start:04d}-{end:04d}"


def parse_seasons(page_html: str) -> list[SeasonOption]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[SeasonOption] = []
    for option in soup.select("select#seasons option"):
        raw = option.get("value")
        if not raw:
            continue
        path = normalize_path(str(raw))
        match = SEASON_RE.search(path)
        if match is None:
            continue
        rows.append(
            SeasonOption(
                region_id=int(match.group("region_id")),
                tournament_id=int(match.group("tournament_id")),
                season_id=int(match.group("season_id")),
                name=option.get_text(" ", strip=True),
                slug=match.group("slug"),
                url=absolute_url(path),
                selected=option.has_attr("selected"),
            )
        )
    return rows


def resolve_season(options: list[SeasonOption], requested: str) -> SeasonOption:
    target = normalize_season_label(requested)
    matches = [row for row in options if normalize_season_label(row.name) == target]
    if not matches:
        available = ", ".join(row.name for row in options[:12]) or "none"
        raise ValueError(f"WhoScored season {requested!r} was not found; available: {available}")
    return matches[0]


def parse_stages(page_html: str) -> list[StageOption]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[StageOption] = []
    seen: set[tuple[int, str]] = set()
    for anchor in soup.select('a[href*="/stages/"]'):
        raw = anchor.get("href")
        if not raw:
            continue
        path = normalize_path(str(raw))
        match = STAGE_RE.search(path)
        if match is None:
            continue
        key = (int(match.group("stage_id")), match.group("section").lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            StageOption(
                region_id=int(match.group("region_id")),
                tournament_id=int(match.group("tournament_id")),
                season_id=int(match.group("season_id")),
                stage_id=int(match.group("stage_id")),
                section=match.group("section").lower(),
                slug=match.group("slug"),
                url=absolute_url(path),
                selected="selected" in (anchor.get("class") or []),
            )
        )
    return rows


def resolve_fixture_stage(
    stages: list[StageOption], *, stage_override: int | None = None
) -> StageOption:
    fixtures = [row for row in stages if row.section == "fixtures"]
    if stage_override is not None:
        override = next((row for row in fixtures if row.stage_id == stage_override), None)
        if override is None:
            raise ValueError(f"Configured WhoScored stage {stage_override} was not present")
        return override
    selected = [row for row in fixtures if row.selected]
    if selected:
        return selected[0]
    if not fixtures:
        raise ValueError("WhoScored season page exposed no fixture stage")
    return fixtures[0]


def catalog_payload(
    seasons: list[SeasonOption], stages: list[StageOption]
) -> dict[str, Any]:
    return {
        "seasons": [asdict(row) for row in seasons],
        "stages": [asdict(row) for row in stages],
    }
