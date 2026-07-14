from __future__ import annotations

import html as html_module
import re
from dataclasses import asdict, dataclass
from datetime import datetime

from bs4 import BeautifulSoup

BASE_URL = "https://www.whoscored.com"
MATCH_PATTERN = re.compile(
    r"/matches/(?P<match_id>\d+)/(?P<page_kind>live|preview|matchreport|show)/(?P<slug>[^\"'#?<\s]+)",
    re.IGNORECASE,
)
FIXTURE_DATE_PATTERN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\b"
)


@dataclass(frozen=True)
class DiscoveredMatch:
    match_id: int
    match_url: str
    source_url: str
    slug: str
    page_kind: str
    fixture_date: str | None = None


def normalize_path(value: str) -> str:
    path = html_module.unescape(value).strip()
    path = re.sub(r"^https?://www\.whoscored\.com", "", path, flags=re.IGNORECASE)
    return path if path.startswith("/") else f"/{path}"


def _fixture_date(anchor: object) -> str | None:
    """Read the date header from the nearest enclosing fixture accordion."""
    current = anchor
    for _ in range(10):
        current = getattr(current, "parent", None)
        if current is None:
            return None
        get_text = getattr(current, "get_text", None)
        if get_text is None:
            continue
        match = FIXTURE_DATE_PATTERN.search(get_text(" ", strip=True))
        if match is None:
            continue
        try:
            return datetime.strptime(
                f"{match.group('month')} {match.group('day')} {match.group('year')}",
                "%B %d %Y",
            ).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(
                    f"{match.group('month')} {match.group('day')} {match.group('year')}",
                    "%b %d %Y",
                ).date().isoformat()
            except ValueError:
                return None
    return None


def discover_match_links(page_html: str, source_url: str) -> list[DiscoveredMatch]:
    soup = BeautifulSoup(page_html, "html.parser")
    matches: dict[int, DiscoveredMatch] = {}
    # WhoScored has emitted both /Matches/ and /matches/ paths over time.
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href:
            continue
        path = normalize_path(str(href))
        match = MATCH_PATTERN.search(path.lower())
        if not match:
            continue
        match_id = int(match.group("match_id"))
        matches.setdefault(
            match_id,
            DiscoveredMatch(
                match_id=match_id,
                match_url=f"{BASE_URL}{path}",
                source_url=source_url,
                slug=match.group("slug"),
                page_kind=match.group("page_kind").lower(),
                fixture_date=_fixture_date(anchor),
            ),
        )
    return list(matches.values())


def discovered_matches_as_dicts(
    page_html: str,
    source_url: str,
) -> list[dict[str, object]]:
    return [asdict(row) for row in discover_match_links(page_html, source_url)]
