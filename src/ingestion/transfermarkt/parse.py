from __future__ import annotations

import re
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

from ingestion.transfermarkt.models import (
    ClubRecord,
    PlayerRecord,
)


CLUB_ID_PATTERN = re.compile(r"/verein/(\d+)")
PLAYER_ID_PATTERN = re.compile(r"/spieler/(\d+)")
DATE_OF_BIRTH_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = " ".join(value.split())
    return cleaned or None


def _absolute_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _canonical_club_path(path: str) -> str:
    parsed = urlparse(path)
    clean_path = parsed.path
    clean_path = re.sub(r"/saison_id/\d{4}.*$", "", clean_path)
    return clean_path.rstrip("/")


def parse_clubs(
    *,
    html: str,
    base_url: str,
    season: int,
) -> list[ClubRecord]:
    tree = HTMLParser(html)
    clubs: dict[int, ClubRecord] = {}

    for anchor in tree.css('a[href*="/startseite/verein/"]'):
        href = anchor.attributes.get("href")
        if not href:
            continue

        match = CLUB_ID_PATTERN.search(href)
        if not match:
            continue

        club_id = int(match.group(1))
        club_name = _clean_text(anchor.text(strip=True))

        if not club_name:
            club_name = _clean_text(anchor.attributes.get("title"))

        if not club_name:
            continue

        profile_path = _canonical_club_path(href)

        clubs[club_id] = ClubRecord(
            club_id=club_id,
            club_name=club_name,
            profile_path=profile_path,
            roster_url=_absolute_url(
                base_url,
                f"{profile_path}/saison_id/{season}",
            ),
        )

    return sorted(clubs.values(), key=lambda item: item.club_id)


def _extract_squad_position(player_cell) -> str | None:
    inline_rows = player_cell.css("table.inline-table tr")

    if len(inline_rows) >= 2:
        position_cell = inline_rows[1].css_first("td")
        if position_cell is not None:
            return _clean_text(position_cell.text(strip=True))

    position_span = player_cell.css_first("span.spieler-zusatz")
    if position_span is not None:
        return _clean_text(position_span.text(strip=True))

    return None


def _extract_name_from_cell(
    player_cell,
    profile_anchor,
) -> str | None:
    portrait = player_cell.css_first(
        'img[title][data-src*="portrait"], '
        'img[alt][data-src*="portrait"], '
        'img[title][src*="portrait"], '
        'img[alt][src*="portrait"]'
    )

    if portrait is not None:
        name = _clean_text(
            portrait.attributes.get("title")
            or portrait.attributes.get("alt")
        )
        if name:
            return name

    name_span = player_cell.css_first("span.spielername")
    if name_span is not None:
        name = _clean_text(name_span.text(strip=True))
        if name:
            return name

    name_anchor = player_cell.css_first(
        'td.hauptlink a[href*="/profil/spieler/"]'
    )
    if name_anchor is not None:
        name = _clean_text(name_anchor.text(strip=True))
        if name:
            return name

    return _clean_text(profile_anchor.text(strip=True))


def _parse_current_squad(
    *,
    tree: HTMLParser,
    club: ClubRecord,
    season: int,
) -> dict[int, PlayerRecord]:
    players: dict[int, PlayerRecord] = {}

    for row in tree.css("table.items tr"):
        player_cell = row.css_first("td.posrela")
        if player_cell is None:
            continue

        profile_anchor = player_cell.css_first(
            'a[href*="/profil/spieler/"]'
        )
        if profile_anchor is None:
            continue

        href = profile_anchor.attributes.get("href")
        if not href:
            continue

        match = PLAYER_ID_PATTERN.search(href)
        if not match:
            continue

        player_id = int(match.group(1))
        position = _extract_squad_position(player_cell)
        player_name = _extract_name_from_cell(
            player_cell,
            profile_anchor,
        )

        if not player_name:
            continue

        if (
            position
            and player_name.casefold().endswith(position.casefold())
        ):
            player_name = player_name[: -len(position)].strip()

        date_of_birth = None

        for centered_cell in row.css("td.zentriert"):
            date_match = DATE_OF_BIRTH_PATTERN.search(
                centered_cell.text(strip=True)
            )
            if date_match:
                date_of_birth = date_match.group(0)
                break

        nationalities: list[str] = []

        for flag in row.css("img.flaggenrahmen"):
            nationality = _clean_text(
                flag.attributes.get("title")
                or flag.attributes.get("alt")
            )

            if nationality and nationality not in nationalities:
                nationalities.append(nationality)

        players[player_id] = PlayerRecord(
            player_id=player_id,
            player_name=player_name,
            profile_path=urlparse(href).path,
            club_id=club.club_id,
            club_name=club.club_name,
            season=season,
            position=position,
            date_of_birth=date_of_birth,
            nationalities=tuple(nationalities),
            roster_status="current_squad",
            source_section="squad_table",
            destination_club_name=None,
        )

    return players


def _parse_top_departures(
    *,
    tree: HTMLParser,
    club: ClubRecord,
    season: int,
) -> dict[int, PlayerRecord]:
    players: dict[int, PlayerRecord] = {}

    for row in tree.css(
        "div.abgaenge-widget table.startseite tbody tr"
    ):
        name_anchor = row.css_first(
            'td.td a[href*="/profil/spieler/"]'
        )
        if name_anchor is None:
            continue

        href = name_anchor.attributes.get("href")
        if not href:
            continue

        match = PLAYER_ID_PATTERN.search(href)
        if not match:
            continue

        player_id = int(match.group(1))

        name_span = name_anchor.css_first("span.spielername")
        position_span = name_anchor.css_first("span.spieler-zusatz")

        player_name = (
            _clean_text(name_span.text(strip=True))
            if name_span is not None
            else None
        )
        position = (
            _clean_text(position_span.text(strip=True))
            if position_span is not None
            else None
        )

        if not player_name:
            portrait = row.css_first("td.foto img")

            if portrait is not None:
                player_name = _clean_text(
                    portrait.attributes.get("title")
                    or portrait.attributes.get("alt")
                )

        if not player_name:
            continue

        destination_club_name = None
        destination_anchor = row.css_first(
            'td.wappen a[href*="/startseite/verein/"]'
        )

        if destination_anchor is not None:
            destination_club_name = _clean_text(
                destination_anchor.attributes.get("title")
            )

            if not destination_club_name:
                destination_image = destination_anchor.css_first("img")

                if destination_image is not None:
                    destination_club_name = _clean_text(
                        destination_image.attributes.get("alt")
                    )

        players[player_id] = PlayerRecord(
            player_id=player_id,
            player_name=player_name,
            profile_path=urlparse(href).path,
            club_id=club.club_id,
            club_name=club.club_name,
            season=season,
            position=position,
            date_of_birth=None,
            nationalities=(),
            roster_status="departed",
            source_section="top_departures",
            destination_club_name=destination_club_name,
        )

    return players


def parse_players(
    *,
    html: str,
    club: ClubRecord,
    season: int,
) -> list[PlayerRecord]:
    tree = HTMLParser(html)

    current_squad = _parse_current_squad(
        tree=tree,
        club=club,
        season=season,
    )

    # Lightweight fallback for simplified HTML fixtures and pages where the
    # table wrappers are missing but profile links are still present.
    if not current_squad:
        for anchor in tree.css('a[href*="/profil/spieler/"]'):
            href = anchor.attributes.get("href")
            if not href:
                continue
            match = PLAYER_ID_PATTERN.search(href)
            player_name = _clean_text(anchor.text(strip=True))
            if not match or not player_name:
                continue
            player_id = int(match.group(1))
            current_squad[player_id] = PlayerRecord(
                player_id=player_id,
                player_name=player_name,
                profile_path=urlparse(href).path,
                club_id=club.club_id,
                club_name=club.club_name,
                season=season,
                position=None,
                date_of_birth=None,
                nationalities=(),
                roster_status="current_squad",
                source_section="profile_link_fallback",
                destination_club_name=None,
            )

    departures = _parse_top_departures(
        tree=tree,
        club=club,
        season=season,
    )

    # Current-squad rows take priority if a player appears in both sections.
    players = dict(current_squad)

    for player_id, player in departures.items():
        players.setdefault(player_id, player)

    return sorted(players.values(), key=lambda item: item.player_id)
