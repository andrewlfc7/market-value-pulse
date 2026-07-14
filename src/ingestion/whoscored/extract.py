from __future__ import annotations

import re
from typing import Any

import orjson
from bs4 import BeautifulSoup


class WhoScoredExtractionError(ValueError):
    """Raised when a page does not contain parseable match-centre data."""


def extract_match_id_from_url(url: str) -> int | None:
    match = re.search(r"/[Mm]atches/(\d+)", url)
    return int(match.group(1)) if match else None


def extract_match_id_from_html(html: str) -> int | None:
    match = re.search(r"matchId\s*:\s*(\d+)", html)
    return int(match.group(1)) if match else None


def _extract_balanced_object(text: str, start_index: int) -> str:
    start = text.find("{", start_index)
    if start < 0:
        raise WhoScoredExtractionError("Could not find the match-centre object")

    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
            continue
        if character in {"'", '"'}:
            in_string = True
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise WhoScoredExtractionError("Match-centre object has no closing brace")


def extract_match_centre_json_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        script.get_text()
        for script in soup.find_all("script")
        if "matchCentreData" in script.get_text()
    ]
    if not candidates:
        if "matchCentreData" not in html:
            raise WhoScoredExtractionError("Page does not contain matchCentreData")
        candidates = [html]
    text = max(candidates, key=len)
    marker_index = text.find("matchCentreData")
    colon_index = text.find(":", marker_index)
    if colon_index < 0:
        raise WhoScoredExtractionError("matchCentreData has no value")
    return _extract_balanced_object(text, colon_index)


def parse_match_centre_data(html: str) -> dict[str, Any]:
    try:
        payload = orjson.loads(extract_match_centre_json_text(html))
    except orjson.JSONDecodeError as exc:
        raise WhoScoredExtractionError(f"Invalid matchCentreData JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WhoScoredExtractionError("matchCentreData is not an object")
    return payload
