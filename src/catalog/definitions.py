from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from catalog.models import DatasetDefinition


DEFAULT_DATASETS_PATH = Path("config/catalog/datasets.json")
DEFAULT_FIELDS_PATH = Path("config/catalog/field_descriptions.json")


def load_dataset_definitions(
    path: Path = DEFAULT_DATASETS_PATH,
) -> list[DatasetDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        DatasetDefinition.from_dict(item)
        for item in payload.get("datasets", [])
    ]


def load_field_descriptions(
    path: Path = DEFAULT_FIELDS_PATH,
) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(name): str(description)
        for name, description in payload.get("fields", {}).items()
    }


def _humanize(name: str) -> str:
    value = name.replace("_", " ").strip()
    replacements = {
        " xg ": " xG ",
        " xgot ": " xGOT ",
        " xa ": " xA ",
        " xt ": " xT ",
        " xpv ": " xPV ",
        " eur": " EUR",
        " ewm": " exponentially weighted mean",
    }
    padded = f" {value.lower()} "
    for source, target in replacements.items():
        padded = padded.replace(source, target)
    return re.sub(r"\s+", " ", padded).strip()


def describe_field(name: str, descriptions: dict[str, str]) -> str:
    exact = descriptions.get(name)
    if exact:
        return exact

    if name.startswith("log_"):
        base = name[4:]
        return (
            f"Natural-log transformed {_humanize(base)} used to reduce skew "
            "during model fitting."
        )

    if name.endswith("_90"):
        base = name[:-3]
        base_description = descriptions.get(base, _humanize(base))
        return f"{base_description.rstrip('.')} normalized per 90 minutes."

    if name.endswith("_per90"):
        base = name[:-6]
        base_description = descriptions.get(base, _humanize(base))
        return f"{base_description.rstrip('.')} normalized per 90 minutes."

    if name.endswith("_average"):
        base = name[:-8]
        base_description = descriptions.get(base, _humanize(base))
        return (
            f"Mean {base_description[0].lower() + base_description[1:]}"
            " across appearances in the valuation interval."
        )

    if name.endswith("_id"):
        return f"Stable identifier for {_humanize(name[:-3])}."

    if name.endswith("_date"):
        return f"Calendar date associated with {_humanize(name[:-5])}."

    if name.endswith("_at"):
        return f"Timestamp associated with {_humanize(name[:-3])}."

    if name.endswith("_version"):
        return f"Version identifier for {_humanize(name[:-8])}."

    if name.endswith("_source"):
        return f"Source or derivation method for {_humanize(name[:-7])}."

    if name.endswith("_pct") or name.endswith("_percentage"):
        return f"Percentage value for {_humanize(name.removesuffix('_pct').removesuffix('_percentage'))}."

    if name.endswith("_share"):
        return f"Proportion of the relevant total represented by {_humanize(name[:-6])}."

    if name.endswith("_component"):
        return f"Standardized {_humanize(name[:-10])} contribution used by the rating model."

    return f"{_humanize(name).capitalize()}."


def infer_tags(name: str, dtype: str) -> list[str]:
    tags: list[str] = []
    lowered = name.lower()

    if lowered.endswith("_id"):
        tags.append("identifier")
    if "date" in lowered or "datetime" in lowered or lowered.endswith("_at"):
        tags.append("temporal")
    if lowered.endswith("_eur") or "market_value" in lowered:
        tags.append("currency")
    if "rating" in lowered:
        tags.append("rating")
    if "percentile" in lowered:
        tags.append("percentile")
    if lowered.endswith("_90") or lowered.endswith("_per90"):
        tags.append("per-90")
    if lowered.startswith("log_") or "predicted" in lowered:
        tags.append("model")
    if dtype == "Null":
        tags.append("future-state")
    return tags


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
