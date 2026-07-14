from __future__ import annotations

import re
from typing import Any


def qualifier_name(qualifier: dict[str, Any]) -> str | None:
    qualifier_type = qualifier.get("type")
    if not isinstance(qualifier_type, dict):
        return None
    value = qualifier_type.get("displayName")
    return str(value) if value is not None else None


def qualifier_value(qualifier: dict[str, Any]) -> str | None:
    value = qualifier.get("value")
    return str(value) if value is not None else None


def _safe_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def flatten_qualifiers(
    qualifiers: list[dict[str, Any]] | None,
) -> dict[str, str | bool]:
    output: dict[str, str | bool] = {}
    for qualifier in qualifiers or []:
        name = qualifier_name(qualifier)
        if not name:
            continue
        value = qualifier_value(qualifier)
        output[f"q_{_safe_column_name(name)}"] = value if value is not None else True
    return output


def qualifier_names(qualifiers: list[dict[str, Any]] | None) -> list[str]:
    return [
        name
        for qualifier in qualifiers or []
        if (name := qualifier_name(qualifier))
    ]
