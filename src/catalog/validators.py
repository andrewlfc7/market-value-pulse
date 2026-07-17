from __future__ import annotations

from collections import Counter
from typing import Any

from catalog.models import DatasetDefinition


class CatalogValidationError(ValueError):
    """Raised when catalog contracts are incomplete or incompatible."""


def validate_definitions(
    definitions: list[DatasetDefinition],
) -> list[str]:
    errors: list[str] = []
    names = [definition.name for definition in definitions]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        errors.append(f"Duplicate dataset definitions: {duplicates}")

    known = set(names)
    for definition in definitions:
        if not definition.paths:
            errors.append(f"{definition.name}: at least one path is required")
        if not definition.primary_key:
            errors.append(f"{definition.name}: primary_key must not be empty")
        unknown_upstream = sorted(set(definition.upstream).difference(known))
        # External/raw nodes are allowed when they are intentionally prefixed.
        unknown_upstream = [
            name
            for name in unknown_upstream
            if not name.startswith(("raw_", "normalized_", "feature_", "valuation_"))
        ]
        if unknown_upstream:
            errors.append(
                f"{definition.name}: unknown upstream datasets {unknown_upstream}"
            )
    return errors


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for dataset in catalog.get("datasets", []):
        name = str(dataset.get("name"))
        status = dataset.get("status")
        if status == "missing" and dataset.get("required", True):
            errors.append(f"{name}: required dataset is missing")
            continue

        field_names = {
            str(field.get("name"))
            for field in dataset.get("fields", [])
        }
        missing_keys = sorted(
            set(dataset.get("primary_key", [])).difference(field_names)
        )
        if missing_keys:
            errors.append(
                f"{name}: primary-key fields missing from schema: {missing_keys}"
            )

        drift = dataset.get("schema_drift", {})
        if drift.get("status") == "failed":
            errors.append(f"{name}: schema drift detected")

    return errors


def require_valid(errors: list[str]) -> None:
    if errors:
        raise CatalogValidationError("\n".join(errors))
