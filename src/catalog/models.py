from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    description: str
    layer: str
    owner: str
    grain: str
    paths: tuple[str, ...]
    primary_key: tuple[str, ...]
    partition_by: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()
    downstream: tuple[str, ...] = ()
    published_aliases: tuple[str, ...] = ()
    required: bool = True
    field_checks: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    quality_rules: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetDefinition":
        field_checks = {
            name: tuple(checks)
            for name, checks in value.get("field_checks", {}).items()
        }
        return cls(
            name=str(value["name"]),
            description=str(value["description"]),
            layer=str(value["layer"]),
            owner=str(value["owner"]),
            grain=str(value["grain"]),
            paths=tuple(str(item) for item in value["paths"]),
            primary_key=tuple(str(item) for item in value.get("primary_key", [])),
            partition_by=tuple(str(item) for item in value.get("partition_by", [])),
            upstream=tuple(str(item) for item in value.get("upstream", [])),
            downstream=tuple(str(item) for item in value.get("downstream", [])),
            published_aliases=tuple(
                str(item) for item in value.get("published_aliases", [])
            ),
            required=bool(value.get("required", True)),
            field_checks=field_checks,
            quality_rules=tuple(
                str(item) for item in value.get("quality_rules", [])
            ),
        )


@dataclass(frozen=True)
class CatalogBuildResult:
    catalog_path: Path
    lineage_path: Path
    markdown_path: Path
    datasets: int
    files: int
    rows: int
    contract_failures: int
