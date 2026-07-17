from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from catalog.definitions import (
    DEFAULT_DATASETS_PATH,
    DEFAULT_FIELDS_PATH,
    describe_field,
    infer_tags,
    load_dataset_definitions,
    load_field_descriptions,
)
from catalog.models import CatalogBuildResult, DatasetDefinition
from catalog.validators import (
    require_valid,
    validate_catalog,
    validate_definitions,
)


CATALOG_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _discover_files(
    root: Path,
    patterns: Iterable[str],
) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file()
        )
    return sorted(files)


def _row_count(path: Path) -> int:
    return int(
        pl.scan_parquet(path)
        .select(pl.len().alias("row_count"))
        .collect()
        .item()
    )


def _null_counts(
    path: Path,
    columns: Iterable[str],
) -> dict[str, int]:
    names = list(columns)
    if not names:
        return {}
    row = (
        pl.scan_parquet(path)
        .select(
            *[
                pl.col(column).null_count().alias(column)
                for column in names
            ]
        )
        .collect()
        .row(0, named=True)
    )
    return {name: int(value) for name, value in row.items()}


def _schema_map(path: Path) -> dict[str, str]:
    return {
        name: str(dtype)
        for name, dtype in pl.read_parquet_schema(path).items()
    }


def _inspect_files(
    files: list[Path],
    *,
    project_root: Path,
) -> dict[str, Any]:
    if not files:
        return {
            "files": [],
            "file_count": 0,
            "row_count": 0,
            "schema": {},
            "null_counts": {},
            "schema_drift": {
                "status": "not_checked",
                "files_with_drift": [],
            },
        }

    schemas: list[tuple[Path, dict[str, str]]] = []
    total_rows = 0
    null_counts: defaultdict[str, int] = defaultdict(int)

    for path in files:
        schema = _schema_map(path)
        schemas.append((path, schema))
        total_rows += _row_count(path)
        for name, value in _null_counts(path, schema).items():
            null_counts[name] += value

    baseline = schemas[0][1]
    drift_files: list[dict[str, Any]] = []
    for path, schema in schemas[1:]:
        if schema != baseline:
            drift_files.append(
                {
                    "path": str(path.relative_to(project_root)),
                    "missing_columns": sorted(
                        set(baseline).difference(schema)
                    ),
                    "extra_columns": sorted(
                        set(schema).difference(baseline)
                    ),
                    "type_changes": {
                        name: {
                            "expected": baseline[name],
                            "actual": schema[name],
                        }
                        for name in sorted(
                            set(baseline).intersection(schema)
                        )
                        if baseline[name] != schema[name]
                    },
                }
            )

    return {
        "files": [
            str(path.relative_to(project_root))
            for path in files
        ],
        "file_count": len(files),
        "row_count": total_rows,
        "schema": baseline,
        "null_counts": dict(null_counts),
        "schema_drift": {
            "status": "failed" if drift_files else "compatible",
            "files_with_drift": drift_files,
        },
    }


def _field_entry(
    name: str,
    dtype: str,
    *,
    null_count: int,
    row_count: int,
    descriptions: dict[str, str],
    checks: tuple[dict[str, Any], ...],
    primary_key: tuple[str, ...],
) -> dict[str, Any]:
    nullable = null_count > 0 or dtype == "Null"
    inferred_checks: list[dict[str, Any]] = list(checks)
    if name in primary_key:
        inferred_checks.insert(0, {"type": "not_null"})
    return {
        "name": name,
        "dtype": dtype,
        "nullable": nullable,
        "null_count": null_count,
        "null_fraction": (
            round(null_count / row_count, 6)
            if row_count > 0
            else None
        ),
        "description": describe_field(name, descriptions),
        "tags": infer_tags(name, dtype),
        "checks": inferred_checks,
    }


def _dataset_entry(
    definition: DatasetDefinition,
    *,
    project_root: Path,
    descriptions: dict[str, str],
) -> dict[str, Any]:
    files = _discover_files(project_root, definition.paths)
    inspection = _inspect_files(files, project_root=project_root)
    row_count = int(inspection["row_count"])
    fields = [
        _field_entry(
            name,
            dtype,
            null_count=int(
                inspection["null_counts"].get(name, 0)
            ),
            row_count=row_count,
            descriptions=descriptions,
            checks=definition.field_checks.get(name, ()),
            primary_key=definition.primary_key,
        )
        for name, dtype in inspection["schema"].items()
    ]

    missing_primary_key = sorted(
        set(definition.primary_key).difference(
            field["name"] for field in fields
        )
    )
    contract_errors = []
    if missing_primary_key:
        contract_errors.append(
            f"Missing primary-key fields: {missing_primary_key}"
        )
    if inspection["schema_drift"]["status"] == "failed":
        contract_errors.append("Partition schemas are incompatible")

    return {
        "name": definition.name,
        "description": definition.description,
        "layer": definition.layer,
        "owner": definition.owner,
        "grain": definition.grain,
        "required": definition.required,
        "status": "available" if files else "missing",
        "paths": list(definition.paths),
        "physical_files": inspection["files"],
        "published_aliases": list(definition.published_aliases),
        "file_count": inspection["file_count"],
        "row_count": row_count,
        "primary_key": list(definition.primary_key),
        "partition_by": list(definition.partition_by),
        "upstream": list(definition.upstream),
        "downstream": list(definition.downstream),
        "quality_rules": list(definition.quality_rules),
        "contract": {
            "status": "failed" if contract_errors else "passed",
            "errors": contract_errors,
        },
        "schema_drift": inspection["schema_drift"],
        "fields": fields,
    }


def _build_lineage(
    datasets: list[dict[str, Any]],
    *,
    generated_at: str,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for dataset in datasets:
        nodes[dataset["name"]] = {
            "id": dataset["name"],
            "label": dataset["name"],
            "layer": dataset["layer"],
            "owner": dataset["owner"],
            "status": dataset["status"],
        }

    for dataset in datasets:
        for upstream in dataset["upstream"]:
            if upstream not in nodes:
                nodes[upstream] = {
                    "id": upstream,
                    "label": upstream,
                    "layer": "external",
                    "owner": "data-engineering",
                    "status": "logical",
                }
            edges.append(
                {
                    "source": upstream,
                    "target": dataset["name"],
                    "relationship": "feeds",
                }
            )

    return {
        "version": CATALOG_VERSION,
        "generated_at": generated_at,
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(
            edges,
            key=lambda item: (
                item["source"],
                item["target"],
            ),
        ),
    }


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _render_markdown(catalog: dict[str, Any]) -> str:
    summary = catalog["summary"]
    lines = [
        "# Market Value Pulse Data Catalog",
        "",
        (
            f"Generated `{catalog['generated_at']}` from the physical "
            "Parquet schemas in the repository."
        ),
        "",
        "## Summary",
        "",
        "| Datasets | Files | Rows | Fields | Contract failures |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {summary['datasets']} | {summary['files']} | "
            f"{summary['rows']:,} | {summary['fields']} | "
            f"{summary['contract_failures']} |"
        ),
        "",
    ]

    for dataset in catalog["datasets"]:
        lines.extend(
            [
                f"## `{dataset['name']}`",
                "",
                dataset["description"],
                "",
                "| Property | Value |",
                "|---|---|",
                f"| Layer | `{dataset['layer']}` |",
                f"| Owner | `{dataset['owner']}` |",
                f"| Grain | {_escape_markdown(dataset['grain'])} |",
                f"| Rows | {dataset['row_count']:,} |",
                f"| Files | {dataset['file_count']} |",
                (
                    "| Primary key | "
                    + ", ".join(
                        f"`{name}`"
                        for name in dataset["primary_key"]
                    )
                    + " |"
                ),
                (
                    "| Partitions | "
                    + (
                        ", ".join(
                            f"`{name}`"
                            for name in dataset["partition_by"]
                        )
                        or "None"
                    )
                    + " |"
                ),
                (
                    "| Upstream | "
                    + (
                        ", ".join(
                            f"`{name}`"
                            for name in dataset["upstream"]
                        )
                        or "None"
                    )
                    + " |"
                ),
                (
                    "| Contract | "
                    f"`{dataset['contract']['status']}` |"
                ),
                (
                    "| Schema drift | "
                    f"`{dataset['schema_drift']['status']}` |"
                ),
                "",
                "### Physical paths",
                "",
            ]
        )
        lines.extend(
            f"- `{path}`"
            for path in dataset["paths"]
        )
        if dataset["published_aliases"]:
            lines.append("")
            lines.append("Published aliases:")
            lines.extend(
                f"- `{path}`"
                for path in dataset["published_aliases"]
            )

        if dataset["quality_rules"]:
            lines.extend(["", "### Dataset quality rules", ""])
            lines.extend(
                f"- {rule}"
                for rule in dataset["quality_rules"]
            )

        lines.extend(
            [
                "",
                "### Fields",
                "",
                "| Field | Type | Nullable | Nulls | Definition |",
                "|---|---|:---:|---:|---|",
            ]
        )
        for field in dataset["fields"]:
            lines.append(
                "| `{}` | `{}` | {} | {:,} | {} |".format(
                    field["name"],
                    field["dtype"],
                    "yes" if field["nullable"] else "no",
                    field["null_count"],
                    _escape_markdown(field["description"]),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_catalog(
    *,
    project_root: Path = Path("."),
    datasets_path: Path = DEFAULT_DATASETS_PATH,
    fields_path: Path = DEFAULT_FIELDS_PATH,
    catalog_path: Path = Path("metadata/catalog.json"),
    lineage_path: Path = Path("metadata/lineage.json"),
    markdown_path: Path = Path("docs/data-catalog.md"),
    strict: bool = True,
) -> CatalogBuildResult:
    project_root = project_root.resolve()
    datasets_path = _project_path(project_root, datasets_path)
    fields_path = _project_path(project_root, fields_path)

    definitions = load_dataset_definitions(datasets_path)
    definition_errors = validate_definitions(definitions)
    require_valid(definition_errors)

    descriptions = load_field_descriptions(fields_path)
    generated_at = _utc_now()
    datasets = [
        _dataset_entry(
            definition,
            project_root=project_root,
            descriptions=descriptions,
        )
        for definition in definitions
    ]

    catalog = {
        "version": CATALOG_VERSION,
        "generated_at": generated_at,
        "summary": {
            "datasets": len(datasets),
            "files": sum(
                int(dataset["file_count"])
                for dataset in datasets
            ),
            "rows": sum(
                int(dataset["row_count"])
                for dataset in datasets
            ),
            "fields": sum(
                len(dataset["fields"])
                for dataset in datasets
            ),
            "contract_failures": sum(
                dataset["contract"]["status"] == "failed"
                for dataset in datasets
            ),
        },
        "datasets": datasets,
    }
    validation_errors = validate_catalog(catalog)
    if strict:
        require_valid(validation_errors)
    catalog["validation"] = {
        "status": "failed" if validation_errors else "passed",
        "errors": validation_errors,
    }

    lineage = _build_lineage(
        datasets,
        generated_at=generated_at,
    )
    markdown = _render_markdown(catalog)

    resolved_catalog_path = _project_path(
        project_root, catalog_path
    )
    resolved_lineage_path = _project_path(
        project_root, lineage_path
    )
    resolved_markdown_path = _project_path(
        project_root, markdown_path
    )

    for path in (
        resolved_catalog_path,
        resolved_lineage_path,
        resolved_markdown_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    resolved_catalog_path.write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_lineage_path.write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return CatalogBuildResult(
        catalog_path=resolved_catalog_path,
        lineage_path=resolved_lineage_path,
        markdown_path=resolved_markdown_path,
        datasets=len(datasets),
        files=int(catalog["summary"]["files"]),
        rows=int(catalog["summary"]["rows"]),
        contract_failures=int(
            catalog["summary"]["contract_failures"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect curated Parquet artifacts and generate the "
            "Market Value Pulse catalog and lineage documentation."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=DEFAULT_DATASETS_PATH,
    )
    parser.add_argument(
        "--fields",
        type=Path,
        default=DEFAULT_FIELDS_PATH,
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write documentation even when contracts fail.",
    )
    args = parser.parse_args()

    result = build_catalog(
        project_root=args.project_root,
        datasets_path=args.datasets,
        fields_path=args.fields,
        strict=not args.no_strict,
    )
    print(
        "Catalog built: "
        f"{result.datasets} datasets, "
        f"{result.files} files, "
        f"{result.rows:,} rows."
    )
    print(f"Catalog: {result.catalog_path}")
    print(f"Lineage: {result.lineage_path}")
    print(f"Documentation: {result.markdown_path}")


if __name__ == "__main__":
    main()
