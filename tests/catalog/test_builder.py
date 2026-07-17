from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from catalog.builder import build_catalog
from catalog.validators import CatalogValidationError


def _write_config(root: Path) -> tuple[Path, Path]:
    datasets = {
        "datasets": [
            {
                "name": "example",
                "description": "Example contract.",
                "layer": "serving",
                "owner": "test",
                "grain": "one row per id",
                "paths": ["data/part=*/example.parquet"],
                "primary_key": ["id"],
                "partition_by": ["part"],
                "field_checks": {
                    "score": [
                        {"type": "range", "min": 0, "max": 100}
                    ]
                },
                "quality_rules": ["IDs are unique."],
            }
        ]
    }
    fields = {
        "fields": {
            "id": "Stable example identifier.",
            "score": "Example bounded score.",
        }
    }
    datasets_path = root / "datasets.json"
    fields_path = root / "fields.json"
    datasets_path.write_text(json.dumps(datasets), encoding="utf-8")
    fields_path.write_text(json.dumps(fields), encoding="utf-8")
    return datasets_path, fields_path


def test_build_catalog_aggregates_partition_rows(
    tmp_path: Path,
) -> None:
    datasets_path, fields_path = _write_config(tmp_path)

    for part, rows in ((1, 2), (2, 3)):
        path = tmp_path / f"data/part={part}/example.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame(
            {
                "id": list(range(part * 10, part * 10 + rows)),
                "score": [50.0] * rows,
            }
        ).write_parquet(path)

    result = build_catalog(
        project_root=tmp_path,
        datasets_path=datasets_path,
        fields_path=fields_path,
        strict=True,
    )

    payload = json.loads(
        result.catalog_path.read_text(encoding="utf-8")
    )
    dataset = payload["datasets"][0]

    assert result.rows == 5
    assert result.files == 2
    assert dataset["schema_drift"]["status"] == "compatible"
    assert dataset["contract"]["status"] == "passed"
    assert dataset["fields"][0]["description"] == (
        "Stable example identifier."
    )
    assert result.markdown_path.exists()


def test_strict_catalog_rejects_partition_schema_drift(
    tmp_path: Path,
) -> None:
    datasets_path, fields_path = _write_config(tmp_path)

    first = tmp_path / "data/part=1/example.parquet"
    second = tmp_path / "data/part=2/example.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)

    pl.DataFrame(
        {"id": [1], "score": [50.0]}
    ).write_parquet(first)
    pl.DataFrame(
        {"id": [2], "score": ["high"]}
    ).write_parquet(second)

    with pytest.raises(CatalogValidationError):
        build_catalog(
            project_root=tmp_path,
            datasets_path=datasets_path,
            fields_path=fields_path,
            strict=True,
        )
