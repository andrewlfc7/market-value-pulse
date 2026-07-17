from pathlib import Path

from catalog.definitions import (
    load_dataset_definitions,
    load_field_descriptions,
)
from catalog.validators import validate_definitions


def test_catalog_definitions_are_unique_and_valid() -> None:
    definitions = load_dataset_definitions(
        Path("config/catalog/datasets.json")
    )

    assert len(definitions) == 7
    assert not validate_definitions(definitions)
    assert len({item.name for item in definitions}) == len(definitions)


def test_core_field_definitions_are_documented() -> None:
    descriptions = load_field_descriptions(
        Path("config/catalog/field_descriptions.json")
    )

    required = {
        "post_match_rating",
        "form_rating_ewm",
        "predicted_market_value_eur",
        "primary_role",
        "percentile",
        "similarity",
    }
    assert required.issubset(descriptions)
