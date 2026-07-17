"""Automated data catalog and lineage generation."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catalog.models import CatalogBuildResult

__all__ = ["CatalogBuildResult", "build_catalog"]


def __getattr__(name: str) -> Any:
    """Load public catalog objects without importing the builder eagerly."""
    if name == "CatalogBuildResult":
        from catalog.models import CatalogBuildResult

        return CatalogBuildResult

    if name == "build_catalog":
        from catalog.builder import build_catalog

        return build_catalog

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
