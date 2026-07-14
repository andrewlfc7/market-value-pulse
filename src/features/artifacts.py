"""Resolve versioned feature-model artifacts from ``models/features``.

Each family/version directory owns a ``metadata.json`` sidecar whose
``model_file`` field names the primary artifact.  This keeps model selection
and provenance out of scoring code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Standardized family -> default version mapping. Used so callers can pass just
# a family and get the current canonical version, while still allowing an
# explicit --model-version override.
DEFAULT_VERSIONS = {
    "xg": "xg_shot_v1",
    "xgot": "xgot_shot_v1",
    "xa": "xa_action_v1",
    "xthreat": "xt_action_v1",
    "goal_probability": "xpv_action_v1",
    "pass_clusters": "pass_cluster_v1",
}


def _default_models_root() -> Path:
    """Locate the self-contained feature model artifacts.

    Resolution order:
      1. ``MVP_FEATURE_MODELS_ROOT`` override,
      2. walk up until ``models/features`` is found.
    """
    env = os.getenv("MVP_FEATURE_MODELS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models" / "features"
        if candidate.is_dir():
            return candidate
    return (here.parents[2] / "models" / "features").resolve()


@dataclass(frozen=True)
class ResolvedArtifact:
    """A located and validated feature-model artifact."""

    family: str
    version: str
    directory: Path
    metadata: dict[str, Any]

    @property
    def model_path(self) -> Path:
        """Primary model file path (from metadata ``model_file``)."""
        model_file = self.metadata.get("model_file")
        if not model_file:
            raise KeyError(
                f"metadata.json for {self.family}/{self.version} is missing "
                f"'model_file'"
            )
        return self.directory / model_file

    @property
    def artifact_version(self) -> str:
        """Version string for logging / status rows."""
        return self.metadata.get("model_version", self.version)

    def path(self, filename: str) -> Path:
        """Resolve a sibling file inside the artifact directory."""
        return self.directory / filename

    def load_model(self) -> Any:
        """Load the primary model file. Dispatches on extension.

        ``.joblib`` -> joblib.load ; ``.json`` -> parsed JSON dict.
        For multi-file (parquet) artifacts there is no single ``model_file``;
        callers should use ``path(...)`` directly.
        """
        p = self.model_path
        if p.suffix == ".joblib":
            import joblib

            return joblib.load(p)
        if p.suffix == ".json":
            return json.loads(p.read_text())
        raise ValueError(f"Unsupported model file extension: {p}")

    def load_extra_joblib(self, filename: str = "metadata.joblib") -> Any:
        """Load an auxiliary joblib (e.g. training feature config for shot models)."""
        import joblib

        p = self.directory / filename
        if not p.is_file():
            raise FileNotFoundError(
                f"Expected auxiliary joblib {filename} for "
                f"{self.family}/{self.version} at {p}"
            )
        return joblib.load(p)

    def log_line(self) -> str:
        return (
            f"[artifact] family={self.family} version={self.artifact_version} "
            f"dir={self.directory} model_file={self.metadata.get('model_file')}"
        )


def resolve_artifact(
    family: str,
    version: Optional[str] = None,
    models_root: Optional[os.PathLike[str] | str] = None,
    require_model_file: bool = True,
) -> ResolvedArtifact:
    """Resolve ``family``/``version`` to a validated inference artifact.

    Parameters
    ----------
    family:
        Model family directory name (e.g. ``xa``, ``xg``, ``xthreat``).
    version:
        Version directory name (e.g. ``xa_action_v1``). If ``None``, the
        canonical default from :data:`DEFAULT_VERSIONS` is used.
    models_root:
        Override the ``models/features`` root (else auto-detected / env).
    require_model_file:
        If True, validate the ``model_file`` named in metadata exists. Set
        False for multi-file (parquet) artifacts like pass clusters.

    Raises
    ------
    FileNotFoundError
        If the artifact directory, metadata, or model file is missing.
    """
    if version is None:
        version = DEFAULT_VERSIONS.get(family)
        if version is None:
            raise ValueError(
                f"No default version for family '{family}'; pass an explicit "
                f"version. Known families: {sorted(DEFAULT_VERSIONS)}"
            )

    root = Path(models_root).resolve() if models_root else _default_models_root()
    directory = root / family / version
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Artifact directory not found: {directory} "
            f"(family={family} version={version}, models_root={root})"
        )

    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json missing for artifact: {meta_path}")
    metadata = json.loads(meta_path.read_text())

    artifact = ResolvedArtifact(
        family=family, version=version, directory=directory, metadata=metadata
    )

    if require_model_file:
        mp = artifact.model_path
        if not mp.is_file():
            raise FileNotFoundError(
                f"Model file named in metadata not found: {mp} "
                f"(family={family} version={version})"
            )

    return artifact


def load_artifact(
    family: str,
    version: Optional[str] = None,
    models_root: Optional[os.PathLike[str] | str] = None,
    require_model_file: bool = True,
    log: bool = True,
) -> ResolvedArtifact:
    """Resolve an artifact and print its version line (convenience for scorers)."""
    artifact = resolve_artifact(
        family, version, models_root=models_root, require_model_file=require_model_file
    )
    if log:
        print(artifact.log_line(), flush=True)
    return artifact
