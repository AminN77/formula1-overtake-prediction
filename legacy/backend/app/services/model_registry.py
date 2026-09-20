"""Load and switch sklearn pipelines from models/artifacts + registry.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class LoadedModel:
    version: str
    pipeline: Any
    meta: dict[str, Any]
    pkl_path: Path
    meta_path: Path


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, default_version: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.default_version = default_version
        self._registry: dict[str, Any] = {}
        self._active: LoadedModel | None = None
        self._load_registry_file()

    def _load_registry_file(self) -> None:
        reg_path = self.artifacts_dir / "registry.json"
        if reg_path.exists():
            self._registry = json.loads(reg_path.read_text())
        else:
            self._registry = {"default": self.default_version, "models": {}}

    def available_versions(self) -> list[str]:
        models = self._registry.get("models") or {}
        return sorted(models.keys())

    def data_version_for(self, version: str) -> str | None:
        models = self._registry.get("models") or {}
        entry = models.get(version) or {}
        return entry.get("data_version")

    def resolve_default(self) -> str:
        return str(self._registry.get("default") or self.default_version)

    def load(self, version: str | None = None) -> LoadedModel:
        ver = version or self.resolve_default()
        models = self._registry.get("models") or {}
        if ver not in models:
            raise FileNotFoundError(f"Unknown model version '{ver}'. Available: {list(models.keys())}")

        entry = models[ver]
        pkl = self.artifacts_dir / entry["pkl"]
        meta_p = self.artifacts_dir / entry["meta"]
        if not pkl.exists() or not meta_p.exists():
            raise FileNotFoundError(f"Missing artifact files for {ver}: {pkl} / {meta_p}")

        pipeline = joblib.load(pkl)
        meta = json.loads(meta_p.read_text())
        self._active = LoadedModel(
            version=ver,
            pipeline=pipeline,
            meta=meta,
            pkl_path=pkl,
            meta_path=meta_p,
        )
        return self._active

    @property
    def active(self) -> LoadedModel:
        if self._active is None:
            self.load()
        assert self._active is not None
        return self._active
