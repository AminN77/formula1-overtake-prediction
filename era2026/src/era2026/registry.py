"""
Model registry: one incumbent pointer, versioned artifacts, full provenance.

A single model ships, so this is a pointer and a history, not a version table.
Every stored version records the rounds it was trained on, the data snapshot it
came from, and the metrics that justified its promotion, so "the model as of
round 14" resolves to exactly one lineage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


REGISTRY_DIR = _project_root() / "models"
MANIFEST = REGISTRY_DIR / "manifest.json"


@dataclass
class ModelVersion:
    version: str
    created_at: str
    trained_on_rounds: list[int]
    train_rows: int
    train_events: int
    data_fingerprint: str
    params: dict
    features: list[str]
    metrics: dict
    status: str = "challenger"  # challenger | incumbent | archived
    note: str = ""
    promoted_over: str | None = None


def data_fingerprint(rows: pd.DataFrame) -> str:
    """Identifies the exact table a model was trained on."""
    payload = f"{len(rows)}|{sorted(rows.columns)}|{sorted(rows['round_number'].unique())}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_manifest() -> dict:
    if not MANIFEST.exists():
        return {"incumbent": None, "versions": []}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_manifest(manifest: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def register(version: ModelVersion, model: object | None = None) -> ModelVersion:
    manifest = _read_manifest()
    manifest["versions"] = [v for v in manifest["versions"] if v["version"] != version.version]
    manifest["versions"].append(asdict(version))
    _write_manifest(manifest)
    if model is not None:
        import joblib

        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, REGISTRY_DIR / f"{version.version}.joblib")
    return version


def incumbent() -> dict | None:
    manifest = _read_manifest()
    if not manifest.get("incumbent"):
        return None
    return next((v for v in manifest["versions"] if v["version"] == manifest["incumbent"]), None)


def promote(version_id: str, over: str | None = None) -> None:
    manifest = _read_manifest()
    for entry in manifest["versions"]:
        if entry["version"] == version_id:
            entry["status"] = "incumbent"
            entry["promoted_over"] = over
        elif entry["status"] == "incumbent":
            entry["status"] = "archived"
    manifest["incumbent"] = version_id
    _write_manifest(manifest)


def archive(version_id: str, note: str = "") -> None:
    manifest = _read_manifest()
    for entry in manifest["versions"]:
        if entry["version"] == version_id:
            entry["status"] = "archived"
            if note:
                entry["note"] = note
    _write_manifest(manifest)


def history() -> pd.DataFrame:
    manifest = _read_manifest()
    return pd.DataFrame(manifest["versions"])


def new_version_id(round_number: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"r{round_number:02d}-{stamp}"
