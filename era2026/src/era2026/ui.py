"""Build the static dashboard by inlining the export into the template."""

from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


UI_DIR = _project_root() / "ui"
TEMPLATE = UI_DIR / "template.html"
DATA = UI_DIR / "data.json"
OUTPUT = UI_DIR / "index.html"

PLACEHOLDER = "__DATA__"


def build() -> Path:
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} missing; run era2026-export first")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    OUTPUT.write_text(template.replace(PLACEHOLDER, DATA.read_text(encoding="utf-8")), encoding="utf-8")
    return OUTPUT


def cli() -> None:
    path = build()
    print(f"built {path} ({path.stat().st_size / 1024:.0f} KB) — open it directly, no server needed")
