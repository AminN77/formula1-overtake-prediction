"""Application settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_repo_root() -> Path:
    """Directory that contains the `pipeline` package (project root in dev, /app in Docker)."""
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "pipeline" / "__init__.py").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_ARTIFACTS = _REPO_ROOT / "models" / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    artifacts_dir: Path = Path(os.environ.get("MODEL_ARTIFACTS_DIR", str(_DEFAULT_ARTIFACTS)))
    default_model: str = os.environ.get("DEFAULT_MODEL", "v5")
    cors_origins: str = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://localhost:5174",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
