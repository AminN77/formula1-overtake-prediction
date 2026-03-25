"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import circuits, health, models, predict, sensitivity, standings
from .services.model_registry import ModelRegistry


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if registry is not None:
            app.state.registry = registry
        else:
            reg = ModelRegistry(settings.artifacts_dir, settings.default_model)
            reg.load()
            app.state.registry = reg
        yield

    app = FastAPI(
        title="F1 Overtake Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(circuits.router, prefix="/api")
    app.include_router(standings.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(sensitivity.router, prefix="/api")
    return app


app = create_app()
