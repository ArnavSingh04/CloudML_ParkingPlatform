"""FastAPI dependency providers.

Services are constructed once in the app lifespan and stashed on
``app.state``. These providers surface them to routes via ``Depends`` and — the
main reason they exist — give tests a single seam to override with fakes
(``app.dependency_overrides[...] = ...``) without touching the real model or
network.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from .config import Settings, get_settings
from .services.camera_client import CameraClient
from .services.carpark_registry import CarParkRegistry
from .services.inference import InferenceService
from .services.request_repository import RequestRepository
from .services.response_cache import TTLCache


def get_app_settings() -> Settings:
    return get_settings()


def get_registry(request: Request) -> CarParkRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Car-park registry unavailable")
    return registry


def get_repository(request: Request) -> RequestRepository:
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Repository unavailable")
    return repository


def get_camera_client(request: Request) -> CameraClient:
    client = getattr(request.app.state, "camera_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Camera client unavailable")
    return client


def get_response_cache(request: Request) -> TTLCache:
    cache = getattr(request.app.state, "response_cache", None)
    if cache is None:
        raise HTTPException(status_code=503, detail="Response cache unavailable")
    return cache


def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Inference service unavailable (model not loaded)"
        )
    return service
