"""Shared test fixtures and fakes.

The whole point of the dependency-injection seam in ``app/dependencies.py`` is
visible here: we override the registry, repository, camera client and inference
service with in-process fakes, so tests never load the real model, read the
supplied images, or make a network call. TestClient is used WITHOUT its context
manager so the app lifespan (which would load the model) never runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import (
    get_app_settings,
    get_camera_client,
    get_inference_service,
    get_registry,
    get_repository,
    get_response_cache,
)
from app.main import create_app
from app.models.schemas import CarParkInfo
from app.services.camera_client import FetchOutcome
from app.services.carpark_registry import CarParkRegistry
from app.services.inference import InferenceResult
from app.services.request_repository import InMemoryRequestRepository
from app.services.response_cache import TTLCache


def carpark_number(carpark_id: str) -> int:
    """Car-park number from a 'CBD_007'-style id."""
    return int(carpark_id.split("_")[1])


def _empty_from_id(carpark: CarParkInfo) -> int:
    """Derive a deterministic 'empty' count from the car park number."""
    return carpark_number(carpark.id)


class FakeCameraClient:
    """Stand-in for CameraClient with no network I/O.

    Encodes the desired 'empty' count for a car park as the image bytes, so the
    fake inference service can produce deterministic, per-car-park results.
    """

    def __init__(self, fail_all: bool = False, fail_photo: bool = False,
                 photo_bytes: bytes = b"7") -> None:
        self.fail_all = fail_all
        self.fail_photo = fail_photo
        self.photo_bytes = photo_bytes

    async def fetch_many(self, carparks: list[CarParkInfo]) -> list[FetchOutcome]:
        outcomes = []
        for cp in carparks:
            if self.fail_all:
                outcomes.append(FetchOutcome(cp, None, "camera down"))
            else:
                outcomes.append(FetchOutcome(cp, str(_empty_from_id(cp)).encode()))
        return outcomes

    async def fetch_photo(self, camera_url: str) -> bytes:
        if self.fail_photo:
            raise RuntimeError("camera down")
        return self.photo_bytes


class FakeInferenceService:
    """Stand-in for InferenceService — no model, no threads."""

    model_loaded = True

    def __init__(self) -> None:
        self.calls = 0

    async def infer(self, image_bytes: bytes, annotate: bool = False) -> InferenceResult:
        self.calls += 1
        empty = int(image_bytes.decode())
        return InferenceResult(
            empty_count=empty,
            occupied_count=2,
            total_spaces=empty + 2,
            confidence_score=round(min(0.99, 0.5 + empty / 100), 4),
            speed_inference=10.0,
            annotated_jpeg=b"ANNOTATED" if annotate else None,
        )


def build_client(
    *,
    num_carparks: int = 24,
    camera_client: FakeCameraClient | None = None,
    inference: FakeInferenceService | None = None,
    uuid_window: int = 30,
    cache_ttl: float = 0.0,
) -> SimpleNamespace:
    """Create a TestClient with all external dependencies overridden.

    ``cache_ttl`` defaults to 0 (caching disabled) so most tests exercise the
    real pipeline on every call; the cache has its own dedicated tests.
    """
    app = create_app()
    registry = CarParkRegistry(num_carparks, "http://camera.test")
    repository = InMemoryRequestRepository()
    camera_client = camera_client or FakeCameraClient()
    inference = inference or FakeInferenceService()
    cache: TTLCache = TTLCache(ttl_seconds=cache_ttl)
    settings = Settings(
        num_carparks=num_carparks,
        uuid_window_seconds=uuid_window,
        camera_base_url="http://camera.test",
        cache_ttl_seconds=cache_ttl,
    )

    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_camera_client] = lambda: camera_client
    app.dependency_overrides[get_inference_service] = lambda: inference
    app.dependency_overrides[get_response_cache] = lambda: cache
    app.dependency_overrides[get_app_settings] = lambda: settings
    # health reads app.state.inference directly (not via a dependency).
    app.state.inference = inference

    return SimpleNamespace(
        client=TestClient(app),
        app=app,
        registry=registry,
        repository=repository,
        camera_client=camera_client,
        inference=inference,
        cache=cache,
        settings=settings,
    )


@pytest.fixture
def ctx() -> SimpleNamespace:
    """Default happy-path client with 24 car parks."""
    return build_client()
