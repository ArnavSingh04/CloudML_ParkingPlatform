"""SmartPark API application factory and lifespan.

The FastAPI *lifespan* builds the long-lived singletons exactly once on startup
(model, HTTP client, registry, repository), stashes them on ``app.state``, and
tears the network client down on shutdown. This is where the assignment's
"load the YOLO model once" requirement is satisfied.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from . import __version__
from .api import core, operations
from .config import Settings, get_settings
from .logging_config import configure_logging, get_logger
from .middleware import RequestContextMiddleware
from .services.camera_client import CameraClient
from .services.carpark_registry import CarParkRegistry
from .services.inference import InferenceService
from .services.request_repository import (
    InMemoryRequestRepository,
    RequestRepository,
)
from .services.response_cache import TTLCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name=settings.service_name)
    logger = get_logger("smartpark.startup")

    # Registry is cheap and always available. camera_base_url_clean normalises
    # away any trailing slash before the camera URLs are built.
    app.state.registry = CarParkRegistry(
        num_carparks=settings.num_carparks,
        camera_base_url=settings.camera_base_url_clean,
    )
    # Repository: process-local memory for single-pod/dev, or shared Firestore
    # for a multi-pod deployment (so operational views aggregate across pods).
    app.state.repository = _build_repository(settings, logger)

    # Short-TTL cache for repeated find-carparks calls from the same user
    # (§4.3 performance optimisation). Disabled when CACHE_TTL_SECONDS=0.
    app.state.response_cache = TTLCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )

    # One reused httpx.AsyncClient for all camera calls (keep-alive pooling).
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.http_timeout_seconds)
    )
    app.state.http_client = http_client
    app.state.camera_client = CameraClient(http_client)

    # Load the model once. If it fails (e.g. weights not mounted), the app still
    # starts in a degraded state; inference routes return 503 until it's fixed.
    app.state.inference = None
    try:
        app.state.inference = InferenceService.load(
            model_path=settings.model_path,
            max_concurrency=settings.model_max_concurrency,
            confidence=settings.confidence_threshold,
        )
        logger.info(
            "startup complete",
            extra={"num_carparks": settings.num_carparks, "model_loaded": True},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "model load failed; inference routes will return 503",
            extra={"model_path": settings.model_path, "error": str(exc)},
        )

    # Best-effort, non-blocking check that the camera service agrees on
    # NUM_CARPARKS. Runs as a background task so a slow or absent camera can
    # never delay the app becoming ready.
    config_check: asyncio.Task | None = None
    if settings.verify_camera_carpark_count:
        config_check = asyncio.create_task(
            _verify_carpark_count(http_client, settings, logger)
        )

    try:
        yield
    finally:
        if config_check is not None and not config_check.done():
            config_check.cancel()
        await http_client.aclose()
        await app.state.repository.close()
        logger.info("shutdown complete")


async def _verify_carpark_count(
    http_client: httpx.AsyncClient, settings: Settings, logger: logging.Logger
) -> None:
    """Warn if the camera service was configured with a different NUM_CARPARKS.

    A mismatch is silent but corrosive: the API would request cameras the
    simulator has never heard of, and those car parks would be permanently
    marked 'error'. Detecting it at startup turns a confusing runtime symptom
    into one obvious log line.
    """
    url = f"{settings.camera_base_url_clean}/cameras"
    try:
        response = await http_client.get(url, timeout=httpx.Timeout(5.0))
        response.raise_for_status()
        camera_count = int(response.json().get("count", -1))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
        logger.warning(
            "could not verify camera car-park count",
            extra={"camera_url": url, "error": str(exc)},
        )
        return

    if camera_count != settings.num_carparks:
        logger.error(
            "NUM_CARPARKS mismatch between API and camera service; car parks "
            "outside the camera's range will always report 'error'",
            extra={
                "api_num_carparks": settings.num_carparks,
                "camera_num_carparks": camera_count,
            },
        )
    else:
        logger.info(
            "camera car-park count verified",
            extra={"num_carparks": camera_count},
        )


def _build_repository(
    settings: Settings, logger: logging.Logger
) -> RequestRepository:
    """Construct the configured repository backend.

    Importing the Firestore implementation lazily keeps ``google-cloud-firestore``
    off the import path for the in-memory/dev/test case.
    """
    if settings.repository_backend == "firestore":
        from .services.firestore_repository import FirestoreRequestRepository

        logger.info(
            "using Firestore repository",
            extra={
                "firestore_database": settings.firestore_database,
                "firestore_project": settings.firestore_project or "<ADC-default>",
            },
        )
        return FirestoreRequestRepository(
            project=settings.firestore_project,
            database=settings.firestore_database,
        )

    logger.info("using in-memory repository (single-pod)")
    return InMemoryRequestRepository()


def create_app() -> FastAPI:
    settings = get_settings()
    # Configure logging eagerly too, so logs emitted before lifespan are JSON.
    configure_logging(level=settings.log_level, service_name=settings.service_name)

    app = FastAPI(
        title="SmartPark API",
        version=__version__,
        description="FIT3184 SmartPark — find available car parks via YOLO detection.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(core.router)
    app.include_router(operations.router)
    return app


app = create_app()
