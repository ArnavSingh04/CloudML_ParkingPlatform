"""Operational API: statuses, recent UUIDs, health, and the dashboard."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from .. import __version__
from ..config import Settings
from ..dashboard import DASHBOARD_HTML
from ..dependencies import (
    get_app_settings,
    get_registry,
    get_repository,
    get_response_cache,
)
from ..models.schemas import (
    AvailabilityResponse,
    CarParkAvailability,
    CarParkInfo,
    HealthResponse,
    RecentUuidsResponse,
    StatusesResponse,
)
from ..plotting import render_availability_png
from ..services.carpark_registry import CarParkRegistry
from ..services.request_repository import RequestRepository
from ..services.response_cache import TTLCache

router = APIRouter(tags=["operations"])


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/api/operations/statuses", response_model=StatusesResponse)
async def all_statuses(
    repository: RequestRepository = Depends(get_repository),
) -> StatusesResponse:
    """Return the latest known status for every car park queried so far."""
    statuses = await repository.list_statuses()
    return StatusesResponse(count=len(statuses), statuses=statuses)


@router.get("/api/operations/recent-uuids", response_model=RecentUuidsResponse)
async def recent_uuids(
    repository: RequestRepository = Depends(get_repository),
    settings: Settings = Depends(get_app_settings),
) -> RecentUuidsResponse:
    """Return distinct UUIDs seen within the configured sliding window (30s)."""
    window = settings.uuid_window_seconds
    uuids = await repository.recent_uuids(window)
    return RecentUuidsResponse(window_seconds=window, count=len(uuids), uuids=uuids)


@router.get("/api/operations/cache-stats")
async def cache_stats(
    cache: TTLCache = Depends(get_response_cache),
) -> dict:
    """Effectiveness of the §4.3 per-user response cache (hits, misses, size).

    Exposed so the caching optimisation can be demonstrated and quantified
    during the benchmark rather than merely asserted.
    """
    return await cache.stats()


@router.get("/api/carparks", response_model=list[CarParkInfo])
async def list_carparks(
    registry: CarParkRegistry = Depends(get_registry),
) -> list[CarParkInfo]:
    """List all configured car parks and their camera URLs."""
    return registry.all()


@router.get("/api/operations/availability", response_model=AvailabilityResponse)
async def availability(
    registry: CarParkRegistry = Depends(get_registry),
    repository: RequestRepository = Depends(get_repository),
) -> AvailabilityResponse:
    """OPS-API-1: every configured car park with its current available spots.

    Merges the full static catalogue (from the registry) with the latest known
    status (from the repository). Car parks never queried show status 'unknown'
    with null counts, so the list always covers *all* car parks.
    """
    latest = {s.carpark_id: s for s in await repository.list_statuses()}

    carparks: list[CarParkAvailability] = []
    for info in registry.all():
        status = latest.get(info.id)
        if status is None:
            carparks.append(CarParkAvailability(carpark_id=info.id, name=info.name))
        elif status.status == "ok":
            carparks.append(
                CarParkAvailability(
                    carpark_id=info.id,
                    name=info.name,
                    available_spots=status.empty_count,
                    total_spaces=status.total_spaces,
                    status="ok",
                    last_seen=status.last_seen,
                )
            )
        else:
            carparks.append(
                CarParkAvailability(
                    carpark_id=info.id,
                    name=info.name,
                    status="error",
                    last_seen=status.last_seen,
                )
            )

    return AvailabilityResponse(
        count=len(carparks), generated_at=_utcnow_iso(), carparks=carparks
    )


@router.get("/api/operations/plot.png", include_in_schema=False)
async def operational_plot(
    repository: RequestRepository = Depends(get_repository),
    settings: Settings = Depends(get_app_settings),
) -> Response:
    """OPS-REQ-2: on-demand matplotlib PNG of current availability + load.

    Rendering is blocking/CPU-bound, so — like inference — it is offloaded to a
    worker thread to keep the event loop responsive.
    """
    window = settings.uuid_window_seconds
    statuses = await repository.list_statuses()
    recent = await repository.recent_uuids(window)
    png = await asyncio.to_thread(
        render_availability_png, statuses, window, len(recent)
    )
    return Response(content=png, media_type="image/png")


def _build_health(request: Request, settings: Settings) -> HealthResponse:
    inference = getattr(request.app.state, "inference", None)
    model_loaded = inference is not None and inference.model_loaded
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service=settings.service_name,
        version=__version__,
        model_loaded=model_loaded,
        num_carparks=settings.num_carparks,
        ready=model_loaded,
    )


@router.get("/api/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Combined health probe (backward-compatible). Reports model-load state."""
    return _build_health(request, settings)


@router.get("/health/live", response_model=HealthResponse, include_in_schema=False)
async def health_live(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Liveness: the process is up and serving. Always 200 while running."""
    return _build_health(request, settings)


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Readiness: 200 only once the YOLO model is loaded, else 503."""
    health_body = _build_health(request, settings)
    if not health_body.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return health_body


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the simple HTML monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
