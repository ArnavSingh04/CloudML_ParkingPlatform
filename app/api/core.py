"""Core business API: find-carparks and annotate-carpark."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import (
    get_camera_client,
    get_inference_service,
    get_registry,
    get_repository,
    get_response_cache,
)
from ..logging_config import get_logger
from ..models.schemas import (
    AnnotateResponse,
    CarParkStatus,
    FindCarParksResponse,
)
from ..services.camera_client import CameraClient, FetchOutcome
from ..services.carpark_registry import CarParkRegistry
from ..services.inference import InferenceService
from ..services.request_repository import RequestRepository
from ..services.response_cache import TTLCache

router = APIRouter(prefix="/api", tags=["core"])
logger = get_logger("smartpark.api")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/find-carparks", response_model=FindCarParksResponse)
async def find_carparks(
    uuid: str = Query(..., min_length=1, description="Caller/session identifier."),
    n: int = Query(
        ...,
        ge=1,
        description="Number of car parks to return. The API queries min(2*n, configured parks).",
    ),
    registry: CarParkRegistry = Depends(get_registry),
    camera: CameraClient = Depends(get_camera_client),
    inference: InferenceService = Depends(get_inference_service),
    repository: RequestRepository = Depends(get_repository),
    cache: TTLCache = Depends(get_response_cache),
) -> FindCarParksResponse:
    """Query up to 2*n distinct random car parks and return the best n by availability.

    Flow: check the per-user cache -> pick min(2*n, catalogue) distinct car parks
    -> fetch their camera photos concurrently -> run inference (serialised on the
    shared model) -> rank by number of 'empty' spaces -> return the top n (or
    fewer if some cameras fail). Cameras that fail or fail to infer are recorded
    as errored and excluded from ranking rather than failing the call.

    Large ``n`` (§4.3, Ed #109): the 2*n sample is capped at the configured
    catalogue. The request still returns 200 with ranked results — it does not
    400/422 just because 2*n exceeds NUM_CARPARKS (max 99).
    """
    # §4.3: repeated requests from the same user are served from a short-TTL
    # cache, skipping camera fetches and YOLO predictions.
    cache_key = (uuid, n)
    hit = await cache.get(cache_key)
    if hit is not None:
        # Still record the sighting: a cached response is a real user request
        # and must count towards OPS-API-2's "users in the last 30 seconds".
        await repository.record_uuid(uuid)
        logger.info(
            "find-carparks served from cache",
            extra={"requested_n": n, "cache_hit": True},
        )
        return hit.model_copy(update={"cached": True})

    k = min(2 * n, registry.count)
    picks = registry.sample(k)

    outcomes = await camera.fetch_many(picks)
    timestamp = _utcnow_iso()

    async def process(outcome: FetchOutcome) -> CarParkStatus:
        carpark_id = outcome.carpark.id
        if not outcome.ok:
            return CarParkStatus(
                carpark_id=carpark_id,
                status="error",
                last_uuid=uuid,
                last_seen=timestamp,
                detail=outcome.error or "camera fetch failed",
            )
        try:
            result = await inference.infer(outcome.image_bytes)
        except Exception as exc:  # noqa: BLE001 - isolate per-car-park failures
            logger.exception(
                "inference failed", extra={"carpark_id": carpark_id, "error": str(exc)}
            )
            return CarParkStatus(
                carpark_id=carpark_id,
                status="error",
                last_uuid=uuid,
                last_seen=timestamp,
                detail=f"inference failed: {exc}",
            )
        return CarParkStatus(
            carpark_id=carpark_id,
            status="ok",
            empty_count=result.empty_count,
            occupied_count=result.occupied_count,
            total_spaces=result.total_spaces,
            confidence_score=result.confidence_score,
            speed_inference=result.speed_inference,
            last_uuid=uuid,
            last_seen=timestamp,
        )

    statuses = await asyncio.gather(*(process(o) for o in outcomes))

    # Persist every status and the UUID sighting for the operational views.
    # Batched: on a network-backed repository this is one round trip instead
    # of 2*n sequential ones.
    await repository.upsert_statuses(list(statuses))
    await repository.record_uuid(uuid)

    successful = [s for s in statuses if s.status == "ok"]
    # Rank by most empty spaces, breaking ties by mean 'empty' confidence.
    ranked = sorted(
        successful,
        key=lambda s: (s.empty_count, s.confidence_score),
        reverse=True,
    )[:n]

    # Top-level speed_inference (COREAPI1 example: "xxx ms"): total model
    # inference time across every car park inferred for this request.
    total_inference_ms = sum(s.speed_inference for s in successful)
    speed_inference = f"{total_inference_ms:.1f} ms"

    logger.info(
        "find-carparks completed",
        extra={
            "requested_n": n,
            "queried": len(picks),
            "succeeded": len(successful),
            "returned": len(ranked),
            "cache_hit": False,
        },
    )

    def _name(carpark_id: str) -> str | None:
        info = registry.get(carpark_id)
        return info.name if info else None

    response = FindCarParksResponse(
        uuid=uuid,
        status="success",
        msg="success",
        speed_inference=speed_inference,
        requested_n=n,
        queried=len(picks),
        returned=len(ranked),
        generated_at=timestamp,
        cached=False,
        results=[s.to_result(name=_name(s.carpark_id)) for s in ranked],
    )
    await cache.set(cache_key, response)
    return response


@router.get("/annotate-carpark", response_model=AnnotateResponse)
async def annotate_carpark(
    carpark_id: str = Query(..., description="Car park id, e.g. 'CBD_001'."),
    uuid: str | None = Query(None, description="Optional caller identifier."),
    registry: CarParkRegistry = Depends(get_registry),
    camera: CameraClient = Depends(get_camera_client),
    inference: InferenceService = Depends(get_inference_service),
    repository: RequestRepository = Depends(get_repository),
) -> AnnotateResponse:
    """Fetch one car park's photo and return a base64 annotated JPEG + counts."""
    carpark = registry.get(carpark_id)
    if carpark is None:
        raise HTTPException(status_code=404, detail=f"Unknown car park: {carpark_id}")

    try:
        image_bytes = await camera.fetch_photo(carpark.camera_url)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "camera fetch failed", extra={"carpark_id": carpark_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=502, detail=f"Camera fetch failed: {exc}"
        ) from exc

    try:
        result = await inference.infer(image_bytes, annotate=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "inference failed", extra={"carpark_id": carpark_id, "error": str(exc)}
        )
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    timestamp = _utcnow_iso()
    await repository.upsert_status(
        CarParkStatus(
            carpark_id=carpark_id,
            status="ok",
            empty_count=result.empty_count,
            occupied_count=result.occupied_count,
            total_spaces=result.total_spaces,
            confidence_score=result.confidence_score,
            speed_inference=result.speed_inference,
            last_uuid=uuid,
            last_seen=timestamp,
        )
    )
    if uuid:
        await repository.record_uuid(uuid)

    annotated = result.annotated_jpeg or b""
    return AnnotateResponse(
        carpark_id=carpark_id,
        status="success",
        msg="success",
        uuid=uuid,
        available_spaces=result.empty_count,
        empty_count=result.empty_count,
        occupied_count=result.occupied_count,
        total_spaces=result.total_spaces,
        confidence_score=result.confidence_score,
        speed_inference=result.speed_inference,
        image_base64=base64.b64encode(annotated).decode("ascii"),
    )
