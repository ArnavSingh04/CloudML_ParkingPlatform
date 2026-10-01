"""Pydantic schemas shared across the API.

These are the wire contracts (request/response bodies and persisted records).
Keeping them in one place makes the API surface easy to review and lets the
in-memory repository be swapped for Firestore later without touching callers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CarParkInfo(BaseModel):
    """Static description of a configured car park."""

    id: str
    name: str
    camera_url: str


class CarParkResult(BaseModel):
    """A single car park in a find-carparks response.

    Field names mirror the assignment's COREAPI1 example output (§4.1):
    ``carpark_id``, ``available_spaces``, ``confidence_score`` (plus an optional
    ``name``). ``available_spaces`` is the count of detections classified
    'empty'.
    """

    carpark_id: str
    name: str | None = Field(
        default=None, description="Human-readable car park name, if known."
    )
    available_spaces: int = Field(
        description="Number of detected 'empty' spaces (greatest-first ranking key).",
    )
    confidence_score: float = Field(
        description="Mean confidence of 'empty' detections, or 0.0 when none.",
    )


class FindCarParksResponse(BaseModel):
    """Response body for GET /api/find-carparks (COREAPI1, §4.1).

    Top-level fields follow the assignment's example output: ``uuid``,
    ``status``, ``msg``, ``speed_inference`` (a string like ``"123.4 ms"``) and
    ``requested_n``. ``queried``/``returned``/``generated_at`` are additive
    operational fields (harmless to the documented contract).
    """

    uuid: str
    status: str = Field(default="success", description="'success' or 'error'.")
    msg: str = Field(default="success", description="Error message if there is an issue.")
    speed_inference: str = Field(
        description='Total model inference time for the request, e.g. "123.4 ms".',
    )
    requested_n: int = Field(description="Number of car parks requested (n).")
    queried: int = Field(description="Distinct car parks actually queried (min(2*n, catalogue)).")
    returned: int = Field(description="Number of car parks returned (<= n).")
    generated_at: str = Field(description="ISO-8601 UTC timestamp.")
    cached: bool = Field(
        default=False,
        description="True when served from the per-user response cache (§4.3).",
    )
    results: list[CarParkResult]


class AnnotateResponse(BaseModel):
    """Response body for GET /api/annotate-carpark (COREAPI2, §4.1).

    The assignment's example requires ``carpark_id``, ``status``, ``msg`` and
    ``image_base64``; the count/confidence/timing fields are additive detail.
    """

    carpark_id: str
    status: str = Field(default="success", description="'success' or 'error'.")
    msg: str = Field(default="success", description="Error message if there is an issue.")
    uuid: str | None = None
    available_spaces: int = Field(description="Number of detected 'empty' spaces.")
    empty_count: int
    occupied_count: int
    total_spaces: int
    confidence_score: float
    speed_inference: float = Field(description="Inference time in milliseconds.")
    content_type: str = "image/jpeg"
    image_base64: str = Field(description="Base64-encoded annotated JPEG.")


class CarParkStatus(BaseModel):
    """Latest known status for a car park (persisted in the repository)."""

    carpark_id: str
    status: Literal["ok", "error"] = "ok"
    empty_count: int = 0
    occupied_count: int = 0
    total_spaces: int = 0
    confidence_score: float = 0.0
    speed_inference: float = 0.0
    last_uuid: str | None = None
    last_seen: str = Field(description="ISO-8601 UTC timestamp of last update.")
    detail: str | None = Field(
        default=None, description="Error detail when status == 'error'."
    )

    def to_result(self, name: str | None = None) -> CarParkResult:
        """Project a status onto the COREAPI1 find-carparks result shape."""
        return CarParkResult(
            carpark_id=self.carpark_id,
            name=name,
            available_spaces=self.empty_count,
            confidence_score=self.confidence_score,
        )


class CarParkAvailability(BaseModel):
    """One car park's current availability for the OPS-API-1 listing.

    Every *configured* car park appears, even if it has never been queried; in
    that case ``status`` is 'unknown' and the count fields are null.
    """

    carpark_id: str
    name: str
    available_spots: int | None = Field(
        default=None,
        description="Current empty spaces, or null if never queried / errored.",
    )
    total_spaces: int | None = Field(
        default=None, description="Total detected spaces, or null if unknown."
    )
    status: Literal["ok", "error", "unknown"] = "unknown"
    last_seen: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp of last update, if any."
    )


class AvailabilityResponse(BaseModel):
    """Response body for OPS-API-1: all car parks + current available spots."""

    count: int = Field(description="Number of configured car parks.")
    generated_at: str = Field(description="ISO-8601 UTC timestamp of this snapshot.")
    carparks: list[CarParkAvailability]


class StatusesResponse(BaseModel):
    """Response body for the operational 'all car-park statuses' view."""

    count: int
    statuses: list[CarParkStatus]


class RecentUuidsResponse(BaseModel):
    """Response body for the operational 'recent UUIDs' view."""

    window_seconds: int
    count: int
    uuids: list[str]


class HealthResponse(BaseModel):
    """Response body for the health / liveness / readiness probes."""

    status: str = "ok"
    service: str
    version: str
    model_loaded: bool
    num_carparks: int
    ready: bool = Field(
        default=True,
        description="True when the service can serve inference (model loaded).",
    )
