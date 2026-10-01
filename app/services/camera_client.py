"""Camera HTTP client.

Talks to the camera-simulator service. A single ``httpx.AsyncClient`` is
created at startup and reused for the process lifetime — this keeps the
connection pool warm (HTTP keep-alive) instead of paying TCP/TLS setup on every
photo, and is the recommended httpx usage pattern.

Photos across many car parks are fetched *concurrently*: the network wait for
one camera overlaps the wait for the others, so a find-carparks call is bound
by the slowest single camera rather than the sum of all of them.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx

from ..logging_config import get_logger, request_id_ctx
from ..models.schemas import CarParkInfo

logger = get_logger("smartpark.camera")


def _trace_headers() -> dict[str, str]:
    """Propagate the current request id to the camera for cross-service tracing."""
    request_id = request_id_ctx.get()
    return {"X-Request-ID": request_id} if request_id else {}


@dataclass(slots=True)
class FetchOutcome:
    """Result of trying to fetch one car park's photo."""

    carpark: CarParkInfo
    image_bytes: bytes | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.image_bytes is not None and self.error is None


class CameraClient:
    """Fetches camera photos over HTTP using a shared async client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch_photo(self, camera_url: str) -> bytes:
        """Fetch and decode a single camera photo. Raises on HTTP/parse errors."""
        response = await self._client.get(camera_url, headers=_trace_headers())
        response.raise_for_status()
        payload = response.json()
        return base64.b64decode(payload["image_base64"])

    async def fetch_many(self, carparks: list[CarParkInfo]) -> list[FetchOutcome]:
        """Fetch photos for many car parks concurrently.

        A failure for one car park is captured on its FetchOutcome rather than
        raising, so one bad camera never fails the whole batch.
        """

        async def _one(carpark: CarParkInfo) -> FetchOutcome:
            try:
                image_bytes = await self.fetch_photo(carpark.camera_url)
                return FetchOutcome(carpark=carpark, image_bytes=image_bytes)
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                logger.error(
                    "camera fetch failed",
                    extra={"carpark_id": carpark.id, "error": str(exc)},
                )
                return FetchOutcome(carpark=carpark, image_bytes=None, error=str(exc))

        return await asyncio.gather(*(_one(cp) for cp in carparks))
