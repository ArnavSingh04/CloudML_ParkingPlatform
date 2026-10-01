"""Camera simulator service.

Stands in for real parking-lot cameras. For every configured car park it exposes
an endpoint ending in ``/api/takephoto`` that returns a *random* supplied JPEG
encoded as base64.

Deliberately standalone: it has NO dependency on the ``app`` package and no ML
libraries, so ``Dockerfile.camera`` builds a small image. The supplied images are
mounted at runtime (via ``IMAGES_DIR``) and are never copied into the image.

Configuration (environment variables):
  IMAGES_DIR      Directory containing the supplied JPEGs. Default ``./images``.
  NUM_CARPARKS    Number of car parks to expose cameras for (10..99). Default 24.
  SERVICE_NAME    Logging service label. Default ``smartpark-camera``.
  LOG_LEVEL       Log level. Default ``INFO``.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

# --- configuration --------------------------------------------------------
# Permitted car-park range from the assignment (§4.2, "10-99"). Must match
# app/config.py so the API and camera agree on the configuration envelope.
MIN_CARPARKS = 10
MAX_CARPARKS = 99

# Car-park id format (§4.1 example output: "CBD_001"). Must stay in sync with
# app.services.carpark_registry — the two services share no code.
CARPARK_ID_PREFIX = "CBD"
CARPARK_ID_DIGITS = 3


def _parse_num_carparks(raw: str) -> int:
    """Parse & validate NUM_CARPARKS, failing fast with a clear startup error."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(
            f"NUM_CARPARKS must be an integer, got {raw!r}"
        )
    if not MIN_CARPARKS <= value <= MAX_CARPARKS:
        raise SystemExit(
            f"NUM_CARPARKS must be between {MIN_CARPARKS} and {MAX_CARPARKS} "
            f"(inclusive), got {value}"
        )
    return value


IMAGES_DIR = os.getenv("IMAGES_DIR", "./images")
NUM_CARPARKS = _parse_num_carparks(os.getenv("NUM_CARPARKS", "24"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "smartpark-camera")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_MAX_REQUEST_ID_LEN = 200


def _sanitise_request_id(raw: str | None) -> str | None:
    """Return a trusted incoming request id, or None to mint a fresh one."""
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > _MAX_REQUEST_ID_LEN:
        return None
    if not candidate.isascii() or not candidate.isprintable():
        return None
    return candidate


# --- structured JSON logging (self-contained) -----------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "endpoint",
            "latency_ms",
            "status_code",
            "carpark_id",
            "images_dir",
            "num_carparks",
            "image_count",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL.upper())
    return logging.getLogger("smartpark.camera")


logger = _configure_logging()


def _carpark_id(index: int) -> str:
    """Must match app.services.carpark_registry.carpark_id."""
    return f"{CARPARK_ID_PREFIX}_{index:0{CARPARK_ID_DIGITS}d}"


def _valid_carpark_ids() -> set[str]:
    return {_carpark_id(i) for i in range(1, NUM_CARPARKS + 1)}


def _load_image_files(images_dir: str) -> list[Path]:
    directory = Path(images_dir)
    if not directory.is_dir():
        logger.error("images directory not found", extra={"images_dir": images_dir})
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    ]
    return sorted(files)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.valid_ids = _valid_carpark_ids()
    app.state.image_files = _load_image_files(IMAGES_DIR)
    logger.info(
        "camera simulator ready",
        extra={
            "status_code": 200,
            "images_dir": IMAGES_DIR,
            "num_carparks": NUM_CARPARKS,
            "image_count": len(app.state.image_files),
        },
    )
    yield


app = FastAPI(title="SmartPark Camera Simulator", lifespan=lifespan)


class _AccessLogMiddleware:
    """Minimal pure-ASGI access logger for the camera service."""

    def __init__(self, asgi_app) -> None:
        self.app = asgi_app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Reuse a valid incoming X-Request-ID (forwarded by the API for
        # cross-service tracing); otherwise mint a new one.
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id")
        request_id = _sanitise_request_id(
            incoming.decode("latin-1") if incoming else None
        ) or uuid4().hex
        path = scope.get("path", "")
        start = time.perf_counter()
        status_holder = {"code": 500}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode())
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request completed",
                extra={
                    "request_id": request_id,
                    "endpoint": path,
                    "latency_ms": latency_ms,
                    "status_code": status_holder["code"],
                },
            )


app.add_middleware(_AccessLogMiddleware)


def _health_payload(request: Request) -> dict:
    # Read state off the *request's* app rather than the module-level global, so
    # the helper works for any app instance (and inside tests).
    image_count = len(getattr(request.app.state, "image_files", []))
    ready = image_count > 0
    return {
        "status": "ok" if ready else "degraded",
        "service": SERVICE_NAME,
        "num_carparks": NUM_CARPARKS,
        "images_available": image_count,
        "ready": ready,
    }


@app.get("/health")
async def health(request: Request) -> dict:
    """Combined health probe (backward-compatible alias of /health/live)."""
    return _health_payload(request)


@app.get("/health/live")
async def health_live(request: Request) -> dict:
    """Liveness: the process is up. Always 200 while running."""
    return _health_payload(request)


@app.get("/health/ready")
async def health_ready(request: Request) -> dict:
    """Readiness: 200 only when at least one usable image is loaded, else 503."""
    payload = _health_payload(request)
    if not payload["ready"]:
        raise HTTPException(
            status_code=503, detail="No usable images available on the camera"
        )
    return payload


@app.get("/cameras")
async def list_cameras(request: Request) -> dict:
    """List the takephoto endpoint for every configured car park.

    The API service also uses the ``count`` field at startup to detect a
    NUM_CARPARKS mismatch between the two services.
    """
    ids = sorted(request.app.state.valid_ids)
    return {
        "count": len(ids),
        "num_carparks": NUM_CARPARKS,
        "cameras": [
            {"carpark_id": cid, "takephoto_url": f"/cameras/{cid}/api/takephoto"}
            for cid in ids
        ],
    }


@app.get("/cameras/{carpark_id}/api/takephoto")
async def take_photo(carpark_id: str, request: Request) -> dict:
    """Return a random supplied image (base64) for the given car park's camera."""
    if carpark_id not in request.app.state.valid_ids:
        raise HTTPException(status_code=404, detail=f"No camera for {carpark_id}")
    image_files: list[Path] = request.app.state.image_files
    if not image_files:
        raise HTTPException(status_code=503, detail="No images available on the camera")

    chosen: Path = random.choice(image_files)
    # read_bytes() is blocking disk I/O (images are on a mounted volume, later a
    # Cloud Storage FUSE mount). Offload it so it never stalls the event loop.
    data = await asyncio.to_thread(chosen.read_bytes)
    content_type = _CONTENT_TYPES.get(chosen.suffix.lower(), "image/jpeg")
    return {
        "carpark_id": carpark_id,
        "filename": chosen.name,
        "content_type": content_type,
        "image_base64": base64.b64encode(data).decode("ascii"),
    }
