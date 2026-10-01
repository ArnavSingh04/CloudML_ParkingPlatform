"""Cross-service request-id tracing (concern 2).

Covers the main API middleware (reuse vs mint) and that CameraClient forwards
the active request id to the camera service.
"""

from __future__ import annotations

import base64

import httpx

from conftest import build_client

from app.logging_config import request_id_ctx
from app.services.camera_client import CameraClient


def test_main_reuses_valid_incoming_request_id():
    ctx = build_client()
    resp = ctx.client.get("/api/health", headers={"X-Request-ID": "trace-abc-123"})
    assert resp.headers["x-request-id"] == "trace-abc-123"


def test_main_generates_request_id_when_absent():
    ctx = build_client()
    resp = ctx.client.get("/api/health")
    rid = resp.headers.get("x-request-id")
    assert rid and len(rid) >= 8  # a freshly minted uuid4 hex


def test_main_replaces_invalid_incoming_request_id():
    ctx = build_client()
    # Over-length id is rejected and a fresh one is minted instead.
    bogus = "x" * 500
    resp = ctx.client.get("/api/health", headers={"X-Request-ID": bogus})
    assert resp.headers["x-request-id"] != bogus


async def test_camera_client_forwards_request_id():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["rid"] = request.headers.get("x-request-id")
        return httpx.Response(200, json={"image_base64": base64.b64encode(b"i").decode()})

    transport = httpx.MockTransport(handler)
    token = request_id_ctx.set("forwarded-rid-42")
    try:
        async with httpx.AsyncClient(transport=transport) as ac:
            await CameraClient(ac).fetch_photo("http://cam/x/api/takephoto")
    finally:
        request_id_ctx.reset(token)

    assert captured["rid"] == "forwarded-rid-42"


async def test_camera_client_omits_header_without_context():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["rid"] = request.headers.get("x-request-id")
        return httpx.Response(200, json={"image_base64": base64.b64encode(b"i").decode()})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as ac:
        await CameraClient(ac).fetch_photo("http://cam/x/api/takephoto")

    assert captured["rid"] is None
