"""Tests for GET /api/annotate-carpark."""

from __future__ import annotations

import base64

from conftest import FakeCameraClient, build_client


def test_returns_base64_annotated_image(ctx):
    resp = ctx.client.get(
        "/api/annotate-carpark", params={"carpark_id": "CBD_001", "uuid": "u2"}
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["carpark_id"] == "CBD_001"
    # COREAPI2 documented fields (§4.1).
    assert body["status"] == "success"
    assert body["msg"] == "success"
    assert body["content_type"] == "image/jpeg"
    # photo_bytes default is b"7" -> fake inference reports empty_count == 7.
    assert body["empty_count"] == 7
    assert body["available_spaces"] == 7
    assert base64.b64decode(body["image_base64"]) == b"ANNOTATED"


def test_records_status_and_uuid(ctx):
    ctx.client.get(
        "/api/annotate-carpark", params={"carpark_id": "CBD_002", "uuid": "u9"}
    )
    statuses = {s["carpark_id"]: s for s in
                ctx.client.get("/api/operations/statuses").json()["statuses"]}
    assert statuses["CBD_002"]["status"] == "ok"
    assert "u9" in ctx.client.get("/api/operations/recent-uuids").json()["uuids"]


def test_unknown_carpark_returns_404(ctx):
    resp = ctx.client.get(
        "/api/annotate-carpark", params={"carpark_id": "CBD_099"}
    )
    assert resp.status_code == 404


def test_camera_failure_returns_502():
    ctx = build_client(camera_client=FakeCameraClient(fail_photo=True))
    resp = ctx.client.get(
        "/api/annotate-carpark", params={"carpark_id": "CBD_001"}
    )
    assert resp.status_code == 502
