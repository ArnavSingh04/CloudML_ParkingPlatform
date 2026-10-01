"""Tests for the standalone camera simulator."""

from __future__ import annotations

import base64
import importlib

import pytest
from fastapi.testclient import TestClient

# Two tiny fake "JPEG" files (content is arbitrary bytes; the simulator just
# streams whatever is on disk).
IMG_A = b"\xff\xd8\xff\xe0AAA"
IMG_B = b"\xff\xd8\xff\xe0BBB"


@pytest.fixture
def camera_module(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(IMG_A)
    (tmp_path / "b.jpg").write_bytes(IMG_B)
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("NUM_CARPARKS", "12")

    import camera_service.main as camera_main
    importlib.reload(camera_main)  # re-read env-driven module globals
    return camera_main


def test_takephoto_returns_random_supplied_jpeg(camera_module):
    with TestClient(camera_module.app) as client:
        resp = client.get("/cameras/CBD_001/api/takephoto")
    assert resp.status_code == 200
    body = resp.json()
    assert body["carpark_id"] == "CBD_001"
    assert body["content_type"] == "image/jpeg"
    assert base64.b64decode(body["image_base64"]) in (IMG_A, IMG_B)


def test_unknown_carpark_returns_404(camera_module):
    with TestClient(camera_module.app) as client:
        # CBD_099 is outside the configured range (NUM_CARPARKS=12).
        resp = client.get("/cameras/CBD_099/api/takephoto")
    assert resp.status_code == 404


def test_lists_a_camera_for_every_configured_carpark(camera_module):
    with TestClient(camera_module.app) as client:
        body = client.get("/cameras").json()
    assert body["count"] == 12
    assert body["cameras"][0]["takephoto_url"].endswith("/api/takephoto")


def test_health_reports_image_count(camera_module):
    with TestClient(camera_module.app) as client:
        body = client.get("/health").json()
    assert body["num_carparks"] == 12
    assert body["images_available"] == 2
