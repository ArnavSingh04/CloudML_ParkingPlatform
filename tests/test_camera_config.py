"""Camera-service configuration, readiness, logging and ID-parity tests.

Covers concern 1 (NUM_CARPARKS validation), concern 3 (readiness), concern 5
(structured-log fields) and concern 8 (car-park id parity across services).
"""

from __future__ import annotations

import importlib
import json
import logging

import pytest
from fastapi.testclient import TestClient


def _reload_camera(monkeypatch, images_dir, num_carparks):
    monkeypatch.setenv("IMAGES_DIR", str(images_dir))
    monkeypatch.setenv("NUM_CARPARKS", str(num_carparks))
    import camera_service.main as camera_main
    return importlib.reload(camera_main)


# --- concern 1: NUM_CARPARKS validation -----------------------------------

@pytest.mark.parametrize("value", [10, 24, 99])
def test_num_carparks_accepts_valid_boundaries(tmp_path, monkeypatch, value):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0A")
    mod = _reload_camera(monkeypatch, tmp_path, value)
    assert mod.NUM_CARPARKS == value


@pytest.mark.parametrize("value", [9, 100, 0, -5])
def test_num_carparks_rejects_out_of_range(tmp_path, monkeypatch, value):
    with pytest.raises(SystemExit):
        _reload_camera(monkeypatch, tmp_path, value)


@pytest.mark.parametrize("value", ["abc", "12.5", ""])
def test_num_carparks_rejects_non_integer(tmp_path, monkeypatch, value):
    with pytest.raises(SystemExit):
        _reload_camera(monkeypatch, tmp_path, value)


# --- concern 3: readiness --------------------------------------------------

def test_ready_503_when_no_images(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    mod = _reload_camera(monkeypatch, empty, 12)
    with TestClient(mod.app) as client:
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200  # live even so


def test_ready_200_when_images_present(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0A")
    mod = _reload_camera(monkeypatch, tmp_path, 12)
    with TestClient(mod.app) as client:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


# --- concern 2: camera reuses/generates request id -------------------------

def test_camera_reuses_incoming_request_id(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0A")
    mod = _reload_camera(monkeypatch, tmp_path, 12)
    with TestClient(mod.app) as client:
        resp = client.get("/health", headers={"X-Request-ID": "cam-trace-9"})
    assert resp.headers["x-request-id"] == "cam-trace-9"


def test_camera_generates_request_id_when_absent(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0A")
    mod = _reload_camera(monkeypatch, tmp_path, 12)
    with TestClient(mod.app) as client:
        resp = client.get("/health")
    assert resp.headers.get("x-request-id")


# --- concern 5: structured-log fields --------------------------------------

def test_json_formatter_emits_semantic_fields(tmp_path, monkeypatch):
    mod = _reload_camera(monkeypatch, tmp_path, 12)
    record = logging.LogRecord(
        name="smartpark.camera", level=logging.INFO, pathname=__file__,
        lineno=1, msg="camera simulator ready", args=(), exc_info=None,
    )
    record.images_dir = str(tmp_path)
    record.num_carparks = 12
    record.image_count = 3
    record.request_id = "rid-1"
    line = mod._JsonFormatter().format(record)

    payload = json.loads(line)  # must be valid JSON
    assert payload["images_dir"] == str(tmp_path)
    assert payload["num_carparks"] == 12
    assert payload["image_count"] == 3
    assert payload["request_id"] == "rid-1"
    assert payload["severity"] == "INFO"
    assert payload["service"] == mod.SERVICE_NAME
    assert "timestamp" in payload


# --- concern 8: car-park id parity across services -------------------------

def test_camera_and_registry_ids_are_identical(tmp_path, monkeypatch):
    mod = _reload_camera(monkeypatch, tmp_path, 12)
    from app.services.carpark_registry import carpark_id as registry_id

    for i in range(1, 100):
        assert mod._carpark_id(i) == registry_id(i)
