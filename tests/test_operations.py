"""Tests for the operational + dashboard endpoints."""

from __future__ import annotations

from conftest import build_client, carpark_number


def test_health_reports_model_and_carparks(ctx):
    body = ctx.client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_carparks"] == 24


def test_ready_returns_200_when_model_loaded(ctx):
    resp = ctx.client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_ready_returns_503_when_model_not_loaded():
    ctx = build_client()
    ctx.app.state.inference = None  # simulate a failed/absent model load
    resp = ctx.client.get("/health/ready")
    assert resp.status_code == 503


def test_live_is_200_even_when_model_not_loaded():
    ctx = build_client()
    ctx.app.state.inference = None
    resp = ctx.client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_statuses_empty_initially(ctx):
    body = ctx.client.get("/api/operations/statuses").json()
    assert body["count"] == 0
    assert body["statuses"] == []


def test_recent_uuids_empty_initially(ctx):
    body = ctx.client.get("/api/operations/recent-uuids").json()
    assert body["count"] == 0
    assert body["uuids"] == []
    assert body["window_seconds"] == 30


def test_list_carparks(ctx):
    body = ctx.client.get("/api/carparks").json()
    assert len(body) == 24
    assert body[0]["id"] == "CBD_001"
    assert body[0]["camera_url"].endswith("/cameras/CBD_001/api/takephoto")


def test_dashboard_served(ctx):
    resp = ctx.client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "SmartPark" in resp.text


def test_request_id_header_present(ctx):
    resp = ctx.client.get("/api/health")
    assert resp.headers.get("x-request-id")


def test_availability_lists_all_carparks_initially_unknown(ctx):
    body = ctx.client.get("/api/operations/availability").json()
    assert body["count"] == 24
    assert len(body["carparks"]) == 24
    first = body["carparks"][0]
    assert first["carpark_id"] == "CBD_001"
    # Never queried yet -> unknown with null counts.
    assert first["status"] == "unknown"
    assert first["available_spots"] is None


def test_availability_reflects_queried_spots(ctx):
    # find-carparks populates statuses; the fake encodes empty=carpark number.
    ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3})
    body = ctx.client.get("/api/operations/availability").json()
    assert body["count"] == 24  # still lists every configured car park
    by_id = {c["carpark_id"]: c for c in body["carparks"]}
    known = [c for c in body["carparks"] if c["status"] == "ok"]
    assert known, "expected at least one queried car park"
    sample = known[0]
    assert sample["available_spots"] == carpark_number(sample["carpark_id"])
    assert sample["last_seen"] is not None


def test_plot_endpoint_returns_png(ctx):
    resp = ctx.client.get("/api/operations/plot.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
