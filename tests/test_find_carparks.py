"""Tests for GET /api/find-carparks."""

from __future__ import annotations

import re

from conftest import FakeCameraClient, build_client


def test_queries_2n_and_returns_top_n(ctx):
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3})
    assert resp.status_code == 200
    body = resp.json()

    assert body["uuid"] == "u1"
    # COREAPI1 documented top-level fields (§4.1 example output).
    assert body["status"] == "success"
    assert body["msg"] == "success"
    assert body["requested_n"] == 3
    assert body["speed_inference"].endswith(" ms")
    assert body["queried"] == 6  # exactly 2 * n distinct car parks
    assert body["returned"] == 3
    assert len(body["results"]) == 3
    # Inference ran once per queried car park.
    assert ctx.inference.calls == 6


def test_results_ranked_by_available_spaces_desc(ctx):
    body = ctx.client.get(
        "/api/find-carparks", params={"uuid": "u1", "n": 4}
    ).json()
    spaces = [r["available_spaces"] for r in body["results"]]
    assert spaces == sorted(spaces, reverse=True)
    # Each result carries exactly the COREAPI1-documented fields (+ optional name).
    for r in body["results"]:
        assert {"carpark_id", "available_spaces", "confidence_score"} <= r.keys()
        assert "name" in r


def test_records_statuses_and_uuid(ctx):
    ctx.client.get("/api/find-carparks", params={"uuid": "seen-me", "n": 3})

    statuses = ctx.client.get("/api/operations/statuses").json()
    assert statuses["count"] == 6  # 2 * n distinct car parks persisted

    recent = ctx.client.get("/api/operations/recent-uuids").json()
    assert recent["uuids"] == ["seen-me"]
    assert recent["window_seconds"] == 30


def test_carpark_ids_use_the_spec_cbd_format(ctx):
    body = ctx.client.get(
        "/api/find-carparks", params={"uuid": "u1", "n": 3}
    ).json()
    for result in body["results"]:
        # §4.1 example output: "CBD_001", "CBD_042", ...
        assert re.fullmatch(r"CBD_\d{3}", result["carpark_id"]), result["carpark_id"]
        assert result["name"]  # human-readable street name, e.g. "Market Street East"


def test_caps_query_at_catalogue_and_still_returns_ranked_results():
    # 10 car parks; n=6 would need 12 distinct. Query all 10, return the top 6.
    ctx = build_client(num_carparks=10)
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 6})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["requested_n"] == 6
    assert body["queried"] == 10
    assert body["returned"] == 6
    assert len(body["results"]) == 6
    assert ctx.inference.calls == 10


# --- §4.3 / Ed #109: large n still returns ranked results -----------------


def test_n_greater_than_100_still_returns_200(ctx):
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 1000})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["requested_n"] == 1000
    assert body["queried"] == 24  # all configured parks, not 2000
    assert body["returned"] == 24
    assert len(body["results"]) == 24
    assert ctx.inference.calls == 24


def test_n_fifty_with_default_catalogue_still_succeeds():
    ctx = build_client(num_carparks=24)
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert body["queried"] == 24
    assert body["returned"] == 24
    assert ctx.inference.calls == 24


def test_max_serviceable_n_still_succeeds():
    # 24 car parks -> n=12 needs exactly 24 distinct car parks.
    ctx = build_client(num_carparks=24)
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 12})
    assert resp.status_code == 200
    assert resp.json()["queried"] == 24


def test_empty_uuid_rejected(ctx):
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "", "n": 3})
    assert resp.status_code == 422


def test_camera_failures_are_isolated_not_fatal():
    # Every camera fails; the request still succeeds with zero results, and
    # every queried car park is recorded with status 'error'.
    ctx = build_client(camera_client=FakeCameraClient(fail_all=True))
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["returned"] == 0
    assert body["results"] == []

    statuses = ctx.client.get("/api/operations/statuses").json()["statuses"]
    assert len(statuses) == 6
    assert all(s["status"] == "error" for s in statuses)


def test_n_must_be_at_least_one(ctx):
    resp = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 0})
    assert resp.status_code == 422  # fails Query(ge=1) validation
