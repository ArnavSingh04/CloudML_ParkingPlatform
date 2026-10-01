"""Tests for CameraClient using httpx's in-memory MockTransport (no network)."""

from __future__ import annotations

import base64

import httpx

from app.models.schemas import CarParkInfo
from app.services.camera_client import CameraClient


def _carpark(i: int) -> CarParkInfo:
    cid = f"CBD_{i:03d}"
    return CarParkInfo(
        id=cid, name=f"Car Park {i}",
        camera_url=f"http://cam/cameras/{cid}/api/takephoto",
    )


def _handler(request: httpx.Request) -> httpx.Response:
    # CBD_002's camera is "down".
    if "CBD_002" in str(request.url):
        return httpx.Response(500, json={"detail": "boom"})
    return httpx.Response(200, json={"image_base64": base64.b64encode(b"img").decode()})


async def test_fetch_many_isolates_failures():
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as ac:
        client = CameraClient(ac)
        outcomes = await client.fetch_many([_carpark(1), _carpark(2), _carpark(3)])

    by_id = {o.carpark.id: o for o in outcomes}
    assert by_id["CBD_001"].ok
    assert by_id["CBD_001"].image_bytes == b"img"
    assert by_id["CBD_003"].ok
    assert not by_id["CBD_002"].ok
    assert by_id["CBD_002"].error is not None


async def test_fetch_photo_decodes_base64():
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as ac:
        client = CameraClient(ac)
        data = await client.fetch_photo("http://cam/cameras/CBD_001/api/takephoto")
    assert data == b"img"
