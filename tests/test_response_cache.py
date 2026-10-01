"""Tests for the §4.3 per-user response cache."""

from __future__ import annotations

import asyncio

import pytest
from conftest import build_client

from app.services.response_cache import TTLCache


# --- endpoint behaviour ---------------------------------------------------


def test_repeat_request_from_same_user_is_served_from_cache():
    ctx = build_client(cache_ttl=30.0)

    first = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3}).json()
    calls_after_first = ctx.inference.calls
    second = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3}).json()

    assert first["cached"] is False
    assert second["cached"] is True
    # The expensive work (2*n inferences) did not run a second time.
    assert ctx.inference.calls == calls_after_first == 6
    assert second["results"] == first["results"]


def test_cache_is_keyed_per_user_and_per_n():
    ctx = build_client(cache_ttl=30.0)

    ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3})
    assert ctx.inference.calls == 6

    # Different user -> miss.
    body = ctx.client.get("/api/find-carparks", params={"uuid": "u2", "n": 3}).json()
    assert body["cached"] is False
    assert ctx.inference.calls == 12

    # Same user, different n -> miss.
    body = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 2}).json()
    assert body["cached"] is False
    assert ctx.inference.calls == 16


def test_cached_hit_still_counts_towards_recent_users(ctx=None):
    # OPS-API-2 must not under-count users just because they were served a
    # cached payload — a cache hit is still a real request.
    ctx = build_client(cache_ttl=30.0)
    ctx.client.get("/api/find-carparks", params={"uuid": "repeat-user", "n": 2})
    ctx.client.get("/api/find-carparks", params={"uuid": "repeat-user", "n": 2})

    recent = ctx.client.get("/api/operations/recent-uuids").json()
    assert recent["uuids"] == ["repeat-user"]


def test_caching_disabled_by_default_in_tests(ctx):
    ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3})
    body = ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 3}).json()
    assert body["cached"] is False
    assert ctx.inference.calls == 12  # ran twice: no caching


def test_cache_stats_endpoint_reports_hits_and_misses():
    ctx = build_client(cache_ttl=30.0)
    ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 2})
    ctx.client.get("/api/find-carparks", params={"uuid": "u1", "n": 2})

    stats = ctx.client.get("/api/operations/cache-stats").json()
    assert stats["enabled"] is True
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["hit_rate"] == 0.5


# --- TTLCache unit tests --------------------------------------------------


def test_ttl_cache_expires_entries():
    async def scenario():
        cache: TTLCache[str] = TTLCache(ttl_seconds=10.0)
        await cache.set(("k",), "value", now=100.0)
        assert await cache.get(("k",), now=105.0) == "value"
        assert await cache.get(("k",), now=115.0) is None

    asyncio.run(scenario())


def test_ttl_cache_disabled_when_ttl_is_zero():
    async def scenario():
        cache: TTLCache[str] = TTLCache(ttl_seconds=0.0)
        assert cache.enabled is False
        await cache.set(("k",), "value")
        assert await cache.get(("k",)) is None

    asyncio.run(scenario())


def test_ttl_cache_evicts_oldest_beyond_max_entries():
    async def scenario():
        cache: TTLCache[int] = TTLCache(ttl_seconds=100.0, max_entries=3)
        for i in range(5):
            await cache.set((i,), i, now=float(i))
        stats = await cache.stats()
        assert stats["entries"] == 3
        # The two oldest keys were evicted first.
        assert await cache.get((0,), now=5.0) is None
        assert await cache.get((4,), now=5.0) == 4

    asyncio.run(scenario())


@pytest.mark.parametrize("ttl", [0.0, 5.0])
def test_ttl_cache_stats_shape(ttl):
    async def scenario():
        cache: TTLCache[str] = TTLCache(ttl_seconds=ttl)
        stats = await cache.stats()
        assert set(stats) == {
            "enabled", "ttl_seconds", "entries",
            "max_entries", "hits", "misses", "hit_rate",
        }

    asyncio.run(scenario())
