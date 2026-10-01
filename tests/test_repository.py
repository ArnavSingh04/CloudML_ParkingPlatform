"""Tests for the repository interface and its in-memory implementation."""

from __future__ import annotations

import asyncio

from app.models.schemas import CarParkStatus
from app.services.request_repository import InMemoryRequestRepository


def _status(carpark_id: str, empty: int = 1) -> CarParkStatus:
    return CarParkStatus(
        carpark_id=carpark_id,
        status="ok",
        empty_count=empty,
        last_seen="2026-01-01T00:00:00+00:00",
    )


def test_bulk_upsert_writes_every_status():
    async def scenario():
        repo = InMemoryRequestRepository()
        await repo.upsert_statuses([_status(f"CBD_{i:03d}", i) for i in range(1, 7)])
        statuses = await repo.list_statuses()
        assert [s.carpark_id for s in statuses] == [f"CBD_{i:03d}" for i in range(1, 7)]

    asyncio.run(scenario())


def test_bulk_upsert_replaces_existing_status():
    async def scenario():
        repo = InMemoryRequestRepository()
        await repo.upsert_status(_status("CBD_001", empty=1))
        await repo.upsert_statuses([_status("CBD_001", empty=9)])
        got = await repo.get_status("CBD_001")
        assert got is not None and got.empty_count == 9
        assert len(await repo.list_statuses()) == 1

    asyncio.run(scenario())


def test_bulk_upsert_of_empty_list_is_a_noop():
    async def scenario():
        repo = InMemoryRequestRepository()
        await repo.upsert_statuses([])
        assert await repo.list_statuses() == []

    asyncio.run(scenario())


def test_recent_uuids_within_window():
    async def scenario():
        repo = InMemoryRequestRepository()
        await repo.record_uuid("a", at=100.0)
        await repo.record_uuid("b", at=110.0)
        await repo.record_uuid("a", at=115.0)  # duplicate collapses
        # Whole window: both users, in first-seen order.
        assert await repo.recent_uuids(30, now=120.0) == ["a", "b"]
        # now=145 -> cutoff 115, so only a's second sighting survives.
        assert await repo.recent_uuids(30, now=145.0) == ["a"]
        assert await repo.recent_uuids(30, now=200.0) == []

    asyncio.run(scenario())


def test_recent_uuids_ignores_out_of_order_expired_sightings():
    # An out-of-order timestamp leaves an expired sighting stranded behind a
    # newer one, so popleft-pruning alone would wrongly report it as recent.
    async def scenario():
        repo = InMemoryRequestRepository()
        await repo.record_uuid("fresh", at=200.0)
        await repo.record_uuid("stale", at=100.0)  # recorded late, already old
        assert await repo.recent_uuids(30, now=210.0) == ["fresh"]

    asyncio.run(scenario())
