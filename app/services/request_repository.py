"""Request/status repository.

Defines an abstract ``RequestRepository`` interface and a concrete in-memory
implementation. Everything the API persists (latest per-car-park status and the
stream of UUID sightings) goes through this interface, so it can be swapped for
a Firestore-backed implementation later without changing any caller.

All methods are ``async`` for exactly that reason — the in-memory version does
not need to await anything, but a Firestore version will.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections import deque

from ..models.schemas import CarParkStatus


class RequestRepository(abc.ABC):
    """Storage interface for car-park statuses and UUID sightings."""

    @abc.abstractmethod
    async def upsert_status(self, status: CarParkStatus) -> None:
        """Insert or replace the latest status for a car park."""

    async def upsert_statuses(self, statuses: list[CarParkStatus]) -> None:
        """Insert or replace many statuses at once.

        ``find-carparks`` produces ``2*n`` statuses per request, so writing them
        one at a time costs ``2*n`` sequential round trips on a network-backed
        store. Implementations that support batching should override this; the
        default simply loops so a backend only has to implement the single-item
        version.
        """
        for status in statuses:
            await self.upsert_status(status)

    @abc.abstractmethod
    async def list_statuses(self) -> list[CarParkStatus]:
        """Return the latest status for every car park seen so far."""

    @abc.abstractmethod
    async def get_status(self, carpark_id: str) -> CarParkStatus | None:
        """Return the latest status for one car park, or None."""

    @abc.abstractmethod
    async def record_uuid(self, uuid: str, at: float | None = None) -> None:
        """Record that ``uuid`` was seen at ``at`` (defaults to now)."""

    @abc.abstractmethod
    async def recent_uuids(
        self, window_seconds: int, now: float | None = None
    ) -> list[str]:
        """Return distinct UUIDs seen within the last ``window_seconds``."""

    async def close(self) -> None:
        """Release any resources (network clients). No-op by default."""
        return None


class InMemoryRequestRepository(RequestRepository):
    """Process-local repository backed by dicts/deques and an asyncio lock."""

    def __init__(self) -> None:
        self._statuses: dict[str, CarParkStatus] = {}
        # (timestamp, uuid) sightings, kept roughly time-ordered.
        self._uuid_sightings: deque[tuple[float, str]] = deque()
        self._lock = asyncio.Lock()

    async def upsert_status(self, status: CarParkStatus) -> None:
        async with self._lock:
            self._statuses[status.carpark_id] = status

    async def upsert_statuses(self, statuses: list[CarParkStatus]) -> None:
        # Take the lock once for the whole batch instead of once per status.
        async with self._lock:
            for status in statuses:
                self._statuses[status.carpark_id] = status

    async def list_statuses(self) -> list[CarParkStatus]:
        async with self._lock:
            return sorted(self._statuses.values(), key=lambda s: s.carpark_id)

    async def get_status(self, carpark_id: str) -> CarParkStatus | None:
        async with self._lock:
            return self._statuses.get(carpark_id)

    async def record_uuid(self, uuid: str, at: float | None = None) -> None:
        timestamp = time.time() if at is None else at
        async with self._lock:
            self._uuid_sightings.append((timestamp, uuid))

    async def recent_uuids(
        self, window_seconds: int, now: float | None = None
    ) -> list[str]:
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        async with self._lock:
            # Reclaim memory by dropping expired sightings from the left. This
            # relies on append order, which is time order for real traffic.
            while self._uuid_sightings and self._uuid_sightings[0][0] < cutoff:
                self._uuid_sightings.popleft()
            # Filter again while reading: callers may record explicit
            # out-of-order timestamps (tests, replayed logs), which would leave
            # an expired sighting stranded behind a newer one above.
            seen: dict[str, None] = {}
            for timestamp, uuid in self._uuid_sightings:
                if timestamp >= cutoff:
                    seen.setdefault(uuid, None)
            return list(seen.keys())
