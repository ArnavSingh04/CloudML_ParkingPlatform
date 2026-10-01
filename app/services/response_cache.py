"""Short-TTL response cache for repeated requests from the same user.

Implements the assignment's performance-optimisation hint (§4.3): *"repeated
requests from the same user can be cached"*. ``find-carparks`` is by far the
most expensive endpoint — one call runs ``2*n`` camera fetches and ``2*n``
serialised YOLO predictions — so a user polling it in a tight loop otherwise
re-does all of that work for an answer that cannot meaningfully have changed.

Design notes:

* Keyed by ``(uuid, n)``. Two different users, or the same user asking for a
  different ``n``, never share an entry.
* The TTL is deliberately short (default 5s). Availability is real-time data;
  a long TTL would trade correctness for throughput. ``CACHE_TTL_SECONDS=0``
  disables the cache entirely, which is what you want when benchmarking raw
  inference throughput with Locust.
* Bounded by ``cache_max_entries`` so a flood of distinct uuids cannot grow the
  cache without limit — expired entries are dropped first, then the oldest.
* Guarded by an ``asyncio.Lock`` rather than a ``threading.Lock``: the lock is
  only ever held by coroutines on the event loop, and blocking that thread is
  exactly what the rest of this codebase works to avoid.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """An async-safe, bounded, time-to-live cache."""

    def __init__(self, ttl_seconds: float, max_entries: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        # OrderedDict preserves insertion order, which gives us cheap
        # oldest-first eviction via popitem(last=False).
        self._entries: OrderedDict[tuple, tuple[float, T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """A zero (or negative) TTL disables the cache completely."""
        return self._ttl > 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    async def get(self, key: tuple, now: float | None = None) -> T | None:
        """Return the cached value for ``key``, or None when absent/expired."""
        if not self.enabled:
            return None
        current = time.monotonic() if now is None else now
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= current:
                # Lazily evict on read rather than running a sweeper task.
                del self._entries[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    async def set(self, key: tuple, value: T, now: float | None = None) -> None:
        """Store ``value`` under ``key`` for the configured TTL."""
        if not self.enabled:
            return
        current = time.monotonic() if now is None else now
        async with self._lock:
            self._entries[key] = (current + self._ttl, value)
            self._entries.move_to_end(key)
            self._evict_locked(current)

    def _evict_locked(self, now: float) -> None:
        """Drop expired entries, then the oldest, until within the size cap.

        Caller must hold ``self._lock``.
        """
        expired = [k for k, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            del self._entries[key]
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    async def stats(self) -> dict[str, float | int | bool]:
        """Snapshot of cache effectiveness, surfaced on the operations API."""
        async with self._lock:
            size = len(self._entries)
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "ttl_seconds": self._ttl,
            "entries": size,
            "max_entries": self._max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
