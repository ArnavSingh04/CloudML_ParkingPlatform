"""Firestore-backed request repository.

A shared, cross-pod implementation of ``RequestRepository`` so that every API
pod reads and writes the *same* store. Without this, each pod keeps its own
in-memory statuses/UUID sightings and the "unique users in the last 30s" view
is only ever correct for whichever pod happens to serve the operational request.

Two collections are used (created lazily by the first write — nothing to set up
by hand):

``carpark_status/{carpark_id}``
    One document per car park holding its latest :class:`CarParkStatus`.

``request_logs/{auto_id}``
    One document per UUID sighting. The ``ts`` field (epoch seconds) is what the
    30-second window query filters on; ``timestamp``/``endpoint``/``request_id``
    are carried for traceability and mirror the structured access logs.

``request_logs`` is append-only and would otherwise grow without bound, so
``recent_uuids`` opportunistically deletes a bounded batch of documents that
fell out of the window (see ``_purge_expired``). For a production deployment
you would *also* configure a Firestore TTL policy on the ``expires_at`` field
written by :meth:`record_uuid`, which lets Google expire documents server-side
at no query cost:

    gcloud firestore fields ttls update expires_at \\
        --collection-group=request_logs --enable-ttl

Authentication uses Application Default Credentials via
``google.cloud.firestore.AsyncClient`` — no JSON key file is read or shipped in
the image. On Cloud Run / GKE this resolves to the attached service account
(which needs the ``roles/datastore.user`` role).
"""

from __future__ import annotations

import inspect
import time

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from ..logging_config import endpoint_ctx, get_logger, request_id_ctx
from ..models.schemas import CarParkStatus
from .request_repository import RequestRepository

logger = get_logger("smartpark.firestore")

# Collection names (see module docstring).
STATUS_COLLECTION = "carpark_status"
REQUEST_LOGS_COLLECTION = "request_logs"

# Firestore caps a single batched write at 500 operations.
_MAX_BATCH_SIZE = 500
# How many expired request_logs documents one opportunistic purge may delete,
# and how often a purge may run. Both are deliberately small: purging is a
# background courtesy, not something an operator request should wait on.
_PURGE_BATCH_SIZE = 200
_PURGE_INTERVAL_SECONDS = 60.0
# Retain sightings well past the 30s window so a longer UUID_WINDOW_SECONDS, or
# a clock skew between pods, can never delete a document that is still needed.
_LOG_RETENTION_SECONDS = 3600.0


class FirestoreRequestRepository(RequestRepository):
    """Shared repository backed by Google Cloud Firestore (Native mode)."""

    def __init__(
        self,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        # AsyncClient integrates with the running event loop, so repository
        # methods stay non-blocking. project=None lets ADC supply the project.
        self._db = firestore.AsyncClient(project=project, database=database)
        self._last_purge = 0.0

    async def upsert_status(self, status: CarParkStatus) -> None:
        doc = self._db.collection(STATUS_COLLECTION).document(status.carpark_id)
        # set() (without merge) replaces the doc with the latest status.
        await doc.set(status.model_dump())

    async def upsert_statuses(self, statuses: list[CarParkStatus]) -> None:
        """Write many statuses in batches instead of one round trip each.

        find-carparks produces 2*n statuses; the default loop in the base class
        would make 2*n sequential calls. A batched write commits up to 500
        documents per round trip.
        """
        if not statuses:
            return
        collection = self._db.collection(STATUS_COLLECTION)
        for start in range(0, len(statuses), _MAX_BATCH_SIZE):
            chunk = statuses[start : start + _MAX_BATCH_SIZE]
            batch = self._db.batch()
            for status in chunk:
                batch.set(collection.document(status.carpark_id), status.model_dump())
            await batch.commit()

    async def list_statuses(self) -> list[CarParkStatus]:
        statuses: list[CarParkStatus] = []
        async for snapshot in self._db.collection(STATUS_COLLECTION).stream():
            data = snapshot.to_dict()
            if data:
                statuses.append(CarParkStatus(**data))
        return sorted(statuses, key=lambda s: s.carpark_id)

    async def get_status(self, carpark_id: str) -> CarParkStatus | None:
        snapshot = (
            await self._db.collection(STATUS_COLLECTION)
            .document(carpark_id)
            .get()
        )
        if not snapshot.exists:
            return None
        data = snapshot.to_dict()
        return CarParkStatus(**data) if data else None

    async def record_uuid(self, uuid: str, at: float | None = None) -> None:
        timestamp = time.time() if at is None else at
        # request_id / endpoint come from the per-request contextvars set by the
        # access-log middleware; they may be None outside a request scope.
        entry = {
            "uuid": uuid,
            "ts": timestamp,
            "timestamp": _iso(timestamp),
            # Native Firestore timestamp so a server-side TTL policy can expire
            # this document without any client involvement.
            "expires_at": _datetime(timestamp + _LOG_RETENTION_SECONDS),
            "request_id": request_id_ctx.get(),
            "endpoint": endpoint_ctx.get(),
        }
        await self._db.collection(REQUEST_LOGS_COLLECTION).add(entry)

    async def recent_uuids(
        self, window_seconds: int, now: float | None = None
    ) -> list[str]:
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        # Range filter + order on the same field needs no composite index; the
        # single-field index Firestore maintains automatically is sufficient.
        query = (
            self._db.collection(REQUEST_LOGS_COLLECTION)
            .where(filter=FieldFilter("ts", ">=", cutoff))
            .order_by("ts")
        )
        seen: dict[str, None] = {}
        async for snapshot in query.stream():
            data = snapshot.to_dict() or {}
            uuid = data.get("uuid")
            if uuid is not None:
                seen.setdefault(uuid, None)

        await self._purge_expired(current)
        return list(seen.keys())

    async def _purge_expired(self, now: float) -> None:
        """Delete a bounded batch of long-expired request_logs documents.

        Keeps the append-only collection from growing forever when no
        server-side TTL policy is configured. Rate-limited and capped so it
        never turns an operator request into a long-running delete, and
        failures are swallowed — purging is housekeeping, not correctness.
        """
        if now - self._last_purge < _PURGE_INTERVAL_SECONDS:
            return
        self._last_purge = now
        cutoff = now - _LOG_RETENTION_SECONDS
        try:
            query = (
                self._db.collection(REQUEST_LOGS_COLLECTION)
                .where(filter=FieldFilter("ts", "<", cutoff))
                .limit(_PURGE_BATCH_SIZE)
            )
            batch = self._db.batch()
            count = 0
            async for snapshot in query.stream():
                batch.delete(snapshot.reference)
                count += 1
            if count:
                await batch.commit()
                logger.info("purged expired request_logs", extra={"deleted": count})
        except Exception as exc:  # noqa: BLE001 - housekeeping must never fail a read
            logger.warning(
                "request_logs purge failed (non-fatal)", extra={"error": str(exc)}
            )

    async def close(self) -> None:
        # AsyncClient.close() is synchronous in some versions and a coroutine in
        # others; handle both so shutdown never leaks the gRPC channel.
        result = self._db.close()
        if inspect.isawaitable(result):
            await result


def _datetime(epoch_seconds: float):
    """Timezone-aware UTC datetime, stored by Firestore as a native timestamp."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def _iso(epoch_seconds: float) -> str:
    """ISO-8601 UTC string for an epoch timestamp (for human-readable logs)."""
    return _datetime(epoch_seconds).isoformat()
