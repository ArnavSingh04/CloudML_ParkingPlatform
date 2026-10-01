"""Firestore connectivity smoke test.

Verifies the FirestoreRequestRepository can authenticate (via Application
Default Credentials) and round-trip both collections against your REAL
Firestore database — run it once before/after deploying.

Prereqs:
  * pip install google-cloud-firestore  (already in requirements.txt)
  * ADC available. Locally:  gcloud auth application-default login
    On Cloud Run / GKE this is automatic via the attached service account.

Usage (from the repo root):
  FIRESTORE_PROJECT=<your-project> python scripts/firestore_smoke.py
  # or rely on ADC to supply the project:
  python scripts/firestore_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import time

from app.models.schemas import CarParkStatus
from app.services.firestore_repository import FirestoreRequestRepository


async def main() -> None:
    repo = FirestoreRequestRepository(
        project=os.environ.get("FIRESTORE_PROJECT"),
        database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
    )
    try:
        now = time.time()

        # 1) carpark_status round-trip.
        await repo.upsert_status(
            CarParkStatus(
                carpark_id="CBD_999",
                status="ok",
                empty_count=5,
                occupied_count=3,
                total_spaces=8,
                confidence_score=0.87,
                speed_inference=12.3,
                last_uuid="smoke-user",
                last_seen="smoke-test",
            )
        )
        got = await repo.get_status("CBD_999")
        assert got is not None and got.empty_count == 5, got
        print("carpark_status round-trip OK:", got.carpark_id, got.empty_count)

        # 2) request_logs + 30s window round-trip.
        await repo.record_uuid("smoke-user-A", at=now)
        await repo.record_uuid("smoke-user-B", at=now)
        await repo.record_uuid("smoke-user-A", at=now)  # duplicate -> collapses
        recent = await repo.recent_uuids(window_seconds=30, now=now + 1)
        assert "smoke-user-A" in recent and "smoke-user-B" in recent, recent
        print("recent_uuids (distinct) OK:", sorted(recent))

        # 3) window excludes old sightings.
        old = await repo.recent_uuids(window_seconds=30, now=now + 999)
        print("recent_uuids after window expiry:", old, "(should not include smoke users)")

        print("\nAll Firestore smoke checks passed.")
    finally:
        await repo.close()


if __name__ == "__main__":
    asyncio.run(main())
