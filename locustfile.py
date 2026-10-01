"""Locust load generator for SmartPark CORE-API-1 / CORE-API-2 (§5).

Simulates concurrent end users hitting the public API. Each Locust user gets
its own UUID (matching the assignment's "each simulated user has a uuid") and
continuously calls ``/api/find-carparks``. A minority of calls follow up with
``/api/annotate-carpark`` on one of the returned ids — the "user then wants to
see a photo" path from §4.1.

Usage (against a running API — local or the GKE LoadBalancer IP):

    pip install -r requirements-dev.txt

    # Headless, for the 1/2/4/8 replica benchmark:
    locust -f locustfile.py --host http://<API-IP> --headless \\
           -u 20 -r 5 -t 3m --csv reports/pods-1

    # Web UI (opens http://localhost:8089):
    locust -f locustfile.py --host http://localhost:8000

Benchmark notes
---------------
* Set ``CACHE_TTL_SECONDS=0`` on the API for the scaling table. Each Locust
  user has a unique UUID, so the per-user cache would still miss on the first
  call — but wait-time repeats from the *same* user would otherwise be served
  from cache and inflate QPS / deflate latency. After the 1/2/4/8 runs, flip
  the cache on and re-run once to quantify §4.3.
* Scale the Deployment between runs, don't rely on HPA for the table:

      kubectl scale deploy/smartpark-api -n smartpark --replicas=N
      kubectl rollout status deploy/smartpark-api -n smartpark
"""

from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, task


# Small enough that 2*n stays within the default 24-car-park catalogue,
# large enough to exercise ranking. Change here if you want a heavier request.
DEFAULT_N = 3


class SmartParkUser(HttpUser):
    """One simulated end user with a stable UUID for the whole session."""

    # Think-time between tasks. Short enough to generate load, long enough
    # that a single user isn't a tight loop of inference.
    wait_time = between(1.0, 3.0)

    def on_start(self) -> None:
        self.user_id = str(uuid.uuid4())
        self.last_carpark_id: str | None = None

    @task(5)
    def find_carparks(self) -> None:
        with self.client.get(
            "/api/find-carparks",
            params={"uuid": self.user_id, "n": DEFAULT_N},
            name="/api/find-carparks",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("response was not JSON")
                return
            results = body.get("results") or []
            if results:
                self.last_carpark_id = results[0].get("carpark_id")
            response.success()

    @task(1)
    def annotate_carpark(self) -> None:
        carpark_id = self.last_carpark_id
        if carpark_id is None:
            # No prior find-carparks result in this session yet; pick a
            # canonical id so the task still generates traffic.
            carpark_id = f"CBD_{random.randint(1, 24):03d}"
        with self.client.get(
            "/api/annotate-carpark",
            params={"carpark_id": carpark_id, "uuid": self.user_id},
            name="/api/annotate-carpark",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}: {response.text[:200]}")
                return
            response.success()
