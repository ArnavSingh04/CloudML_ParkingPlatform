"""Stepped Locust load for the 1/2/4/8 replica benchmark.

Imported by the runner. Stages are cumulative from t=0.
"""

from locust import LoadTestShape

from locustfile import SmartParkUser  # noqa: F401  — Locust discovers HttpUser here


class StepLoad(LoadTestShape):
    """Hold each user count long enough for YOLO requests to complete."""

    stages = [
        {"duration": 70, "users": 5, "spawn_rate": 5},
        {"duration": 140, "users": 10, "spawn_rate": 5},
        {"duration": 210, "users": 20, "spawn_rate": 5},
        {"duration": 280, "users": 40, "spawn_rate": 5},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None
