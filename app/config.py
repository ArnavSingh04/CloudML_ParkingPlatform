"""Application configuration.

All configuration is sourced from environment variables (optionally via a
`.env` file) so the same image can be promoted across environments without
code changes. Every field below maps to an UPPER_SNAKE_CASE environment
variable of the same name (e.g. ``model_path`` <- ``MODEL_PATH``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the SmartPark API service."""

    # ``protected_namespaces=()`` silences pydantic's warning about the
    # ``model_`` field prefix (``model_path``), which is meaningful here.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Identity ---------------------------------------------------------
    service_name: str = "smartpark-api"

    # --- Model ------------------------------------------------------------
    # Path to the supplied Ultralytics weights. Never baked into the image;
    # mounted at runtime (see docker-compose.yml).
    model_path: str = "./model/model.pt"
    # Minimum detection confidence forwarded to model.predict().
    confidence_threshold: float = 0.25
    # Concurrency guard for the single shared model. The assignment requires
    # this to be 1 (asyncio.Semaphore(1)); exposed as a setting for clarity.
    model_max_concurrency: int = 1

    # --- Car parks / cameras ---------------------------------------------
    # Number of simulated car parks. Constrained to the assignment's 10..99.
    num_carparks: int = 24
    # Base URL of the camera simulator service. Each car park's camera lives
    # at ``{camera_base_url}/cameras/{carpark_id}/api/takephoto``.
    camera_base_url: str = "http://localhost:8001"
    # Timeout (seconds) for a single camera HTTP request.
    http_timeout_seconds: float = 10.0
    # Best-effort startup check that the camera service was configured with the
    # same NUM_CARPARKS. Logs a warning on mismatch; never blocks startup.
    verify_camera_carpark_count: bool = True

    # --- Operations -------------------------------------------------------
    # Sliding window (seconds) used by the "recent UUIDs" operational view.
    uuid_window_seconds: int = 30

    # --- Caching (§4.3 "repeated requests from the same user can be cached")
    # TTL for cached find-carparks responses, keyed by (uuid, n). A short TTL
    # absorbs bursty repeat polling from one user without serving stale
    # availability. Set to 0 to disable caching entirely (useful when
    # benchmarking raw inference throughput with Locust).
    cache_ttl_seconds: float = 5.0
    # Hard cap on cache entries so a flood of distinct uuids cannot grow the
    # cache without bound. Expired entries are purged first, then the oldest.
    cache_max_entries: int = 1024

    # --- Persistence ------------------------------------------------------
    # Where per-car-park statuses and UUID sightings are stored. "memory" is a
    # process-local store (fine for a single pod / local dev / tests); with
    # multiple API pods each pod would see only its own requests, so the cloud
    # deployment must set REPOSITORY_BACKEND=firestore for a shared store.
    repository_backend: Literal["memory", "firestore"] = "memory"
    # GCP project id for Firestore. Left as None so Application Default
    # Credentials can supply it automatically (Cloud Run / GKE Workload
    # Identity). Set FIRESTORE_PROJECT explicitly only to override.
    firestore_project: str | None = None
    # Firestore database id. "(default)" matches the database created in the
    # console; a named database would go here instead.
    firestore_database: str = "(default)"

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("num_carparks")
    @classmethod
    def _validate_num_carparks(cls, value: int) -> int:
        if not 10 <= value <= 99:
            raise ValueError("NUM_CARPARKS must be between 10 and 99 (inclusive)")
        return value

    @field_validator("model_max_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MODEL_MAX_CONCURRENCY must be >= 1")
        return value

    @field_validator("cache_ttl_seconds")
    @classmethod
    def _validate_cache_ttl(cls, value: float) -> float:
        if value < 0:
            raise ValueError("CACHE_TTL_SECONDS must be >= 0 (0 disables caching)")
        return value

    @property
    def camera_base_url_clean(self) -> str:
        """Camera base URL without a trailing slash."""
        return self.camera_base_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (evaluated once per process)."""
    return Settings()
