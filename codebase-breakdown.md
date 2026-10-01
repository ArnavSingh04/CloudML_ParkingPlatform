# SmartPark — Complete Codebase Breakdown

A deep, file-by-file explanation of the FIT3184 Assignment 1 SmartPark project:
what each part is, why it exists, and how it works together.

---

## Table of contents

1. [What SmartPark is](#1-what-smartpark-is)
2. [The big picture (architecture)](#2-the-big-picture-architecture)
3. [End-to-end request flow](#3-end-to-end-request-flow)
4. [Directory tree](#4-directory-tree)
5. [The API service — `app/`](#5-the-api-service--app)
   - [`app/config.py`](#appconfigpy--configuration)
   - [`app/logging_config.py`](#applogging_configpy--structured-json-logging)
   - [`app/middleware.py`](#appmiddlewarepy--per-request-context--access-log)
   - [`app/models/schemas.py`](#appmodelsschemaspy--the-data-contracts)
   - [`app/services/`](#appservices--the-service-layer)
   - [`app/dependencies.py`](#appdependenciespy--dependency-injection-seam)
   - [`app/api/`](#appapi--the-routers)
   - [`app/dashboard.py`](#appdashboardpy--the-monitoring-ui)
   - [`app/main.py`](#appmainpy--assembly--lifespan)
6. [The camera service — `camera_service/`](#6-the-camera-service--camera_service)
7. [The tests — `tests/`](#7-the-tests--tests)
8. [Packaging & ops files](#8-packaging--ops-files)
9. [Cross-cutting design decisions](#9-cross-cutting-design-decisions)
10. [Environment variables reference](#10-environment-variables-reference)
11. [Requirement → implementation map](#11-requirement--implementation-map)

---

## 1. What SmartPark is

SmartPark answers one question: **"Which nearby car parks have the most free
spaces right now?"**

Every car park has a camera pointed at it. A supplied YOLO object-detection
model (`model/model.pt`) looks at a camera photo and draws a box around each
parking space, labelling it **`empty`** or **`occupied`**. SmartPark queries a
random sample of car parks, runs the model on each photo, counts the `empty`
spaces, and returns the best ones.

The model is used **exactly as supplied** — never trained, fine-tuned, or
replaced — and neither the model file nor the images are ever copied into a
Docker image (they are mounted at runtime).

---

## 2. The big picture (architecture)

There are **two separate services**, each independently deployable:

```
                    ┌──────────────────────────────────────────┐
   HTTP client  ───▶│  API service  (app/)          port 8000   │
   (curl, browser)  │                                           │
                    │   • /api/find-carparks                    │
                    │   • /api/annotate-carpark                 │
                    │   • /api/operations/*  (monitoring)       │
                    │   • /  (HTML dashboard)                   │
                    │                                           │
                    │   ┌───────────────┐   ┌────────────────┐  │
                    │   │ YOLO model.pt │   │ httpx AsyncClient│ │
                    │   │ (loaded once) │   │ (reused pool)   │  │
                    │   └───────────────┘   └───────┬────────┘  │
                    └───────────────────────────────┼──────────┘
                                                     │ HTTP (concurrent)
                                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  Camera service (camera_service/) 8001    │
                    │                                           │
                    │   • /cameras/{id}/api/takephoto           │
                    │     -> returns a random supplied JPEG     │
                    │        as base64                          │
                    │   (no ML dependencies -> small image)     │
                    └───────────────────────────────────────────┘
                                     │ reads
                                     ▼
                              ./images/*.jpg  (mounted)
```

**Why two services?** It mirrors reality: cameras are physically separate
devices from the analytics backend. Splitting them means:

- the camera image ships **without** torch/ultralytics (it's tiny),
- the two can scale and be tested independently,
- the API talks to cameras over HTTP exactly as it would to real hardware.

They only agree on one thing: the car-park **id scheme** (`CBD_001` …
`CBD_0NN`), so no shared database or state is needed.

---

## 3. End-to-end request flow

### `GET /api/find-carparks?uuid=<caller>&n=<count>`

```
1. Middleware        mint request_id, pull `uuid` into logging context, start timer
        │
2. Registry          pick EXACTLY 2*n distinct random car parks   (400 if 2*n > NUM_CARPARKS)
        │
3. Camera client     fetch all 2*n photos CONCURRENTLY (asyncio.gather, one httpx client)
        │            → a failed camera is captured, not raised
        │
4. Inference         for each photo: asyncio.to_thread(model.predict) under Semaphore(1)
        │            → count "empty", mean confidence, inference ms
        │            → a failed prediction is captured, not raised
        │
5. Repository        upsert every car park's status; record the uuid sighting
        │
6. Rank & return     sort OK car parks by (empty_count, confidence) desc, take top n
        │
7. Middleware        log one JSON line: request_id, uuid, endpoint, latency, status
```

The response looks like:

```json
{
  "uuid": "me", "n": 3, "queried": 6, "returned": 3,
  "generated_at": "2026-09-08T06:46:38+00:00",
  "results": [
    { "carpark_id": "CBD_011", "empty_count": 36, "occupied_count": 4,
      "total_spaces": 40, "confidence_score": 0.9338, "speed_inference": 161.383 }
  ]
}
```

### `GET /api/annotate-carpark?carpark_id=<id>&uuid=<caller>`

Same idea but for **one** car park, and it also returns the model's *annotated*
image (boxes drawn on the photo) as a base64 JPEG. Camera failure → `502`,
inference failure → `500`, unknown car park → `404`.

---

## 4. Directory tree

```
release/
├── app/                         # THE API SERVICE
│   ├── __init__.py              # package marker + __version__
│   ├── main.py                  # app factory + lifespan (loads model once)
│   ├── config.py                # env-driven settings (pydantic-settings)
│   ├── logging_config.py        # JSON log formatter + request contextvars
│   ├── middleware.py            # per-request id / access logging (pure ASGI)
│   ├── dependencies.py          # DI providers (the test seam)
│   ├── dashboard.py             # the monitoring HTML page (as a string)
│   ├── api/
│   │   ├── core.py              # /api/find-carparks, /api/annotate-carpark
│   │   └── operations.py        # statuses, recent-uuids, health, dashboard
│   ├── services/
│   │   ├── inference.py         # YOLO wrapper: to_thread + Semaphore(1)
│   │   ├── camera_client.py     # reused httpx.AsyncClient, concurrent fetch
│   │   ├── carpark_registry.py  # car parks -> camera URLs
│   │   └── request_repository.py# interface + in-memory impl (Firestore-ready)
│   └── models/
│       └── schemas.py           # every request/response/record shape
│
├── camera_service/              # THE CAMERA SIMULATOR (standalone, no ML)
│   └── main.py                  # /cameras/{id}/api/takephoto
│
├── tests/                       # pytest — YOLO + camera HTTP fully mocked
│   ├── conftest.py              # fakes + TestClient factory
│   ├── test_find_carparks.py
│   ├── test_annotate.py
│   ├── test_operations.py
│   ├── test_inference_parsing.py
│   ├── test_camera_client.py
│   └── test_camera_service.py
│
├── Dockerfile                   # API image (installs ML deps)
├── Dockerfile.camera            # camera image (tiny, no ML)
├── docker-compose.yml           # runs both; mounts model + images read-only
├── .dockerignore                # keeps model/images OUT of images
├── requirements.txt             # API runtime deps (pinned)
├── requirements-camera.txt      # camera runtime deps (pinned, minimal)
├── requirements-dev.txt         # test tooling (pinned)
├── .env.example                 # documents every environment variable
├── pytest.ini                   # pytest config (pythonpath, asyncio mode)
└── README.md                    # setup + curl examples
```

Supplied assets that are **not** part of the code and are mounted at runtime:
`model/` (weights), `images/` (1,010 JPEGs), `main.py`, `main_new.ipynb`.

---

## 5. The API service — `app/`

### `app/config.py` — configuration

**Purpose:** one typed, validated place for all settings, sourced from
environment variables.

- Uses **`pydantic-settings`** `BaseSettings`. Each field maps to an
  UPPER_SNAKE_CASE env var of the same name (`model_path` ← `MODEL_PATH`).
- `protected_namespaces=()` silences pydantic's warning about the `model_`
  prefix (we legitimately have `model_path`).
- **Validation happens at load:** `num_carparks` must be 10–99, and
  `model_max_concurrency` must be ≥ 1 — bad config fails fast instead of causing
  a confusing error later.
- `get_settings()` is wrapped in `@lru_cache` so the settings object is built
  **once per process** and reused everywhere.

Key fields: `model_path`, `confidence_threshold`, `model_max_concurrency`,
`num_carparks`, `camera_base_url`, `http_timeout_seconds`, `uuid_window_seconds`,
`service_name`, `log_level`.

### `app/logging_config.py` — structured JSON logging

**Purpose:** make every log line a single JSON object so it can be shipped to a
log aggregator (Cloud Logging, ELK, …) and queried without regex.

- **`JsonFormatter`** turns a standard `LogRecord` into JSON with
  `timestamp, severity, service, logger, message`, plus anything passed via
  `extra={...}`, plus exception text when present.
- **Context variables** (`request_id_ctx`, `uuid_ctx`, `endpoint_ctx`) are
  Python `contextvars`. The middleware sets them at the start of each request;
  the formatter reads them, so **any** log emitted while handling a request is
  automatically tagged with the request id / uuid / endpoint — even a log deep
  inside the camera client.
- **`configure_logging()`** installs the formatter on the root logger and routes
  uvicorn's own loggers through it, so *everything* is uniform JSON.

This is the machinery behind assignment requirement 14 (structured JSON logs
containing timestamp, severity, service, request id, uuid, endpoint, latency,
status code).

### `app/middleware.py` — per-request context + access log

**Purpose:** wrap every HTTP request to (a) give it an id, (b) populate the
logging context, and (c) emit exactly one access-log line with latency + status.

- It is written as **pure ASGI middleware** (a class with
  `async __call__(scope, receive, send)`), **not** Starlette's
  `BaseHTTPMiddleware`. This matters: `BaseHTTPMiddleware` runs the endpoint in a
  *separate task*, which breaks `contextvars` propagation. Pure ASGI runs in the
  same task, so the contextvars we set here are visible inside the route.
- It mints `request_id = uuid4().hex`, parses the `uuid` query param, sets the
  three contextvars, and starts a `perf_counter` timer.
- A `send_wrapper` intercepts the response start to capture the **status code**
  and inject the **`X-Request-ID`** response header.
- On completion it logs `"request completed"` with `method, status_code,
  latency_ms`; on an unhandled exception it logs `"request failed"` and re-raises
  (so Starlette's error handling still produces a 500). A `finally` block resets
  the contextvars so they never leak between requests.

### `app/models/schemas.py` — the data contracts

**Purpose:** every shape that crosses the wire or is persisted, in one file, as
Pydantic models. Centralising them makes the API surface easy to review and lets
the repository be swapped without touching callers.

- `CarParkInfo` — static description of a car park (`id`, `name`, `camera_url`).
- `CarParkResult` — one row in a find-carparks response (`empty_count`,
  `occupied_count`, `total_spaces`, `confidence_score`, `speed_inference`).
- `FindCarParksResponse` — the find-carparks body (`uuid`, `n`, `queried`,
  `returned`, `generated_at`, `results`).
- `AnnotateResponse` — the annotate body (counts + `image_base64`).
- `CarParkStatus` — the **persisted** record: everything in a result plus
  `status` (`"ok"`/`"error"`), `last_uuid`, `last_seen`, `detail`. Has a
  `to_result()` helper that projects it onto `CarParkResult`.
- `StatusesResponse`, `RecentUuidsResponse`, `HealthResponse` — operational
  bodies.

### `app/services/` — the service layer

This is where the real work lives. Each service is a small, single-purpose class
with no FastAPI knowledge, which makes them easy to unit-test.

#### `services/carpark_registry.py`

**Purpose:** the catalogue of car parks and their camera URLs.

- `carpark_id(i)` → `"CBD_001"` … the canonical, zero-padded id used by
  **both** services.
- `CarParkRegistry(num_carparks, camera_base_url)` builds a dict of
  `CarParkInfo`, where each camera URL is
  `{base}/cameras/{id}/api/takephoto`.
- `sample(k)` returns **k distinct** random car parks (via `random.sample`) and
  raises `ValueError` if `k` exceeds the number configured — that's what turns
  "`2*n` too big" into a clean `400`.

#### `services/camera_client.py`

**Purpose:** fetch camera photos over HTTP, concurrently, resiliently.

- Wraps a **single shared `httpx.AsyncClient`** (created once in the lifespan)
  so the connection pool stays warm — no TCP/TLS setup per photo.
- `fetch_photo(url)` GETs the camera endpoint, raises on HTTP errors, and
  base64-decodes the `image_base64` field into raw bytes.
- `fetch_many(carparks)` runs all fetches with `asyncio.gather`. Each is wrapped
  in try/except and returns a **`FetchOutcome`** (`carpark`, `image_bytes`,
  `error`), so **one broken camera never fails the whole batch** — it just comes
  back with `ok == False`.

#### `services/inference.py`

**Purpose:** own the shared YOLO model and mediate *all* access to it. This is
the heart of the concurrency rules.

- `InferenceService(model, max_concurrency, confidence)` holds the model and an
  **`asyncio.Semaphore(1)`**.
- `InferenceService.load(model_path, …)` lazily imports `ultralytics` (so the
  module can be imported for tests without torch installed) and loads the model.
- `infer(image_bytes, annotate=False)`:
  ```python
  async with self._semaphore:                 # only 1 prediction at a time
      return await asyncio.to_thread(self._run, image_bytes, annotate)
  ```
  → the CPU-bound prediction runs **off the event loop** in a worker thread, and
  the semaphore guarantees the single shared model is never touched
  concurrently.
- `_run` = `_decode` (bytes → PIL image) then `model.predict` then `_parse`.
- **`_parse(result, annotate)`** is the pure counting logic (kept free of PIL /
  torch so it can be unit-tested with a lightweight fake):
  - iterates `result.boxes`, resolves each box's class **name** via
    `result.names[int(box.cls[0])]`,
  - counts a detection as empty when the name is exactly `"empty"`,
  - `confidence_score` = mean confidence of empty boxes, else `0.0`,
  - `speed_inference` = `result.speed["inference"]` (milliseconds),
  - returns an `InferenceResult` dataclass.
- `_render_annotated` (used only when `annotate=True`) turns `result.plot()`
  (a BGR numpy array) into a JPEG via Pillow.

> Matching by class **name** rather than index makes the code robust regardless
> of how the model orders its classes. (Verified against the real model:
> `names = {0: 'empty', 1: 'occupied'}`.)

#### `services/request_repository.py`

**Purpose:** store what the operational views need, behind an interface that can
later become Firestore.

- **`RequestRepository`** (abstract base class) defines the contract:
  `upsert_status`, `list_statuses`, `get_status`, `record_uuid`,
  `recent_uuids`. Every method is `async` — the in-memory version doesn't need
  it, but a Firestore version will, and callers shouldn't have to change.
- **`InMemoryRequestRepository`** implements it with:
  - a `dict` of latest `CarParkStatus` per car park,
  - a `deque` of `(timestamp, uuid)` sightings,
  - an `asyncio.Lock` guarding both.
  `recent_uuids(window)` prunes sightings older than the window off the left of
  the deque and returns the distinct uuids that remain.

Swapping to Firestore later means writing one new class that implements
`RequestRepository`; nothing else changes.

### `app/dependencies.py` — dependency-injection seam

**Purpose:** hand services to routes via FastAPI's `Depends`, and give tests a
single place to substitute fakes.

- Each provider (`get_registry`, `get_repository`, `get_camera_client`,
  `get_inference_service`) reads its object from `request.app.state` (populated
  by the lifespan) and raises `503` if it's missing.
- In tests, `app.dependency_overrides[get_camera_client] = lambda: FakeCamera()`
  replaces the real one — so tests never touch the model, the network, or the
  images.

### `app/api/` — the routers

#### `api/core.py` — the business endpoints

- **`find_carparks`** implements the flow in §3. Notable details:
  - `Query(..., ge=1)` on `n` → `422` for `n < 1` automatically.
  - `registry.sample(2*n)` → `400` on over-request.
  - A local `process()` coroutine converts each `FetchOutcome` into a
    `CarParkStatus`, isolating camera/inference failures as `status="error"`.
  - All statuses are persisted, the uuid is recorded, then OK ones are ranked by
    `(empty_count, confidence_score)` descending and the top `n` returned.
- **`annotate_carpark`** fetches one photo, runs inference with `annotate=True`,
  persists the status, and returns the base64 JPEG + counts, with precise error
  codes (`404/502/500`).

#### `api/operations.py` — monitoring endpoints

- `GET /api/operations/statuses` — every car park's latest status (req 15).
- `GET /api/operations/recent-uuids` — distinct uuids in the last
  `UUID_WINDOW_SECONDS` (30) seconds (req 15).
- `GET /api/carparks` — list configured car parks + camera URLs (handy for
  debugging).
- `GET /api/health` (and `/health`) — liveness/readiness; reports whether the
  model finished loading. Reads `app.state.inference` **directly** (not via the
  503-raising dependency) so health still answers when the model failed to load.
- `GET /` and `/dashboard` — serve the HTML dashboard.

### `app/dashboard.py` — the monitoring UI

**Purpose:** a simple, dependency-free HTML page (req 17), stored as a Python
string so it ships inside the image with no static-file mount or template
engine.

- Plain vanilla JS, **no external CDN/assets** (works offline / in Docker).
- Every 3 seconds it fetches `/api/operations/statuses`,
  `/api/operations/recent-uuids`, and `/api/health`, and renders a status table
  (with a confidence bar per car park), a recent-UUID panel, and a model-loaded
  badge.

### `app/main.py` — assembly & lifespan

**Purpose:** build the FastAPI app and manage startup/shutdown.

- **`lifespan`** (an async context manager) is where the singletons are created
  **once**:
  1. `configure_logging(...)`,
  2. build the `CarParkRegistry` and `InMemoryRequestRepository`,
  3. create the **one** `httpx.AsyncClient` and wrap it in `CameraClient`,
  4. **load the YOLO model once** via `InferenceService.load(...)` — wrapped in
     try/except so a missing model logs an error and starts the app *degraded*
     (inference routes then return 503) rather than crash-looping,
  5. stash everything on `app.state`,
  6. on shutdown, `await http_client.aclose()`.
- **`create_app()`** builds the app, adds `RequestContextMiddleware`, and
  includes the two routers. `app = create_app()` at module scope is what
  `uvicorn app.main:app` imports.

---

## 6. The camera service — `camera_service/`

`camera_service/main.py` is a **single, standalone file** with **no dependency
on `app/`** and **no ML libraries** — that's why `Dockerfile.camera` produces a
tiny image.

What it does:

- Reads config from env directly (`IMAGES_DIR`, `NUM_CARPARKS`, `SERVICE_NAME`,
  `LOG_LEVEL`) — no pydantic-settings needed.
- Has its **own compact JSON logger** and a minimal pure-ASGI access-log
  middleware (a little duplication is the price of being fully standalone).
- In its lifespan it caches the list of valid car-park ids and the list of image
  files on disk.
- Endpoints:
  - `GET /cameras/{carpark_id}/api/takephoto` — validates the id against the
    configured set (`404` if unknown), picks a **random** supplied JPEG, and
    returns `{carpark_id, filename, content_type, image_base64}`. This is the
    "camera simulator endpoint for every configured car park, ending in
    `/api/takephoto`" (req 10).
  - `GET /cameras` — lists the takephoto endpoint for every car park.
  - `GET /health` — reports car-park count and number of images found.

The `CBD_0NN` id format here is deliberately identical to
`app/services/carpark_registry.py::carpark_id`, which is the only contract the
two services share.

---

## 7. The tests — `tests/`

Run with `pytest` (config in `pytest.ini`). They mock **YOLO inference and
camera HTTP entirely** (req 18), so they need no model, no images, and no
network.

- **`conftest.py`** — the test harness:
  - `FakeCameraClient` — returns image bytes that *encode* the desired empty
    count (or simulates failures), so results are deterministic.
  - `FakeInferenceService` — decodes those bytes into an `InferenceResult` with
    no model and no threads.
  - `build_client(...)` — creates the app and **overrides all four dependency
    providers** with fakes, then returns a `TestClient`. Crucially it uses
    `TestClient(app)` **without** the `with` context manager, so the app lifespan
    (which would load the real model) never runs.
- **`test_find_carparks.py`** — verifies `queried == 2*n`, `returned == n`,
  ranking order, that statuses + uuid are recorded, the `400` on over-request,
  the `422` on `n < 1`, and that an all-cameras-down request still returns `200`
  with every car park marked `error`.
- **`test_annotate.py`** — verifies the base64 image round-trips, status/uuid are
  recorded, `404` for unknown car park, `502` on camera failure.
- **`test_operations.py`** — health, empty-initial statuses/uuids, car-park
  listing, dashboard HTML, and the `X-Request-ID` header.
- **`test_inference_parsing.py`** — feeds a lightweight fake `Results` object to
  `InferenceService._parse` to prove the empty/occupied counting, mean
  confidence, and ms extraction; also drives the async `infer()` path with a
  fake model.
- **`test_camera_client.py`** — uses httpx's in-memory `MockTransport` to prove
  `fetch_many` isolates a failing camera and `fetch_photo` decodes base64.
- **`test_camera_service.py`** — points the simulator at a temp image dir and
  checks takephoto, the 404 for out-of-range ids, the `/cameras` listing, and
  health.

---

## 8. Packaging & ops files

- **`Dockerfile`** (API image): `python:3.12-slim`, installs the `libgl1` /
  `libglib2.0-0` system libs that ultralytics/opencv need, `pip install -r
  requirements.txt`, then **`COPY app ./app` only** — never the model or images.
  Defaults `MODEL_PATH=/models/model.pt`.
- **`Dockerfile.camera`** (camera image): `python:3.12-slim`, installs the tiny
  `requirements-camera.txt`, then `COPY camera_service` only. Defaults
  `IMAGES_DIR=/images`.
- **`.dockerignore`** — excludes `model/`, `images/`, `sample/`, `result/`,
  `.venv/`, tests, notebooks, etc. This is a **second line of defence** ensuring
  the model and images never end up in an image (req 3).
- **`docker-compose.yml`** — runs both services, **bind-mounts** `./model` →
  `/models` and `./images` → `/images` **read-only**, wires
  `CAMERA_BASE_URL=http://camera:8001`, and keeps `NUM_CARPARKS` in sync via a
  `${NUM_CARPARKS:-24}` default.
- **`requirements*.txt`** — all versions **pinned** (req 19). Split three ways:
  full API runtime, minimal camera runtime, and dev/test tooling.
- **`.env.example`** — copy to `.env`; documents every variable.
- **`pytest.ini`** — `pythonpath = .` (so `import app` works),
  `asyncio_mode = auto` (so `async def` tests just run).

---

## 9. Cross-cutting design decisions

| Decision | Why |
|---|---|
| **Fetch cameras concurrently, one shared `AsyncClient`** | Camera calls are I/O-bound; a request is then bound by the *slowest* camera, not the sum. A reused client keeps connections warm. |
| **`asyncio.to_thread` + `Semaphore(1)` for inference** | `model.predict` is CPU-bound and not async-safe. Offloading keeps the event loop responsive; the semaphore guarantees the single shared model is used by one caller at a time (req 13). |
| **Model loaded once in the lifespan** | Loading weights is expensive; do it at startup and share via `app.state` (req 1). |
| **Repository behind an abstract interface** | Lets the in-memory store be replaced by Firestore later with zero changes to callers (req 16). |
| **Dependency-injection providers** | Single seam for tests to inject fakes; routes stay ignorant of construction. |
| **Pure-ASGI middleware + contextvars** | Reliable propagation of request id / uuid into every log line, which `BaseHTTPMiddleware` cannot guarantee. |
| **Match detections by class *name*** | Robust to class-index ordering in the model. |
| **Per-item failure isolation** | One bad camera or prediction degrades gracefully instead of failing the whole request. |
| **Two independent services / images** | Camera image needs no ML deps; realistic separation of camera hardware from analytics. |

---

## 10. Environment variables reference

**API service** (`app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `./model/model.pt` | Path to supplied weights (mounted at runtime). |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence for `model.predict()`. |
| `MODEL_MAX_CONCURRENCY` | `1` | Concurrent predictions on the shared model (must be 1). |
| `NUM_CARPARKS` | `24` | Number of car parks (10–99). Keep equal on both services. |
| `CAMERA_BASE_URL` | `http://localhost:8001` | Base URL of the camera service. |
| `HTTP_TIMEOUT_SECONDS` | `10.0` | Per-camera HTTP timeout. |
| `UUID_WINDOW_SECONDS` | `30` | Window for the recent-UUIDs view. |
| `SERVICE_NAME` | `smartpark-api` | `service` field in logs. |
| `LOG_LEVEL` | `INFO` | Log verbosity. |

**Camera service** (`camera_service/main.py`):

| Variable | Default | Meaning |
|---|---|---|
| `IMAGES_DIR` | `./images` | Directory of supplied JPEGs (mounted at runtime). |
| `NUM_CARPARKS` | `24` | Camera endpoints to expose (match the API). |
| `SERVICE_NAME` | `smartpark-camera` | `service` field in logs. |
| `LOG_LEVEL` | `INFO` | Log verbosity. |

---

## 11. Requirement → implementation map

| # | Requirement | Where |
|---|---|---|
| 1 | Load YOLO once via lifespan | `app/main.py` `lifespan` → `InferenceService.load` |
| 2 | `MODEL_PATH` from env | `app/config.py` |
| 3 | Model/images not in any image | `Dockerfile*` (explicit `COPY`) + `.dockerignore` + compose volumes |
| 4 | `GET /api/find-carparks` with `uuid`, `n` | `app/api/core.py::find_carparks` |
| 5 | Query exactly `2*n` distinct car parks | `registry.sample(2*n)` |
| 6 | Count class name exactly `"empty"` | `inference.py::_parse` |
| 7 | `confidence_score` = mean empty conf, else 0.0 | `inference.py::_parse` |
| 8 | `speed_inference` in ms | `result.speed["inference"]` in `_parse` |
| 9 | `GET /api/annotate-carpark` base64 image | `app/api/core.py::annotate_carpark` |
| 10 | Camera endpoint per car park ending `/api/takephoto` | `camera_service/main.py` |
| 11 | 10–99 car parks via `NUM_CARPARKS` | `config.py` validator + registry |
| 12 | Concurrent fetch via reused `httpx.AsyncClient` | `camera_client.py` + lifespan |
| 13 | Never `predict` in async route; `to_thread` + `Semaphore(1)` | `inference.py::infer` |
| 14 | Structured JSON logs (all fields) | `logging_config.py` + `middleware.py` |
| 15 | Operational endpoints (statuses, recent uuids/30s) | `app/api/operations.py` + repository |
| 16 | In-memory repo behind an interface | `request_repository.py` |
| 17 | HTML monitoring dashboard | `app/dashboard.py` + `/` route |
| 18 | Tests mock YOLO + camera HTTP | `tests/` |
| 19 | Pinned dependency versions | `requirements*.txt` |
| 20 | Setup + curl examples in README | `README.md` |

---

*Kubernetes is intentionally not included in this version, as specified.*
