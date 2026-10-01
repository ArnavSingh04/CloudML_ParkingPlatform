# SmartPark — FIT3184 Assignment 1

A local, production-quality FastAPI service that finds available car parks. Each
car park has a camera; a supplied YOLO model (`model.pt`) classifies detected
parking spaces as **`empty`** or **`occupied`**. The API queries a random subset
of car parks, runs inference, and returns those with the most availability.

The supplied model is used **as-is** — it is never trained, fine-tuned, or
replaced, and neither the model nor the images are copied into any Docker image.

---

## Architecture

Two independently deployable services:

| Service | Path | Port | Purpose |
|---|---|---|---|
| **API** | `app/` | 8000 | Business + operational endpoints, model inference, dashboard |
| **Camera simulator** | `camera_service/` | 8001 | One `/api/takephoto` endpoint per car park; returns a random supplied JPEG (base64) |

```
client ──▶ API (app)  ──HTTP──▶ Camera simulator (camera_service) ──▶ supplied JPEGs
                │
                └── YOLO model.pt (loaded once, in a worker thread)
```

```
app/
  main.py                     # app factory + lifespan (loads model once)
  config.py                   # env-driven settings (pydantic-settings)
  logging_config.py           # structured JSON logging + request contextvars
  middleware.py               # per-request id / access log (pure ASGI)
  dependencies.py             # DI providers (the test seam)
  dashboard.py                # self-contained monitoring HTML
  api/core.py                 # /api/find-carparks, /api/annotate-carpark
  api/operations.py           # statuses, recent-uuids, health, dashboard
  services/inference.py       # YOLO wrapper: to_thread + Semaphore(1)
  services/camera_client.py   # reused httpx.AsyncClient, concurrent fetch
  services/carpark_registry.py# car parks -> camera URLs
  services/request_repository.py  # interface + in-memory impl (Firestore-ready)
  models/schemas.py           # pydantic request/response/record models
camera_service/main.py        # standalone camera simulator (no ML deps)
tests/                        # pytest; YOLO + camera HTTP fully mocked
```

---

## Local setup (without Docker)

Requires **Python 3.12**.

```bash
# 1. Create a 3.12 virtualenv and install deps
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env               # defaults work for local runs

# 3. Terminal A — start the camera simulator (serves ./images)
IMAGES_DIR=./images NUM_CARPARKS=24 \
  uvicorn camera_service.main:app --port 8001

# 4. Terminal B — start the API (loads ./model/model.pt)
MODEL_PATH=./model/model.pt NUM_CARPARKS=24 CAMERA_BASE_URL=http://localhost:8001 \
  uvicorn app.main:app --port 8000
```

> On Windows PowerShell, set variables with `$env:MODEL_PATH="./model/model.pt"` etc.
> before the `uvicorn` call, or rely on the values in `.env`.

Open the dashboard at <http://localhost:8000/> and interactive docs at
<http://localhost:8000/docs>.

## Run with Docker

```bash
docker compose up --build
# API      -> http://localhost:8000
# Camera   -> http://localhost:8001
```

The model (`./model`) and images (`./images`) are bind-mounted read-only; the
images are lean and contain only application code.

---

## curl examples

```bash
# Health (also reports whether the model finished loading)
curl -s http://localhost:8000/api/health | jq

# List configured car parks + their camera URLs
curl -s http://localhost:8000/api/carparks | jq

# Find the 3 best car parks for a caller (queries 2*3 = 6 at random)
curl -s "http://localhost:8000/api/find-carparks?uuid=abc-123&n=3" | jq

# Annotated detection for one car park (base64 JPEG in the response)
curl -s "http://localhost:8000/api/annotate-carpark?carpark_id=CBD_001&uuid=abc-123" | jq

# Decode the annotated image to a file
curl -s "http://localhost:8000/api/annotate-carpark?carpark_id=CBD_001" \
  | jq -r .image_base64 | base64 -d > annotated.jpg

# Operational: latest status of every car park queried so far
curl -s http://localhost:8000/api/operations/statuses | jq

# Operational: every configured car park + current availability (OPS-API-1)
curl -s http://localhost:8000/api/operations/availability | jq

# Operational: distinct UUIDs seen in the last 30 seconds (OPS-API-2)
curl -s http://localhost:8000/api/operations/recent-uuids | jq

# Operational: response-cache effectiveness (hits / misses / hit rate)
curl -s http://localhost:8000/api/operations/cache-stats | jq

# Hit the camera simulator directly
curl -s http://localhost:8001/cameras/CBD_001/api/takephoto | jq '.carpark_id, .filename'
```

### Car-park identifiers

Car parks are `CBD_001` … `CBD_0NN`, matching the assignment's example output
(§4.1), and each carries a human-readable street name such as
`Market Street East`. Both services derive ids from the same
`CBD_%03d` format independently — see `app/services/carpark_registry.py` and
`camera_service/main.py`, which are kept in sync by a test.

### `find-carparks` response shape (COREAPI1)

```json
{
  "uuid": "abc-123",
  "status": "success",
  "msg": "success",
  "speed_inference": "128.3 ms",
  "requested_n": 3,
  "queried": 6,
  "returned": 3,
  "generated_at": "2026-09-08T12:00:00+00:00",
  "cached": false,
  "results": [
    {
      "carpark_id": "CBD_007",
      "name": "Little Lonsdale Lane",
      "available_spaces": 18,
      "confidence_score": 0.91
    }
  ]
}
```

Top-level `speed_inference` is the **total** model inference time for the
request (a string like `"128.3 ms"`, per the assignment example);
`available_spaces` is the number of detected `empty` spaces (the ranking key);
`confidence_score` is the mean confidence over `empty` detections (0.0 if none).
`queried`/`returned`/`generated_at`/`cached` are additive operational fields.

### Performance optimisations (§4.3)

**Repeated requests are cached.** `find-carparks` responses are cached for
`CACHE_TTL_SECONDS` (default 5s), keyed by `(uuid, n)`, so a user polling in a
loop skips `2*n` camera fetches and `2*n` YOLO predictions. Measured locally:
**1343 ms → 63 ms** on the repeat call. A cache hit still records the UUID
sighting, so OPS-API-2's "users in the last 30s" stays accurate. Set
`CACHE_TTL_SECONDS=0` when benchmarking raw inference throughput with Locust,
and watch `/api/operations/cache-stats` to quantify the effect.

**Large `n` still returns ranked results.** `find-carparks` queries
`min(2*n, NUM_CARPARKS)` distinct parks and always responds **200** with the
best available ranking. If `2*n` exceeds the catalogue (max 99), every
configured park is queried and up to `n` results are returned — the request
does not 400/422. Only `n < 1` is rejected (422).

**Writes are batched.** The `2*n` car-park statuses produced by one request are
written through `upsert_statuses()`, which is a single batched commit on
Firestore instead of `2*n` sequential round trips.

### Health endpoints

`/health/live` (always 200 while the process runs) and `/health/ready`
(200 once the YOLO model is loaded, else **503**). `/health` and `/api/health`
remain as combined aliases. The camera service mirrors this: `/health/ready`
returns 503 until at least one usable image is loaded.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock YOLO inference and camera HTTP entirely, so they need neither the
model, the images, nor a running camera service.

## Load generation (Locust, §5)

`locustfile.py` simulates concurrent end users. Each Locust user gets its own
UUID and continuously calls `/api/find-carparks`, with a minority of requests
following up on `/api/annotate-carpark`.

```bash
pip install -r requirements-dev.txt

# Web UI — open http://localhost:8089
locust -f locustfile.py --host http://localhost:8000

# Headless, for the 1/2/4/8 replica benchmark against the GKE LoadBalancer:
locust -f locustfile.py --host http://<EXTERNAL-IP> --headless \
       -u 20 -r 5 -t 3m --csv reports/pods-1
```

Scale the API between runs (`kubectl scale deploy/smartpark-api -n smartpark --replicas=N`).
Keep `CACHE_TTL_SECONDS=0` (the ConfigMap default) so the table is raw
inference; then optionally flip the cache on and re-run once to quantify §4.3.

---

## Configuration

See `.env.example` for every variable and its default. Highlights:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `./model/model.pt` | Path to supplied weights (mounted at runtime) |
| `NUM_CARPARKS` | `24` | Car parks to simulate (10–99); keep equal on both services |
| `CAMERA_BASE_URL` | `http://localhost:8001` | Camera simulator base URL |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence for `predict()` |
| `MODEL_MAX_CONCURRENCY` | `1` | Concurrent predictions on the shared model (must be 1) |
| `HTTP_TIMEOUT_SECONDS` | `10.0` | Per-camera HTTP timeout |
| `UUID_WINDOW_SECONDS` | `30` | Window for the recent-UUIDs view |
| `CACHE_TTL_SECONDS` | `5` | Per-user response cache TTL; `0` disables |
| `CACHE_MAX_ENTRIES` | `1024` | Bound on cached responses |
| `VERIFY_CAMERA_CARPARK_COUNT` | `true` | Warn at startup if the camera disagrees on `NUM_CARPARKS` |
| `IMAGES_DIR` | `./images` | Camera simulator's image directory |
| `LOG_LEVEL` | `INFO` | Log verbosity |

A `NUM_CARPARKS` mismatch between the two services is otherwise silent but
corrosive — the API would request cameras the simulator has never heard of and
those car parks would permanently report `error`. At startup the API
best-effort queries the camera's `/cameras` count in a background task and logs
an ERROR on mismatch; it never delays or blocks readiness.

Kubernetes is intentionally **not** included in this version.
