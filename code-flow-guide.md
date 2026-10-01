# SmartPark — Code Flow Guide

A practical tour of `camera_service/` and `app/`: how the pieces fit together, what happens in
what order, and the handful of techniques that genuinely matter (`async`, worker threads, the
concurrency guard, middleware, contextvars).

This is the **lighter companion** to `camera-and-core-api-explained.md`. That document covers
every line; this one covers the shape of the system. Read this first.

---

## Contents

- [1. The system in one picture](#1-the-system-in-one-picture)
- [2. What happens at startup](#2-what-happens-at-startup)
- [3. The main request flow: find-carparks](#3-the-main-request-flow-find-carparks)
- [4. The other flows](#4-the-other-flows)
- [5. How data changes shape as it moves](#5-how-data-changes-shape-as-it-moves)
- [6. The concepts that actually matter](#6-the-concepts-that-actually-matter)
- [7. What each file does](#7-what-each-file-does)
- [8. Error handling: the rules](#8-error-handling-the-rules)
- [9. Quick reference](#9-quick-reference)

---

## 1. The system in one picture

Two independent services, two Docker images, two ports.

```
      ┌─────────────────────────────────────────────────┐
      │  API SERVICE          app/         :8000        │
      │                                                 │
 ───▶ │  routes ─▶ registry ─▶ camera client ─▶ YOLO    │
      │                             │          model    │
      │                             │            │      │
      │                        repository ◀──────┘      │
      │                       (memory | Firestore)      │
      └─────────────────────────────┬───────────────────┘
                                    │ HTTP, concurrent, keep-alive
                                    ▼
      ┌─────────────────────────────────────────────────┐
      │  CAMERA SIMULATOR     camera_service/   :8001    │
      │  returns a random supplied JPEG as base64 JSON  │
      └─────────────────────────────┬───────────────────┘
                                    ▼
                          /images  (mounted volume)
```

**Why split them?** The camera image needs only FastAPI (~100 MB). The API image needs
torch + ultralytics (~1 GB). Keeping them separate means camera replicas stay cheap, and each
service scales on its own signal — cameras are I/O-bound, the API is CPU-bound.

The camera service has **zero imports from `app`**. That's enforced by `Dockerfile.camera`, which
only copies `camera_service/`. Any cross-import would crash the container.

**The one rule that shapes everything else:** the API runs on a single-threaded `asyncio` event
loop. Anything that blocks that thread freezes *every* in-flight request. Every blocking
operation in this codebase is therefore pushed onto a worker thread. More on that in
[section 6](#6-the-concepts-that-actually-matter).

---

## 2. What happens at startup

### API service (`app/main.py`)

`uvicorn app.main:app` imports the module, which calls `create_app()`, which builds the FastAPI
object and registers the **lifespan**. The lifespan then runs before the port accepts traffic.

```
 import time
 ├─ get_settings()              read env vars, validate, cache  (lru_cache)
 ├─ configure_logging()         install JSON formatter early
 └─ create_app()
     ├─ FastAPI(lifespan=...)
     ├─ add_middleware(RequestContextMiddleware)
     └─ include_router(core), include_router(operations)

 lifespan startup   ← uvicorn runs this before opening the port
 ├─ 1. settings + logging      (again; idempotent)
 ├─ 2. CarParkRegistry          builds CBD_001 … CBD_0NN + camera URLs
 ├─ 3. repository               memory or Firestore, per REPOSITORY_BACKEND
 ├─ 4. TTLCache                 per-user response cache (§4.3)
 ├─ 5. httpx.AsyncClient        ONE client for the process lifetime
 │      └─ CameraClient(client)
 ├─ 6. InferenceService.load()  loads the YOLO model ONCE
 │      └─ on failure: log the error, set inference = None, KEEP RUNNING
 └─ 7. spawn background task    verify the camera agrees on NUM_CARPARKS
        └─ create_task, never awaited → cannot delay readiness

 ─── serving ───

 lifespan shutdown
 ├─ cancel the config-check task if still running
 ├─ await http_client.aclose()
 └─ await repository.close()
```

Step 7 exists because a `NUM_CARPARKS` mismatch between the two services is silent but
corrosive: the API would request cameras the simulator has never heard of, and those car parks
would permanently report `error`. The check queries the camera's `/cameras` count and logs an
ERROR on mismatch. It is `asyncio.create_task`'d rather than awaited, so a slow or absent camera
can never hold up readiness, and it's cancelled on shutdown.

Everything built here is stashed on **`app.state`** and lives for the whole process. Nothing
expensive is ever constructed per request.

Two decisions in that sequence are worth understanding:

**The model loads exactly once.** It takes seconds and tens of MB. Loading per request would be
catastrophic, so it happens here and the instance is shared.

**A model failure does not crash the app.** `app.state.inference` is set to `None` and startup
continues:

```python
    app.state.inference = None
    try:
        app.state.inference = InferenceService.load(...)
    except Exception:
        logger.error("model load failed; inference routes will return 503", ...)
```

Why? A crashed container tells you almost nothing — it restarts in a loop and you lose the logs.
A running-but-degraded one lets you hit `/health` (which reports `model_loaded: false`), read the
logs, and see the dashboard. Inference routes cleanly return 503 via the dependency; everything
else keeps working.

**One shared `httpx.AsyncClient`** is the biggest performance decision here. A new client per
request means a new TCP handshake per request. A `find-carparks` call with n=10 makes 20 camera
requests — that's 20 handshakes avoided by reusing a warm connection pool.

### Camera service (`camera_service/main.py`)

Simpler, and it fails differently:

```
 import time
 ├─ read IMAGES_DIR, NUM_CARPARKS, SERVICE_NAME, LOG_LEVEL from env
 ├─ _parse_num_carparks()   → SystemExit if invalid   ← HARD FAIL
 └─ configure JSON logging

 lifespan startup
 ├─ app.state.valid_ids     {CBD_001 … CBD_0NN}
 └─ app.state.image_files   scan IMAGES_DIR once, sorted
        └─ directory missing? log an error, return []  ← SOFT FAIL
```

Note the deliberate asymmetry. A bad `NUM_CARPARKS` **exits immediately** — it's unfixable without
a restart, so fail loudly. A missing images directory **doesn't** — the volume can be remounted,
and meanwhile the service stays up and diagnosable with readiness reporting 503.

The directory is scanned **once at startup**, not per request. Listing a directory is a syscall
per entry; doing it in the hot path would add disk I/O to every photo.

---

## 3. The main request flow: find-carparks

`GET /api/find-carparks?uuid=alice&n=3`

### The pipeline

```
 1. MIDDLEWARE       mint/reuse request_id, pull uuid from query,
                     set contextvars, start latency timer
        │
 2. VALIDATION       FastAPI checks uuid non-empty, 1 <= n <= 49  (422 if not)
        │
 3. DEPENDENCIES     pull registry / camera / inference / repository / cache
                     from app.state                        (503 if missing)
        │
 4. LIMIT GUARD      reject n this deployment can't serve   (400, §4.3)
        │            ── no camera contacted, no model touched ──
 5. CACHE            cache.get((uuid, n))  →  HIT? record uuid, return early
        │            ══ skips steps 6-10 entirely ══
 6. SAMPLE           registry.sample(2 * n)  →  6 distinct random car parks
        │
 7. FETCH            camera.fetch_many(picks)
                     ══ 6 concurrent HTTP GETs ══
        │            failures become FetchOutcome(error=...), not exceptions
        │
 8. INFER            asyncio.gather(process(o) for o in outcomes)
                     ══ 6 coroutines, but Semaphore(1) ══
                     ══ serialises the actual predictions ══
        │            each prediction runs in a worker thread
        │
 9. PERSIST          upsert_statuses(6)  ← ONE batched write
        │            record the uuid sighting
10. RANK             keep status == "ok", sort by (empty_count, confidence)
        │            descending, take the top n
11. RESPOND          project each status → CarParkResult, store in cache
        │
12. MIDDLEWARE       log "request completed" + latency, reset contextvars
```

### Why 2n?

You query **twice** as many car parks as requested so there's a real pool to rank. Querying
exactly n would mean returning whatever you happened to get, merely sorted — not "the best n".

### Steps 4 and 5: the §4.3 optimisations

The spec asks two pointed questions under *Performance Optimisation*, and these two steps are
the answers.

**"What if the user sends a large n (n>100)?"** Two layers reject it before any work happens.
FastAPI's `le=49` constraint catches absurd values during validation (422) — 49 being the
largest `n` serviceable at the maximum `NUM_CARPARKS` of 99, since `2*n` parks must be queried.
Step 4 then catches values within that ceiling that *this* deployment still can't serve, and the
error names the real limit:

```
n=13 requires querying 26 distinct car parks but only 24 are configured (maximum n is 12)
```

**"Repeated requests from the same user can be cached."** Step 5 checks a short-TTL cache keyed
by `(uuid, n)`. A hit skips `2*n` camera fetches and `2*n` YOLO predictions — measured locally,
**1343 ms → 63 ms**.

One subtlety: a cache hit **still records the UUID sighting** before returning. OPS-API-2 counts
distinct users in the last 30 seconds from the request logs, so if cached responses skipped that
write, a user polling every second would vanish from the operational view after their first
call. The optimisation would silently corrupt an unrelated metric.

Set `CACHE_TTL_SECONDS=0` to disable caching when benchmarking raw inference throughput;
`/api/operations/cache-stats` reports hits, misses, and hit rate.

### The concurrency, drawn out

This is the interesting part. Step 5 is fully parallel; step 6 is deliberately not.

```
 t=0ms    fetch_many launches all 6 HTTP requests at once
          CBD_003  ├────────────┤
          CBD_011  ├─────────────┤        network waits OVERLAP
          CBD_007  ├───────────┤          total ≈ the slowest one,
          CBD_019  ├──────────────┤       not the sum
          CBD_002  ├────────────┤
          CBD_014  ├───────╳ timeout → recorded as an error
 t≈55ms   5 photos in hand, 1 failure captured

 t=55ms   6 process() coroutines start; the 5 good ones call infer()
          Semaphore(1) admits exactly one at a time:

          CBD_003  [worker thread: predict 40ms]
          CBD_011                               [predict 40ms]
          CBD_007                                             [predict 40ms]
          CBD_019                                                    [40ms]
          CBD_002                                                       [40ms]
          CBD_014  (error path — returns instantly, never touches the model)

 t≈255ms  all done
```

Two different strategies, for two different reasons:

- **Network waits are parallelised** because they're pure waiting. 6 × 50 ms sequential would be
  300 ms; overlapped it's ~55 ms.
- **Predictions are serialised** because the YOLO model is **not thread-safe**. Two threads
  predicting at once can interleave mutations of shared internal state and produce garbage or
  crash in native code.

And crucially: while those predictions are queued and running, **the event loop is free**. Other
requests — `/health`, `/dashboard`, other users' camera fetches — are served throughout, because
each prediction runs in a worker thread rather than on the loop.

### The key code

**Concurrent fetching** (`services/camera_client.py`):

```python
        async def _one(carpark: CarParkInfo) -> FetchOutcome:
            try:
                image_bytes = await self.fetch_photo(carpark.camera_url)
                return FetchOutcome(carpark=carpark, image_bytes=image_bytes)
            except Exception as exc:
                logger.error("camera fetch failed", extra={"carpark_id": carpark.id, ...})
                return FetchOutcome(carpark=carpark, image_bytes=None, error=str(exc))

        return await asyncio.gather(*(_one(cp) for cp in carparks))
```

`asyncio.gather` schedules every coroutine at once and returns results **in input order**,
regardless of which finished first. The `try/except` inside `_one` is what makes one dead camera
survivable — failures become *result objects* rather than exceptions, so the batch never fails as
a whole.

**The concurrency guard** (`services/inference.py`) — four lines carrying the entire strategy:

```python
    async def infer(self, image_bytes: bytes, annotate: bool = False) -> InferenceResult:
        """Run prediction off the event loop, one caller at a time."""
        async with self._semaphore:
            return await asyncio.to_thread(self._run, image_bytes, annotate)
```

- `async with self._semaphore` — at most one prediction at a time. It **suspends the coroutine**
  while waiting, so the loop keeps running other work.
- `asyncio.to_thread(self._run, ...)` — the actual prediction happens in a worker thread, so the
  event loop is never blocked.

Note the order: acquire the permit **first**, then dispatch to a thread. The reverse would occupy
a thread-pool slot just to sit and wait.

**Ranking** (`api/core.py`):

```python
    successful = [s for s in statuses if s.status == "ok"]
    ranked = sorted(
        successful,
        key=lambda s: (s.empty_count, s.confidence_score),
        reverse=True,
    )[:n]
```

A tuple key sorts by empty spaces first and uses confidence **only to break ties** — two car parks
both showing 5 free spaces are ordered by how sure the model was. `[:n]` is safe even when fewer
than n cameras worked; you simply get fewer results instead of an error.

---

## 4. The other flows

### annotate-carpark

`GET /api/annotate-carpark?carpark_id=CBD_007`

```
 registry.get(carpark_id)        → 404 if unknown
 camera.fetch_photo(url)         → 502 if the camera fails
 inference.infer(..., annotate=True)  → 500 if inference fails
     └─ additionally: result.plot() → flip BGR→RGB → encode JPEG
 repository.upsert_status(...)
 respond with base64 annotated image + counts
```

**The error handling here is deliberately different from `find-carparks`.** There, one failing
camera was survivable because 19 others remained, so it became a recorded error. Here there's only
one car park, so a failure means the request can't be fulfilled — and it becomes an HTTP error.

The 502-vs-500 split is the useful bit: **502** means the *camera service* failed (we were a
gateway to it), **500** means *our own* inference failed. An operator knows which service to
investigate without reading a single log line.

This is also the only place `annotate=True` is passed. Rendering boxes and encoding a JPEG is
expensive, so `find-carparks` skips it entirely — that's 20 image encodes avoided per call.

### The dashboard

`GET /` or `/dashboard` returns a self-contained HTML page (no CDN, no static-file mount, no
template engine — it's a Python string in `app/dashboard.py`). The browser then polls:

```
 every 3s:
   Promise.allSettled([ /api/operations/statuses,
                        /api/operations/recent-uuids,
                        /api/health ])
   then: /api/operations/plot.png?t=<timestamp>
```

Two details worth copying:

**`Promise.allSettled`, not `Promise.all`.** `all` rejects as soon as any promise rejects, so one
failing endpoint would stop the other two from updating. `allSettled` waits for all of them
regardless — partial failure degrades one panel instead of the whole dashboard. It's the
browser-side twin of `fetch_many`'s per-camera error isolation.

**The cache-busting `?t=`.** Browsers cache by URL, so re-setting the same `src` every 3 seconds
would show a frozen image. The timestamp makes each URL unique; the server ignores it.

### The plot endpoint

`GET /api/operations/plot.png` is the third place the offload pattern appears:

```python
    png = await asyncio.to_thread(
        render_availability_png, statuses, window, len(recent)
    )
    return Response(content=png, media_type="image/png")
```

matplotlib is blocking CPU work — a 24-bar chart takes 50–200 ms. On the event loop that would
freeze the server; in a worker thread it doesn't.

Two things inside `plotting.py` matter:

```python
    import matplotlib
    matplotlib.use("Agg")          # headless — no display server needed
    import matplotlib.pyplot as plt
```

The order is mandatory. `pyplot` picks its backend at import time, so `use("Agg")` must come
first, or matplotlib tries to load a GUI backend, finds no display in the container, and fails.

```python
    plt.close(fig)  # release the figure so repeated calls don't leak memory
```

matplotlib keeps a global registry of open figures. The dashboard polls every 3 seconds, so
without this you'd leak ~1,200 figures an hour and eventually OOM.

### Health probes — and why there are three

Both services expose `/health`, `/health/live`, and `/health/ready`. The liveness/readiness split
is not ceremony; getting it backwards causes outages.

| Probe | Question | On failure | Returns 503 when |
|---|---|---|---|
| **liveness** | Is the process wedged? | Kubernetes **kills and restarts** the pod | essentially never |
| **readiness** | Can it serve traffic *now*? | pod is **pulled from the load balancer** | model not loaded / no images |

If liveness checked for the model, a missing volume mount would cause an infinite restart loop and
you'd lose the logs explaining why. Because readiness handles it instead, the pod stays up, stays
inspectable, and simply receives no traffic.

---

## 5. How data changes shape as it moves

One image's journey through the system:

```
  /images/lot_04.jpg
        │  camera: read_bytes() in a worker thread
        ▼
  raw JPEG bytes
        │  camera: base64.b64encode()
        ▼
  {"carpark_id", "filename", "content_type", "image_base64"}   ← JSON over HTTP
        │  CameraClient.fetch_photo(): response.json() + b64decode()
        ▼
  raw JPEG bytes
        │  wrapped as FetchOutcome(carpark, image_bytes, error)
        ▼
  InferenceService._run(): PIL decode → RGB → model.predict()
        │
        ▼
  Ultralytics Results  ──  _parse()  ──▶  InferenceResult
                                          empty_count, occupied_count,
                                          total_spaces, confidence_score,
                                          speed_inference, annotated_jpeg?
        │  api/core.py maps it
        ▼
  CarParkStatus         ← the PERSISTED shape (goes to the repository)
        │  .to_result(name=...)  projects it
        ▼
  CarParkResult         ← the PUBLIC shape (only 4 fields, per the spec)
```

Three shapes, three jobs:

- **`InferenceResult`** — a plain `@dataclass`, internal only, never crosses the wire.
- **`CarParkStatus`** — the pydantic record that gets stored. Rich: counts, confidence, timing,
  who asked, when, and an error detail.
- **`CarParkResult`** — the narrow public projection: `carpark_id`, `name`, `available_spaces`,
  `confidence_score`. `to_result()` is the deliberate narrowing, so internal fields can't leak.

**Why base64?** JSON has no binary type. It costs ~33% more bytes, but it buys a self-describing
envelope carrying `filename` and `content_type` alongside the pixels, and the client side is just
`response.json()`.

**Counting logic** (`_parse`) is simple: every detection labelled `"empty"` is a free space and its
confidence is collected; everything else counts as occupied. `confidence_score` is the mean
confidence across the empty detections — that's the tie-breaker in ranking.

---

## 6. The concepts that actually matter

### The event loop, and why blocking is fatal

An `asyncio` app runs on **one thread**. The loop runs each task until it hits an `await` that
suspends, then switches. That gives huge concurrency for I/O — but one absolute rule:

> **Never block the event loop thread.**

A synchronous call that takes 200 ms means the loop runs *nothing else* for 200 ms. Every other
in-flight request stalls.

This codebase has exactly three blocking operations, and **all three are offloaded**:

| Operation | Where | Why it blocks |
|---|---|---|
| `Path.read_bytes()` | camera `take_photo` | disk / network filesystem I/O |
| `model.predict()` | `InferenceService.infer` | CPU-bound native code |
| `fig.savefig()` | `operational_plot` | CPU-bound rendering |

The tool is always the same:

```python
result = await asyncio.to_thread(some_blocking_function, arg1, arg2)
```

Run it in a worker thread, suspend this coroutine, let the loop serve other requests, resume when
the thread finishes. Note the function is passed **without parentheses** — you hand over the
function object and its arguments separately.

> **About the GIL.** Python's Global Interpreter Lock normally stops threads running Python
> bytecode in parallel, so why does `to_thread` help CPU-bound work? Because the heavy libraries
> here (torch, numpy, PIL, matplotlib's C core) **release the GIL** while in native code. During
> `predict()` the worker thread is in C++ with the GIL released, so the event loop runs freely.
> For pure-Python CPU work you'd reach for a process pool instead.

### `async` / `await`, briefly

`async def` creates a coroutine that does nothing until awaited. `await` means "suspend here, let
the loop do other work, resume when this finishes". A coroutine that never awaits anything gains
nothing from being async.

Which is why it's worth noting that **`InMemoryRequestRepository`'s methods are `async` even though
they never await anything**. That's intentional, and the docstring says so:

> All methods are `async` for exactly that reason — the in-memory version does not need to await
> anything, but a Firestore version will.

The interface is shaped for the *slowest* implementation. Routes are written as
`await repository.upsert_status(...)` from day one, so swapping in Firestore changes nothing at any
call site. Had the interface been synchronous, adding Firestore would have meant rewriting every
route — you can't `await` inside a `def`.

### Semaphore vs Lock — the distinction that matters

| Primitive | What it blocks | Use for |
|---|---|---|
| `asyncio.Semaphore(n)` | **suspends the coroutine** | at most *n* concurrent holders — here, 1 prediction |
| `asyncio.Lock()` | **suspends the coroutine** | mutual exclusion — here, in-memory repository state |
| `threading.Lock()` | **blocks the thread** | coordinating real threads — never hold one on the loop |

Using `threading.Lock` on the event loop reintroduces the exact freeze you were avoiding: the loop
thread would sit there waiting. Both the inference guard and the in-memory repository use the
`asyncio` variants for this reason.

**Why does single-threaded async code need a lock at all?** Because `await` is a yield point.
Between two `await`s another coroutine can run and mutate shared state. In `recent_uuids`, which
prunes the deque and then reads it, an interleaved append would produce inconsistent results. The
lock makes each operation atomic with respect to other coroutines.

### Middleware — and why it's pure ASGI

Both services wrap every request in middleware that assigns a request id, times the request, and
emits one structured log line. The API's version (`app/middleware.py`) also pulls `uuid` out of the
query string.

The notable decision is in the docstring:

> Implemented as *pure ASGI* middleware (rather than Starlette's `BaseHTTPMiddleware`) so the
> contextvars we set here reliably propagate into the route handler — `BaseHTTPMiddleware` runs the
> endpoint in a separate task, which breaks contextvar propagation.

`BaseHTTPMiddleware` is friendlier to write (you get a `Request` object and `await call_next`), but
it runs the downstream app in a **separate asyncio task**. The practical consequence: a
`request_id` set in the middleware would be **invisible** to `logger.info(...)` inside a route
handler. Every log line from the business logic would lose its request id — defeating the entire
logging design.

Pure ASGI middleware runs in the **same task** as the endpoint, so contextvars work. The cost is
handling raw `scope`/`receive`/`send`, which looks like this:

```python
    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":       # pass lifespan/websocket through untouched
            await self.app(scope, receive, send)
            return
        ...
        async def send_wrapper(message):   # intercept the response to capture status
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            ... log at ERROR with traceback ...
            raise                          # re-raise: observe, don't handle
        else:
            ... log at INFO with the real status ...
        finally:
            ... reset the contextvars ...
```

Three things to take from that skeleton:

- **`send_wrapper`** is the standard ASGI interception pattern — hand the app a replacement `send`,
  watch what flows through, forward it.
- **`except` re-raises.** The middleware *observes and records*; it does not handle. Swallowing the
  exception would leave the client hanging with no response.
- **`finally` resets the contextvars.** Without this, values leak between requests handled by the
  same task, and request B could log request A's id.

### contextvars — the invisible plumbing

This is what makes the logging design work, and it's worth understanding properly.

**The problem:** you want every log line emitted while handling a request to carry that request's
id — including lines from deep inside `inference.py` or `camera_client.py`, which have no `Request`
object. Thread-locals don't work (async multiplexes many requests onto one thread, so they'd
collide). Passing a context object through every signature pollutes every API in the codebase.

**The solution:** `contextvars.ContextVar` holds a value scoped to the current *execution context*,
and asyncio gives each task its own copied context. So request A sees `"aaa"`, request B sees
`"bbb"`, interleaved on the same thread, with no locking. Coroutines spawned via `asyncio.gather`
inherit the parent's context automatically.

The flow:

```
 middleware sets  request_id_ctx, uuid_ctx, endpoint_ctx
        │
        ├─▶ JsonFormatter reads them → every log line is decorated automatically
        ├─▶ camera_client._trace_headers() reads request_id → forwards X-Request-ID
        │        └─▶ the camera service reuses it → SAME id in both services' logs
        └─▶ FirestoreRepository.record_uuid() reads them → persisted with each sighting
        │
 middleware resets them in `finally`
```

The payoff: you can grep one request id and see the complete story across two services, the log
aggregator, and Firestore — without a single function signature mentioning it.

### Lifespan singletons + dependency injection

Expensive things are built **once** in the lifespan and stashed on `app.state`. Routes reach them
through small provider functions in `app/dependencies.py`:

```python
def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(503, detail="Inference service unavailable (model not loaded)")
    return service
```

Routes then declare `inference: InferenceService = Depends(get_inference_service)`.

Two things this buys:

1. **It converts the degraded startup state into a clean 503** on exactly the routes that need the
   model, while health, the dashboard, and the operational views keep working.
2. **It's the test seam** — and this is the main reason the file exists. A test writes one line:

   ```python
   app.dependency_overrides[get_inference_service] = lambda: FakeInference()
   ```

   and exercises the full HTTP stack — routing, validation, serialisation, middleware — with no
   model, no network, no Firestore. Without this seam, testing the ranking logic would mean loading
   a real YOLO model.

### The repository pattern

An abstract `RequestRepository` with two implementations, selected by one environment variable:

```
 REPOSITORY_BACKEND=memory     → InMemoryRequestRepository    dev, tests, single pod
 REPOSITORY_BACKEND=firestore  → FirestoreRequestRepository   multi-pod cloud
```

Routes only ever see the abstract type. The lifespan picks; `_build_repository` is the only code
that knows both exist.

**Why it's needed:** with 3 API pods behind a load balancer, `find-carparks` calls scatter across
all three, but a `/recent-uuids` request lands on exactly one. With in-memory storage you'd see
roughly a third of your users, and the number would jump around depending on routing. Firestore
gives all pods one shared store.

The in-memory version uses a `dict` for statuses (natural upsert semantics) and a `deque` of
`(timestamp, uuid)` sightings, pruned from the left on each read — lazy garbage collection, no
background task needed. Firestore mirrors that with two collections:

```
 carpark_status/{carpark_id}   one doc per car park, overwritten   ← bounded
 request_logs/{auto_id}        one doc per sighting, append-only   ← grows
```

Both implementations deduplicate UUIDs the same way — using a `dict` as an **ordered set**, because
`dict` preserves insertion order and `set` doesn't:

```python
        seen: dict[str, None] = {}
        for _, uuid in self._uuid_sightings:
            seen.setdefault(uuid, None)
        return list(seen.keys())
```

Only the keys matter; the values are meaningless. The result is distinct UUIDs in first-seen order.

### Lazy imports

Heavy dependencies are imported **inside functions**, not at module scope:

```python
    @classmethod
    def load(cls, model_path: str, ...) -> "InferenceService":
        from ultralytics import YOLO   # imported here to avoid a hard import cost
        ...
```

| Module | Deferred | Why |
|---|---|---|
| `services/inference.py` | `ultralytics`, `PIL` | unit-test the parsing logic without torch installed |
| `services/firestore_repository.py` | (imported lazily by `main.py`) | skip the gRPC stack on the memory backend |
| `plotting.py` | `matplotlib` | keep imports cheap when no plot is requested |

Importing torch takes seconds and hundreds of MB. Several tests genuinely run on machines without
it. Python caches modules in `sys.modules`, so only the first call does real work.

### Structured JSON logging

Every log line is one JSON object, written to stdout — never to a file. The container writes to
stdout and the platform routes it; a service managing its own log files has to solve rotation,
permissions, and disk-full, all of which are the platform's job.

The payoff over plain text: *"every request from uuid X that took over 500 ms"* is a query
(`uuid="X" AND latency_ms>500`) rather than an unmaintainable regex.

Fields are named for the aggregator, not for Python — `severity` rather than `levelname`, because
that's what Google Cloud Logging recognises. And `configure_logging` reroutes uvicorn's own loggers
through the same handler, so you don't end up with a stream that's half JSON and half not.

### Distributed tracing across the two services

```
 client ──▶ API middleware          mints request_id (or reuses a valid inbound one)
                 │                  stores it in request_id_ctx
                 ▼
            camera_client           _trace_headers() reads the contextvar
                 │                  sends X-Request-ID: <id>
                 ▼
            camera middleware       reuses the inbound id after sanitising it
                 │
                 ▼
            both services log the SAME request_id
```

The inbound header is sanitised before use — it's attacker-controlled input that ends up in log
files, so control characters (especially newlines, which would let someone forge log lines) are
rejected. Invalid ids are simply replaced with a fresh one rather than rejecting the request.

---

## 7. What each file does

### `camera_service/`

| File | What it does |
|---|---|
| `__init__.py` | package marker; docstring records the "no ML dependencies" constraint |
| `main.py` | the entire simulator: config, JSON logging, access-log middleware, 4 endpoints |

`main.py` is self-contained by design — config via `os.getenv` (four settings don't justify a
settings framework), its own small JSON formatter, its own middleware. The interesting line is the
offloaded disk read in `take_photo`.

### `app/` — foundations

| File | What it does | Worth knowing |
|---|---|---|
| `__init__.py` | version string | single source of truth, used by `/health` and OpenAPI |
| `config.py` | all settings from env vars via `pydantic-settings` | `@lru_cache` makes it a singleton; validators fail fast at startup |
| `logging_config.py` | JSON formatter + the three contextvars | auto-decorates every log line with request context |
| `middleware.py` | request id, uuid extraction, access log | pure ASGI so contextvars propagate |
| `dependencies.py` | five `Depends` providers | the test seam; converts missing singletons into 503s |
| `main.py` | app factory + lifespan | builds every singleton once; loads the model once |

On `config.py`: thirteen settings with types, defaults, validation, and documentation in one place
— which is why it uses a settings class where the camera service uses `os.getenv`. Two validators
fail fast at startup (`NUM_CARPARKS` in 10–99, `MODEL_MAX_CONCURRENCY >= 1`), so bad config never
reaches a live listener.

### `app/models/`

| File | What it does |
|---|---|
| `schemas.py` | every pydantic wire contract in one file |

The models split into three groups: **inputs/outputs** (`FindCarParksResponse`, `AnnotateResponse`,
`HealthResponse`, …), the **persisted record** (`CarParkStatus`), and the **static catalogue entry**
(`CarParkInfo`). `CarParkStatus.to_result()` is the internal→public projection.

`CarParkAvailability` is the one with a three-state status (`ok` / `error` / `unknown`), because
OPS-API-1 must list *every configured* car park including ones never queried. Those get `None`
counts rather than `0` — "we never looked" and "there are no spaces" are genuinely different
answers.

### `app/services/`

| File | What it does | Key mechanic |
|---|---|---|
| `carpark_registry.py` | the static catalogue: `CBD_001` … `CBD_0NN` → camera URLs + street names | precomputed at startup; `sample()` draws *distinct* car parks |
| `camera_client.py` | fetches photos over HTTP | shared `AsyncClient`; `gather` for concurrency; failures → result objects |
| `inference.py` | owns the shared YOLO model | `Semaphore(1)` + `to_thread`; lazy torch import |
| `request_repository.py` | abstract interface + in-memory impl | all-async interface; `deque` + `asyncio.Lock`; batched `upsert_statuses` |
| `firestore_repository.py` | shared cross-pod impl | `AsyncClient`; ADC auth, no key file; `WriteBatch` + bounded log purge |
| `response_cache.py` | per-user TTL cache (§4.3) | monotonic clock, lazy expiry, bounded eviction |

Three details from this layer:

**`registry.sample()` uses `random.sample`, not `random.choices`** — `sample` draws *without*
replacement, so the 2n car parks are guaranteed distinct. `choices` would happily return the same
car park three times.

**`CameraClient` borrows its HTTP client rather than creating one.** That's why it has no `close()`
method (the lifespan owns the client) and why tests can pass an `AsyncClient` with a mock transport
and never touch the network.

**`inference.py` mediates all access to the model.** Nothing outside the class touches
`self._model`, which means the concurrency guard cannot be bypassed. The parsing logic (`_parse`)
is deliberately free of torch and PIL types so it can be unit-tested with a fake result object.

### `app/api/`

| File | Endpoints |
|---|---|
| `core.py` | `find-carparks`, `annotate-carpark` — the two assignment endpoints |
| `operations.py` | statuses, recent-uuids, carparks, availability, plot.png, health ×3, dashboard ×2 |

`core.py` uses `APIRouter(prefix="/api", tags=["core"])`; `operations.py` uses no prefix because it
serves paths at several levels (`/api/operations/…`, `/health`, `/`).

`operations.py` has one neat trick — stacked decorators registering one function at two paths:

```python
@router.get("/api/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health(...):
```

Same for `/` and `/dashboard`. `include_in_schema=False` keeps the duplicate out of `/docs`.

Its health helper reads `app.state` **directly rather than via `Depends`**, because the dependency
*raises* 503 when the model is missing — and health needs to *report* that state, not fail on it.

### `app/` — presentation

| File | What it does |
|---|---|
| `plotting.py` | renders the availability chart as PNG bytes |
| `dashboard.py` | the whole monitoring UI as one Python string |

`render_availability_png` is a plain `def` (it runs in a worker thread) taking plain data and
returning bytes — no repository, no request, no I/O, so it's trivially testable. It draws a
**stacked** bar chart so you see availability and total capacity at once, and it has a real empty
state ("No data yet — call /api/find-carparks") rather than a blank chart that looks like a bug.

`dashboard.py` ships the UI inside the image with no static-file mount, no template engine, and no
CDN — so it works in an air-gapped network and can't break because a third party changed. Its
`esc()` helper is a genuine XSS guard: `uuid` comes straight from user input (`?uuid=...`) and gets
interpolated into HTML.

---

## 8. Error handling: the rules

The consistent principle: **degrade when the system can still do useful work; fail fast when it
can't.**

| Failure | What happens | Why |
|---|---|---|
| Model fails to load | app starts; inference routes 503; `/health` says `model_loaded: false` | logs and dashboard stay reachable |
| Images directory missing | camera starts; readiness 503 | the mount can be fixed without a restart loop |
| One camera unreachable | that car park marked `error`; the rest returned normally | 19 good results beat 1 error |
| `NUM_CARPARKS` invalid | **hard exit** | unfixable without a restart — fail loudly |
| Camera fails on `annotate` | 502 | only one car park; nothing to degrade to |
| Inference fails on `annotate` | 500 | our process, our bug |

### Status codes, chosen deliberately

| Code | Where | Meaning |
|---|---|---|
| 400 | `find-carparks` when 2n > configured | client asked for the impossible |
| 404 | unknown car park | that resource doesn't exist |
| 422 | automatic, from FastAPI | a parameter failed validation |
| 500 | `annotate-carpark` inference failure | our bug, in our process |
| 502 | `annotate-carpark` camera failure | upstream service failed |
| 503 | readiness probes, missing dependencies | temporarily unable; retry later |

### The two error strategies, side by side

```
 find-carparks          many car parks → one failure is survivable
                        → capture it as FetchOutcome(error=...) / CarParkStatus(status="error")
                        → persist it (the dashboard shows WHY a car park is unavailable)
                        → exclude it from ranking
                        → return the successful ones

 annotate-carpark       one car park → a failure means the request can't be fulfilled
                        → raise HTTPException with a code identifying WHICH service failed
```

Both broad `except Exception` blocks are marked `# noqa: BLE001` with a comment stating the intent
— they're deliberate, not sloppy. Anything can go wrong with a network call (connection refused,
DNS, timeout, malformed JSON, bad base64) and the response to all of them is identical.

---

## 9. Quick reference

### Endpoints

**API service, port 8000**

| Path | Purpose |
|---|---|
| `/api/find-carparks?uuid=&n=` | COREAPI1 — query 2n car parks, return the best n |
| `/api/annotate-carpark?carpark_id=&uuid=` | COREAPI2 — annotated image + counts |
| `/api/carparks` | list configured car parks and camera URLs |
| `/api/operations/statuses` | latest status for every car park queried so far |
| `/api/operations/recent-uuids` | distinct UUIDs in the 30-second window |
| `/api/operations/availability` | OPS-API-1 — **all** car parks with current availability |
| `/api/operations/cache-stats` | response-cache hits, misses, hit rate (§4.3) |
| `/api/operations/plot.png` | OPS-REQ-2 — matplotlib chart |
| `/api/health`, `/health` | combined probe |
| `/health/live` · `/health/ready` | liveness · readiness |
| `/` , `/dashboard` | HTML monitoring dashboard |
| `/docs` | auto-generated API documentation |

**Camera simulator, port 8001**

| Path | Purpose |
|---|---|
| `/cameras/{carpark_id}/api/takephoto` | random supplied image as base64 JSON |
| `/cameras` | list every camera's URL |
| `/health`, `/health/live`, `/health/ready` | probes |

### Configuration

**API** — `SERVICE_NAME`, `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `MODEL_MAX_CONCURRENCY`,
`NUM_CARPARKS`, `CAMERA_BASE_URL`, `HTTP_TIMEOUT_SECONDS`, `UUID_WINDOW_SECONDS`,
`MAX_REQUESTED_N`, `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES`,
`VERIFY_CAMERA_CARPARK_COUNT`, `REPOSITORY_BACKEND`, `FIRESTORE_PROJECT`,
`FIRESTORE_DATABASE`, `LOG_LEVEL`

**Camera** — `IMAGES_DIR`, `NUM_CARPARKS`, `SERVICE_NAME`, `LOG_LEVEL`

`NUM_CARPARKS` must match across both services, or requests for the extra car parks 404 forever.
`docker-compose.yml` avoids this by feeding both from the same `${NUM_CARPARKS:-24}` variable,
and the API logs an ERROR at startup if the camera disagrees.

Neither Dockerfile bakes in the model weights or the images — both arrive as mounted volumes at
runtime. That keeps the images small and lets you swap a model without a rebuild.

### Where to look for a given concern

| If you're changing… | Look at |
|---|---|
| how car parks are ranked | `api/core.py`, the `sorted(...)` call |
| car-park ids or names | `services/carpark_registry.py` **and** `camera_service/main.py` (kept in sync by a test) |
| what counts as an empty space | `services/inference.py`, `_parse` and `EMPTY_CLASS_NAME` |
| where data is stored | `services/request_repository.py`, `services/firestore_repository.py` |
| caching / request limits | `services/response_cache.py`, `api/core.py`, `config.py` |
| the response shape | `models/schemas.py` |
| what gets logged | `logging_config.py`, `middleware.py` |
| the dashboard UI | `dashboard.py` (HTML/CSS/JS), `plotting.py` (the chart) |
| a new setting | `config.py`, then `.env.example` and `docker-compose.yml` |
| camera behaviour | `camera_service/main.py` |

### The five patterns to remember

1. **Blocking work goes to a worker thread** — `asyncio.to_thread`, used in exactly three places.
2. **The model is loaded once and access is serialised** — `Semaphore(1)` inside `infer()`.
3. **Independent I/O runs concurrently, and per-item failures become data** — `asyncio.gather` over
   coroutines that never raise.
4. **Request context travels on contextvars**, so logs self-decorate and the trace id crosses
   service boundaries.
5. **Singletons are built once in the lifespan and injected via `Depends`** — which is also the
   seam every test uses.
