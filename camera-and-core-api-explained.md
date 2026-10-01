# SmartPark — Camera Service & Core API, Explained Line by Line

This document explains **everything** inside `camera_service/` and `app/`: what each file is,
why it exists, what every block of code does, and why particular Python/FastAPI techniques
(`async`, `asyncio.to_thread`, `Semaphore(1)`, `contextvars`, lazy imports, `lru_cache`, …)
were chosen instead of the obvious alternative.

If you only read one section, read [Part 0](#part-0--the-mental-model) and
[Part 5](#part-5--cross-cutting-concepts-the-why-behind-the-patterns).

---

## Table of contents

- [Part 0 — The mental model](#part-0--the-mental-model)
- [Part 1 — The camera service](#part-1--the-camera-service)
  - [`camera_service/__init__.py`](#camera_service__init__py)
  - [`camera_service/main.py`](#camera_servicemainpy)
- [Part 2 — The core API: foundations](#part-2--the-core-api-foundations)
  - [`app/__init__.py`](#app__init__py)
  - [`app/config.py`](#appconfigpy)
  - [`app/logging_config.py`](#apploggingconfigpy)
  - [`app/middleware.py`](#appmiddlewarepy)
  - [`app/dependencies.py`](#appdependenciespy)
  - [`app/main.py`](#appmainpy)
- [Part 3 — The core API: data & services](#part-3--the-core-api-data--services)
  - [`app/models/__init__.py`](#appmodels__init__py)
  - [`app/models/schemas.py`](#appmodelsschemaspy)
  - [`app/services/__init__.py`](#appservices__init__py)
  - [`app/services/carpark_registry.py`](#appservicescarpark_registrypy)
  - [`app/services/camera_client.py`](#appservicescamera_clientpy)
  - [`app/services/inference.py`](#appservicesinferencepy)
  - [`app/services/request_repository.py`](#appservicesrequest_repositorypy)
  - [`app/services/firestore_repository.py`](#appservicesfirestore_repositorypy)
  - [`app/services/response_cache.py`](#appservicesresponse_cachepy)
- [Part 4 — The core API: routes & presentation](#part-4--the-core-api-routes--presentation)
  - [`app/api/__init__.py`](#appapi__init__py)
  - [`app/api/core.py`](#appapicorepy)
  - [`app/api/operations.py`](#appapioperationspy)
  - [`app/plotting.py`](#appplottingpy)
  - [`app/dashboard.py`](#appdashboardpy)
- [Part 5 — Cross-cutting concepts (the "why" behind the patterns)](#part-5--cross-cutting-concepts-the-why-behind-the-patterns)
- [Part 6 — End-to-end request walkthroughs](#part-6--end-to-end-request-walkthroughs)
- [Part 7 — Endpoint reference](#part-7--endpoint-reference)
- [Part 8 — Observations, sharp edges, and deliberate trade-offs](#part-8--observations-sharp-edges-and-deliberate-trade-offs)

---

## Part 0 — The mental model

There are **two separate web services** in this repository, each with its own Docker image,
its own dependency list, and its own port.

```
                 ┌──────────────────────────────┐
  browser /      │  API service  (app/)         │
  marker    ───▶ │  uvicorn app.main:app :8000  │
                 │                              │
                 │  • YOLO model (loaded once)  │
                 │  • ranking / annotation      │
                 │  • operational views         │
                 │  • HTML dashboard            │
                 └──────────┬───────────────────┘
                            │  HTTP GET (concurrent, keep-alive)
                            │  /cameras/CBD_007/api/takephoto
                            ▼
                 ┌──────────────────────────────┐
                 │  Camera simulator            │
                 │  (camera_service/)  :8001    │
                 │                              │
                 │  • returns a random supplied │
                 │    JPEG as base64 JSON       │
                 │  • NO ML dependencies        │
                 └──────────┬───────────────────┘
                            │  reads from a mounted volume
                            ▼
                    /images/*.jpg  (never baked into the image)
```

**Why two services instead of one?**

1. **Realism.** In a real deployment the cameras are physically separate devices reachable over
   the network. Splitting them forces the API to deal with real network latency, partial
   failures, and timeouts — which is the interesting engineering problem.
2. **Image size.** The camera image only needs `fastapi` + `uvicorn` (~100 MB). The API image
   needs `torch` + `ultralytics` + `opencv` (~1 GB even with CPU-only wheels). If they were one
   service you would pay the big image for every camera replica.
3. **Independent scaling.** Cameras are I/O-bound and cheap; the API is CPU-bound and expensive.
   Separate deployments can be scaled on different signals.

**The key architectural rule that shapes almost all of this code:** the API is an `asyncio`
application running on a **single-threaded event loop**. Anything that blocks that thread
(disk reads, YOLO inference, matplotlib rendering) stalls *every* in-flight request, not just
the one doing the work. So every blocking operation in this codebase is pushed onto a worker
thread with `asyncio.to_thread(...)`, and every I/O wait is done with `await`.

### Directory map

```
release/
├── app/                          ← the core API service
│   ├── __init__.py               version string
│   ├── main.py                   app factory + lifespan (builds all singletons)
│   ├── config.py                 env-var-driven settings
│   ├── dependencies.py           FastAPI Depends() providers (test seam)
│   ├── middleware.py             pure-ASGI request-context + access logging
│   ├── logging_config.py         structured JSON logging + contextvars
│   ├── dashboard.py              the HTML/CSS/JS dashboard as a Python string
│   ├── plotting.py               matplotlib PNG rendering
│   ├── api/
│   │   ├── __init__.py
│   │   ├── core.py               COREAPI1 find-carparks, COREAPI2 annotate-carpark
│   │   └── operations.py         statuses, recent-uuids, availability, health, plot, dashboard
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            every pydantic wire contract
│   └── services/
│       ├── __init__.py
│       ├── carpark_registry.py   static catalogue of car parks → camera URLs
│       ├── camera_client.py      shared httpx.AsyncClient, concurrent fetching
│       ├── inference.py          the shared YOLO model + concurrency guard
│       ├── request_repository.py abstract storage interface + in-memory impl
│       └── firestore_repository.py  shared cross-pod storage impl
└── camera_service/               ← the camera simulator service
    ├── __init__.py
    └── main.py                   the entire simulator, self-contained
```

---

# Part 1 — The camera service

## `camera_service/__init__.py`

```python
"""Camera simulator service (standalone, no ML dependencies)."""
```

That is the **entire file** — a single docstring, no code.

**What it's for.** A directory only becomes an importable Python *package* when it contains an
`__init__.py`. Its presence is what makes `uvicorn camera_service.main:app` work and what lets
the tests do `from camera_service.main import app`.

**Why it's empty.** Two reasons:

1. Anything you put in `__init__.py` runs on *every* import of the package, even
   `import camera_service.something_unrelated`. Keeping it empty keeps imports cheap and
   side-effect-free.
2. The docstring states the single most important property of this module — **no ML
   dependencies** — right where someone browsing the package will see it. That constraint is
   what keeps `Dockerfile.camera` small, and it's easy to accidentally violate by adding one
   convenience import.

> Technically, Python 3.3+ supports "namespace packages" without `__init__.py`. An explicit
> `__init__.py` is still preferred: it's unambiguous, it makes the package findable by tooling,
> and it prevents a stray directory on `sys.path` from silently merging into your package.

---

## `camera_service/main.py`

The whole simulator: configuration, logging, middleware, and four endpoints, in 272 lines with
**zero dependency on the `app` package**. That independence is deliberate and load-bearing —
`Dockerfile.camera` only copies `camera_service/`, so any `from app...` import would crash the
container at startup.

### The module docstring (lines 1–16)

```python
"""Camera simulator service.

Stands in for real parking-lot cameras. For every configured car park it exposes
an endpoint ending in ``/api/takephoto`` that returns a *random* supplied JPEG
encoded as base64.
...
Configuration (environment variables):
  IMAGES_DIR      Directory containing the supplied JPEGs. Default ``./images``.
  NUM_CARPARKS    Number of car parks to expose cameras for (10..99). Default 24.
  SERVICE_NAME    Logging service label. Default ``smartpark-camera``.
  LOG_LEVEL       Log level. Default ``INFO``.
"""
```

The docstring documents the **full environment contract** in one place. For a containerised
service the env vars *are* the public API — someone writing the Kubernetes manifest needs this
list and should not have to grep for `os.getenv`.

### `from __future__ import annotations` (line 18)

```python
from __future__ import annotations
```

This makes **all type annotations lazy**: Python stores them as strings instead of evaluating
them at function-definition time. Three concrete benefits here:

- You can write modern syntax like `str | None` and `list[Path]` even on older interpreters,
  because the annotation is never actually executed.
- It removes a tiny amount of import-time work (annotations aren't built into objects).
- It prevents `NameError` from forward references (using a class in its own method signature).

It's in **every** module of this project for consistency.

### Imports (lines 20–33)

```python
import asyncio          # to_thread — offload the blocking file read
import base64           # encode JPEG bytes for JSON transport
import datetime as _dt  # UTC timestamps in log lines
import json             # serialise each log record to one JSON line
import logging          # stdlib logging, reconfigured with a JSON formatter
import os               # os.getenv for configuration
import random           # random.choice to pick an image
import sys              # sys.stdout for the log handler
import time             # time.perf_counter for latency measurement
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
```

Note the alias `import datetime as _dt`. The leading underscore signals "module-private" and
avoids shadowing the common local variable name `datetime`.

`time.perf_counter` rather than `time.time`: `perf_counter` is a **monotonic** high-resolution
clock, so measured latency can't go negative or jump if NTP adjusts the wall clock mid-request.
`time.time` is still the right choice for *timestamps* (it's absolute); `perf_counter` is the
right choice for *durations*.

### Car-park bounds (lines 35–39)

```python
MIN_CARPARKS = 10
MAX_CARPARKS = 99
```

Named constants rather than magic numbers buried in an `if`. They mirror the validator in
`app/config.py` so the API and camera agree on the same configuration envelope. The comment
above them cites the assignment section, so a future reader knows the bound is a requirement,
not an arbitrary choice.

The two IDs are derived from the same format (`CBD_%03d`) on both sides — see
`_carpark_id` below — which is why 99 is the ceiling: a 3-digit id would break the zero-padded
two-character format.

### `_parse_num_carparks` (lines 42–55)

```python
def _parse_num_carparks(raw: str) -> int:
    """Parse & validate NUM_CARPARKS, failing fast with a clear startup error."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"NUM_CARPARKS must be an integer, got {raw!r}")
    if not MIN_CARPARKS <= value <= MAX_CARPARKS:
        raise SystemExit(
            f"NUM_CARPARKS must be between {MIN_CARPARKS} and {MAX_CARPARKS} "
            f"(inclusive), got {value}"
        )
    return value
```

Line by line:

- **`int(raw)` in a `try`.** `int("abc")` raises `ValueError`; `int(None)` raises `TypeError`.
  Both are caught because `os.getenv` can hand back either a string or `None`.
- **`raise SystemExit(...)` rather than `ValueError`.** `SystemExit` propagates straight out of
  the interpreter and sets a non-zero exit code *without* printing a Python traceback. The
  operator sees one clean line, and the container immediately enters `CrashLoopBackOff` — which
  is exactly the signal you want for a misconfiguration. Compare the alternative: booting with a
  silently wrong value and serving 404s for half your car parks.
- **`{raw!r}`** uses `repr()`, so `''` shows as `''` rather than as invisible whitespace. When
  debugging env vars this distinction matters constantly.
- **`if not MIN <= value <= MAX`.** Python's chained comparison, evaluated as
  `MIN <= value and value <= MAX` but with `value` evaluated only once.
- **Fail fast at import time.** This runs at module import (line 59), *before* uvicorn binds the
  port. A bad config can never reach a live listener.

### Reading the environment (lines 58–61)

```python
IMAGES_DIR = os.getenv("IMAGES_DIR", "./images")
NUM_CARPARKS = _parse_num_carparks(os.getenv("NUM_CARPARKS", "24"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "smartpark-camera")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

Plain `os.getenv` with defaults — no `pydantic-settings` here, because pulling in pydantic would
contradict the "minimal dependencies" goal of this service. Four settings don't justify a
settings framework.

Every value has a sensible default, so `python -m uvicorn camera_service.main:app` works with no
environment at all. `docker-compose.yml` then overrides them:

```yaml
environment:
  NUM_CARPARKS: ${NUM_CARPARKS:-24}
  IMAGES_DIR: /images
  SERVICE_NAME: smartpark-camera
  LOG_LEVEL: ${LOG_LEVEL:-INFO}
```

These are **module-level constants read once at import**. The service therefore does not pick up
env changes at runtime — correct behaviour for a container, where changing config means a new
pod.

### Image type allow-lists (lines 63–65)

```python
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_MAX_REQUEST_ID_LEN = 200
```

`_ALLOWED_SUFFIXES` is a **`set`**, not a list, because it's only ever used for membership
testing (`suffix in _ALLOWED_SUFFIXES`), which is O(1) on a set and O(n) on a list. With three
elements the difference is irrelevant to performance, but the data structure documents the
intent: this is a membership check, not an ordered sequence.

It's an **allow-list**, not a deny-list. The directory is a mounted volume that may contain
`.DS_Store`, `Thumbs.db`, `README.txt`, or a stray `.pt` file. Enumerating what's acceptable is
always safer than trying to enumerate what isn't.

`_CONTENT_TYPES` maps the suffix to the MIME type reported to the client, so a PNG isn't
mislabelled as a JPEG.

### `_sanitise_request_id` (lines 68–77)

```python
def _sanitise_request_id(raw: str | None) -> str | None:
    """Return a trusted incoming request id, or None to mint a fresh one."""
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > _MAX_REQUEST_ID_LEN:
        return None
    if not candidate.isascii() or not candidate.isprintable():
        return None
    return candidate
```

The API forwards its `X-Request-ID` so a single logical request can be traced across both
services. But that header is **attacker-controlled input that ends up in log files**, so it gets
validated:

- **`if not raw`** covers both `None` and `""` in one check (empty strings are falsy).
- **`.strip()`** removes surrounding whitespace, then the result is re-checked for emptiness —
  a header of `"   "` is not a usable id.
- **`len(...) > 200`** caps the size. Without this, a client could push a 10 MB header value into
  every log line and inflate your logging bill (or fill the disk).
- **`.isascii()` and `.isprintable()`** are the important ones. They reject control characters —
  most critically `\n` and `\r`. If a newline reached the logger, an attacker could inject a
  **fake log line** ("log forging"), e.g. making it look like an admin action occurred. They also
  reject ANSI escape sequences, which can rewrite a terminal when someone `cat`s the log.
- **Returning `None` instead of raising.** An invalid id isn't worth rejecting the request over;
  the caller just doesn't get to choose the trace id. The call site pairs this with `or
  uuid4().hex`.

This function is duplicated verbatim in `app/middleware.py`. That's a conscious trade: the
camera service must not import from `app`, and a shared package for ten lines would be
over-engineering.

### `_JsonFormatter` (lines 81–104)

```python
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
        }
```

A `logging.Formatter` subclass turns a `LogRecord` into the string that gets written. Overriding
`format()` is the documented extension point.

- **`record.created`** is the epoch float recorded when the log call happened.
  `fromtimestamp(..., tz=timezone.utc)` makes it a *timezone-aware* UTC datetime, and
  `.isoformat()` renders `2026-09-16T01:47:00.123456+00:00`. Always logging UTC with an explicit
  offset removes all ambiguity when correlating logs from pods in different regions.
- **`record.levelname`** is `"INFO"`, `"ERROR"`, etc. It's emitted under the key `severity`
  because that's the field name **Google Cloud Logging** recognises for automatic log-level
  colouring and filtering. The key names here are chosen for the log aggregator, not for Python.
- **`record.getMessage()`** applies any `%s`-style args to the format string. Using this rather
  than `record.msg` means `logger.info("hi %s", name)` renders correctly.

```python
        for key in (
            "request_id", "endpoint", "latency_ms", "status_code",
            "carpark_id", "images_dir", "num_carparks", "image_count",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)
```

When you call `logger.info("msg", extra={"carpark_id": "CBD_007"})`, the stdlib **sets those
keys as attributes on the `LogRecord`**. This loop walks an explicit allow-list of known field
names and copies any that are present.

- **`getattr(record, key, None)`** with a default avoids `AttributeError` for fields that weren't
  supplied on this particular call.
- **`if value is not None`** keeps absent fields out of the JSON rather than emitting a wall of
  `null`s.
- **`json.dumps(..., default=str)`** — `default` is the fallback called for any object `json`
  can't serialise. `str` means an unexpected type (a `Path`, a `datetime`, an exception) is
  stringified instead of blowing up **inside the logger**. A logging call that raises can take
  down a request, so this is a genuine safety net.

The allow-list approach here differs from `app/logging_config.py`, which copies *any* non-reserved
attribute. The camera's fixed list is simpler; the API's dynamic approach is more flexible.

### `_configure_logging` (lines 107–116)

```python
def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL.upper())
    return logging.getLogger("smartpark.camera")


logger = _configure_logging()
```

- **`StreamHandler(sys.stdout)`** — logs go to stdout, never to a file. This is
  [12-factor](https://12factor.net/logs) discipline: the container writes to stdout, and Docker /
  Kubernetes / Cloud Run captures and routes it. A service that manages its own log files has to
  solve rotation, permissions, and disk-full, all of which are the platform's job.
- **`root.handlers = [handler]`** — *replacing* the list rather than `addHandler()`. uvicorn
  installs its own handlers; appending would print every line twice, once JSON and once plain.
  Replacement guarantees exactly one format.
- **`root.setLevel(LOG_LEVEL.upper())`** — `setLevel` accepts a level name string, but only in
  upper case, so `LOG_LEVEL=debug` in the env still works.
- **Returning a named child logger** (`"smartpark.camera"`). Records carry the logger name, so
  you can filter by subsystem later, and child loggers propagate up to the root handler
  automatically.
- **`logger = _configure_logging()`** at module level configures logging **at import time**, so
  even messages emitted during import are JSON-formatted.

### ID generation and validation (lines 119–125)

```python
CARPARK_ID_PREFIX = "CBD"
CARPARK_ID_DIGITS = 3


def _carpark_id(index: int) -> str:
    """Must match app.services.carpark_registry.carpark_id."""
    return f"{CARPARK_ID_PREFIX}_{index:0{CARPARK_ID_DIGITS}d}"


def _valid_carpark_ids() -> set[str]:
    return {_carpark_id(i) for i in range(1, NUM_CARPARKS + 1)}
```

The id format comes straight from the assignment's COREAPI1 example output (§4.1, which shows
`CBD_001`, `CBD_042`, `CBD_015`).

`{index:0{CARPARK_ID_DIGITS}d}` is a **nested** f-string format spec: the inner `{...}` is
substituted first to give `03d`, which then means "decimal integer, zero-padded to 3
characters". So `1 → "CBD_001"`, `24 → "CBD_024"`. Writing the width as a named constant
instead of hard-coding `03d` keeps the prefix and the width defined together as one decision.

**Why zero-pad at all?** Sorting. `sorted(["CBD_1", "CBD_10", "CBD_2"])` puts `CBD_10` in the
middle because it's a string comparison. With zero padding the lexicographic order matches the
numeric order, which is why `/cameras` and the operational views come out in the order a human
expects. Three digits rather than two leaves room if the 99-car-park ceiling is ever raised.

The docstring is a **cross-service contract note**. These two services never share code, so if
someone changes one format, this comment is the only warning that every camera lookup will start
404ing.

`_valid_carpark_ids` is a **set comprehension** (braces with no colon). Again a set for O(1)
lookups — this one is checked on every single `takephoto` call.

`range(1, NUM_CARPARKS + 1)` is 1-based: car parks are numbered for humans, starting at 1.

### `_load_image_files` (lines 128–137)

```python
def _load_image_files(images_dir: str) -> list[Path]:
    directory = Path(images_dir)
    if not directory.is_dir():
        logger.error("images directory not found", extra={"images_dir": images_dir})
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    ]
    return sorted(files)
```

- **`Path`** instead of string paths gives `.suffix`, `.name`, `.read_bytes()`, and correct
  separator handling on every OS.
- **Missing directory logs and returns `[]` rather than raising.** A deliberate choice: the
  service still starts, `/health/live` still answers 200, and `/health/ready` returns 503. An
  operator can see the *service* is healthy but the *volume mount* is wrong — far more
  diagnosable than a crash loop where you can't even reach the health endpoint. (Contrast with
  `NUM_CARPARKS`, which *does* hard-exit: a bad number is unfixable without a restart, whereas a
  volume can be remounted.)
- **`p.is_file()`** excludes subdirectories, and a directory named `photos.jpg` would otherwise
  sneak through the suffix check.
- **`.suffix.lower()`** makes `PHOTO.JPG` work; mounted volumes often come from case-sensitive
  filesystems.
- **`sorted(files)`** gives a **deterministic order**. The selection is random, but the *list* is
  stable — so the same `random.seed` produces the same picks, which makes tests reproducible.
  Directory iteration order is otherwise filesystem-dependent and essentially arbitrary.
- **Scanning once at startup, not per request.** Listing a directory is a syscall per entry; doing
  it inside `takephoto` would add disk I/O to the hot path. The trade-off is that images added
  after startup aren't picked up, which is correct for an immutable container.

### `lifespan` (lines 140–153)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.valid_ids = _valid_carpark_ids()
    app.state.image_files = _load_image_files(IMAGES_DIR)
    logger.info(
        "camera simulator ready",
        extra={
            "status_code": 200,
            "images_dir": IMAGES_DIR,
            "num_carparks": NUM_CARPARKS,
            "image_count": len(app.state.image_files),
        },
    )
    yield
```

The **lifespan protocol** is FastAPI's modern replacement for the deprecated `@app.on_event`
hooks. It is an async context manager where everything before `yield` is startup and everything
after is shutdown.

- **`@asynccontextmanager`** turns an async generator into an async context manager, so you don't
  have to hand-write a class with `__aenter__`/`__aexit__`.
- **`app.state`** is a simple namespace object Starlette provides for exactly this: storing
  application-scoped singletons where request handlers can reach them via `request.app.state`.
  It's how you avoid module-level globals that tests can't replace.
- **`yield` with nothing after it** means there's no shutdown work. Correct here: the simulator
  holds no sockets, no connection pools, no file handles.
- **The startup log line** records the resolved config and the image count. When someone reports
  "the camera returns 503", this one line tells you whether `IMAGES_DIR` was wrong or the volume
  was empty.

### Creating the app (line 156)

```python
app = FastAPI(title="SmartPark Camera Simulator", lifespan=lifespan)
```

A module-level `app` object, because `uvicorn camera_service.main:app` resolves `module:attribute`.
Note the API service uses a `create_app()` factory instead — see the discussion in
[`app/main.py`](#appmainpy) — but the simulator is simple enough that the direct form is fine.

`title` shows up in the auto-generated OpenAPI docs at `/docs`.

### `_AccessLogMiddleware` (lines 159–200)

```python
class _AccessLogMiddleware:
    """Minimal pure-ASGI access logger for the camera service."""

    def __init__(self, asgi_app) -> None:
        self.app = asgi_app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
```

This is **pure ASGI middleware**, the lowest-level form. The contract is simply: be a callable
that accepts `(scope, receive, send)`.

- **`scope`** — a dict of request metadata (type, path, method, headers, query string).
- **`receive`** — an async callable to pull request body chunks.
- **`send`** — an async callable to push response messages.

The `scope["type"] != "http"` guard passes through lifespan and WebSocket traffic untouched.
Without it, the lifespan startup message would be treated as a request and
`scope["path"]` would `KeyError`.

```python
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id")
        request_id = _sanitise_request_id(
            incoming.decode("latin-1") if incoming else None
        ) or uuid4().hex
```

- ASGI headers are a **list of `(bytes, bytes)` tuples**, always lowercased by the server.
  `dict(...)` converts that to a lookup — note the `b"..."` byte-string key.
- **`.decode("latin-1")`** is the encoding the ASGI spec mandates for headers. Unlike UTF-8,
  latin-1 maps every possible byte 0–255 to a character, so **decoding can never raise** on
  malformed input. Sanitisation then rejects anything unreasonable.
- **`... or uuid4().hex`** — the classic "use theirs if valid, else mint one" idiom, relying on
  `None` being falsy. `.hex` gives `"a3f2..."` (32 chars, no dashes) rather than the dashed form.

```python
        path = scope.get("path", "")
        start = time.perf_counter()
        status_holder = {"code": 500}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode())
                )
            await send(message)
```

`send_wrapper` is the standard ASGI interception pattern: you hand the downstream app a
*replacement* `send`, observe what flows through, then forward it.

- **`status_holder = {"code": 500}`** — a **mutable container to work around closure scoping**.
  A nested function can *read* an enclosing variable but assigning to it creates a new local.
  Mutating a dict sidesteps that. (`app/middleware.py` solves the same problem with the `nonlocal`
  keyword, which is arguably cleaner; both are correct.)
- **Defaulting to 500** means that if the app crashes before sending any response, the log line
  says 500 — which is what the client will observe.
- **`http.response.start`** is the ASGI message carrying status and headers; the body arrives
  separately as `http.response.body`. Headers must be captured here, before they're sent.
- **`message.setdefault("headers", []).append(...)`** returns the existing header list or inserts
  a new empty one, then appends in a single expression. The echoed `X-Request-ID` lets a caller
  correlate its own logs with the server's.

```python
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request completed",
                extra={
                    "request_id": request_id,
                    "endpoint": path,
                    "latency_ms": latency_ms,
                    "status_code": status_holder["code"],
                },
            )
```

**`try/finally` with no `except`** — exceptions still propagate (which is what you want; the
server turns them into a 500), but the log line is guaranteed to be written either way. An
access log with holes in it is worse than useless during an incident.

`(perf_counter() - start) * 1000` converts seconds to milliseconds; `round(..., 2)` trims
float noise like `12.340000000000001`.

```python
app.add_middleware(_AccessLogMiddleware)
```

Starlette instantiates the class with the downstream app as its first argument, which matches
`__init__(self, asgi_app)`.

### Health endpoints (lines 206–238)

```python
def _health_payload() -> dict:
    image_count = len(app.state.image_files)
    ready = image_count > 0
    return {
        "status": "ok" if ready else "degraded",
        "service": SERVICE_NAME,
        "num_carparks": NUM_CARPARKS,
        "images_available": image_count,
        "ready": ready,
    }
```

One builder, three endpoints — the payload shape can never drift between them.

```python
@app.get("/health")
async def health() -> dict:
    """Combined health probe (backward-compatible alias of /health/live)."""
    return _health_payload()


@app.get("/health/live")
async def health_live() -> dict:
    """Liveness: the process is up. Always 200 while running."""
    return _health_payload()


@app.get("/health/ready")
async def health_ready() -> dict:
    """Readiness: 200 only when at least one usable image is loaded, else 503."""
    payload = _health_payload()
    if not payload["ready"]:
        raise HTTPException(
            status_code=503, detail="No usable images available on the camera"
        )
    return payload
```

**Liveness vs readiness is the key distinction**, and getting it backwards causes outages:

| Probe | Question | On failure | Must return 503 when… |
|---|---|---|---|
| **Liveness** | Is the process wedged? | Kubernetes **kills and restarts** the pod | essentially never |
| **Readiness** | Can it serve traffic *right now*? | Pod is **removed from the load balancer**, not killed | images are missing |

If liveness checked for images, a bad volume mount would cause an infinite restart loop and you'd
lose the logs that tell you why. Because readiness handles it instead, the pod stays up, stays
inspectable, and simply receives no traffic.

`/health` is kept as a plain alias so older probe configs don't break.

Raising `HTTPException` rather than returning a status code is the FastAPI idiom — it aborts the
handler and produces `{"detail": "..."}` with the right status.

**A subtlety worth knowing:** `_health_payload()` reads the *module-level* `app`, not a
request-scoped one, and `app.state.image_files` only exists after lifespan has run. In production
that's guaranteed (lifespan completes before the port accepts traffic), but a test that calls
`_health_payload()` without entering the lifespan context will get `AttributeError`. Using
`TestClient` as a context manager runs lifespan properly and avoids this.

### `GET /cameras` (lines 241–251)

```python
@app.get("/cameras")
async def list_cameras() -> dict:
    """List the takephoto endpoint for every configured car park."""
    ids = sorted(app.state.valid_ids)
    return {
        "count": len(ids),
        "cameras": [
            {"carpark_id": cid, "takephoto_url": f"/cameras/{cid}/api/takephoto"}
            for cid in ids
        ],
    }
```

A discovery endpoint — useful for debugging ("is the camera really configured for 24 car
parks?") without calling 24 separate endpoints.

`sorted(app.state.valid_ids)` is needed because **sets have no defined order**; iterating one
directly would produce output that shuffles between runs. This is exactly where the zero-padded
ID format pays off.

### `GET /cameras/{carpark_id}/api/takephoto` (lines 254–272)

The endpoint everything else exists to serve.

```python
@app.get("/cameras/{carpark_id}/api/takephoto")
async def take_photo(carpark_id: str) -> dict:
    """Return a random supplied image (base64) for the given car park's camera."""
    if carpark_id not in app.state.valid_ids:
        raise HTTPException(status_code=404, detail=f"No camera for {carpark_id}")
    if not app.state.image_files:
        raise HTTPException(status_code=503, detail="No images available on the camera")
```

`{carpark_id}` in the path becomes a function parameter — FastAPI matches by name and coerces to
the annotated type.

Two distinct failures, two distinct status codes, and the difference matters:

- **404 Not Found** — you asked for a camera that doesn't exist. *Your* fault; retrying won't
  help.
- **503 Service Unavailable** — the camera exists but the server can't serve it. *Our* fault;
  retrying after the mount is fixed will help.

```python
    chosen: Path = random.choice(app.state.image_files)
    # read_bytes() is blocking disk I/O (images are on a mounted volume, later a
    # Cloud Storage FUSE mount). Offload it so it never stalls the event loop.
    data = await asyncio.to_thread(chosen.read_bytes)
```

**This is the single most important line in the file.**

`Path.read_bytes()` is a synchronous, blocking call. Called directly inside an `async def`, it
would occupy the event loop thread for its entire duration — **freezing every other in-flight
request** in the process. On a local SSD that's maybe 1 ms and you'd never notice. But the
comment names the real scenario: on a **Cloud Storage FUSE mount**, that read is a network round
trip that can take 50–500 ms. With the API firing 20+ concurrent photo requests, serialising
them on the event loop turns a 100 ms operation into a 2-second one.

`asyncio.to_thread(fn, *args)` runs `fn` in a thread from the default `ThreadPoolExecutor` and
returns an awaitable. While the thread blocks on disk, the event loop is free to serve other
requests. Note that `read_bytes` is passed **without parentheses** — you're handing over the
function object for `to_thread` to call, not calling it yourself.

This is the same pattern used for YOLO inference and matplotlib rendering in the API. See
[Part 5](#the-event-loop-and-why-blocking-is-fatal).

```python
    content_type = _CONTENT_TYPES.get(chosen.suffix.lower(), "image/jpeg")
    return {
        "carpark_id": carpark_id,
        "filename": chosen.name,
        "content_type": content_type,
        "image_base64": base64.b64encode(data).decode("ascii"),
    }
```

- **`.get(key, default)`** never raises on an unexpected suffix, defaulting to JPEG.
- **`base64.b64encode(data)`** returns `bytes`; **`.decode("ascii")`** makes it a `str` so it can
  go into JSON. Base64 output is ASCII by definition, so this decode can't fail.
- **Why base64 at all?** JSON has no binary type. Encoding costs ~33% more bytes than raw binary,
  which is a real cost — but it buys a self-describing JSON envelope carrying `filename` and
  `content_type` alongside the pixels, and it means the API client is just `response.json()`.
  For a simulator serving a handful of images that trade is clearly worth it.
- **`filename`** is included purely for debugging: it tells you which of the supplied images
  produced a given detection result.

---

# Part 2 — The core API: foundations

## `app/__init__.py`

```python
"""SmartPark FastAPI application package."""

__version__ = "1.0.0"
```

Marks `app/` as a package and defines the **single source of truth for the version string**.
It's consumed in two places: `app/main.py` passes it to `FastAPI(version=...)` (so it appears in
the OpenAPI spec and `/docs`), and `app/api/operations.py` puts it in the health response.

A `__version__` module attribute is the long-standing Python convention (PEP 396). Defining it in
`__init__.py` means any module can `from .. import __version__` without a circular-import risk,
because `__init__.py` has no imports of its own.

---

## `app/config.py`

Centralises **all** configuration, sourced from environment variables, using `pydantic-settings`.

### Why a settings class instead of `os.getenv` everywhere?

The camera service uses bare `os.getenv` because it has four settings. The API has thirteen, and
they need types, defaults, validation, and documentation. Scattering `os.getenv("MODEL_PATH")`
across modules gives you: no type coercion (everything's a string), no validation, no single
place to see what's configurable, and no way to catch a typo'd variable name.

`Settings` gives you all of that plus IDE autocompletion and `mypy` checking.

### The imports and `Settings` class (lines 11–28)

```python
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the SmartPark API service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )
```

`BaseSettings` is a pydantic model that **populates itself from the environment**. Each field
name maps to an env var of the same name, case-insensitively — so `model_path` reads
`MODEL_PATH`.

The `model_config` options:

- **`env_file=".env"`** — also read a `.env` file if present. Environment variables take
  precedence over the file, which is the correct priority: local dev uses the file, production
  uses real env vars injected by the orchestrator, and the file is never committed.
- **`env_file_encoding="utf-8"`** — explicit, because the default depends on the OS locale.
- **`extra="ignore"`** — *don't* error on unrecognised env vars. Essential: a container's
  environment contains `PATH`, `HOSTNAME`, `KUBERNETES_SERVICE_HOST`, and dozens of others. The
  strict default would refuse to start.
- **`protected_namespaces=()`** — this one needs explaining. Pydantic v2 reserves the `model_`
  prefix for its own methods (`model_dump`, `model_validate`, …) and warns loudly if your fields
  use it. Here `model_path` and `model_max_concurrency` refer to the *ML model*, which is the
  natural domain name. Setting the protected namespaces to an empty tuple disables the warning —
  the alternative would be renaming the fields to something less clear purely to appease a linter.

### Identity and model settings (lines 30–41)

```python
    service_name: str = "smartpark-api"

    model_path: str = "./model/model.pt"
    confidence_threshold: float = 0.25
    model_max_concurrency: int = 1
```

- **`service_name`** appears in every JSON log line, so you can filter one service out of a
  shared log stream.
- **`model_path`** — note the comment: *"Never baked into the image; mounted at runtime."* Model
  weights are large binary artefacts. Baking them in bloats the image, forces a rebuild for every
  model update, and (in a real product) ships your IP to anyone who can pull the image.
  `Dockerfile` sets `MODEL_PATH=/models/model.pt` and `docker-compose.yml` mounts the volume
  there.
- **`confidence_threshold: float = 0.25`** is passed to `model.predict(conf=...)`. Detections
  below it are discarded. 0.25 is the Ultralytics default: low enough to catch genuine parking
  spaces, high enough to suppress noise. Raise it for fewer false positives, lower it for fewer
  misses.
- **`model_max_concurrency: int = 1`** — how many predictions may touch the shared model at once.
  The assignment mandates 1. It's exposed as a setting rather than hard-coded so the constraint is
  *visible* in the config surface, and so a future thread-safe model could relax it without a code
  change.

### Car parks and cameras (lines 43–50)

```python
    num_carparks: int = 24
    camera_base_url: str = "http://localhost:8001"
    http_timeout_seconds: float = 10.0
```

- **`num_carparks`** must match the camera service's `NUM_CARPARKS`, or the API will request
  cameras that return 404. `docker-compose.yml` feeds both from the same `${NUM_CARPARKS:-24}`
  shell variable, which is how you prevent that drift.
- **`camera_base_url`** defaults to localhost for bare-metal dev; compose overrides it with the
  service DNS name (`http://camera:8001`).
- **`http_timeout_seconds: float = 10.0`** — **critical for resilience.** Without a timeout, a
  camera that accepts a TCP connection and then never responds would hang the request *forever*,
  holding a connection and a coroutine. With 20 such requests you've exhausted the pool and the
  whole API stops. A bounded timeout converts an unbounded hang into a recorded per-car-park
  error.

### Operations and persistence (lines 52–68)

```python
    uuid_window_seconds: int = 30

    repository_backend: Literal["memory", "firestore"] = "memory"
    firestore_project: str | None = None
    firestore_database: str = "(default)"
```

- **`uuid_window_seconds: int = 30`** is the sliding window for "unique users recently". A
  setting, not a constant, so the dashboard's window can be tuned without a redeploy of new code.
- **`repository_backend: Literal["memory", "firestore"]`** — `Literal` restricts the value to
  exactly those two strings, and **pydantic validates it at startup**. `REPOSITORY_BACKEND=firestoer`
  fails immediately with a clear message rather than silently falling through to the in-memory
  branch and producing wrong numbers in production. The comment explains the stakes precisely:
  with multiple API pods, each in-memory store only sees its own pod's requests, so the
  "unique users" count would be wrong by a factor of however many pods you run.
- **`firestore_project: str | None = None`** — `None` is meaningful here. It tells the Firestore
  client to **infer the project from Application Default Credentials**, which is exactly what you
  want on Cloud Run or GKE. Hard-coding a project id would break promotion between dev and prod.
- **`firestore_database: str = "(default)"`** — the literal string Google uses for the unnamed
  default database. Parentheses and all.

### Logging (lines 70–71)

```python
    log_level: str = "INFO"
```

`DEBUG` in development, `INFO` or `WARNING` in production (log volume costs money at scale).

### Validators (lines 73–85)

```python
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
```

`@field_validator("field_name")` registers a function pydantic calls when that field is set. The
decorator order matters: **`@field_validator` must be on the outside, `@classmethod` on the
inside** — pydantic's docs specify this and the reverse silently misbehaves.

A validator **must return the value** (it's allowed to transform it); forgetting the `return`
sets the field to `None`.

Why `model_max_concurrency >= 1`? A `Semaphore(0)` would block forever — every `infer()` call
would hang with no way to ever acquire a permit. Negative values raise inside `asyncio` with a
much less helpful message. Catching it here turns a mysterious production hang into a startup
error naming the env var.

Validation failures happen at `Settings()` construction, which happens at import, which happens
before uvicorn binds the port. Bad config never reaches a live listener.

### A derived property (lines 87–90)

```python
    @property
    def camera_base_url_clean(self) -> str:
        """Camera base URL without a trailing slash."""
        return self.camera_base_url.rstrip("/")
```

Guards against `CAMERA_BASE_URL=http://camera:8001/` producing
`http://camera:8001//cameras/...`. A double slash usually still works but it's ugly in logs and
some proxies normalise it inconsistently.

(In practice `CarParkRegistry.__init__` does its own `.rstrip("/")`, so this property is currently
belt-and-braces.)

### `get_settings` and the `lru_cache` trick (lines 93–96)

```python
@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (evaluated once per process)."""
    return Settings()
```

Small function, three jobs:

1. **Singleton via `@lru_cache`.** The decorator caches the return value against the arguments;
   with no arguments, there's one cache slot, so `Settings()` is constructed exactly once and
   every later call returns the same object. Without it, each `Depends(get_app_settings)` on each
   request would re-read the environment, re-parse `.env` from disk, and re-run validators — real
   overhead on a hot path.
2. **Deferred construction.** `Settings()` is *not* executed at import time; it runs on first
   call. That means a test can set env vars before touching the app.
3. **A testing seam.** `get_settings.cache_clear()` drops the cached instance so a test can
   construct fresh settings with different env vars. This is the standard FastAPI pattern.

---

## `app/logging_config.py`

Structured JSON logging, with per-request fields attached automatically.

### Why JSON logs?

Plain-text logs require regex to query: *"show me every request from uuid X that took over 500 ms"*
becomes an unmaintainable pattern. With JSON, each line is a document with typed fields, and log
aggregators (Cloud Logging, Elasticsearch, Datadog) let you query
`uuid="X" AND latency_ms>500` directly. The one-line-per-object rule matters too: multi-line
output would be split into separate records by the collector.

### The context variables (lines 18–28)

```python
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
uuid_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "uuid", default=None
)
endpoint_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "endpoint", default=None
)
```

**`contextvars` are the heart of the logging design**, so it's worth understanding properly.

The problem: you want *every* log line emitted while handling a request to carry that request's
id — including lines from deep inside `inference.py` or `camera_client.py`, which have no
`Request` object. The naive fixes are all bad:

- **Thread-locals** don't work. Async code multiplexes many requests onto one thread, so all
  requests would share (and overwrite) the same slot.
- **Passing a context object through every function signature** pollutes every API in the
  codebase with a parameter that only logging cares about.

`contextvars.ContextVar` (PEP 567) solves this. A `ContextVar` holds a value that is scoped to the
current **execution context** — and crucially, `asyncio` gives each task its own context and
**copies the current context when a task is created**. So:

- Request A's coroutine sets `request_id_ctx` to `"aaa"`.
- Request B's coroutine, running interleaved on the same thread, sets it to `"bbb"`.
- Each sees only its own value. No locking, no collisions.
- Any coroutine spawned from A (via `asyncio.gather`, for example) inherits `"aaa"` automatically.

`default=None` means reading outside a request scope returns `None` rather than raising
`LookupError`. That matters for startup and shutdown logs.

`ContextVar("request_id", ...)` — the string is a human-readable name used in `repr()` for
debugging; it's not a lookup key.

### The service name global (lines 30–45)

```python
_SERVICE_NAME = "smartpark"


def set_service_name(name: str) -> None:
    """Set the ``service`` field emitted on every log line."""
    global _SERVICE_NAME
    _SERVICE_NAME = name
```

A module-level global (leading underscore = private) with a setter, because the formatter needs
it on every line and reading settings inside `format()` would be wasteful. `global` is required
to rebind a module-level name from inside a function.

### `_RESERVED_LOGRECORD_KEYS` (lines 32–39)

```python
_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}
```

This enables the flexible-field behaviour. A `LogRecord` object mixes stdlib bookkeeping
attributes with whatever the caller passed via `extra={...}` — they're indistinguishable by type.
This set enumerates the stdlib ones so the formatter can treat **everything else** as a
user-supplied structured field.

The payoff: `logger.info("msg", extra={"anything": "at all"})` works without registering the field
name anywhere. Contrast the camera service's fixed allow-list, which requires editing the
formatter to add a field.

### `JsonFormatter.format` (lines 48–80)

```python
class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = _dt.datetime.fromtimestamp(
            record.created, tz=_dt.timezone.utc
        ).isoformat()

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "severity": record.levelname,
            "service": _SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }
```

Same base fields as the camera service, plus **`logger: record.name`** — the dotted logger name
(`smartpark.inference`, `smartpark.access`, `smartpark.camera`). That's what lets you filter to
one subsystem, which is more valuable in the API where there are six distinct loggers.

```python
        if (rid := request_id_ctx.get()) is not None:
            payload["request_id"] = rid
        if (uuid := uuid_ctx.get()) is not None:
            payload["uuid"] = uuid
        if (endpoint := endpoint_ctx.get()) is not None:
            payload["endpoint"] = endpoint
```

**This is the magic.** Every log line, from anywhere in the codebase, is automatically decorated
with the current request's context. No call site has to pass anything.

The `:=` **walrus operator** (PEP 572, Python 3.8+) assigns and tests in one expression. Without
it you'd need two lines per field:

```python
rid = request_id_ctx.get()
if rid is not None:
    payload["request_id"] = rid
```

Note `is not None` rather than a truthiness test — an empty-string uuid should still be recorded
as present rather than silently dropped.

```python
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOGRECORD_KEYS and key not in payload:
                payload[key] = value
```

Walks every attribute on the record, skipping the stdlib ones and anything already set. The
second condition, `key not in payload`, means **context values win over `extra` values** — a
caller can't accidentally clobber the real request id.

```python
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)
```

`record.exc_info` is populated by `logger.exception(...)` or `logger.error(..., exc_info=True)`.
`formatException` (inherited from the base `Formatter`) renders the full traceback as a string, so
it becomes **one JSON field on one line** rather than 20 unattached lines the aggregator can't
correlate.

### `configure_logging` (lines 83–98)

```python
def configure_logging(level: str = "INFO", service_name: str = "smartpark") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    set_service_name(service_name)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
```

**"Idempotent"** — calling it twice produces the same state, because `root.handlers = [handler]`
*replaces* rather than appends. That matters because `app/main.py` calls it twice on purpose (once
in `create_app`, once in `lifespan`).

The uvicorn loop is the interesting part. uvicorn installs its own handlers with its own
plain-text format, which would give you a log stream that's half JSON and half not — breaking
every downstream parser. The fix has two halves:

- **`logger.handlers = []`** removes uvicorn's own output path.
- **`logger.propagate = True`** lets records bubble up to the root logger, where *our* JSON
  handler picks them up.

Result: uvicorn's startup banner and access logs come out as JSON too.

### `get_logger` (lines 101–103)

```python
def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper mirroring ``logging.getLogger``."""
    return logging.getLogger(name)
```

A thin alias. Its value is **indirection**: every module in the app imports `get_logger` from
here, so if the logging strategy ever changes (adding a default prefix, switching to `structlog`),
there's exactly one function to change instead of thirty import lines.

---

## `app/middleware.py`

Wraps every HTTP request to establish the logging context and emit one access-log line.

### Why pure ASGI instead of `BaseHTTPMiddleware`? (lines 1–13)

This is the most important design decision in the file, and the docstring calls it out:

> Implemented as *pure ASGI* middleware (rather than Starlette's `BaseHTTPMiddleware`) so the
> contextvars we set here reliably propagate into the route handler — `BaseHTTPMiddleware` runs
> the endpoint in a separate task, which breaks contextvar propagation.

Starlette's `BaseHTTPMiddleware` is friendlier (you get a `Request` object and write
`response = await call_next(request)`), but internally it runs the downstream app in a
**separate `asyncio` task** connected by a memory stream. And `contextvars` propagate *into* newly
created tasks by copy, not *back out* — worse, the middleware's own context isn't the one the
endpoint task inherits in the way you'd need here.

The practical consequence: with `BaseHTTPMiddleware`, `request_id_ctx.set(...)` in the middleware
would **not** be visible to `logger.info(...)` inside the route handler. Every log line from the
business logic would lose its request id — which defeats the entire point of the logging design.

Pure ASGI middleware runs **in the same task** as the endpoint, so contextvars work exactly as
intended. The cost is dealing with raw `scope`/`receive`/`send`, which is a fair price.

### `_sanitise_request_id` (lines 31–47)

```python
_MAX_REQUEST_ID_LEN = 200


def _sanitise_request_id(raw: str | None) -> str | None:
    """Return a trusted incoming request id, or None to mint a fresh one.

    Accept a non-empty, reasonably short, printable ASCII value so a caller (or
    upstream gateway) can supply its own trace id; otherwise reject it.
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > _MAX_REQUEST_ID_LEN:
        return None
    if not candidate.isascii() or not candidate.isprintable():
        return None
    return candidate
```

Identical to the camera service's version — see
[that explanation](#_sanitise_request_id-lines-6877) for the log-injection reasoning. The
duplication is intentional: `camera_service` must not import from `app`.

### The middleware class (lines 50–67)

```python
class RequestContextMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id")
        request_id = _sanitise_request_id(
            incoming.decode("latin-1") if incoming else None
        ) or uuid4().hex
        path = scope.get("path", "")
        method = scope.get("method", "")
```

Same ASGI shape as the camera's logger. The non-HTTP passthrough is essential — without it,
lifespan startup would crash on the missing `path` key.

This reuse of an inbound `X-Request-ID` is the **receiving half** of distributed tracing;
`camera_client.py` implements the sending half.

### Extracting the `uuid` query parameter (lines 69–71)

```python
        query = scope.get("query_string", b"").decode("latin-1")
        uuid_values = parse_qs(query).get("uuid", [])
        uuid_value = uuid_values[0] if uuid_values else None
```

The `uuid` query parameter identifies the caller/session, and putting it in the logging context
lets you trace one user's entire journey across many requests.

- `query_string` is `bytes` in the ASGI scope, hence the decode.
- `parse_qs("uuid=abc&n=3")` returns `{"uuid": ["abc"], "n": ["3"]}` — **values are lists**,
  because a parameter can legitimately appear multiple times (`?tag=a&tag=b`).
- `.get("uuid", [])` then `[0] if ... else None` takes the first occurrence, or `None` when the
  parameter is absent. Doing it in the middleware means even a request that 422s on validation
  still has its uuid logged.

### Setting the context (lines 73–78)

```python
        rid_token = request_id_ctx.set(request_id)
        uuid_token = uuid_ctx.set(uuid_value)
        ep_token = endpoint_ctx.set(path)

        status_code = 500
        start = time.perf_counter()
```

**`ContextVar.set()` returns a `Token`**, and that token is what `reset()` needs to restore the
previous value. Keeping them is not optional bookkeeping — without the reset in the `finally`
block, values would leak between requests handled by the same task, so request B could log
request A's id.

`status_code = 500` as the pessimistic default, same reasoning as the camera service.

### `send_wrapper` and `nonlocal` (lines 80–86)

```python
        async def send_wrapper(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode()))
            await send(message)
```

**`nonlocal status_code`** declares that assignment targets the enclosing function's variable
rather than creating a new local one. This is the cleaner solution to the closure-scoping problem
the camera service solves with a mutable dict.

(`global` does the same thing for module-level names; `nonlocal` is for enclosing *function*
scopes.)

Note `headers` here shadows the outer `headers` dict from line 61 — harmless, since the outer one
isn't needed again, but worth spotting when reading.

### The try/except/else/finally block (lines 88–110)

```python
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            _logger.exception(
                "request failed",
                extra={"method": method, "status_code": 500, "latency_ms": latency_ms},
            )
            raise
        else:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            _logger.info(
                "request completed",
                extra={
                    "method": method,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                },
            )
        finally:
            request_id_ctx.reset(rid_token)
            uuid_ctx.reset(uuid_token)
            endpoint_ctx.reset(ep_token)
```

All four clauses, each earning its place:

- **`try`** runs the rest of the application.
- **`except Exception`** catches an unhandled error, logs at ERROR with the full traceback via
  `_logger.exception(...)` (which automatically attaches `exc_info`), then **`raise` re-raises**.
  Re-raising is critical: swallowing the exception here would leave the client hanging with no
  response at all. The middleware observes and records; it does not handle.
- **`else`** runs only when no exception occurred, logging at INFO with the *real* status code.
  Splitting success and failure into `else`/`except` means the severity is right automatically —
  errors are ERROR with a traceback, successes are INFO.
- **`finally`** resets all three contextvars **no matter what happened**. This is the cleanup that
  prevents cross-request context leakage.

Note `except Exception`, not `except BaseException`: `KeyboardInterrupt` and `SystemExit` derive
from `BaseException` and should pass straight through so shutdown isn't logged as a request
failure.

---

## `app/dependencies.py`

Five tiny functions that are the **seam between the application's singletons and its routes**.

### Why this file exists (lines 1–8)

> Services are constructed once in the app lifespan and stashed on `app.state`. These providers
> surface them to routes via `Depends` and — the main reason they exist — give tests a single seam
> to override with fakes (`app.dependency_overrides[...] = ...`) without touching the real model
> or network.

FastAPI's dependency injection is the point. A route could reach `request.app.state.inference`
directly, but then a test would have to construct a real `InferenceService` (loading a ~50 MB
YOLO model) just to test the ranking logic.

With `Depends`, a test writes one line:

```python
app.dependency_overrides[get_inference_service] = lambda: FakeInference()
```

and the route receives the fake with no changes to production code. Every test in `tests/` relies
on this.

### The settings provider (lines 21–22)

```python
def get_app_settings() -> Settings:
    return get_settings()
```

Wraps the `lru_cache`d `get_settings` so routes depend on a *function in this module*. That gives
tests one consistent override target for every injected dependency, instead of some coming from
`config` and some from `dependencies`.

### The `app.state` providers (lines 25–52)

All four follow one shape:

```python
def get_registry(request: Request) -> CarParkRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Car-park registry unavailable")
    return registry
```

- **`request: Request`** — FastAPI recognises the type annotation and injects the current request.
  `request.app` is the FastAPI instance; `request.app.state` is the namespace lifespan populated.
- **`getattr(..., None)`** rather than attribute access, so a missing attribute yields `None`
  instead of `AttributeError` (which would surface as an opaque 500).
- **`raise HTTPException(503)`** — "Service Unavailable" is the honest code: the request is fine,
  the server isn't ready. A 500 would suggest a bug and wouldn't tell a load balancer to retry.

The four providers cover `registry`, `repository`, `camera_client`, and `inference`.

The last one is the one that actually fires in practice:

```python
def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Inference service unavailable (model not loaded)"
        )
    return service
```

`app/main.py` deliberately sets `app.state.inference = None` and lets startup continue if the
model fails to load. **This dependency is what converts that degraded state into a clean 503** on
inference routes, while `/health`, `/dashboard`, and the operational views keep working. The
detail message names the actual cause, so an operator doesn't have to guess.

---

## `app/main.py`

The composition root: it builds every singleton once, wires them together, and tears them down.

### The docstring (lines 1–7)

> The FastAPI *lifespan* builds the long-lived singletons exactly once on startup (model, HTTP
> client, registry, repository), stashes them on `app.state`, and tears the network client down on
> shutdown. This is where the assignment's "load the YOLO model once" requirement is satisfied.

### `lifespan` — startup (lines 30–69)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name=settings.service_name)
    logger = get_logger("smartpark.startup")
```

Order matters: **settings first, logging second, everything else third.** Logging has to be
configured before anything interesting happens, or the first errors you'd want to see come out
unformatted.

```python
    app.state.registry = CarParkRegistry(
        num_carparks=settings.num_carparks,
        camera_base_url=settings.camera_base_url,
    )
```

The registry is built first because it's pure computation — a dict of 24 objects, no I/O, can't
fail. The comment says "cheap and always available".

```python
    app.state.repository = _build_repository(settings, logger)
```

Delegated to a helper (below) so the lifespan stays readable.

```python
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.http_timeout_seconds)
    )
    app.state.http_client = http_client
    app.state.camera_client = CameraClient(http_client)
```

**One `AsyncClient` for the entire process lifetime.** This is the single most impactful
performance decision in the API.

A new client per request means a new TCP handshake per request (and a TLS handshake in
production) — roughly 1 RTT for TCP plus 2 more for TLS. A `find-carparks` call with n=10 makes
20 camera requests; that's 20 handshakes you simply don't need. The shared client keeps a
**connection pool** with HTTP keep-alive, so subsequent requests reuse warm connections.

It's stored on state under **two** names: `http_client` for the lifespan to close, and wrapped in
`CameraClient` for the routes to use. The raw handle is kept because `CameraClient` deliberately
doesn't own the client — it borrows it, which is what makes it trivially testable with a mock
transport.

```python
    app.state.inference = None
    try:
        app.state.inference = InferenceService.load(
            model_path=settings.model_path,
            max_concurrency=settings.model_max_concurrency,
            confidence=settings.confidence_threshold,
        )
        logger.info(
            "startup complete",
            extra={"num_carparks": settings.num_carparks, "model_loaded": True},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "model load failed; inference routes will return 503",
            extra={"model_path": settings.model_path, "error": str(exc)},
        )
```

**The model is loaded exactly once, here** — satisfying the assignment's core requirement. Loading
takes seconds and tens of MB of RAM; doing it per request would be catastrophic.

The **graceful-degradation pattern** is the subtle part:

- `app.state.inference = None` is set *first*, so the attribute always exists.
- The load is attempted inside `try`.
- On failure the app **still starts**. It logs an error naming `model_path` (the usual culprit is
  a missing volume mount) and carries on.

Why not crash? Because a crashed container tells you almost nothing. A running one in a degraded
state lets you hit `/health` (reports `model_loaded: false`), read the logs, see the dashboard,
and query the operational endpoints. You get a diagnosable system instead of a restart loop. The
inference routes cleanly return 503 via `get_inference_service`.

`# noqa: BLE001` suppresses the "blind except" lint warning, documenting that the broad catch is
intentional — `YOLO()` can raise almost anything depending on what's wrong with the file.

### `lifespan` — shutdown (lines 71–76)

```python
    try:
        yield
    finally:
        await http_client.aclose()
        await app.state.repository.close()
        logger.info("shutdown complete")
```

`yield` hands control to the running application; execution resumes here on shutdown.

**`try/finally` around the `yield`** guarantees cleanup even if the app raises during its
lifetime.

- **`await http_client.aclose()`** closes every pooled connection. Skipping it leaks sockets and
  prints `Unclosed client session` warnings.
- **`await app.state.repository.close()`** releases storage resources. It's a no-op for the
  in-memory repository (the base class provides an empty implementation) and closes the gRPC
  channel for Firestore. The caller doesn't need to know which — that's the abstraction working.
- Both are `await`ed because closing network resources involves I/O.

### `_build_repository` (lines 79–101)

```python
def _build_repository(settings, logger) -> RequestRepository:
    """Construct the configured repository backend.

    Importing the Firestore implementation lazily keeps ``google-cloud-firestore``
    off the import path for the in-memory/dev/test case.
    """
    if settings.repository_backend == "firestore":
        from .services.firestore_repository import FirestoreRequestRepository
        ...
        return FirestoreRequestRepository(
            project=settings.firestore_project,
            database=settings.firestore_database,
        )

    logger.info("using in-memory repository (single-pod)")
    return InMemoryRequestRepository()
```

A **factory function** implementing the strategy pattern. The return type is the abstract
`RequestRepository`, so callers are guaranteed not to depend on which one they got.

**The lazy import is the notable bit.** `from .services.firestore_repository import ...` sits
*inside* the `if`, so `google-cloud-firestore` and its gRPC stack are only imported when actually
configured. Benefits: faster startup in dev, and tests run without the package installed at all.

The log line records which backend was chosen and includes
`settings.firestore_project or "<ADC-default>"` — a nice touch, since `None` in a log is
ambiguous, whereas `<ADC-default>` says "we're intentionally letting credentials supply this".

### `create_app` (lines 104–118)

```python
def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name=settings.service_name)

    app = FastAPI(
        title="SmartPark API",
        version=__version__,
        description="FIT3184 SmartPark — find available car parks via YOLO detection.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(core.router)
    app.include_router(operations.router)
    return app


app = create_app()
```

The **application factory pattern**: a function that builds and returns a fresh app. Tests can
call `create_app()` for an isolated instance rather than sharing one global.

`configure_logging` is called here *as well as* in the lifespan. Deliberate, and the comment says
why: *"Configure logging eagerly too, so logs emitted before lifespan are JSON."* `create_app()`
runs at import; lifespan runs later, when uvicorn starts the app. Anything logged in between
would otherwise be unformatted. The function is idempotent, so the second call is harmless.

`app.add_middleware(RequestContextMiddleware)` wraps the whole app. With a single middleware,
ordering isn't yet a concern — but note that Starlette applies middleware in reverse order of
registration, so the last one added is outermost.

The two routers keep business endpoints (`api/core.py`) and operational endpoints
(`api/operations.py`) in separate files with separate OpenAPI tags, so `/docs` groups them
sensibly.

`app = create_app()` at module level is what `uvicorn app.main:app` resolves.

---

# Part 3 — The core API: data & services

## `app/models/__init__.py`

```python
"""Pydantic data models / API schemas."""
```

Package marker, empty by design. See the discussion under
[`camera_service/__init__.py`](#camera_service__init__py).

---

## `app/models/schemas.py`

Every wire contract in one file: request bodies, response bodies, and persisted records.

### Why pydantic models rather than plain dicts?

Four things you get for free:

1. **Validation** — a wrong type is caught at the boundary with a clear 422, not three layers
   deep as an `AttributeError`.
2. **Serialisation** — FastAPI converts models to JSON automatically, including nested models.
3. **Documentation** — every field and `description=` flows into the OpenAPI schema and renders
   in `/docs`.
4. **Type safety** — `status.empty_count` is checkable by an IDE and `mypy`; `status["empty_count"]`
   is not.

Keeping them in one module means the entire API surface can be reviewed on one screen.

### `CarParkInfo` (lines 15–20)

```python
class CarParkInfo(BaseModel):
    """Static description of a configured car park."""

    id: str
    name: str
    camera_url: str
```

The static catalogue entry produced by `CarParkRegistry`. Immutable configuration — no counts, no
timestamps, nothing that changes at runtime.

### `CarParkResult` (lines 23–41)

```python
class CarParkResult(BaseModel):
    carpark_id: str
    name: str | None = Field(
        default=None, description="Human-readable car park name, if known."
    )
    available_spaces: int = Field(
        description="Number of detected 'empty' spaces (greatest-first ranking key).",
    )
    confidence_score: float = Field(
        description="Mean confidence of 'empty' detections, or 0.0 when none.",
    )
```

One entry in a `find-carparks` response. The docstring notes the field names **mirror the
assignment's COREAPI1 example output exactly** — that's a contract, not a preference.

`Field(...)` attaches metadata. With `default=None` it's optional; with only a `description` it
stays required but gains documentation. Those descriptions are the text a consumer reads in
`/docs`, which is why they explain semantics (*"mean confidence of 'empty' detections"*) rather
than restating the type.

### `FindCarParksResponse` (lines 44–63)

```python
class FindCarParksResponse(BaseModel):
    uuid: str
    status: str = Field(default="success", description="'success' or 'error'.")
    msg: str = Field(default="success", description="Error message if there is an issue.")
    speed_inference: str = Field(
        description='Total model inference time for the request, e.g. "123.4 ms".',
    )
    requested_n: int = Field(description="Number of car parks requested (n).")
    queried: int = Field(description="Distinct car parks actually queried (2*n).")
    returned: int = Field(description="Number of car parks returned (<= n).")
    generated_at: str = Field(description="ISO-8601 UTC timestamp.")
    results: list[CarParkResult]
```

Note **`speed_inference: str`**, not a float. The assignment's example output shows `"xxx ms"`, so
the contract is a formatted string. It's typed `float` on `AnnotateResponse` and on
`CarParkStatus` — the inconsistency is intentional, matching two different specified shapes.

The docstring is explicit about which fields are required by the spec and which are additive:
`queried`, `returned`, and `generated_at` are extras that make the response more debuggable without
breaking the documented contract.

`results: list[CarParkResult]` — a nested model. Pydantic validates each element and FastAPI
serialises the whole tree.

### `AnnotateResponse` (lines 66–84)

```python
class AnnotateResponse(BaseModel):
    carpark_id: str
    status: str = Field(default="success", ...)
    msg: str = Field(default="success", ...)
    uuid: str | None = None
    available_spaces: int = Field(description="Number of detected 'empty' spaces.")
    empty_count: int
    occupied_count: int
    total_spaces: int
    confidence_score: float
    speed_inference: float = Field(description="Inference time in milliseconds.")
    content_type: str = "image/jpeg"
    image_base64: str = Field(description="Base64-encoded annotated JPEG.")
```

`available_spaces` and `empty_count` hold the **same value** under two names — `available_spaces`
because the spec's example uses it, `empty_count` for symmetry with `occupied_count` and
`total_spaces`. Slight redundancy in exchange for satisfying both the contract and readability.

`uuid: str | None = None` — optional here, because annotating a single car park doesn't require
identifying the caller (unlike `find-carparks`, where the uuid is mandatory).

### `CarParkStatus` (lines 87–110)

```python
class CarParkStatus(BaseModel):
    """Latest known status for a car park (persisted in the repository)."""

    carpark_id: str
    status: Literal["ok", "error"] = "ok"
    empty_count: int = 0
    occupied_count: int = 0
    total_spaces: int = 0
    confidence_score: float = 0.0
    speed_inference: float = 0.0
    last_uuid: str | None = None
    last_seen: str = Field(description="ISO-8601 UTC timestamp of last update.")
    detail: str | None = Field(
        default=None, description="Error detail when status == 'error'."
    )
```

The **persisted** record — the one model that isn't purely a wire format. It's what both
repository implementations store and what the operational views read.

`Literal["ok", "error"]` restricts the value to two strings and enforces it on load as well as on
save, so a corrupted Firestore document is caught at read time.

Zero defaults on the counts mean an error status can be constructed with just
`carpark_id`, `status="error"`, `last_seen`, and `detail` — the numeric fields sensibly default
to zero rather than requiring explicit noise at the call site.

`last_uuid` records who last triggered the update, and `detail` carries the error message (`None`
on success).

```python
    def to_result(self, name: str | None = None) -> CarParkResult:
        """Project a status onto the COREAPI1 find-carparks result shape."""
        return CarParkResult(
            carpark_id=self.carpark_id,
            name=name,
            available_spaces=self.empty_count,
            confidence_score=self.confidence_score,
        )
```

A **method on a model** — pydantic models are normal classes, so behaviour can live alongside
data. This is the internal-to-external projection: the rich internal record is narrowed to the
four fields the spec requires, dropping `occupied_count`, `detail`, and the rest.

`name` is a parameter rather than a field because the status record doesn't carry the name — only
`CarParkRegistry` knows it. The route supplies it.

### `CarParkAvailability` and `AvailabilityResponse` (lines 113–140)

```python
class CarParkAvailability(BaseModel):
    carpark_id: str
    name: str
    available_spots: int | None = Field(default=None, ...)
    total_spaces: int | None = Field(default=None, ...)
    status: Literal["ok", "error", "unknown"] = "unknown"
    last_seen: str | None = Field(default=None, ...)
```

Note the **three-state status**, adding `"unknown"` to `CarParkStatus`'s two. That's the whole
point of this model: OPS-API-1 must list *every configured* car park, including ones never
queried. Those get `status="unknown"` and `None` counts.

`int | None` rather than `0` is a meaningful distinction: `0` means "we looked and there are no
free spaces"; `None` means "we have never looked". Reporting an unqueried car park as 0 available
would be actively misleading.

### The remaining response models (lines 143–169)

```python
class StatusesResponse(BaseModel):
    count: int
    statuses: list[CarParkStatus]


class RecentUuidsResponse(BaseModel):
    window_seconds: int
    count: int
    uuids: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    model_loaded: bool
    num_carparks: int
    ready: bool = Field(default=True, ...)
```

Every collection response carries an explicit `count` alongside the list. Redundant (the client
can call `len()`), but it makes logs and manual `curl` checks readable at a glance.

`RecentUuidsResponse` echoes `window_seconds` so the client knows what window produced the number
without having to know the server's config.

`HealthResponse` separates **`model_loaded`** (a specific fact) from **`ready`** (the derived
decision). Today they're the same value, but keeping them distinct means adding a second
readiness condition later doesn't change the meaning of either field.

---

## `app/services/__init__.py`

```python
"""Service layer: model inference, camera client, registry, repository."""
```

Package marker. The docstring enumerates the four services, which is a useful table of contents
when you open the package.

---

## `app/services/carpark_registry.py`

The static catalogue: what car parks exist and where their cameras are.

### The module-level ID and name functions

```python
CARPARK_ID_PREFIX = "CBD"
CARPARK_ID_DIGITS = 3


def carpark_id(index: int) -> str:
    """Canonical id for the *1-based* car park ``index`` (e.g. 1 -> 'CBD_001')."""
    return f"{CARPARK_ID_PREFIX}_{index:0{CARPARK_ID_DIGITS}d}"


def carpark_name(index: int) -> str:
    """Human-readable name for the *1-based* car park ``index``."""
    name = _STREET_NAMES[(index - 1) % len(_STREET_NAMES)]
    cycle = (index - 1) // len(_STREET_NAMES)
    return name if cycle == 0 else f"{name} {cycle + 1}"
```

Module-level rather than methods, so they can be imported and used without constructing a
registry — the tests do exactly that to verify the id format matches the camera service's
`_carpark_id`.

`carpark_id` **is the cross-service contract**. Both services generate IDs independently and
they agree only because both use `CBD_%03d`. There is a test
(`test_camera_and_registry_ids_are_identical`) that walks all 99 possible indices and asserts
the two implementations agree, so a change to one without the other fails the suite rather than
silently 404ing every camera call.

`carpark_name` gives each car park a street name like `Market Street East`, matching the
assignment's example output where `CBD_042` carries exactly that name. `_STREET_NAMES` holds 33
entries, and the modulo/floor-division pair cycles through them: the first 33 car parks get a
bare name, the next 33 get `"<name> 2"`, and so on. That keeps every name unique up to the
99-car-park maximum without maintaining a 99-entry list.

### `CarParkRegistry.__init__` (lines 24–33)

```python
class CarParkRegistry:
    """In-memory catalogue of car parks and their camera URLs."""

    def __init__(self, num_carparks: int, camera_base_url: str) -> None:
        base = camera_base_url.rstrip("/")
        self._carparks: dict[str, CarParkInfo] = {}
        for i in range(1, num_carparks + 1):
            cid = carpark_id(i)
            self._carparks[cid] = CarParkInfo(
                id=cid,
                name=carpark_name(i),
                camera_url=f"{base}/cameras/{cid}/api/takephoto",
            )
```

- **`.rstrip("/")`** once, at construction, rather than on every URL build.
- **A `dict` keyed by id**, not a list. `get()` is the most common operation (`annotate-carpark`
  does it on every call) and dict lookup is O(1) versus a list scan. The insertion order is
  preserved (Python 3.7+), so `all()` still comes back in car-park order.
- **`self._carparks`** with a leading underscore signals "private"; access goes through the
  methods below, so the internal representation can change freely.
- **Everything is precomputed at startup.** 24 objects, built once. Building URLs per request
  would be wasted work on the hot path.
- **`camera_url` is a full absolute URL**, so `CameraClient` needs no knowledge of the camera
  service's address at all. The registry owns that mapping entirely.

### The accessors (lines 35–49)

```python
    def __len__(self) -> int:
        return len(self._carparks)

    @property
    def count(self) -> int:
        return len(self._carparks)

    def all(self) -> list[CarParkInfo]:
        return list(self._carparks.values())

    def ids(self) -> list[str]:
        return list(self._carparks.keys())

    def get(self, carpark_id: str) -> CarParkInfo | None:
        return self._carparks.get(carpark_id)
```

- **`__len__`** is the **dunder method** that makes `len(registry)` work — implementing the
  standard protocol rather than inventing a bespoke `.size()`.
- **`count`** as a `@property` is the same value with attribute syntax (`registry.count`), which
  reads better in f-strings. Two ways to ask the same question is mild redundancy, but both are
  idiomatic in different contexts.
- **`all()` and `ids()` return new lists** via `list(...)`. This is defensive copying: handing out
  the internal `.values()` view would let a caller's mutation corrupt the registry, and the view
  would also change under them if the registry ever did.
- **`get()` returns `None`** for unknown IDs rather than raising, mirroring `dict.get`. The caller
  in `api/core.py` turns that `None` into a 404.

### `sample` (lines 51–62)

```python
    def sample(self, k: int, rng: random.Random | None = None) -> list[CarParkInfo]:
        """Return ``k`` *distinct* random car parks.

        Raises ValueError if ``k`` exceeds the number of configured car parks.
        """
        if k > len(self._carparks):
            raise ValueError(
                f"Requested {k} distinct car parks but only "
                f"{len(self._carparks)} are configured"
            )
        chooser = rng or random
        return chooser.sample(self.all(), k)
```

The method `find-carparks` depends on.

- **`random.sample`, not `random.choices`.** `sample` draws **without replacement**, guaranteeing
  distinct results. `choices` samples *with* replacement and would happily return the same car
  park three times — which would mean querying one camera repeatedly and ranking duplicates.
- **The explicit `ValueError`.** `random.sample` would itself raise
  `ValueError: Sample larger than population`, but that message doesn't say *what* population.
  This version names both numbers, and the route turns it into a **400 Bad Request** with that
  text — so a caller asking for `n=20` against 24 car parks (needing 40) gets told exactly why.
- **`rng: random.Random | None = None`** is **dependency injection for randomness**. Production
  passes nothing and gets the global `random`. A test passes `random.Random(42)` and gets
  deterministic, reproducible picks. Without this seam you'd be reduced to monkey-patching the
  `random` module.
- **`chooser = rng or random`** relies on the module object being truthy and `None` being falsy.
  It works because `random.Random` instances and the `random` module expose the same `.sample`
  interface — structural typing in action.

---

## `app/services/camera_client.py`

The HTTP client that fetches photos, concurrently.

### The docstring (lines 1–11)

> A single `httpx.AsyncClient` is created at startup and reused for the process lifetime — this
> keeps the connection pool warm (HTTP keep-alive) instead of paying TCP/TLS setup on every photo…
>
> Photos across many car parks are fetched *concurrently*: the network wait for one camera overlaps
> the wait for the others, so a find-carparks call is bound by the slowest single camera rather
> than the sum of all of them.

That second paragraph quantifies the win. With 20 cameras at 50 ms each:

- **Sequential:** 20 × 50 ms = **1000 ms**
- **Concurrent:** ≈ **50 ms** (the slowest one)

A 20× improvement from one `asyncio.gather`.

### `_trace_headers` (lines 27–30)

```python
def _trace_headers() -> dict[str, str]:
    """Propagate the current request id to the camera for cross-service tracing."""
    request_id = request_id_ctx.get()
    return {"X-Request-ID": request_id} if request_id else {}
```

The **sending half** of distributed tracing. It reads the contextvar the middleware set and
forwards it, so the camera service logs the *same* request id — and you can grep both services'
logs for one id and see the complete picture of a single user request.

Returning `{}` when there's no context (startup, a script, a test) is correct: no header is better
than `X-Request-ID: None`.

This is contextvars paying off. The function takes no arguments and no caller has to thread a
trace id through `CameraClient.fetch_photo`'s signature.

### `FetchOutcome` (lines 33–43)

```python
@dataclass(slots=True)
class FetchOutcome:
    """Result of trying to fetch one car park's photo."""

    carpark: CarParkInfo
    image_bytes: bytes | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.image_bytes is not None and self.error is None
```

A **result type** — the alternative to exceptions for per-item failures in a batch.

- **`@dataclass`** generates `__init__`, `__repr__`, and `__eq__` from the annotations.
- **`slots=True`** (Python 3.10+) generates `__slots__`, which stores attributes in a fixed array
  rather than a per-instance `__dict__`. That's less memory and faster attribute access. Worth it
  here because one of these is created per car park per request. The trade-off — you can't add
  arbitrary attributes later — is a feature for a fixed-shape value object.
- **A plain `@dataclass`, not a pydantic model.** This never crosses the wire, so pydantic's
  validation overhead would buy nothing.
- **`ok` as a property** centralises the "did this succeed" logic. Callers write `if outcome.ok`
  instead of repeating the two-part condition and eventually getting it subtly wrong.

Why a result type at all? Because with 20 concurrent fetches, one failing camera must not fail the
whole request. Exceptions are all-or-nothing; a result object lets each car park carry its own
outcome and the route decide what to do.

### `CameraClient.__init__` and `fetch_photo` (lines 46–57)

```python
class CameraClient:
    """Fetches camera photos over HTTP using a shared async client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
```

**The client is injected, not created.** `CameraClient` borrows a client it doesn't own — which is
why it has no `close()` method (the lifespan closes the client it created) and why tests can pass
an `AsyncClient` with a `MockTransport` and never touch the network.

```python
    async def fetch_photo(self, camera_url: str) -> bytes:
        """Fetch and decode a single camera photo. Raises on HTTP/parse errors."""
        response = await self._client.get(camera_url, headers=_trace_headers())
        response.raise_for_status()
        payload = response.json()
        return base64.b64decode(payload["image_base64"])
```

Four lines, each doing one thing:

- **`await self._client.get(...)`** — the `await` is what makes concurrency possible. While this
  coroutine waits for the network, the event loop runs others.
- **`response.raise_for_status()`** turns a 4xx/5xx into an `httpx.HTTPStatusError`. Without it,
  a 404 body would fall through to `.json()` and fail with a confusing `KeyError` instead of a
  clear HTTP error.
- **`response.json()`** parses the body. The timeout configured on the client applies to this whole
  operation.
- **`base64.b64decode(...)`** reverses the camera's encoding, returning the original JPEG bytes.

The docstring is explicit that this **raises**, because `fetch_many` depends on that to convert
failures into `FetchOutcome`s.

### `fetch_many` — the concurrency engine (lines 59–77)

```python
    async def fetch_many(self, carparks: list[CarParkInfo]) -> list[FetchOutcome]:
        """Fetch photos for many car parks concurrently.

        A failure for one car park is captured on its FetchOutcome rather than
        raising, so one bad camera never fails the whole batch.
        """

        async def _one(carpark: CarParkInfo) -> FetchOutcome:
            try:
                image_bytes = await self.fetch_photo(carpark.camera_url)
                return FetchOutcome(carpark=carpark, image_bytes=image_bytes)
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                logger.error(
                    "camera fetch failed",
                    extra={"carpark_id": carpark.id, "error": str(exc)},
                )
                return FetchOutcome(carpark=carpark, image_bytes=None, error=str(exc))

        return await asyncio.gather(*(_one(cp) for cp in carparks))
```

**`_one` is a nested function** because it closes over `self` and is meaningful only here. It
wraps `fetch_photo` in a try/except that converts any exception into a failed `FetchOutcome`.

The `except Exception` is **deliberately broad**, and the `noqa` comment says so. Anything can go
wrong with a network call — connection refused, DNS failure, timeout, malformed JSON, a missing
key, bad base64 — and the response to all of them is identical: record it, keep going. Listing
every exception type would be a maintenance burden with no benefit.

`str(exc)` is stored so the error text reaches the client via `CarParkStatus.detail`, and logged
so it's in the aggregator with full request context attached.

**The final line is the payoff.** Breaking it down:

- **`(_one(cp) for cp in carparks)`** is a generator expression producing coroutine objects.
  Calling `_one(cp)` creates a coroutine but doesn't run it.
- **`*`** unpacks them into positional arguments.
- **`asyncio.gather(...)`** schedules them all as tasks on the event loop and returns a single
  awaitable.
- **`await`** waits for all of them.

`gather` **preserves input order** in its results regardless of completion order, which is why the
route can safely pair outcomes with the car parks it asked for.

Note there's no `return_exceptions=True` — unnecessary, because `_one` already guarantees it never
raises. Belt-and-braces would mean results typed `FetchOutcome | Exception` and a messier caller.

---

## `app/services/inference.py`

The shared YOLO model, and the discipline around touching it.

### The design constraints (lines 1–14)

> * The Ultralytics model is loaded exactly once (see the FastAPI lifespan) and shared across
>   requests.
> * `model.predict` is CPU-bound and NOT async-safe, so it must never run directly inside an
>   async route. We offload it to a worker thread with `asyncio.to_thread` and serialise access
>   with an `asyncio.Semaphore(1)` to guarantee only one prediction touches the model at a time.
>
> Heavy dependencies (ultralytics/torch, Pillow, numpy) are imported lazily so this module — and
> the pure-parsing logic in `_parse` — can be imported and unit-tested without those packages
> installed.

Everything in the file follows from those three points.

### Imports and the class-name constant (lines 16–27)

```python
import asyncio
import io
import statistics
from dataclasses import dataclass

from ..logging_config import get_logger

logger = get_logger("smartpark.inference")

EMPTY_CLASS_NAME = "empty"
```

**Note what is *not* imported at module level: no `ultralytics`, no `torch`, no `PIL`, no `numpy`.**
Those come in lazily inside the functions that need them. Importing `torch` takes seconds and
hundreds of MB of RAM.

`EMPTY_CLASS_NAME = "empty"` is the class label the supplied model emits for a free space. A named
constant instead of a string literal buried in a comparison, because if the model is retrained
with different labels there's one place to change.

`statistics` is stdlib — used for `fmean`.

### `InferenceResult` (lines 30–39)

```python
@dataclass(slots=True)
class InferenceResult:
    """Outcome of a single prediction."""

    empty_count: int
    occupied_count: int
    total_spaces: int
    confidence_score: float
    speed_inference: float  # milliseconds
    annotated_jpeg: bytes | None = None
```

A plain dataclass with `slots=True`, same reasoning as `FetchOutcome`: internal-only, fixed shape,
created per car park per request.

`annotated_jpeg: bytes | None = None` is optional because annotation is expensive (rendering
boxes, JPEG encoding) and only `annotate-carpark` needs it. `find-carparks` skips it entirely.

The trailing `# milliseconds` comment on `speed_inference` supplies the unit, which the type can't.

### `__init__` and the semaphore (lines 42–49)

```python
class InferenceService:
    """Owns the shared model and mediates all access to it."""

    def __init__(self, model, max_concurrency: int = 1, confidence: float = 0.25):
        self._model = model
        # Semaphore(1) — only one prediction on the shared model at a time.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._confidence = confidence
```

**"Mediates all access"** is the whole design. Nothing outside this class touches `self._model`;
every path goes through `infer()`, which means the concurrency guard cannot be bypassed.

**`asyncio.Semaphore(1)`** is a counter with `acquire`/`release`. Initialised to 1, it permits
exactly one holder at a time, with the rest queuing.

Why is that necessary? Ultralytics `YOLO` objects are **not thread-safe**. Internal state
(input buffers, the `.speed` dict, cached tensors) is mutated during `predict()`. Two threads
predicting simultaneously can interleave those mutations and produce garbage results, or crash
in native code. Since inference runs in worker threads (below), something must serialise it — and
that something is this semaphore.

**Why not a `threading.Lock`?** Because a `threading.Lock` blocks the *thread*. If a coroutine on
the event loop tried to acquire it while another thread held it, the entire event loop would
freeze — the exact problem we're avoiding. `asyncio.Semaphore` **suspends the coroutine** and
lets the loop run other work while waiting. That distinction is the crux of mixing async and
threads correctly.

Note `model` is passed in rather than loaded here, so tests can inject a fake with a `.predict()`
method.

### `load` (lines 51–60)

```python
    @classmethod
    def load(
        cls, model_path: str, max_concurrency: int = 1, confidence: float = 0.25
    ) -> "InferenceService":
        """Load the Ultralytics model from disk (lazy import keeps torch optional)."""
        from ultralytics import YOLO  # imported here to avoid a hard import cost

        logger.info("loading YOLO model", extra={"model_path": model_path})
        model = YOLO(model_path)
        return cls(model, max_concurrency=max_concurrency, confidence=confidence)
```

An **alternative constructor** as a `@classmethod`: `__init__` takes a model object,
`load()` takes a path. Using `cls(...)` rather than `InferenceService(...)` means a subclass would
get its own type back.

**The lazy `from ultralytics import YOLO` inside the method** is the key line. It means the module
can be imported — and `_parse` unit-tested with a fake result object — on a machine with no
`torch` installed at all. Several tests in `tests/` depend on this.

The return type is the **string** `"InferenceService"` because the class doesn't exist yet while
its own body is being executed. (`from __future__ import annotations` makes all annotations lazy
anyway, so the quotes are belt-and-braces.)

### `model_loaded` (lines 62–64)

```python
    @property
    def model_loaded(self) -> bool:
        return self._model is not None
```

Used by the health endpoint. A property rather than exposing `_model`, so callers can't reach in
and bypass the semaphore.

### `infer` — the only public entry point (lines 66–69)

```python
    async def infer(self, image_bytes: bytes, annotate: bool = False) -> InferenceResult:
        """Run prediction off the event loop, one caller at a time."""
        async with self._semaphore:
            return await asyncio.to_thread(self._run, image_bytes, annotate)
```

**Four lines that encode the entire concurrency strategy.**

- **`async with self._semaphore`** acquires a permit, suspending the coroutine if none is
  available, and releases it on exit — including if an exception is raised. The `async with` form
  is what makes the release leak-proof; a manual `acquire()`/`release()` pair would need its own
  try/finally.
- **`await asyncio.to_thread(self._run, image_bytes, annotate)`** runs the synchronous `_run` in a
  worker thread. The event loop stays free the entire time the model is chewing on the image.

Put the two together and you get: *only one prediction at a time (semaphore), and that prediction
never blocks the event loop (to_thread)*. Ten concurrent `find-carparks` requests will queue on
the semaphore, but the server continues serving `/health`, `/dashboard`, and camera fetches
throughout.

**What would go wrong without each piece?**

| Missing | Consequence |
|---|---|
| The semaphore | Concurrent threads mutate shared model state → corrupt results or a native crash |
| `to_thread` | Inference runs on the event loop → every other request freezes for its duration |
| Both | Both failure modes at once |

Note the ordering: acquire the semaphore **first**, then dispatch to a thread. The reverse would
occupy a thread-pool slot just to sit and wait for a permit.

### `_run` (lines 73–78)

```python
    def _run(self, image_bytes: bytes, annotate: bool) -> InferenceResult:
        image = self._decode(image_bytes)
        results = self._model.predict(
            image, conf=self._confidence, verbose=False
        )
        return self._parse(results[0], annotate)
```

**A `def`, not an `async def`** — it runs inside the worker thread, where normal blocking code is
exactly right. The section comment above it (`synchronous internals (run inside the worker
thread)`) marks the boundary clearly.

- **`conf=self._confidence`** applies the configured threshold inside the model, so low-confidence
  boxes never reach Python. Filtering in the model is cheaper than filtering afterwards.
- **`verbose=False`** suppresses Ultralytics' own `print()` output, which would otherwise spew
  unstructured text into the JSON log stream on every single prediction.
- **`results[0]`** — `predict()` accepts batches and returns a list. One image in, take the first
  (and only) result out.

### `_decode` (lines 80–84)

```python
    @staticmethod
    def _decode(image_bytes: bytes):
        from PIL import Image  # lazy import

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
```

- **`io.BytesIO(image_bytes)`** wraps the in-memory bytes in a file-like object, so PIL can decode
  without ever touching disk.
- **`.convert("RGB")`** normalises the colour mode. Source images may be RGBA (with transparency),
  grayscale (`L`), or palette-based (`P`); YOLO expects three channels. Without this, an RGBA PNG
  would arrive as a 4-channel array and either error or silently produce nonsense.
- **`@staticmethod`** because it uses no instance state.

### `_parse` (lines 86–123)

```python
    def _parse(self, result, annotate: bool) -> InferenceResult:
        """Turn an Ultralytics Results object into an InferenceResult.

        Kept dependency-free (no PIL/torch) except for the optional annotate
        branch, so it can be unit-tested with a lightweight fake result.
        """
        names = result.names
        empty_confidences: list[float] = []
        occupied_count = 0

        for box in result.boxes:
            class_index = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = names[class_index]
            if class_name == EMPTY_CLASS_NAME:
                empty_confidences.append(confidence)
            else:
                occupied_count += 1
```

The business logic that turns detections into counts — and it's **deliberately free of torch and
PIL types**, which is why `tests/test_inference_parsing.py` can exercise it with a hand-rolled
fake object.

- **`result.names`** maps class index → label, e.g. `{0: "empty", 1: "occupied"}`.
- **`result.boxes`** is the collection of detections that survived the confidence threshold.
- **`box.cls[0]` and `box.conf[0]`** — these are **tensors**, not plain numbers (hence the `[0]`
  indexing and the explicit `int()`/`float()` conversions). Keeping tensors around would leak
  torch types into `InferenceResult` and break JSON serialisation.
- **Collecting `empty_confidences` as a list but only counting `occupied`.** Asymmetric on
  purpose: the mean confidence is reported for empty spaces (that's the number the ranking cares
  about), so the individual values are needed. Occupied spaces only need a tally.
- **`else: occupied_count += 1`** treats every non-`empty` class as occupied — robust to a model
  that emits several vehicle classes.

```python
        empty_count = len(empty_confidences)
        total_spaces = empty_count + occupied_count
        confidence_score = (
            statistics.fmean(empty_confidences) if empty_confidences else 0.0
        )
```

- **`statistics.fmean`** rather than `mean`: `fmean` converts to float and is significantly faster,
  which is what you want for a list of floats.
- **The `if empty_confidences else 0.0` guard is required** — `fmean([])` raises
  `StatisticsError`. A car park with zero free spaces is a completely normal case, not an error.

```python
        speed = getattr(result, "speed", None) or {}
        speed_inference = round(float(speed.get("inference", 0.0)), 3)
```

Ultralytics attaches a `speed` dict like
`{"preprocess": 1.2, "inference": 45.6, "postprocess": 0.8}` (milliseconds).

**Triple defensiveness** in two lines, because this is optional metadata that varies by version:
`getattr(..., None)` handles a missing attribute, `or {}` handles it being `None`, and
`.get("inference", 0.0)` handles a missing key. Timing metadata should never be able to break a
prediction.

`round(..., 3)` keeps microsecond precision without float noise.

```python
        annotated_jpeg = self._render_annotated(result) if annotate else None

        return InferenceResult(
            empty_count=empty_count,
            occupied_count=occupied_count,
            total_spaces=total_spaces,
            confidence_score=round(confidence_score, 4),
            speed_inference=speed_inference,
            annotated_jpeg=annotated_jpeg,
        )
```

The conditional expression means `_render_annotated` — and therefore the PIL import and the JPEG
encode — is **never touched** on the `find-carparks` path. With 20 car parks per request that's 20
image encodes avoided.

`round(confidence_score, 4)` gives `0.8734` rather than `0.8734231948852539`, which is plenty of
precision and much nicer in logs and JSON.

### `_render_annotated` (lines 125–134)

```python
    @staticmethod
    def _render_annotated(result) -> bytes:
        """Encode the model's annotated frame as JPEG bytes."""
        from PIL import Image  # lazy import

        # result.plot() returns a BGR numpy array; reverse the last axis to RGB.
        rgb = result.plot()[:, :, ::-1]
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG")
        return buffer.getvalue()
```

- **`result.plot()`** is Ultralytics' built-in renderer, drawing boxes and labels onto the image
  and returning a numpy array.
- **`[:, :, ::-1]`** is the line that needs the comment. It's numpy slicing on three axes: all
  rows, all columns, and **the channel axis reversed**. Ultralytics returns **BGR** (an OpenCV
  convention) while PIL expects **RGB**, so the channels must be flipped. Skip this and the output
  looks correct in shape but has red and blue swapped — blue cars become orange. The same trick
  appears in the supplied `main.py` reference script.
- **`io.BytesIO()` as the save target** keeps everything in memory; no temp file, no cleanup, no
  disk permissions to worry about.
- **`.getvalue()`** extracts the complete JPEG bytes from the buffer.

---

## `app/services/request_repository.py`

The storage abstraction, plus the in-memory implementation.

### Why an abstract interface? (lines 1–10)

> Defines an abstract `RequestRepository` interface and a concrete in-memory implementation…
> so it can be swapped for a Firestore-backed implementation later without changing any caller.
>
> All methods are `async` for exactly that reason — the in-memory version does not need to await
> anything, but a Firestore version will.

That second paragraph explains an otherwise puzzling design choice. `InMemoryRequestRepository`
performs no I/O; its methods could be plain `def`. Making them `async` anyway means **the
interface is shaped for the slowest implementation**, so routes are written as
`await repository.upsert_status(...)` from day one. Swapping in Firestore then changes nothing at
any call site.

Had the interface been synchronous, adding Firestore would have required rewriting every route —
and you cannot `await` inside a `def`, so it would have been a genuinely invasive refactor.

### The abstract base class (lines 22–49)

```python
class RequestRepository(abc.ABC):
    """Storage interface for car-park statuses and UUID sightings."""

    @abc.abstractmethod
    async def upsert_status(self, status: CarParkStatus) -> None:
        """Insert or replace the latest status for a car park."""

    @abc.abstractmethod
    async def list_statuses(self) -> list[CarParkStatus]: ...

    @abc.abstractmethod
    async def get_status(self, carpark_id: str) -> CarParkStatus | None: ...

    @abc.abstractmethod
    async def record_uuid(self, uuid: str, at: float | None = None) -> None: ...

    @abc.abstractmethod
    async def recent_uuids(
        self, window_seconds: int, now: float | None = None
    ) -> list[str]: ...

    async def close(self) -> None:
        """Release any resources (network clients). No-op by default."""
        return None
```

- **`abc.ABC`** makes the class abstract: Python **refuses to instantiate** a subclass that hasn't
  implemented every `@abc.abstractmethod`, and it fails at construction with a message naming the
  missing methods. That's much better than discovering the gap when a route calls it.
- **The bodies are just docstrings.** No `pass` needed — a docstring is a valid body, and it
  doubles as the contract documentation.
- **`close()` is NOT abstract.** It has a default no-op implementation, so `InMemoryRequestRepository`
  doesn't have to write an empty method just to satisfy the interface, while the lifespan can still
  call `await repository.close()` unconditionally. This is the "optional hook with a sensible
  default" pattern.

**The `at` and `now` parameters** deserve a note. `record_uuid(uuid, at=None)` and
`recent_uuids(window, now=None)` both accept an optional timestamp that defaults to "now". This is
**time injection**: a test can record a sighting at `t=1000` and query at `t=1020` to verify the
window logic *instantly*, with no `time.sleep(30)`. Testing a 30-second window without this would
mean a 30-second test.

### `InMemoryRequestRepository.__init__` (lines 52–59)

```python
class InMemoryRequestRepository(RequestRepository):
    """Process-local repository backed by dicts/deques and an asyncio lock."""

    def __init__(self) -> None:
        self._statuses: dict[str, CarParkStatus] = {}
        self._uuid_sightings: deque[tuple[float, str]] = deque()
        self._lock = asyncio.Lock()
```

- **`_statuses` as a dict keyed by `carpark_id`** gives natural upsert semantics: assignment
  either inserts or replaces, exactly matching "latest status per car park".
- **`_uuid_sightings` as a `collections.deque`** of `(timestamp, uuid)` tuples. A deque is a
  double-ended queue with **O(1) appends and pops at both ends**. That's precisely the access
  pattern: append new sightings on the right, pop expired ones from the left. A list would make
  `pop(0)` an O(n) operation that shifts every remaining element.
- **`asyncio.Lock`, not `threading.Lock`** — same reasoning as the inference semaphore. An
  `asyncio.Lock` suspends the coroutine; a `threading.Lock` would block the event loop thread.

**Why lock at all in single-threaded async code?** Because `await` is a yield point. Between two
`await`s, another coroutine can run and mutate shared state. The classic hazard here is
`recent_uuids`, which prunes the deque and then reads it — without the lock, another coroutine
could append in the middle of that and produce inconsistent results. The lock makes each operation
atomic with respect to other coroutines.

### The status methods (lines 61–71)

```python
    async def upsert_status(self, status: CarParkStatus) -> None:
        async with self._lock:
            self._statuses[status.carpark_id] = status

    async def list_statuses(self) -> list[CarParkStatus]:
        async with self._lock:
            return sorted(self._statuses.values(), key=lambda s: s.carpark_id)

    async def get_status(self, carpark_id: str) -> CarParkStatus | None:
        async with self._lock:
            return self._statuses.get(carpark_id)
```

`async with self._lock` acquires and releases with exception safety.

`sorted(..., key=lambda s: s.carpark_id)` gives a stable, human-friendly order — and again, the
zero-padded IDs make the string sort match the numeric one. It also returns a **new list**, so
callers can't mutate the internal store.

### `record_uuid` (lines 73–76)

```python
    async def record_uuid(self, uuid: str, at: float | None = None) -> None:
        timestamp = time.time() if at is None else at
        async with self._lock:
            self._uuid_sightings.append((timestamp, uuid))
```

- **`time.time() if at is None else at`** — note this checks `is None` rather than falsiness,
  because `at=0.0` is a legitimate timestamp that a falsy check would wrongly replace.
- **Computing the timestamp *before* acquiring the lock** keeps the critical section as small as
  possible. Nothing that doesn't need the lock is done while holding it.
- **Append-only, with no deduplication.** The same uuid appearing ten times creates ten entries.
  Deduplication happens at read time in `recent_uuids`, which is the right place: a uuid seen at
  `t=0` and `t=25` must still count at `t=29` even though the first sighting has expired.

### `recent_uuids` (lines 78–91)

```python
    async def recent_uuids(
        self, window_seconds: int, now: float | None = None
    ) -> list[str]:
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        async with self._lock:
            # Drop sightings older than the window from the left of the deque.
            while self._uuid_sightings and self._uuid_sightings[0][0] < cutoff:
                self._uuid_sightings.popleft()
            # Distinct, preserving first-seen order within the window.
            seen: dict[str, None] = {}
            for _, uuid in self._uuid_sightings:
                seen.setdefault(uuid, None)
            return list(seen.keys())
```

The most interesting method in the file. Two phases:

**Phase 1 — prune.** The `while` loop pops expired sightings off the left end. The condition
checks `self._uuid_sightings` first (a deque is falsy when empty) before indexing `[0][0]`, which
is the timestamp of the oldest entry. Because entries are appended in time order, the moment the
oldest is inside the window, everything after it is too — so the loop can stop.

This is **lazy garbage collection**: memory is reclaimed as a side effect of reading, with no
background task to manage. It also bounds memory: the deque can never hold more than one window's
worth of sightings (after any read).

**Phase 2 — deduplicate while preserving order.** `seen: dict[str, None]` is a dict used as an
**ordered set**. Python 3.7+ dicts preserve insertion order; `set` does not. So:

- `set` would give distinct values in arbitrary order.
- `dict` gives distinct values in first-seen order.

`seen.setdefault(uuid, None)` inserts only if absent, leaving the original position intact. The
values are all `None` and meaningless — only the keys matter. `list(seen.keys())` extracts them.

The `for _, uuid in ...` unpacks each `(timestamp, uuid)` tuple, with `_` marking the timestamp as
deliberately unused.

Complexity is O(n) in the window's contents, with the amortised prune cost being O(1) per sighting
over its lifetime.

---

## `app/services/firestore_repository.py`

The shared, cross-pod implementation of the same interface.

### Why it exists (lines 1–23)

> Without this, each pod keeps its own in-memory statuses/UUID sightings and the "unique users in
> the last 30s" view is only ever correct for whichever pod happens to serve the operational
> request.

That's the concrete failure. With 3 API pods behind a load balancer, `find-carparks` calls scatter
across all three, but a `/recent-uuids` request lands on exactly one — so you'd see roughly a
third of your users and the number would jump around depending on routing.

### The data model (lines 9–18)

```
carpark_status/{carpark_id}   one doc per car park, the latest CarParkStatus
request_logs/{auto_id}        one doc per UUID sighting
```

Two collections, two different shapes:

- **`carpark_status`** is keyed by `carpark_id`, so writing the same car park twice **overwrites**.
  Exactly the upsert semantics the dict provides in the in-memory version, and the document count
  is bounded by the number of car parks.
- **`request_logs`** uses **auto-generated IDs** because every sighting is a new record, not a
  replacement. It's append-only, mirroring the deque.

*"created lazily by the first write — nothing to set up by hand"* is a genuinely useful Firestore
property: there's no schema or migration step.

### Authentication (lines 19–23)

> Authentication uses Application Default Credentials via
> `google.cloud.firestore.AsyncClient` — no JSON key file is read or shipped in the image. On
> Cloud Run / GKE this resolves to the attached service account (which needs the
> `roles/datastore.user` role).

**ADC** is the credential-resolution chain: `GOOGLE_APPLICATION_CREDENTIALS` if set, then
`gcloud auth application-default login` credentials for local dev, then the attached service
account on GCP compute.

**Never shipping a JSON key** is the important security property. A key baked into an image is
extractable by anyone who can pull it, doesn't expire, and is a common source of real-world
breaches. A workload identity has none of those problems.

### Imports and constants (lines 27–39)

```python
import inspect
import time

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from ..logging_config import endpoint_ctx, request_id_ctx
from ..models.schemas import CarParkStatus
from .request_repository import RequestRepository

STATUS_COLLECTION = "carpark_status"
REQUEST_LOGS_COLLECTION = "request_logs"
```

Note these imports are **module-level** here — safe, because `app/main.py` only imports this module
inside the `if settings.repository_backend == "firestore"` branch. The laziness lives one level up.

Collection names as constants so a typo becomes a `NameError` rather than a silently empty
collection.

### The constructor (lines 45–52)

```python
    def __init__(
        self,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        # AsyncClient integrates with the running event loop, so repository
        # methods stay non-blocking. project=None lets ADC supply the project.
        self._db = firestore.AsyncClient(project=project, database=database)
```

**`AsyncClient`, not `Client`.** The synchronous client would block the event loop on every
Firestore round trip — the same mistake as calling `read_bytes()` inline. `AsyncClient` uses
async gRPC and integrates with the running loop, so `await` actually yields.

`project=None` defers to ADC, as discussed.

### `upsert_status` (lines 54–57)

```python
    async def upsert_status(self, status: CarParkStatus) -> None:
        doc = self._db.collection(STATUS_COLLECTION).document(status.carpark_id)
        # set() (without merge) replaces the doc with the latest status.
        await doc.set(status.model_dump())
```

- **`.document(status.carpark_id)`** uses the car-park id as the document id, which is what makes
  this an upsert.
- **`set()` without `merge=True`** performs a **full replace**. That's correct here: the new status
  is complete, and a merge would leave a stale `detail` field lingering after a car park recovered
  from an error.
- **`status.model_dump()`** is pydantic v2's "model → dict" method (v1 called it `.dict()`).
  Firestore stores plain dicts.

### `list_statuses` (lines 59–65)

```python
    async def list_statuses(self) -> list[CarParkStatus]:
        statuses: list[CarParkStatus] = []
        async for snapshot in self._db.collection(STATUS_COLLECTION).stream():
            data = snapshot.to_dict()
            if data:
                statuses.append(CarParkStatus(**data))
        return sorted(statuses, key=lambda s: s.carpark_id)
```

**`async for`** iterates an *asynchronous* iterator, awaiting each batch. `stream()` pages results
from the server, so documents arrive incrementally rather than all at once — and crucially, the
event loop is free between pages.

**`CarParkStatus(**data)`** validates on the way in. A malformed document raises a pydantic
`ValidationError` at the boundary rather than producing a broken object that fails somewhere else
later.

`if data:` guards against `to_dict()` returning `None` for a deleted document.

The same `sorted(...)` as the in-memory version — **consistent ordering across implementations**
is part of the contract, even though the interface can't enforce it.

### `get_status` (lines 67–76)

```python
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
```

**`snapshot.exists`** distinguishes "document absent" from "document present but empty" —
Firestore returns a snapshot object either way, and skipping this check would mean calling
`to_dict()` on a non-existent document.

Returns `None` for a miss, matching `dict.get` and the in-memory implementation exactly.

### `record_uuid` (lines 78–89)

```python
    async def record_uuid(self, uuid: str, at: float | None = None) -> None:
        timestamp = time.time() if at is None else at
        entry = {
            "uuid": uuid,
            "ts": timestamp,
            "timestamp": _iso(timestamp),
            "request_id": request_id_ctx.get(),
            "endpoint": endpoint_ctx.get(),
        }
        await self._db.collection(REQUEST_LOGS_COLLECTION).add(entry)
```

**The timestamp is stored twice, on purpose:**

- **`ts`** — epoch float, for *querying*. Numeric range comparisons are efficient and
  index-friendly.
- **`timestamp`** — ISO-8601 string, for *humans*. Nobody reading the Firestore console can
  interpret `1758000420.123`.

**`request_id` and `endpoint` come from the contextvars.** This means every persisted sighting can
be joined back to the structured access logs — you can go from a Firestore record to the exact log
line that produced it. The comment notes they may be `None` outside a request scope (a script, a
test), which Firestore stores as null.

**`.add(entry)`** creates a document with an auto-generated id, the append-only equivalent of
`deque.append`.

### `recent_uuids` (lines 91–109)

```python
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
        return list(seen.keys())
```

**The comment about indexes is the valuable part.** Firestore requires a manually-created
*composite index* when a query filters on one field and orders by a different one. By filtering
and ordering on the **same** field (`ts`), the query is served by the single-field index Firestore
maintains automatically — so there's no `firestore.indexes.json` to deploy and no "this query
requires an index" error the first time it runs in production.

**`FieldFilter("ts", ">=", cutoff)`** is the modern filter API. The older positional form
(`.where("ts", ">=", cutoff)`) is deprecated and emits a warning.

The `seen` dict-as-ordered-set is **identical to the in-memory implementation**, which is what
keeps the two backends behaviourally interchangeable. Combined with `.order_by("ts")`, the
first-seen ordering is genuinely chronological.

Note one real difference from the in-memory version: **expired documents are never deleted**.
`request_logs` grows forever. In production you'd add a Firestore TTL policy on a timestamp field
to expire them automatically — worth knowing about.

### `close` (lines 111–116)

```python
    async def close(self) -> None:
        # AsyncClient.close() is synchronous in some versions and a coroutine in
        # others; handle both so shutdown never leaks the gRPC channel.
        result = self._db.close()
        if inspect.isawaitable(result):
            await result
```

A defensive compatibility shim. `inspect.isawaitable(result)` checks whether the return value can
be awaited; if so, await it. This handles both library versions without pinning to one.

Not closing the client leaks the gRPC channel and its background threads, which shows up as a
process that won't exit cleanly on `SIGTERM` — and then gets `SIGKILL`ed by the orchestrator.

### `_iso` (lines 119–123)

```python
def _iso(epoch_seconds: float) -> str:
    """ISO-8601 UTC string for an epoch timestamp (for human-readable logs)."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()
```

A module-level helper (not a method — it needs no state). The import is function-local, a minor
consistency with the lazy-import style used throughout.

---

## `app/services/response_cache.py`

A short-TTL, bounded, async-safe cache implementing the assignment's performance-optimisation
hint (§4.3: *"repeated requests from the same user can be cached"*).

### Why it exists

`find-carparks` is by far the most expensive endpoint: one call runs `2*n` camera fetches and
`2*n` serialised YOLO predictions. A user polling it in a loop otherwise re-does all of that for
an answer that cannot meaningfully have changed in the intervening second. Measured locally, the
repeat call drops from **1343 ms to 63 ms**.

### `TTLCache` (generic, so it isn't tied to one response type)

```python
T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float, max_entries: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple, tuple[float, T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
```

- **`Generic[T]`** with a `TypeVar` makes the cache type-safe for whatever it stores, so
  `TTLCache[FindCarParksResponse].get(...)` is known to return `FindCarParksResponse | None`
  rather than `Any`.
- **`OrderedDict`** rather than a plain dict. Both preserve insertion order in modern Python,
  but `OrderedDict` adds `popitem(last=False)` (pop the *oldest*) and `move_to_end()`, which are
  exactly what bounded eviction needs.
- **Each entry is `(expires_at, value)`** — the absolute expiry is computed once at write time,
  so reads are a single comparison rather than a subtraction.
- **`asyncio.Lock`, not `threading.Lock`** — the same rule as everywhere else in this codebase.
  The lock is only ever held by coroutines on the event loop, and blocking that thread is what
  the whole design avoids.

### Reading

```python
    async def get(self, key: tuple, now: float | None = None) -> T | None:
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
                del self._entries[key]      # lazily evict on read
                self.misses += 1
                return None
            self.hits += 1
            return value
```

- **`time.monotonic()`, not `time.time()`.** A TTL is a *duration*, and a monotonic clock can't
  jump backwards when NTP adjusts the wall clock — which would otherwise make entries appear to
  live far longer (or expire instantly).
- **`now` is injectable** for the same reason the repository accepts `at`/`now`: it lets tests
  verify expiry arithmetic instantly instead of sleeping.
- **Expiry is lazy, on read.** No background sweeper task to manage, cancel on shutdown, or
  reason about. The size cap handles anything that's never read again.
- **`enabled` is `self._ttl > 0`**, so `CACHE_TTL_SECONDS=0` turns the cache into a no-op
  without any conditional logic at the call sites.

### Writing and eviction

```python
    def _evict_locked(self, now: float) -> None:
        expired = [k for k, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            del self._entries[key]
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
```

Two-stage eviction: drop everything expired first (free, and usually enough), then evict
oldest-first until within the cap. The cap is what makes a flood of distinct UUIDs — an obvious
way to attack a naive per-user cache — bounded rather than a memory leak.

The `_locked` suffix is a convention meaning *"caller must already hold the lock"*. It's a plain
`def`, not `async def`, precisely because it must not contain an `await` that could let another
coroutine in mid-eviction.

### How the route uses it

```python
    cache_key = (uuid, n)
    hit = await cache.get(cache_key)
    if hit is not None:
        # Still record the sighting: a cached response is a real user request
        # and must count towards OPS-API-2's "users in the last 30 seconds".
        await repository.record_uuid(uuid)
        return hit.model_copy(update={"cached": True})
```

Two details worth calling out:

**The UUID is still recorded on a cache hit.** This is the subtle correctness point. OPS-API-2
counts distinct users in the last 30 seconds, derived from the request logs. If cached responses
skipped `record_uuid`, a user polling every second would vanish from the operational view after
their first call — the optimisation would silently corrupt the metric it's supposed to be
independent of.

**`model_copy(update={"cached": True})`** returns a *modified copy* of the cached pydantic model
rather than mutating the stored one. Mutating it would permanently flip the stored copy's flag,
and more importantly would mean the cache hands out a shared mutable object.

`/api/operations/cache-stats` exposes hits, misses, hit rate, and size, so the optimisation can
be demonstrated and quantified in the benchmark rather than merely asserted.

---

# Part 4 — The core API: routes & presentation

## `app/api/__init__.py`

```python
"""API routers: core (business) and operations (monitoring)."""
```

Package marker, empty. The docstring names the split: business endpoints versus monitoring
endpoints.

---

## `app/api/core.py`

The two assignment endpoints: `find-carparks` (COREAPI1) and `annotate-carpark` (COREAPI2).

### Router setup (lines 30–35)

```python
router = APIRouter(prefix="/api", tags=["core"])
logger = get_logger("smartpark.api")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
```

- **`APIRouter`** groups related routes; `main.py` mounts it with `include_router`.
- **`prefix="/api"`** is applied to every route in the file, so `@router.get("/find-carparks")`
  becomes `/api/find-carparks`. One place to change the prefix.
- **`tags=["core"]`** groups these endpoints under a "core" heading in `/docs`.
- **`_utcnow_iso`** always uses **timezone-aware UTC**. `datetime.now()` without a tz gives a naive
  local time, which is ambiguous the moment you have servers in two regions. `.isoformat()`
  includes the `+00:00` offset so the consumer can't misread it.

### `find_carparks` — the signature (lines 38–47)

```python
MAX_N_CEILING = 49


@router.get("/find-carparks", response_model=FindCarParksResponse)
async def find_carparks(
    uuid: str = Query(..., min_length=1, description="Caller/session identifier."),
    n: int = Query(
        ...,
        ge=1,
        le=MAX_N_CEILING,
        description="Number of car parks to return (the API queries 2*n).",
    ),
    registry: CarParkRegistry = Depends(get_registry),
    camera: CameraClient = Depends(get_camera_client),
    inference: InferenceService = Depends(get_inference_service),
    repository: RequestRepository = Depends(get_repository),
    cache: TTLCache = Depends(get_response_cache),
    settings: Settings = Depends(get_app_settings),
) -> FindCarParksResponse:
```

- **`response_model=FindCarParksResponse`** makes FastAPI validate and **filter** the response
  against the schema — any field not declared on the model is stripped, so internal data can't leak
  by accident. It also generates the response schema in OpenAPI.
- **`Query(...)`** — the literal `...` is Python's `Ellipsis`, which pydantic/FastAPI use to mean
  **required**. A missing `uuid` produces an automatic 422 with a precise error.
- **`ge=1, le=MAX_N_CEILING`** are validation constraints: `n` must be between 1 and 49. FastAPI
  rejects `n=0`, `n=-5`, and `n=1000` with a 422 *before your code runs* — no dependency is
  resolved, no camera contacted, no model touched. The constraints also appear in the OpenAPI
  schema. Declarative validation beats an `if` in the body.
- **`min_length=1` on `uuid`** rejects `?uuid=` (present but empty), which would otherwise be
  recorded as a real user with a blank identifier.
- **Six `Depends(...)` parameters** inject the singletons. Each is a test override point.

**Why 49?** The endpoint must query `2*n` distinct car parks, and `NUM_CARPARKS` tops out at 99
(§4.2), so 49 is the largest `n` that could ever be serviceable. It's the *absolute* ceiling;
the per-deployment limit is checked inside the route.

### The docstring (lines 49–54)

> Flow: pick 2\*n distinct car parks -> fetch their camera photos concurrently -> run inference
> (serialised on the shared model) -> rank by number of 'empty' spaces -> return the top n.
> Cameras that fail or fail to infer are recorded as errored and excluded from ranking rather than
> failing the call.

**Why 2n?** You query twice as many as requested so you have a genuine pool to rank. Querying
exactly n would mean returning whatever you got, sorted — not "the best n".

### Guarding large n (§4.3)

```python
    if n > settings.max_requested_n:
        raise HTTPException(status_code=400, detail=...)
    if 2 * n > registry.count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"n={n} requires querying {2 * n} distinct car parks but only "
                f"{registry.count} are configured (maximum n is "
                f"{registry.max_requestable_n})"
            ),
        )
```

The assignment explicitly asks *"what if the user sends a large n (n>100)?"* (§4.3). There are
two layers, answering different questions:

1. **`le=MAX_N_CEILING` on the query parameter** catches absurd values (`n=1000`) during
   validation, producing a 422 before the route body runs at all.
2. **These two checks** catch values that are structurally plausible but exceed what *this*
   deployment can serve. With 24 car parks, `n=13` needs 26 distinct ones — so it fails, and the
   message names the real maximum (12) rather than just saying "too big".

Both run **before** the cache lookup and before any camera call, so a flood of oversized
requests costs essentially nothing.

### Serving from cache (§4.3)

```python
    cache_key = (uuid, n)
    hit = await cache.get(cache_key)
    if hit is not None:
        await repository.record_uuid(uuid)
        logger.info(
            "find-carparks served from cache",
            extra={"requested_n": n, "cache_hit": True},
        )
        return hit.model_copy(update={"cached": True})
```

Keyed by `(uuid, n)`, so different users — and the same user asking for a different `n` — never
share an entry. See [`response_cache.py`](#appservicesresponse_cachepy) for the mechanics, and
for why `record_uuid` must still run on a hit.

### Sampling

```python
    k = 2 * n
    try:
        picks = registry.sample(k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

Translating the domain exception into an HTTP error at the boundary. The service layer raises a
plain `ValueError` (it knows nothing about HTTP); the route maps it to **400 Bad Request** because
asking for more car parks than exist is a client error.

**`from exc`** sets `__cause__`, preserving the original exception in the traceback. Without it
you'd see "During handling of the above exception, another exception occurred" and lose the chain.

`str(exc)` passes through the service's specific message ("Requested 40 distinct car parks but
only 24 are configured"), so the caller learns the actual limit.

### Concurrent fetch (lines 61–62)

```python
    outcomes = await camera.fetch_many(picks)
    timestamp = _utcnow_iso()
```

All 2n photos fetched concurrently in one call. One timestamp is captured for the entire batch, so
every status from this request shares a single `last_seen` — they logically happened together, and
per-car-park timestamps would imply a precision that doesn't exist.

### The `process` inner function (lines 64–97)

```python
    async def process(outcome: FetchOutcome) -> CarParkStatus:
        carpark_id = outcome.carpark.id
        if not outcome.ok:
            return CarParkStatus(
                carpark_id=carpark_id,
                status="error",
                last_uuid=uuid,
                last_seen=timestamp,
                detail=outcome.error or "camera fetch failed",
            )
```

A nested async function closing over `uuid`, `timestamp`, `inference`, and `logger` — so
`asyncio.gather` can map it over the outcomes without a long parameter list.

The camera-failure branch produces an error `CarParkStatus` rather than raising. That status is
still persisted, so the dashboard can show *why* a car park is unavailable. `outcome.error or
"camera fetch failed"` supplies a fallback message.

```python
        try:
            result = await inference.infer(outcome.image_bytes)
        except Exception as exc:  # noqa: BLE001 - isolate per-car-park failures
            logger.exception(
                "inference failed", extra={"carpark_id": carpark_id, "error": str(exc)}
            )
            return CarParkStatus(
                carpark_id=carpark_id,
                status="error",
                last_uuid=uuid,
                last_seen=timestamp,
                detail=f"inference failed: {exc}",
            )
```

Second failure mode, same treatment — one corrupt image must not fail all 20 car parks. The `noqa`
comment states the intent: *isolate per-car-park failures*.

`logger.exception` (not `.error`) captures the full traceback, which for an inference failure is
genuinely needed for diagnosis.

Note `await inference.infer(...)` is called **without `annotate=True`**, skipping the expensive
render.

```python
        return CarParkStatus(
            carpark_id=carpark_id,
            status="ok",
            empty_count=result.empty_count,
            occupied_count=result.occupied_count,
            total_spaces=result.total_spaces,
            confidence_score=result.confidence_score,
            speed_inference=result.speed_inference,
            last_uuid=uuid,
            last_seen=timestamp,
        )
```

The success path maps `InferenceResult` → `CarParkStatus`, the persisted shape.

### Running the pipeline (line 99)

```python
    statuses = await asyncio.gather(*(process(o) for o in outcomes))
```

All 2n `process` coroutines are launched concurrently. **The semaphore inside `infer` still
serialises the actual predictions**, so this doesn't violate the one-at-a-time rule — the
coroutines simply queue for their turn, and the event loop stays responsive throughout. The
error-path coroutines complete immediately without ever touching the model.

### Persistence

```python
    await repository.upsert_statuses(list(statuses))
    await repository.record_uuid(uuid)
```

**Batched.** An earlier version looped and awaited `upsert_status` once per car park, which on
Firestore meant `2*n` sequential network round trips per request. `upsert_statuses` is one call:
the Firestore implementation commits a `WriteBatch` (chunked at Firestore's 500-operation
limit), and the in-memory one takes its lock once for the whole batch instead of `2*n` times.

The base class provides a default `upsert_statuses` that simply loops, so a new backend only has
to implement the single-item version to be correct — batching is an optimisation it can opt into.

`record_uuid(uuid)` records the sighting once per request, feeding the "unique users in 30s" view.

### Ranking (lines 106–112)

```python
    successful = [s for s in statuses if s.status == "ok"]
    ranked = sorted(
        successful,
        key=lambda s: (s.empty_count, s.confidence_score),
        reverse=True,
    )[:n]
```

- **Filter first.** Errored car parks have `empty_count=0` and would sort to the bottom anyway, but
  filtering makes the intent explicit and keeps them out of the response entirely.
- **A tuple sort key** sorts by `empty_count` first and uses `confidence_score` **only to break
  ties**. Two car parks both showing 5 free spaces are ordered by how confident the model was —
  which is a genuinely better signal than arbitrary order.
- **`reverse=True`** makes it descending: most spaces first.
- **`[:n]`** takes the top n. Slicing beyond the list length is safe in Python, so if only 3 of 20
  cameras worked and `n=10`, you get 3 results rather than an error.

### The speed field (lines 114–117)

```python
    total_inference_ms = sum(s.speed_inference for s in successful)
    speed_inference = f"{total_inference_ms:.1f} ms"
```

Summed over **`successful`**, not `ranked` — this reports the total work done, including car parks
that were inferred but didn't make the cut. That's the honest measure of what the request cost.

`f"{value:.1f} ms"` formats to one decimal and appends the unit, matching the spec's `"xxx ms"`.

### Logging and the response (lines 119–143)

```python
    logger.info(
        "find-carparks completed",
        extra={
            "requested_n": n,
            "queried": len(picks),
            "succeeded": len(successful),
            "returned": len(ranked),
        },
    )
```

One structured summary line. Because of the contextvars, it automatically carries `request_id`,
`uuid`, and `endpoint` too — so you can query "requests where succeeded < queried" to find camera
problems.

```python
    def _name(carpark_id: str) -> str | None:
        info = registry.get(carpark_id)
        return info.name if info else None

    response = FindCarParksResponse(
        uuid=uuid,
        status="success",
        msg="success",
        speed_inference=speed_inference,
        requested_n=n,
        queried=len(picks),
        returned=len(ranked),
        generated_at=timestamp,
        cached=False,
        results=[s.to_result(name=_name(s.carpark_id)) for s in ranked],
    )
    await cache.set(cache_key, response)
    return response
```

`_name` is a tiny lookup helper handling the `None` case — it supplies the street name
(`Market Street East`) that the registry knows but the status record doesn't. The list
comprehension calls `to_result()` on each ranked status — the internal→external projection
defined on the model.

The response is built, **then** stored in the cache, **then** returned. Storing the finished
model (rather than the raw statuses) means a cache hit skips the ranking and projection work
too, not just the camera and inference calls.

### `annotate_carpark`

```python
@router.get("/annotate-carpark", response_model=AnnotateResponse)
async def annotate_carpark(
    carpark_id: str = Query(..., description="Car park id, e.g. 'CBD_001'."),
    uuid: str | None = Query(None, description="Optional caller identifier."),
    ...
) -> AnnotateResponse:
```

`uuid` is `Query(None, ...)` — **optional**, unlike in `find-carparks`. Annotating one car park is
a debugging/visualisation operation that doesn't need caller tracking.

```python
    carpark = registry.get(carpark_id)
    if carpark is None:
        raise HTTPException(status_code=404, detail=f"Unknown car park: {carpark_id}")
```

Validate before doing any work. **404** because the resource doesn't exist.

```python
    try:
        image_bytes = await camera.fetch_photo(carpark.camera_url)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "camera fetch failed", extra={"carpark_id": carpark_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=502, detail=f"Camera fetch failed: {exc}"
        ) from exc
```

**Here the error handling differs from `find-carparks` in an important way.** There, one camera
failing was survivable because 19 others remained. Here there's only one car park, so a failure
means the request can't be fulfilled — and it becomes an HTTP error.

**502 Bad Gateway** is precisely the right code: this server acted as a gateway to the camera
service and got an invalid response. Not 500 (which would imply *our* bug) and not 404 (the car
park exists).

```python
    try:
        result = await inference.infer(image_bytes, annotate=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "inference failed", extra={"carpark_id": carpark_id, "error": str(exc)}
        )
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
```

**`annotate=True`** — the one place it's set, triggering `_render_annotated`.

**500** here, not 502: inference happens in *this* process, so a failure is genuinely our problem.
The distinction between 502 and 500 across these two blocks tells an operator immediately which
service to investigate.

```python
    timestamp = _utcnow_iso()
    await repository.upsert_status(
        CarParkStatus(
            carpark_id=carpark_id,
            status="ok",
            ...
        )
    )
    if uuid:
        await repository.record_uuid(uuid)
```

The result is persisted just like `find-carparks`, so the dashboard reflects annotate calls too.
`if uuid:` guards the optional parameter.

```python
    annotated = result.annotated_jpeg or b""
    return AnnotateResponse(
        carpark_id=carpark_id,
        status="success",
        msg="success",
        uuid=uuid,
        available_spaces=result.empty_count,
        empty_count=result.empty_count,
        ...
        image_base64=base64.b64encode(annotated).decode("ascii"),
    )
```

`result.annotated_jpeg or b""` defends against `None` (which shouldn't happen with
`annotate=True`, but `b64encode(None)` would raise `TypeError` and turn a working response into a
500). `b64encode(b"")` returns `b""`, so the client gets an empty string rather than an error.

`available_spaces` and `empty_count` both receive `result.empty_count` — the dual naming discussed
in the schemas section.

---

## `app/api/operations.py`

The monitoring surface: statuses, recent UUIDs, availability, the plot, health probes, and the
dashboard.

### Router setup (lines 27–31)

```python
router = APIRouter(tags=["operations"])
```

**No `prefix`** here, unlike `core.py`, because this router serves paths at several levels:
`/api/operations/...`, `/api/carparks`, `/health`, and `/`. Each route declares its full path.

### `GET /api/operations/statuses` (lines 34–40)

```python
@router.get("/api/operations/statuses", response_model=StatusesResponse)
async def all_statuses(
    repository: RequestRepository = Depends(get_repository),
) -> StatusesResponse:
    """Return the latest known status for every car park queried so far."""
    statuses = await repository.list_statuses()
    return StatusesResponse(count=len(statuses), statuses=statuses)
```

A thin read-through to the repository. Note **"queried so far"** — this returns only car parks that
have actually been looked at, which is exactly what distinguishes it from the `availability`
endpoint below.

### `GET /api/operations/recent-uuids` (lines 43–51)

```python
@router.get("/api/operations/recent-uuids", response_model=RecentUuidsResponse)
async def recent_uuids(
    repository: RequestRepository = Depends(get_repository),
    settings: Settings = Depends(get_app_settings),
) -> RecentUuidsResponse:
    """Return distinct UUIDs seen within the configured sliding window (30s)."""
    window = settings.uuid_window_seconds
    uuids = await repository.recent_uuids(window)
    return RecentUuidsResponse(window_seconds=window, count=len(uuids), uuids=uuids)
```

The "unique users recently" view. The window comes from settings rather than a literal, and it's
echoed in the response so the client knows what it's looking at.

### `GET /api/carparks` (lines 54–59)

```python
@router.get("/api/carparks", response_model=list[CarParkInfo])
async def list_carparks(
    registry: CarParkRegistry = Depends(get_registry),
) -> list[CarParkInfo]:
    """List all configured car parks and their camera URLs."""
    return registry.all()
```

`response_model=list[CarParkInfo]` — a bare list is a valid response model. Useful for verifying
the registry's configuration and the camera URLs it built.

### `GET /api/operations/availability` — OPS-API-1 (lines 62–103)

```python
    latest = {s.carpark_id: s for s in await repository.list_statuses()}
```

A **dict comprehension building a lookup index**. Without it the loop below would do a linear scan
per car park (O(n²)); with it each lookup is O(1).

Note the `await` inside the comprehension — legal in an async function and evaluated once, before
iteration begins.

```python
    carparks: list[CarParkAvailability] = []
    for info in registry.all():
        status = latest.get(info.id)
        if status is None:
            carparks.append(CarParkAvailability(carpark_id=info.id, name=info.name))
        elif status.status == "ok":
            carparks.append(
                CarParkAvailability(
                    carpark_id=info.id,
                    name=info.name,
                    available_spots=status.empty_count,
                    total_spaces=status.total_spaces,
                    status="ok",
                    last_seen=status.last_seen,
                )
            )
        else:
            carparks.append(
                CarParkAvailability(
                    carpark_id=info.id,
                    name=info.name,
                    status="error",
                    last_seen=status.last_seen,
                )
            )
```

**Iterating the registry, not the repository** — that's what guarantees *every configured* car park
appears, which is the OPS-API-1 requirement. Three branches:

1. **Never queried** (`status is None`) → status defaults to `"unknown"`, counts stay `None`.
2. **Queried successfully** → real counts and timestamp.
3. **Queried but errored** → `status="error"` with a timestamp but **no counts**, because the last
   known counts would be stale and misleading.

The merge of "static catalogue" and "dynamic observations" is the whole job of this endpoint.

### `GET /api/operations/plot.png` — OPS-REQ-2 (lines 106–122)

```python
@router.get("/api/operations/plot.png", include_in_schema=False)
async def operational_plot(
    repository: RequestRepository = Depends(get_repository),
    settings: Settings = Depends(get_app_settings),
) -> Response:
    """OPS-REQ-2: on-demand matplotlib PNG of current availability + load.

    Rendering is blocking/CPU-bound, so — like inference — it is offloaded to a
    worker thread to keep the event loop responsive.
    """
    window = settings.uuid_window_seconds
    statuses = await repository.list_statuses()
    recent = await repository.recent_uuids(window)
    png = await asyncio.to_thread(
        render_availability_png, statuses, window, len(recent)
    )
    return Response(content=png, media_type="image/png")
```

- **`include_in_schema=False`** hides it from `/docs`. It's a browser-facing asset, not part of the
  JSON API, and it would clutter the spec.
- **`asyncio.to_thread(render_availability_png, ...)`** — the **third** application of the pattern,
  after camera disk reads and YOLO inference. matplotlib is pure blocking CPU work; rendering a
  24-bar chart takes 50–200 ms, and doing that on the event loop would freeze the server. Note the
  function is passed **unparenthesised** with its arguments following.
- **`Response(content=png, media_type="image/png")`** returns raw bytes with an explicit content
  type, bypassing JSON serialisation entirely.
- The return type is `Response` rather than a model, because this isn't JSON.

### Health probes (lines 125–166)

```python
def _build_health(request: Request, settings: Settings) -> HealthResponse:
    inference = getattr(request.app.state, "inference", None)
    model_loaded = inference is not None and inference.model_loaded
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service=settings.service_name,
        version=__version__,
        model_loaded=model_loaded,
        num_carparks=settings.num_carparks,
        ready=model_loaded,
    )
```

**A double check**: `inference is not None` (did the lifespan set it?) **and**
`inference.model_loaded` (does it actually hold a model?). Short-circuit evaluation means the
second is only reached when the first is true, so there's no `AttributeError` on `None`.

Crucially this accesses `app.state` **directly rather than via `Depends(get_inference_service)`** —
because that dependency *raises 503* when the model is missing, and health must be able to *report*
that state rather than fail on it.

```python
@router.get("/api/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Combined health probe (backward-compatible). Reports model-load state."""
    return _build_health(request, settings)
```

**Stacked decorators register the same function at two paths.** `/api/health` is documented;
`/health` is the conventional probe path and is hidden from the schema to avoid a duplicate entry.

```python
@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Readiness: 200 only once the YOLO model is loaded, else 503."""
    health_body = _build_health(request, settings)
    if not health_body.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return health_body
```

Same liveness/readiness split as the camera service, and here the consequences are even clearer.
Because the lifespan lets the app start without a model:

- **Liveness returns 200** → Kubernetes doesn't restart the pod, so the logs explaining the failure
  survive and you can still reach the dashboard.
- **Readiness returns 503** → the pod is pulled from the load balancer, so no user hits an
  inference endpoint that can't work.

Exactly the intended behaviour for a missing model mount.

### The dashboard routes (lines 169–173)

```python
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the simple HTML monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)
```

Stacked decorators again, so both `/` and `/dashboard` serve the monitoring UI. Making the root
path useful means someone who opens the service in a browser immediately sees something meaningful
rather than `{"detail":"Not Found"}`.

`response_class=HTMLResponse` sets `Content-Type: text/html` so the browser renders rather than
displays the markup.

---

## `app/plotting.py`

matplotlib rendering, isolated in its own module.

### The docstring (lines 1–10)

> matplotlib is a blocking, CPU-bound library, so — exactly like model inference — the render must
> never run directly on the event loop. The route offloads it with `asyncio.to_thread`… We select
> the headless 'Agg' backend so no display server / GUI is required in a container.
>
> matplotlib is imported lazily inside the function so importing this module (and running the rest
> of the app/tests) stays cheap when no plot is requested.

### Colour constants (lines 18–20)

```python
_AVAILABLE = "#3fb950"
_OCCUPIED = "#d29922"
```

Green and amber, and the comment notes they **mirror the HTML dashboard**. Since the PNG is
embedded in that dashboard, mismatched palettes would look broken.

### Function signature and the lazy backend (lines 23–36)

```python
def render_availability_png(
    statuses: list[CarParkStatus],
    window_seconds: int,
    recent_user_count: int,
) -> bytes:
```

**A synchronous `def`**, because it runs inside a worker thread. Consistent with
`InferenceService._run`.

It takes plain data and returns `bytes` — no repository, no request, no I/O. That makes it trivially
unit-testable: pass a list of statuses, assert you got PNG bytes back.

```python
    import matplotlib

    matplotlib.use("Agg")  # headless: no display/GUI needed
    import matplotlib.pyplot as plt
```

**The order here is mandatory.** `matplotlib.use("Agg")` must be called **before**
`import matplotlib.pyplot`, because pyplot selects and initialises its backend at import time.
Reversed, matplotlib would try to load a GUI backend (Tk, Qt), find no display in the container,
and either crash or emit warnings.

**"Agg"** is the Anti-Grain Geometry backend — a pure software rasteriser that writes to a buffer
with no windowing system involved. It's the correct choice for any server-side rendering.

### Building the chart (lines 38–54)

```python
    ok = [s for s in statuses if s.status == "ok"]

    fig, ax = plt.subplots(figsize=(max(6.0, len(ok) * 0.5), 4.0), dpi=100)
```

Only successful statuses are plotted — an errored car park has no meaningful counts.

**`figsize=(max(6.0, len(ok) * 0.5), 4.0)`** makes the width **adaptive**: half an inch per car
park, with a 6-inch floor. With 24 car parks that's 12 inches wide, so the labels stay legible; the
floor stops a 2-car-park chart from being a sliver. Height is fixed at 4 inches, and `dpi=100`
makes the pixel dimensions predictable.

```python
    if ok:
        labels = [s.carpark_id.replace("carpark-", "") for s in ok]
        empty = [s.empty_count for s in ok]
        occupied = [s.occupied_count for s in ok]
        positions = range(len(ok))

        ax.bar(positions, empty, color=_AVAILABLE, label="Available")
        ax.bar(positions, occupied, bottom=empty, color=_OCCUPIED, label="Occupied")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Car park")
        ax.set_ylabel("Parking spaces")
        ax.legend(loc="upper right", fontsize=8)
```

- **`.replace("carpark-", "")`** shortens labels to `01`, `02`, … Full IDs at 24 bars would overlap
  into mush.
- **`bottom=empty` on the second `bar` call** is what makes it a **stacked** chart: the occupied bar
  starts where the available bar ends, so the total height is the total spaces. That's more
  informative than side-by-side bars — you see availability *and* capacity in one glance.
- **`set_xticks` before `set_xticklabels`** is required by matplotlib; setting labels without first
  fixing the tick positions produces a warning and can misalign them.

```python
    else:
        ax.text(
            0.5, 0.5,
            "No data yet — call /api/find-carparks",
            ha="center", va="center", fontsize=11,
        )
        ax.set_axis_off()
```

The empty state. Coordinates `(0.5, 0.5)` with centred alignment put the message in the middle, and
`set_axis_off()` hides the meaningless empty axes. **This is good UX in an unexpected place** — a
blank chart looks like a bug, whereas a message that names the endpoint to call tells the user
exactly what to do.

### Title, layout, and export (lines 66–75)

```python
    ax.set_title(
        f"SmartPark availability · {recent_user_count} "
        f"user(s) in last {window_seconds}s"
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)  # release the figure so repeated calls don't leak memory
    return buffer.getvalue()
```

The title folds the **load metric** into the availability chart, satisfying OPS-REQ-2's request for
both in one artefact.

**`fig.tight_layout()`** adjusts padding so labels and the title aren't clipped — without it, long
axis labels get cut off at the figure edge.

**`plt.close(fig)` is essential and easy to forget.** matplotlib keeps a global registry of open
figures; every `plt.subplots()` adds one, and without an explicit close they accumulate forever.
The dashboard polls this endpoint every 3 seconds, so a leak here would be roughly 1,200 leaked
figures an hour and eventual OOM. The comment flags exactly this.

`savefig` to a `BytesIO` keeps everything in memory, and `.getvalue()` returns the PNG bytes.

---

## `app/dashboard.py`

The monitoring UI: HTML, CSS, and JavaScript in a single Python string.

### Why a Python string? (lines 1–6)

> Kept as a Python string so it ships inside the app image with no static-file mount or template
> engine. It is fully self-contained (no external CDN/assets) and polls the operational JSON
> endpoints from the browser.

Three real benefits:

1. **No static file serving** — no `StaticFiles` mount, no path configuration, no risk of the file
   being missing from the image.
2. **No template engine** — no Jinja2 dependency for a page with no server-side variables.
3. **No CDN** — the dashboard works in an air-gapped network and can't break because a third-party
   CDN changed or went down.

The trade-off is no syntax highlighting or linting for the embedded HTML/CSS/JS. For a single
self-contained page that's acceptable.

### The CSS (lines 14–47)

```css
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: #0f1720; color: #e6edf3;
}
```

- **`color-scheme: light dark`** tells the browser the page supports both, so native form controls
  and scrollbars match.
- **`* { box-sizing: border-box }`** — the near-universal CSS reset making padding and borders count
  *inside* an element's declared width. Without it, `width: 100%` plus padding overflows its
  container.
- **`font-family: system-ui, ...`** uses the OS's native UI font. No web font download, instant
  render, and it looks native on every platform. The fallback chain covers older browsers.
- The palette is a **dark theme** (`#0f1720` background) — appropriate for a monitoring dashboard
  that might be on a wall display.

```css
main { padding: 24px; display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
```

**CSS Grid** with a 2:1 split — the status table gets twice the width of the UUID list, reflecting
their relative information density. The media query collapses to a single column below 900 px, so
it's usable on a tablet or phone.

```css
th { color: #8aa0b6; font-weight: 600; position: sticky; top: 0; background: #111c2b; }
```

**`position: sticky; top: 0`** keeps the table header visible while scrolling — important with 24
rows in a `max-height: 60vh` container. The explicit `background` is required, or rows would show
through the header as it sticks.

```css
.bar { height: 6px; border-radius: 3px; background: #22364a; overflow: hidden; margin-top: 4px; }
.bar > span { display: block; height: 100%; background: #3fb950; }
```

A miniature progress bar built from two elements: a track and a fill whose width is set inline from
JavaScript. Pure CSS, no library.

### The HTML structure (lines 50–83)

```html
<header>
  <h1>🅿️ SmartPark Monitoring</h1>
  <span class="pill" id="health">health: …</span>
  <span class="pill" id="count">0 car parks</span>
  <span class="pill" id="refresh">next refresh …</span>
</header>
```

Three status "pills" giving an at-a-glance summary. Each has an `id` so JavaScript can update it.
The initial text (`health: …`) is a **loading state** — the page never looks broken while the first
fetch is in flight.

```html
<tbody id="rows"><tr><td colspan="8" class="muted">No data yet.</td></tr></tbody>
```

The empty state is in the HTML itself, so it shows immediately on page load rather than after the
first failed fetch.

```html
<section style="grid-column: 1 / -1;">
  <h2>Availability plot (matplotlib, on demand)</h2>
  <img id="plot" alt="Availability plot" style="width: 100%; ..." />
</section>
```

**`grid-column: 1 / -1`** spans the section across all grid columns (from line 1 to the last line),
giving the chart full width. The `alt` attribute is there for accessibility.

### The JavaScript (lines 84–158)

```javascript
const REFRESH_MS = 3000;
let countdown = REFRESH_MS / 1000;

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
```

**`esc` is an XSS guard**, and it matters. Values like `carpark_id`, `detail`, and especially
`uuid` originate from user input (`?uuid=...`) and are interpolated into HTML strings below. A uuid
of `<script>alert(1)</script>` would otherwise execute.

The implementation: a regex with the `g` flag replaces every `&`, `<`, and `>` using an arrow
function that looks each character up in an object literal. `String(s)` coerces non-strings
defensively.

Escaping `&` **first** matters conceptually — doing `<` first would produce `&lt;` and then the `&`
pass would double-escape it to `&amp;lt;`. The single-pass regex avoids that entirely.

```javascript
async function loadStatuses() {
  const res = await fetch('/api/operations/statuses');
  const data = await res.json();
  document.getElementById('count').textContent = data.count + ' car parks';
  const rows = document.getElementById('rows');
  if (!data.statuses.length) {
    rows.innerHTML = '<tr><td colspan="8" class="muted">No data yet — call /api/find-carparks.</td></tr>';
    return;
  }
```

Browser `async`/`await`, the same concept as Python's. `fetch` returns a promise for the response;
`res.json()` returns another for the parsed body — hence two `await`s.

The empty-state message names the endpoint to call, mirroring the matplotlib placeholder.

```javascript
  rows.innerHTML = data.statuses.map(s => {
    const conf = (s.confidence_score * 100).toFixed(0);
    const badge = s.status === 'ok' ? 'ok' : 'error';
    const when = s.last_seen ? esc(s.last_seen.replace('T', ' ').slice(0, 19)) : '—';
    const confCell = s.status === 'ok'
      ? conf + '%<div class="bar"><span style="width:' + conf + '%"></span></div>'
      : esc(s.detail || 'error');
    return '<tr><td>' + esc(s.carpark_id) + '</td>' + ...
  }).join('');
```

- **`.map(...).join('')`** builds the whole table body as one string and assigns it in a single
  `innerHTML` write. One DOM update instead of 24 — far fewer reflows.
- **`(s.confidence_score * 100).toFixed(0)`** converts `0.8734` to `"87"`.
- **`s.last_seen.replace('T', ' ').slice(0, 19)`** turns
  `2026-09-16T01:47:00.123456+00:00` into `2026-09-16 01:47:00` — dropping the `T` separator and
  truncating sub-second precision and the offset for readability.
- **`'—'`** (an em dash) for missing timestamps reads better than "null".
- **The confidence cell is conditional**: a percentage plus the mini progress bar on success, the
  error detail on failure. Same column, two meanings, which keeps the table narrow.
- **`esc()` on every interpolated value** — `carpark_id`, `detail`, and the timestamp. The numeric
  fields are not escaped because they come from the API as numbers.

```javascript
async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    const el = document.getElementById('health');
    el.textContent = 'model: ' + (h.model_loaded ? 'loaded' : 'not loaded');
    el.className = 'pill ' + (h.model_loaded ? 'ok' : 'bad');
  } catch (e) {
    const el = document.getElementById('health');
    el.textContent = 'health: unreachable'; el.className = 'pill bad';
  }
}
```

A `try/catch` because the *server being down* is a distinct, expected failure that the other loaders
don't handle. It's surfaced as "health: unreachable" with a red pill, so the dashboard degrades
visibly rather than silently freezing.

`textContent` (not `innerHTML`) is used here, which is inherently XSS-safe.

```javascript
function loadPlot() {
  // Cache-bust so the browser re-fetches a freshly rendered PNG each cycle.
  document.getElementById('plot').src = '/api/operations/plot.png?t=' + Date.now();
}
```

**Cache-busting.** The browser caches by URL, so setting the same `src` every 3 seconds would show
a frozen image. Appending `?t=<timestamp>` makes each URL unique and forces a real fetch. The
server ignores the parameter.

```javascript
async function refresh() {
  await Promise.allSettled([loadStatuses(), loadUuids(), loadHealth()]);
  loadPlot();
  countdown = REFRESH_MS / 1000;
}
```

**`Promise.allSettled`, not `Promise.all`** — and the difference is the point. `Promise.all`
**rejects immediately if any promise rejects**, so one failing endpoint would prevent the other two
from updating. `allSettled` waits for all of them regardless of outcome. Partial failure degrades
one panel instead of the whole dashboard.

This is the browser-side twin of `fetch_many`'s per-camera error isolation — the same resilience
principle at a different layer.

`loadPlot()` runs after, so the chart reflects the data just fetched.

```javascript
setInterval(() => {
  countdown -= 1;
  document.getElementById('refresh').textContent =
    'next refresh ' + Math.max(countdown, 0) + 's';
  if (countdown <= 0) refresh();
}, 1000);

refresh();
```

A **1-second timer driving a 3-second refresh**. Why not just `setInterval(refresh, 3000)`? Because
the 1-second tick also updates the visible countdown, which is genuine feedback: the user can see
the page is alive and knows when new data is due. A dashboard that silently updates leaves you
wondering whether it's frozen.

`Math.max(countdown, 0)` prevents a negative number flashing if a refresh takes longer than a tick.

The bare `refresh()` at the end fires immediately on load, so there's no 3-second wait for the first
data.

---

# Part 5 — Cross-cutting concepts (the "why" behind the patterns)

## The event loop, and why blocking is fatal

An `asyncio` application runs on **one thread** executing an **event loop**. The loop holds a queue
of ready tasks and runs each until it hits an `await` that actually suspends, then switches to the
next.

That model delivers enormous concurrency for I/O-bound work — thousands of connections on one
thread — but it has one absolute rule:

> **Never block the event loop thread.**

If a coroutine calls a synchronous function that takes 200 ms, the loop cannot run *anything else*
for those 200 ms. Every other in-flight request stalls. With ten concurrent requests each doing
200 ms of blocking work, the last one waits two full seconds.

This codebase has exactly three blocking operations, and **all three are offloaded**:

| Operation | Where | Why it blocks | Offloaded at |
|---|---|---|---|
| `Path.read_bytes()` | `camera_service/main.py` | disk / network filesystem I/O | `take_photo` |
| `model.predict()` | `app/services/inference.py` | CPU-bound native code | `InferenceService.infer` |
| `fig.savefig()` | `app/plotting.py` | CPU-bound rendering | `operational_plot` |

The tool in all three cases is `asyncio.to_thread(fn, *args)`: run `fn` in a worker thread from the
default `ThreadPoolExecutor`, return an awaitable that resolves to its result. The calling coroutine
suspends, the loop runs other tasks, and the result is delivered when the thread finishes.

**A note on the GIL.** Python's Global Interpreter Lock normally prevents threads from running
Python bytecode in parallel — so how does `to_thread` help CPU-bound work? Because the heavy
libraries here (`torch`, `numpy`, `PIL`, matplotlib's C core) **release the GIL** while executing
native code. During `model.predict()` the worker thread is in C++/CUDA land with the GIL released,
so the event loop thread runs freely. For pure-Python CPU work `to_thread` would help much less,
and you'd reach for `ProcessPoolExecutor`.

## `async` / `await` in one paragraph

`async def` defines a **coroutine function**; calling it returns a coroutine object that does
nothing until awaited. `await` means "suspend here, let the loop run other work, resume when this
completes". You can only `await` inside an `async def`. A coroutine that never awaits anything
gains nothing from being async — which is why every `async def` here either awaits I/O, awaits a
`to_thread` offload, or is an interface method shaped for a future async implementation
(`InMemoryRequestRepository`).

## `asyncio.gather` — structured concurrency

```python
results = await asyncio.gather(*(coro(x) for x in items))
```

Schedules every coroutine as a task, runs them concurrently, and returns results **in input order**
regardless of completion order. Used in `CameraClient.fetch_many` (fetch 2n photos at once) and in
`find_carparks` (process all outcomes at once).

Default behaviour is fail-fast: the first exception propagates. This code sidesteps that by making
the mapped coroutines never raise — errors become result objects instead.

## `asyncio.Semaphore` vs `asyncio.Lock` vs `threading.Lock`

| Primitive | Blocks | Use when |
|---|---|---|
| `asyncio.Semaphore(n)` | suspends the coroutine | at most *n* concurrent holders (here: 1 prediction) |
| `asyncio.Lock()` | suspends the coroutine | mutual exclusion between coroutines (here: repository state) |
| `threading.Lock()` | **blocks the thread** | coordinating real threads — **never** hold one on the event loop |

Using `threading.Lock` on the event loop reintroduces the exact freeze you were trying to avoid.
That's why both the inference guard and the in-memory repository use the `asyncio` variants.

## `contextvars` — the invisible plumbing

`ContextVar` values are scoped to the current execution context, and `asyncio` gives each task its
own copied context. This is what lets `logging_config.JsonFormatter` decorate *every* log line with
the current `request_id`, `uuid`, and `endpoint` without a single function signature mentioning
them.

The flow:

1. `RequestContextMiddleware` sets all three and keeps the returned `Token`s.
2. Anything downstream — routes, services, the Firestore repository, the camera client — reads them
   via `.get()`.
3. The middleware's `finally` block calls `.reset(token)` to prevent leakage between requests.

`camera_client._trace_headers()` uses the same mechanism to forward the id to the camera service,
which is how one trace id spans both services.

## Lazy imports

Four modules import heavy dependencies **inside functions** rather than at module scope:

| Module | Deferred import | Why |
|---|---|---|
| `services/inference.py` | `ultralytics`, `PIL` | test `_parse` without torch installed |
| `services/firestore_repository.py` | (imported lazily by `main.py`) | skip gRPC when using memory backend |
| `plotting.py` | `matplotlib` | keep import cheap when no plot is requested |
| `firestore_repository._iso` | `datetime` | stylistic consistency |

The costs avoided are real: importing `torch` takes seconds and hundreds of MB. The trade-off is a
tiny per-call lookup (Python caches modules in `sys.modules`, so only the first call does real
work) and the risk of an `ImportError` surfacing at runtime instead of startup — acceptable here,
because the failure paths are all handled.

## The repository pattern

An abstract `RequestRepository` with two implementations, chosen by one environment variable:

```
REPOSITORY_BACKEND=memory     → InMemoryRequestRepository    (dev, tests, single pod)
REPOSITORY_BACKEND=firestore  → FirestoreRequestRepository   (multi-pod cloud)
```

Routes only ever see the abstract type. The lifespan picks the implementation; `_build_repository`
is the only place that knows both exist. This is what makes the "unique users in 30s" view correct
across multiple pods without changing a line of route code.

## Dependency injection and the test seam

Every singleton reaches a route through a `Depends(...)` provider in `app/dependencies.py`. Beyond
tidiness, this exists so tests can write:

```python
app.dependency_overrides[get_inference_service] = lambda: FakeInference()
app.dependency_overrides[get_camera_client] = lambda: FakeCamera()
```

and exercise the full HTTP stack — routing, validation, serialisation, middleware — with no model,
no network, and no Firestore. Without this seam, testing the ranking logic would require loading a
real YOLO model.

## Graceful degradation

Failures are classified by whether the system can still do useful work:

| Failure | Response | Why |
|---|---|---|
| Model fails to load | app starts; inference routes 503; `/health` reports `model_loaded: false` | logs and dashboard stay reachable |
| Images directory missing | camera starts; readiness 503 | the mount can be fixed without a restart loop |
| One camera unreachable | that car park marked `error`; others returned normally | 19 good results beat 1 error |
| `NUM_CARPARKS` invalid | **hard exit** | unfixable without a restart; fail fast and loudly |
| Firestore unreachable | exception propagates | data loss is worse than an error response |

The consistent principle: **degrade when the system can still do useful work; fail fast when it
can't.**

## HTTP status codes, chosen deliberately

| Code | Where | Meaning here |
|---|---|---|
| 400 | `find-carparks` when 2n > configured | client asked for the impossible |
| 404 | `annotate-carpark`, camera `takephoto` | that car park doesn't exist |
| 422 | automatic, from FastAPI | parameter failed validation (missing `uuid`, `n=0`) |
| 500 | `annotate-carpark` inference failure | our bug, in our process |
| 502 | `annotate-carpark` camera failure | upstream service failed |
| 503 | readiness probes, unavailable dependencies | temporarily unable; retry later |

The 500/502 split on the two `annotate-carpark` failure paths is the sharpest example: it tells an
operator which service to go look at, without reading a single log line.

---

# Part 6 — End-to-end request walkthroughs

## `GET /api/find-carparks?uuid=alice&n=3`

1. **uvicorn** receives the request and builds the ASGI `scope`.
2. **`RequestContextMiddleware.__call__`** runs. No inbound `X-Request-ID`, so it mints
   `uuid4().hex`. It parses `uuid=alice` out of the query string, sets three contextvars, and
   starts a `perf_counter`.
3. **FastAPI routing** matches `/api/find-carparks` and validates the parameters. `n=3` passes
   `ge=1`; a missing `uuid` would 422 here.
4. **Dependencies resolve** — `get_registry`, `get_camera_client`, `get_inference_service`,
   `get_repository`, `get_app_settings` each pull from `app.state`. If the model failed to load,
   this is where the 503 comes from.
5. **`registry.sample(6)`** returns 6 distinct random `CarParkInfo` objects (2 × 3).
6. **`camera.fetch_many(picks)`** launches 6 concurrent `_one` coroutines via `asyncio.gather`.
   Each issues `GET {camera_url}` on the **shared** `AsyncClient` with an `X-Request-ID` header
   from the contextvar. The six network waits overlap.
7. **The camera service** handles each: its own middleware reuses the forwarded request id,
   `take_photo` validates the car park id, picks a random image, offloads `read_bytes` to a thread,
   base64-encodes it, and returns JSON. It logs one line carrying the **same** request id.
8. **Back in `fetch_many`**, each response is parsed and base64-decoded into a `FetchOutcome`. Say
   camera 4 timed out — that one becomes `FetchOutcome(image_bytes=None, error="...")` and is
   logged; the other five succeed.
9. **`asyncio.gather(*(process(o) for o in outcomes))`** runs 6 `process` coroutines. The failed one
   returns an error `CarParkStatus` immediately. The other five call `inference.infer(...)`, which
   **queues on `Semaphore(1)`** — so predictions run strictly one at a time, each inside a worker
   thread via `to_thread`, while the event loop stays free.
10. **Each `_parse`** counts `empty` vs other detections, computes the mean empty confidence, and
    reads `speed["inference"]`.
11. **Persistence** — all 6 statuses are upserted, then `record_uuid("alice")` appends a sighting.
12. **Ranking** — the 5 successful statuses are sorted by `(empty_count, confidence_score)`
    descending, and the top 3 are taken.
13. **`speed_inference`** is the sum across all 5 successful inferences, formatted as `"142.3 ms"`.
14. **The response model** is built; `to_result()` projects each status, and `response_model`
    filters the output.
15. **The middleware's `else` branch** logs `request completed` with the status code and latency,
    then `finally` resets the three contextvars.

Every log line from steps 2–15, in **both services**, carries the same `request_id` and
`uuid="alice"`.

## `GET /api/annotate-carpark?carpark_id=CBD_007`

Same middleware entry. Then: `registry.get("CBD_007")` (404 if unknown) → a single
`camera.fetch_photo` (502 on failure) → `inference.infer(..., annotate=True)` (500 on failure),
which additionally calls `result.plot()`, flips BGR→RGB, and JPEG-encodes → the status is persisted
→ the response carries the base64 annotated image.

## `GET /dashboard`

Returns `DASHBOARD_HTML` immediately. The browser then polls `/api/operations/statuses`,
`/api/operations/recent-uuids`, and `/api/health` in parallel via `Promise.allSettled` every 3
seconds, and re-fetches `/api/operations/plot.png` with a cache-busting timestamp. That PNG request
reads both repository views and offloads the matplotlib render to a worker thread.

---

# Part 7 — Endpoint reference

### Core API service (port 8000)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/find-carparks?uuid=&n=` | COREAPI1 — query 2n car parks, return the best n |
| GET | `/api/annotate-carpark?carpark_id=&uuid=` | COREAPI2 — annotated image + counts for one car park |
| GET | `/api/carparks` | list all configured car parks and camera URLs |
| GET | `/api/operations/statuses` | latest status for every car park queried so far |
| GET | `/api/operations/recent-uuids` | distinct UUIDs in the 30-second window |
| GET | `/api/operations/availability` | OPS-API-1 — all car parks with current availability |
| GET | `/api/operations/plot.png` | OPS-REQ-2 — matplotlib availability chart |
| GET | `/api/health`, `/health` | combined health probe |
| GET | `/health/live` | liveness — 200 while the process runs |
| GET | `/health/ready` | readiness — 503 until the model is loaded |
| GET | `/`, `/dashboard` | HTML monitoring dashboard |
| GET | `/docs`, `/openapi.json` | auto-generated API documentation |

### Camera simulator (port 8001)

| Method | Path | Purpose |
|---|---|---|
| GET | `/cameras/{carpark_id}/api/takephoto` | random supplied image as base64 JSON |
| GET | `/cameras` | list every camera's takephoto URL |
| GET | `/health`, `/health/live` | liveness |
| GET | `/health/ready` | 503 when no usable images are loaded |

---

# Part 8 — Observations, sharp edges, and deliberate trade-offs

Things worth knowing if you're extending or defending this code.

### Intentional duplication

`_sanitise_request_id` exists verbatim in both `app/middleware.py` and `camera_service/main.py`.
The camera service must not import from `app` (its Docker image doesn't contain `app/`), and a
shared package for ten lines would cost more than it saves. Same story for the
`CBD_%03d` ID format, which is duplicated with a cross-reference comment in each location.

### Resolved: items fixed after the first review

The following were flagged in an earlier pass and have since been addressed. They're recorded
here because the *reasoning* is still worth knowing.

**Large `n` is now bounded (spec §4.3).** `n` previously had only `ge=1`, so `n=1000` was
rejected — but only after dependency resolution, by a generic "sample larger than population"
error. It now hits a `le=MAX_N_CEILING` (49) constraint during FastAPI validation → 422, and a
value within the ceiling but beyond *this* deployment's capacity returns 400 naming the real
limit. Neither path contacts a camera or the model.

**Per-user response caching (spec §4.3).** `find-carparks` responses are cached for
`CACHE_TTL_SECONDS` keyed by `(uuid, n)` via `services/response_cache.py`. A cache hit still
calls `record_uuid`, so OPS-API-2's "users in the last 30 seconds" is not under-counted by
caching. `CACHE_TTL_SECONDS=0` disables it for clean benchmark runs.

**`request_logs` no longer grows without bound.** `record_uuid` writes an `expires_at`
timestamp (so a server-side Firestore TTL policy can expire documents at no query cost), and
`recent_uuids` opportunistically deletes a bounded batch of long-expired documents. The purge is
rate-limited, capped, and swallows its own errors — housekeeping must never fail a read.

**Status writes are batched.** `RequestRepository.upsert_statuses()` was added with a default
loop implementation; Firestore overrides it with a `WriteBatch` (chunked at Firestore's 500-op
limit) and the in-memory version takes its lock once for the whole batch. `find-carparks` now
makes one call instead of 2n.

**`_health_payload()` takes the request.** It reads `request.app.state` rather than the
module-level `app`, so it works for any app instance and can't raise `AttributeError` outside a
lifespan context.

**The in-memory deque tolerates out-of-order timestamps.** `recent_uuids` still prunes from the
left to reclaim memory, but now *also* filters each remaining sighting against the cutoff while
reading, so an entry recorded late with an old explicit timestamp can't be stranded behind a
newer one and wrongly reported as recent.

**`NUM_CARPARKS` mismatches are detected.** At startup the API queries the camera's `/cameras`
count in a background task and logs an ERROR on mismatch. It's `asyncio.create_task`'d rather
than awaited, so a slow or absent camera can never delay readiness, and the task is cancelled on
shutdown.

**Small cleanups.** `_build_repository` now annotates its parameters; the registry is built from
`settings.camera_base_url_clean` so that property is no longer dead; and `find_carparks`'s
`settings` dependency is now genuinely used (the large-`n` ceiling and cache TTL).

### Still open: `dict(scope["headers"])` collapses duplicate headers

Both middlewares build a dict from the ASGI header list, so a repeated header name keeps only the
last value. Irrelevant for `X-Request-ID`, but worth knowing if you ever read a header that can
legitimately repeat (like `Set-Cookie`).

### Still open: caching interacts with the benchmark

The cache is a genuine §4.3 optimisation, but it also makes Locust numbers look better than the
underlying inference throughput if the load script reuses a small pool of UUIDs. Run the
benchmark with `CACHE_TTL_SECONDS=0` for the raw scaling curve, then optionally a second run
with caching on to quantify the gain — `/api/operations/cache-stats` reports the hit rate.

### Still open: `random.sample` on every request

`find-carparks` samples 2n car parks per call, so under sustained load the same popular car park
is inferred repeatedly across different users with no shared result reuse. A short-TTL
*per-car-park* inference cache (rather than per-response) would cut redundant model work
further. Not implemented, because it changes the freshness semantics of every result rather than
just repeat calls from one user.

### Model weights and images are never baked into images

Both Dockerfiles copy only code:

```dockerfile
COPY app ./app            # never the model or images
COPY camera_service ./camera_service
```

Weights arrive via a volume at `MODEL_PATH=/models/model.pt`; images via `IMAGES_DIR=/images`. This
keeps images small, lets you swap a model without a rebuild, and keeps large binaries out of the
registry.

### The API Dockerfile installs CPU-only torch first, deliberately

```dockerfile
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt
```

Installing torch from the CPU index **before** `ultralytics` stops pip from pulling the default
CUDA wheels and their `nvidia-*` runtime packages — roughly **6–8 GB** of image size for GPU
support that is never used. Ordering here is the entire optimisation.

`libgl1` and `libglib2.0-0` are installed via apt because opencv (a transitive ultralytics
dependency) links against them at runtime, even headless.
