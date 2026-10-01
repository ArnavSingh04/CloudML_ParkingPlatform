"""Request-scoped logging middleware.

Implemented as *pure ASGI* middleware (rather than Starlette's
``BaseHTTPMiddleware``) so the contextvars we set here reliably propagate into
the route handler — BaseHTTPMiddleware runs the endpoint in a separate task,
which breaks contextvar propagation.

Responsibilities:
  * reuse a valid incoming ``X-Request-ID`` header (for cross-service tracing)
    or mint a new one, and expose it as the ``X-Request-ID`` response header,
  * pull the ``uuid`` query parameter into the logging context when present,
  * emit one structured access-log line per request with latency + status.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs
from uuid import uuid4

from .logging_config import (
    endpoint_ctx,
    get_logger,
    request_id_ctx,
    uuid_ctx,
)

_logger = get_logger("smartpark.access")

# Upper bound on an accepted inbound request id; longer/!printable -> we mint one.
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


class RequestContextMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Reuse a valid incoming X-Request-ID for cross-service correlation;
        # otherwise mint a fresh one. Header names in ASGI scope are lowercase.
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id")
        request_id = _sanitise_request_id(
            incoming.decode("latin-1") if incoming else None
        ) or uuid4().hex
        path = scope.get("path", "")
        method = scope.get("method", "")

        query = scope.get("query_string", b"").decode("latin-1")
        uuid_values = parse_qs(query).get("uuid", [])
        uuid_value = uuid_values[0] if uuid_values else None

        rid_token = request_id_ctx.set(request_id)
        uuid_token = uuid_ctx.set(uuid_value)
        ep_token = endpoint_ctx.set(path)

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode()))
            await send(message)

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
