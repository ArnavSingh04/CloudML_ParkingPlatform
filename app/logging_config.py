"""Structured JSON logging.

Every log line is a single JSON object so it can be shipped to and queried by
log aggregators (Cloud Logging, ELK, etc.) without regex parsing. Per-request
fields (request id, uuid, endpoint, ...) are carried on ``contextvars`` so any
log emitted while handling a request is automatically decorated with them.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import sys
from typing import Any

# Context variables populated by RequestContextMiddleware for the lifetime of
# a single request. They default to None outside of a request scope.
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
uuid_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "uuid", default=None
)
endpoint_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "endpoint", default=None
)

_SERVICE_NAME = "smartpark"

# Attributes present on every stdlib LogRecord; anything *not* in this set that
# a caller attaches via ``extra={...}`` is treated as a structured field.
_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


def set_service_name(name: str) -> None:
    """Set the ``service`` field emitted on every log line."""
    global _SERVICE_NAME
    _SERVICE_NAME = name


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

        # Decorate with per-request context when available.
        if (rid := request_id_ctx.get()) is not None:
            payload["request_id"] = rid
        if (uuid := uuid_ctx.get()) is not None:
            payload["uuid"] = uuid
        if (endpoint := endpoint_ctx.get()) is not None:
            payload["endpoint"] = endpoint

        # Merge any structured fields passed via ``extra=...``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOGRECORD_KEYS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", service_name: str = "smartpark") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    set_service_name(service_name)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Route uvicorn's own loggers through our handler for uniform output.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper mirroring ``logging.getLogger``."""
    return logging.getLogger(name)
