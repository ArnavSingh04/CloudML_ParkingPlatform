"""YOLO inference service.

Design constraints (from the assignment):
  * The Ultralytics model is loaded exactly once (see the FastAPI lifespan) and
    shared across requests.
  * ``model.predict`` is CPU-bound and NOT async-safe, so it must never run
    directly inside an async route. We offload it to a worker thread with
    ``asyncio.to_thread`` and serialise access with an ``asyncio.Semaphore(1)``
    to guarantee only one prediction touches the model at a time.

Heavy dependencies (ultralytics/torch, Pillow, numpy) are imported lazily so
this module — and the pure-parsing logic in ``_parse`` — can be imported and
unit-tested without those packages installed.
"""

from __future__ import annotations

import asyncio
import io
import statistics
from dataclasses import dataclass

from ..logging_config import get_logger

logger = get_logger("smartpark.inference")

EMPTY_CLASS_NAME = "empty"


@dataclass(slots=True)
class InferenceResult:
    """Outcome of a single prediction."""

    empty_count: int
    occupied_count: int
    total_spaces: int
    confidence_score: float
    speed_inference: float  # milliseconds
    annotated_jpeg: bytes | None = None


class InferenceService:
    """Owns the shared model and mediates all access to it."""

    def __init__(self, model, max_concurrency: int = 1, confidence: float = 0.25):
        self._model = model
        # Semaphore(1) — only one prediction on the shared model at a time.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._confidence = confidence

    @classmethod
    def load(
        cls, model_path: str, max_concurrency: int = 1, confidence: float = 0.25
    ) -> "InferenceService":
        """Load the Ultralytics model from disk (lazy import keeps torch optional)."""
        from ultralytics import YOLO  # imported here to avoid a hard import cost

        logger.info("loading YOLO model", extra={"model_path": model_path})
        model = YOLO(model_path)
        return cls(model, max_concurrency=max_concurrency, confidence=confidence)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    async def infer(self, image_bytes: bytes, annotate: bool = False) -> InferenceResult:
        """Run prediction off the event loop, one caller at a time."""
        async with self._semaphore:
            return await asyncio.to_thread(self._run, image_bytes, annotate)

    # --- synchronous internals (run inside the worker thread) --------------

    def _run(self, image_bytes: bytes, annotate: bool) -> InferenceResult:
        image = self._decode(image_bytes)
        results = self._model.predict(
            image, conf=self._confidence, verbose=False
        )
        return self._parse(results[0], annotate)

    @staticmethod
    def _decode(image_bytes: bytes):
        from PIL import Image  # lazy import

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

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

        empty_count = len(empty_confidences)
        total_spaces = empty_count + occupied_count
        confidence_score = (
            statistics.fmean(empty_confidences) if empty_confidences else 0.0
        )

        speed = getattr(result, "speed", None) or {}
        speed_inference = round(float(speed.get("inference", 0.0)), 3)

        annotated_jpeg = self._render_annotated(result) if annotate else None

        return InferenceResult(
            empty_count=empty_count,
            occupied_count=occupied_count,
            total_spaces=total_spaces,
            confidence_score=round(confidence_score, 4),
            speed_inference=speed_inference,
            annotated_jpeg=annotated_jpeg,
        )

    @staticmethod
    def _render_annotated(result) -> bytes:
        """Encode the model's annotated frame as JPEG bytes."""
        from PIL import Image  # lazy import

        # result.plot() returns a BGR numpy array; reverse the last axis to RGB.
        rgb = result.plot()[:, :, ::-1]
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG")
        return buffer.getvalue()
