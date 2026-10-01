"""Unit tests for InferenceService result parsing (YOLO fully faked).

These exercise the class-name counting, mean-confidence and speed extraction
without importing torch/ultralytics or Pillow.
"""

from __future__ import annotations

from app.services.inference import InferenceResult, InferenceService

NAMES = {0: "empty", 1: "occupied"}


class FakeBox:
    def __init__(self, cls: int, conf: float) -> None:
        self.cls = [cls]
        self.conf = [conf]


class FakeResult:
    def __init__(self, boxes, names=None, speed=None) -> None:
        self.boxes = boxes
        self.names = names or NAMES
        self.speed = speed or {"preprocess": 1.0, "inference": 12.5, "postprocess": 0.5}


def test_counts_empty_and_occupied_by_class_name():
    svc = InferenceService(model=None)
    result = FakeResult(
        boxes=[
            FakeBox(0, 0.9),  # empty
            FakeBox(0, 0.8),  # empty
            FakeBox(0, 0.7),  # empty
            FakeBox(1, 0.6),  # occupied
            FakeBox(1, 0.5),  # occupied
        ]
    )
    parsed = svc._parse(result, annotate=False)

    assert parsed.empty_count == 3
    assert parsed.occupied_count == 2
    assert parsed.total_spaces == 5
    assert parsed.confidence_score == 0.8  # mean(0.9, 0.8, 0.7)
    assert parsed.speed_inference == 12.5
    assert parsed.annotated_jpeg is None


def test_confidence_is_zero_when_no_empty_detections():
    svc = InferenceService(model=None)
    result = FakeResult(boxes=[FakeBox(1, 0.6), FakeBox(1, 0.5)])
    parsed = svc._parse(result, annotate=False)

    assert parsed.empty_count == 0
    assert parsed.occupied_count == 2
    assert parsed.confidence_score == 0.0


async def test_infer_offloads_and_parses():
    """The async infer() path (semaphore + to_thread) returns a parsed result."""

    class FakeModel:
        def predict(self, image, conf=None, verbose=False):
            return [FakeResult(boxes=[FakeBox(0, 1.0), FakeBox(1, 0.4)])]

    svc = InferenceService(model=FakeModel())
    # Bypass Pillow decoding; the fake model ignores the image anyway.
    svc._decode = lambda image_bytes: image_bytes

    result = await svc.infer(b"ignored")
    assert isinstance(result, InferenceResult)
    assert result.empty_count == 1
    assert result.occupied_count == 1
    assert result.confidence_score == 1.0
