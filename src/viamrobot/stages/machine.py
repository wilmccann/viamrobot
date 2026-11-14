"""Machine stages: services running on the Viam machine (armfarm01), reached over the network."""

from __future__ import annotations

from typing import Sequence

from viam.media.video import ViamImage
from viam.proto.service.vision import Classification, Detection
from viam.robot.client import RobotClient
from viam.services.vision import Vision, VisionClient

from viamrobot.pipeline import BoundingBox, Frame


def anomaly_score(classifications: Sequence[Classification]) -> float:
    """Recover the raw PatchCore score from a viam-labs:anomalib classification.

    The service returns one label split at its own `threshold` attribute:
    "anomalous" with confidence = score, or "normal" with confidence = 1 - score.
    Undoing that gives the raw score, so the service threshold doesn't matter here
    and our own Thresholds in decision.py make the call. Clamped to [0, 1].
    """
    if not classifications:
        raise ValueError("anomaly service returned no classification")
    top = classifications[0]
    if top.class_name == "anomalous":
        score = top.confidence
    elif top.class_name == "normal":
        score = 1.0 - top.confidence
    else:
        raise ValueError(f"unexpected anomaly class {top.class_name!r}")
    return min(max(score, 0.0), 1.0)


class MachinePatchCoreScorer:
    """Anomaly score: PatchCore on the machine, through its viam-labs:anomalib vision service.

    On armfarm01 that is `vision-1`, backed by `ov` (sortit-patchcore on the NUC GPU, f32).
    Sends the frame the pipeline captured (frame.color must be a ViamImage), so the score
    matches what the detector saw. The whole frame is scored, as in training; `box` is unused.
    """

    def __init__(self, vision: Vision, service_name: str = "vision-1"):
        self.vision = vision
        self.runs = f"PatchCore on the machine, via vision service '{service_name}'"

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str = "vision-1") -> MachinePatchCoreScorer:
        return cls(VisionClient.from_robot(robot, name), name)

    async def score(self, frame: Frame, box: BoundingBox) -> float:
        if not isinstance(frame.color, ViamImage):
            raise TypeError(f"MachinePatchCoreScorer needs a ViamImage frame, got {type(frame.color).__name__}")
        return anomaly_score(await self.vision.get_classifications(frame.color, count=1))


_PIXEL_FIELDS = ("x_min", "y_min", "x_max", "y_max")


def boxes_from_detections(
    detections: Sequence[Detection],
    min_confidence: float = 0.0,
    labels: Sequence[str] | None = None,
) -> list[BoundingBox]:
    """Every detection as a BoundingBox in pixel coordinates, highest confidence first.

    Drops detections below `min_confidence` and, when `labels` is given, any whose
    class_name is not in it. The pixel fields are optional in the proto; a detection
    without them raises rather than silently becoming a box at (0, 0).
    """
    boxes = []
    for d in detections:
        if d.confidence < min_confidence or (labels is not None and d.class_name not in labels):
            continue
        missing = [f for f in _PIXEL_FIELDS if not d.HasField(f)]
        if missing:
            raise ValueError(f"detection {d.class_name!r} has no pixel coordinates ({', '.join(missing)})")
        boxes.append(BoundingBox(d.x_min, d.y_min, d.x_max, d.y_max, confidence=d.confidence, label=d.class_name))
    return sorted(boxes, key=lambda b: b.confidence, reverse=True)


class MachineDetector:
    """Find the item: YOLO on the machine, through its vision service.

    On armfarm01 that is `vision-7`, backed by `mlmodel-7` (yolov8n-part-v2, one class
    "part", on the NUC NPU via OpenVINO); it replaced vision-6 (yolov8n-part). It is
    also the detector inside vision-9, which adds a normal/anomalous/hold verdict per
    part and is what viamrobot.cell uses; vision-9's labels are verdicts, not "part", so
    this class points at vision-7. Sends the frame the pipeline captured (frame.color
    must be a ViamImage), so the boxes line up with that frame.
    """

    def __init__(
        self,
        vision: Vision,
        service_name: str = "vision-7",
        min_confidence: float = 0.5,
        labels: Sequence[str] | None = ("part",),
    ):
        self.vision = vision
        self.min_confidence = min_confidence
        self.labels = labels
        self.runs = f"YOLO on the machine, via vision service '{service_name}'"

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str = "vision-7", **kwargs) -> MachineDetector:
        return cls(VisionClient.from_robot(robot, name), name, **kwargs)

    async def find_items(self, frame: Frame) -> list[BoundingBox]:
        """Every item in view, highest confidence first."""
        if not isinstance(frame.color, ViamImage):
            raise TypeError(f"MachineDetector needs a ViamImage frame, got {type(frame.color).__name__}")
        detections = await self.vision.get_detections(frame.color)
        return boxes_from_detections(detections, self.min_confidence, self.labels)

    async def find_item(self, frame: Frame) -> BoundingBox | None:
        """The pipeline's Detector stage: the most confident item, or None if nothing is in view."""
        boxes = await self.find_items(frame)
        return boxes[0] if boxes else None
