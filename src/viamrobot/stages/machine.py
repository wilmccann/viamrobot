"""Machine stages: services running on the Viam machine (armfarm01), reached over the network."""

from __future__ import annotations

from typing import Sequence

from viam.media.video import ViamImage
from viam.proto.service.vision import Classification
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
