"""MachinePatchCoreScorer against a fake vision service: no machine needed."""

import pytest
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.service.vision import Classification

from viamrobot.pipeline import BoundingBox, Frame
from viamrobot.stages.machine import MachinePatchCoreScorer, anomaly_score

BOX = BoundingBox(0, 0, 10, 10, confidence=1.0)
IMAGE = ViamImage(b"\xff\xd8 fake jpeg", CameraMimeType.JPEG)


class FakeVision:
    def __init__(self, classifications):
        self.classifications, self.calls = classifications, []

    async def get_classifications(self, image, count, **kwargs):
        self.calls.append((image, count))
        return self.classifications


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("anomalous", 0.83, 0.83),
        ("normal", 0.9, 0.1),
        ("anomalous", 1.2, 1.0),  # clamped
        ("normal", 1.1, 0.0),  # clamped
    ],
)
def test_anomaly_score_recovers_raw_score(label, confidence, expected):
    assert anomaly_score([Classification(class_name=label, confidence=confidence)]) == pytest.approx(expected)


def test_anomaly_score_rejects_empty_and_unknown():
    with pytest.raises(ValueError, match="no classification"):
        anomaly_score([])
    with pytest.raises(ValueError, match="unexpected"):
        anomaly_score([Classification(class_name="bottle", confidence=0.5)])


async def test_scorer_scores_the_captured_frame():
    vision = FakeVision([Classification(class_name="normal", confidence=0.7)])
    score = await MachinePatchCoreScorer(vision).score(Frame(color=IMAGE), BOX)

    assert score == pytest.approx(0.3)
    assert vision.calls == [(IMAGE, 1)]


async def test_scorer_needs_a_viam_image():
    with pytest.raises(TypeError, match="ViamImage"):
        await MachinePatchCoreScorer(FakeVision([])).score(Frame(color="not an image"), BOX)
