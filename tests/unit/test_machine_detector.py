"""MachineDetector against a fake vision service: no machine needed.

The fixture mirrors a vision-6 (yolov8n-part) result on a 640x480 `cam` frame with two
parts in view: a bolt at 0.56 and a blue part at 0.84, as seen in the CONTROL tab.
"""

import pytest
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.service.vision import Detection

from viamrobot.pipeline import BoundingBox, Frame
from viamrobot.stages.machine import MachineDetector, boxes_from_detections

IMAGE = ViamImage(b"\xff\xd8 fake jpeg", CameraMimeType.JPEG)
W, H = 640, 480


def detection(x_min, y_min, x_max, y_max, confidence, class_name="part"):
    """A Detection shaped like vision-6 returns it: pixel and normalized corners."""
    return Detection(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        confidence=confidence,
        class_name=class_name,
        x_min_normalized=x_min / W,
        y_min_normalized=y_min / H,
        x_max_normalized=x_max / W,
        y_max_normalized=y_max / H,
    )


BOLT = detection(334, 108, 390, 150, 0.56)
BLUE_PART = detection(390, 243, 425, 302, 0.84)
TWO_PARTS = [BOLT, BLUE_PART]  # service order, not confidence order


class FakeVision:
    def __init__(self, detections):
        self.detections, self.calls = detections, []

    async def get_detections(self, image, **kwargs):
        self.calls.append(image)
        return self.detections


def test_extracts_every_box_with_its_coordinates():
    boxes = boxes_from_detections(TWO_PARTS)

    assert boxes == [
        BoundingBox(390, 243, 425, 302, confidence=pytest.approx(0.84), label="part"),
        BoundingBox(334, 108, 390, 150, confidence=pytest.approx(0.56), label="part"),
    ]


def test_boxes_lie_inside_the_frame_and_match_normalized_corners():
    for d, box in zip(sorted(TWO_PARTS, key=lambda d: -d.confidence), boxes_from_detections(TWO_PARTS)):
        assert 0 <= box.x_min < box.x_max <= W
        assert 0 <= box.y_min < box.y_max <= H
        assert box.x_min / W == pytest.approx(d.x_min_normalized)
        assert box.y_max / H == pytest.approx(d.y_max_normalized)


def test_min_confidence_drops_weak_boxes():
    boxes = boxes_from_detections(TWO_PARTS, min_confidence=0.6)

    assert [(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes] == [(390, 243, 425, 302)]


def test_labels_filter_keeps_only_named_classes():
    hand = detection(0, 0, 200, 190, 0.9, class_name="hand")

    boxes = boxes_from_detections([hand, *TWO_PARTS], labels=["part"])

    assert [b.label for b in boxes] == ["part", "part"]


def test_nothing_in_view_gives_no_boxes():
    assert boxes_from_detections([]) == []


def test_detection_without_pixel_coordinates_is_an_error():
    normalized_only = Detection(class_name="part", confidence=0.8, x_min_normalized=0.5, y_min_normalized=0.5)

    with pytest.raises(ValueError, match="no pixel coordinates"):
        boxes_from_detections([normalized_only])


async def test_detector_finds_all_items_in_the_captured_frame():
    vision = FakeVision(TWO_PARTS)
    detector = MachineDetector(vision, min_confidence=0.5)

    boxes = await detector.find_items(Frame(color=IMAGE))

    assert [(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes] == [(390, 243, 425, 302), (334, 108, 390, 150)]
    assert vision.calls == [IMAGE]


async def test_detector_stage_returns_the_most_confident_item():
    box = await MachineDetector(FakeVision(TWO_PARTS)).find_item(Frame(color=IMAGE))

    assert (box.x_min, box.y_min, box.x_max, box.y_max, box.label) == (390, 243, 425, 302, "part")


async def test_detector_stage_returns_none_when_nothing_clears_the_threshold():
    detector = MachineDetector(FakeVision(TWO_PARTS), min_confidence=0.9)

    assert await detector.find_item(Frame(color=IMAGE)) is None


async def test_detector_needs_a_viam_image():
    with pytest.raises(TypeError, match="ViamImage"):
        await MachineDetector(FakeVision([])).find_items(Frame(color="not an image"))
