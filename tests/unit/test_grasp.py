"""Grasp math: depth decoding, part location and grasp heights. No machine needed."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from viam.proto.common import Geometry, Pose, RectangularPrism, Vector3

from viamrobot.grasp import (
    Intrinsics,
    box_center,
    decode_depth,
    deproject,
    fingertip_reach_mm,
    grasp_heights,
    holds_part,
    iou,
    jaw_turn_deg,
    part_axis,
    part_depth_mm,
    wrap_half_turn,
    table_depth_mm,
)
from viamrobot.pipeline import BoundingBox

FIXTURE = Path(__file__).parents[2] /"fixtures/armfarm01/unlabeled/current-view_084614_1/cam_depth.dep"
K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def encode_depth(depth: np.ndarray) -> bytes:
    h, w = depth.shape
    return b"DEPTHMAP" + w.to_bytes(8, "big") + h.to_bytes(8, "big") + depth.astype(">u2").tobytes()


def test_decode_depth_round_trip():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert np.array_equal(decode_depth(encode_depth(depth)), depth)


def test_decode_depth_fixture_is_640x480_table_at_inspect_height():
    depth = decode_depth(FIXTURE.read_bytes())
    assert depth.shape == (480, 640)
    assert 250 < table_depth_mm(depth) < 400  # captured at inspect pose, camera ~323 mm above the table


def test_part_depth_prefers_the_part_over_table_and_ignores_holes():
    depth = np.full((480, 640), 400, dtype=np.uint16)  # table
    depth[220:260, 300:340] = 360  # 40 mm tall part
    depth[238:242, 318:322] = 0  # missing readings
    box = BoundingBox(290, 210, 350, 270, confidence=0.9)  # a bit bigger than the part
    assert part_depth_mm(depth, box) == pytest.approx(360)
    assert table_depth_mm(depth) == pytest.approx(400)


def test_part_depth_with_no_readings_raises():
    with pytest.raises(ValueError):
        part_depth_mm(np.zeros((480, 640), dtype=np.uint16), BoundingBox(0, 0, 10, 10, confidence=1))


def test_deproject_center_and_offset():
    assert deproject(320, 240, 500, K) == (0, 0, 500)
    x, y, z = deproject(380, 180, 300, K)
    assert (x, y, z) == pytest.approx((30, -30, 300))
    assert box_center(BoundingBox(300, 200, 340, 260, confidence=1)) == (320, 230)


CARDBOARD_BGR = (120, 150, 175)


def scene(parts, seed=0):
    """A 640x480 cardboard-coloured frame with noise, and rotated rectangles drawn on it.

    parts: (centre_u, centre_v, length, width, angle_deg, bgr). Angles are image angles:
    0 = pointing right, 90 = pointing down.
    """
    rng = np.random.default_rng(seed)
    img = np.clip(np.array(CARDBOARD_BGR, np.float32) + rng.normal(0, 6, (480, 640, 3)), 0, 255).astype(np.uint8)
    for u, v, length, width, angle, bgr in parts:
        corners = cv2.boxPoints(((u, v), (length, width), angle)).astype(np.int32)
        cv2.fillPoly(img, [corners], bgr)
    return img


def angle_off(a, b):
    return abs(wrap_half_turn(a - b))


@pytest.mark.parametrize("angle", [0, 20, 45, 70, 90, 115, 135, 160])
def test_part_axis_finds_a_silver_bolts_direction(angle):
    img = scene([(320, 240, 70, 9, angle, (185, 185, 190))])
    box = BoundingBox(320 - 45, 240 - 45, 320 + 45, 240 + 45, confidence=1)
    axis = part_axis(img, box)
    assert axis is not None
    assert angle_off(axis.angle_deg, angle) < 4
    assert axis.elongation > 4
    assert (axis.u, axis.v) == pytest.approx((320, 240), abs=2)


def test_part_axis_blue_taped_bolt_and_its_ends():
    img = scene([(300, 200, 80, 14, 90, (170, 90, 30))])  # vertical, blue
    axis = part_axis(img, BoundingBox(280, 150, 320, 250, confidence=1))
    assert angle_off(axis.angle_deg, 90) < 4
    (u0, v0), (u1, v1) = axis.ends()
    assert sorted([v0, v1]) == pytest.approx([160, 240], abs=4) and u0 == pytest.approx(300, abs=2)


def test_part_axis_square_part_is_not_elongated():
    img = scene([(320, 240, 24, 24, 0, (185, 185, 190))])
    axis = part_axis(img, BoundingBox(300, 220, 340, 260, confidence=1))
    assert axis is not None and axis.elongation < 1.3


def test_part_axis_empty_box_is_none():
    assert part_axis(scene([]), BoundingBox(300, 220, 340, 260, confidence=1)) is None


def test_part_axis_ignores_a_neighbour_at_the_box_edge():
    # the bolt we want, plus the end of another part poking into the box corner
    img = scene([(320, 240, 70, 9, 30, (185, 185, 190)), (380, 300, 60, 10, 0, (170, 90, 30))])
    axis = part_axis(img, BoundingBox(280, 210, 360, 270, confidence=1))
    assert angle_off(axis.angle_deg, 30) < 5


@pytest.mark.parametrize(
    "part, jaw, turn",
    [
        (0, 0, -90),  # part along x, jaws along x: quarter turn (either way; -90 by convention)
        (0, 90, 0),  # already across
        (30, 0, -60),  # wants jaws at 120 == -60
        (90, 10, -10),
        (-45, 0, 45),
        (170, 85, -5),
    ],
)
def test_jaw_turn_is_at_most_a_quarter_turn(part, jaw, turn):
    got = jaw_turn_deg(part, jaw)
    assert -90 <= got < 90
    assert got == pytest.approx(turn)
    assert angle_off(jaw + got, part + 90) < 1e-9


def test_iou():
    a = BoundingBox(0, 0, 10, 10, confidence=1)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, BoundingBox(5, 0, 15, 10, confidence=1)) == pytest.approx(50 / 150)
    assert iou(a, BoundingBox(20, 20, 30, 30, confidence=1)) == 0.0


def test_fingertip_reach_from_armfarm01_geometries():
    def box(z_center, z_dim, label):
        return Geometry(center=Pose(z=z_center, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=45, y=120, z=z_dim)), label=label)

    assert fingertip_reach_mm([box(-55, 110, "case-gripper"), box(-6, 112, "claws")]) == pytest.approx(50)
    assert fingertip_reach_mm([], default=42) == 42


@pytest.mark.parametrize(
    "position, expected",
    [
        (0, False),  # closed on nothing
        (25, False),  # within the margin of an empty close
        (30, True),
        (150, True),  # stopped on a part
        (777, True),
        (790, False),  # within the margin of open
        (807, False),  # still open: the armfarm01 run where the arm lifted too early
    ],
)
def test_holds_part_only_between_empty_close_and_open(position, expected):
    assert holds_part(position, empty_closed_position=0, open_position=807, margin=30) is expected


def test_grasp_heights_below_part_top():
    above, grasp = grasp_heights(part_top_z=40, table_z=0, reach_mm=50, grip_depth_mm=15, approach_mm=100)
    assert grasp == pytest.approx(40 - 15 + 50)
    assert above == pytest.approx(grasp + 100)


def test_grasp_heights_keep_fingertips_off_the_table():
    _, grasp = grasp_heights(part_top_z=8, table_z=0, reach_mm=50, grip_depth_mm=15, approach_mm=100, table_clearance_mm=5)
    assert grasp - 50 == pytest.approx(5)
