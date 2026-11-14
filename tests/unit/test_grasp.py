"""Grasp math: depth decoding, part location and grasp heights. No machine needed."""

from pathlib import Path

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
    part_depth_mm,
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
