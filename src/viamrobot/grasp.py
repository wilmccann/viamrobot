"""Locate a detected part in 3D and plan a top-down grasp on it. Pure math, no machine calls.

The wrist RealSense (`cam`) returns color and depth aligned to the same 640x480 grid, so a
`vision-6` box in the color frame indexes straight into the depth frame. The camera frame is
the usual pinhole one (+x right, +y down, +z out of the lens), in mm, which is the frame
`robot.transform_pose(..., "world")` expects for `cam`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viamrobot.pipeline import BoundingBox

# Viam's raw depth format (image/vnd.viam.dep): 8-byte magic, width and height as
# big-endian uint64, then big-endian uint16 depth in mm, row-major. 0 means no reading.
_DEP_HEADER = 24


def decode_depth(data: bytes) -> np.ndarray:
    """An image/vnd.viam.dep payload as an (height, width) uint16 array in mm."""
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    pixels = np.frombuffer(data, dtype=">u2", count=width * height, offset=_DEP_HEADER)
    return pixels.reshape(height, width).astype(np.uint16)


def part_depth_mm(depth: np.ndarray, box: BoundingBox, inner: float = 0.5, percentile: float = 25.0) -> float:
    """Distance from the camera to the top of the part in `box`, in mm.

    Looks only at the middle `inner` fraction of the box, so table pixels around the
    part's edges mostly drop out, ignores zero (no reading) pixels, and takes a low
    percentile: the part is closer to the camera than whatever table is left.
    """
    h, w = depth.shape
    cx, cy = (box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2
    half_w, half_h = (box.x_max - box.x_min) * inner / 2, (box.y_max - box.y_min) * inner / 2
    x0, x1 = max(int(cx - half_w), 0), min(int(np.ceil(cx + half_w)), w)
    y0, y1 = max(int(cy - half_h), 0), min(int(np.ceil(cy + half_h)), h)
    patch = depth[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]
    valid = patch[patch > 0]
    if valid.size == 0:
        raise ValueError(f"no depth readings inside box {box}")
    return float(np.percentile(valid, percentile))


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_proto(cls, p) -> Intrinsics:
        """From a camera's get_properties().intrinsic_parameters."""
        return cls(p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px)


def deproject(u: float, v: float, depth_mm: float, k: Intrinsics) -> tuple[float, float, float]:
    """Pixel (u, v) at `depth_mm` as an (x, y, z) point in the camera frame, mm."""
    return ((u - k.cx) * depth_mm / k.fx, (v - k.cy) * depth_mm / k.fy, depth_mm)


def box_center(box: BoundingBox) -> tuple[float, float]:
    return (box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2


def fingertip_reach_mm(geometries: Sequence, default: float = 50.0) -> float:
    """How far the gripper's geometry reaches past the gripper frame origin along +z, in mm.

    Motion moves the `gripper` frame origin, not the fingertips. On armfarm01 the "claws"
    box (center z=-6, 112 tall) ends 50 mm past the origin. Uses the furthest box end
    reported by gripper.get_geometries(), or `default` if there are no boxes.
    """
    ends = [g.center.z + g.box.dims_mm.z / 2 for g in geometries if g.HasField("box")]
    return max(ends) if ends else default


def holds_part(closed_position: float, empty_closed_position: float, open_position: float, margin: float) -> bool:
    """Whether the fingers stopped on something: at least `margin` short of where they
    stop closing on nothing, and at least `margin` short of open.

    Both bounds matter. Fingers that haven't closed yet are also "short of an empty
    close": on armfarm01 a read of 807 (still open, empty close 0) passed a one-sided
    check and the arm lifted without the part.

    UFactory gripper positions run from ~0 (closed) to ~810-840 (open), ~0.1 mm per unit.
    """
    return empty_closed_position + margin <= closed_position <= open_position - margin


def table_depth_mm(depth: np.ndarray) -> float:
    """Distance from the camera to the table: the median of every valid pixel.

    At the inspect pose the table fills most of the frame and parts are small, so the
    median lands on the table.
    """
    valid = depth[depth > 0]
    if valid.size == 0:
        raise ValueError("depth frame has no readings")
    return float(np.median(valid))


def grasp_heights(
    part_top_z: float,
    table_z: float,
    reach_mm: float,
    grip_depth_mm: float,
    approach_mm: float,
    table_clearance_mm: float = 5.0,
) -> tuple[float, float]:
    """World z for the gripper frame origin: (above the part, at the grasp).

    At the grasp the fingertips sit `grip_depth_mm` below the part's top surface, but
    never closer than `table_clearance_mm` to the table. The approach is `approach_mm`
    straight above the grasp.

    The table limit matters: on armfarm01 the configured table obstacle's top is at
    z=-23 while the real table is near z=0, so the motion planner alone would let the
    fingertips go into it.
    """
    fingertips = max(part_top_z - grip_depth_mm, table_z + table_clearance_mm)
    grasp = fingertips + reach_mm
    return grasp + approach_mm, grasp
