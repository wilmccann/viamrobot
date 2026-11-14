"""Locate a detected part in 3D and plan a top-down grasp on it. Pure math, no machine calls.

The wrist RealSense (`cam`) returns color and depth aligned to the same 640x480 grid, so a
`vision-9` box in the color frame indexes straight into the depth frame. The camera frame is
the usual pinhole one (+x right, +y down, +z out of the lens), in mm, which is the frame
`robot.transform_pose(..., "world")` expects for `cam`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
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


@dataclass(frozen=True)
class PartAxis:
    """A part's outline in the image, as a rotated rectangle."""

    u: float  # centre, image px
    v: float
    angle_deg: float  # direction of the long side in the image, [0, 180): 0 = +u (right), 90 = +v (down)
    length_px: float
    width_px: float

    @property
    def elongation(self) -> float:
        return self.length_px / max(self.width_px, 1e-6)

    def ends(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two ends of the long axis, image px."""
        a = np.radians(self.angle_deg)
        du, dv = np.cos(a) * self.length_px / 2, np.sin(a) * self.length_px / 2
        return (self.u - du, self.v - dv), (self.u + du, self.v + dv)


def part_axis(bgr: np.ndarray, box: BoundingBox, pad: int = 6, min_contrast: float = 25.0) -> PartAxis | None:
    """The long axis of the part in `box`, from the colour image (BGR, as cv2 decodes it).

    The part is whatever differs most from the background, where the background is the
    median colour of the border of the (padded) box: brown cardboard around a silver or
    blue part. Otsu splits the colour distance into part and background, and the largest
    blob gets a rotated rectangle (cv2.minAreaRect). Returns None when nothing in the box
    stands out by at least `min_contrast`.
    """
    h, w = bgr.shape[:2]
    x0, y0 = max(int(box.x_min) - pad, 0), max(int(box.y_min) - pad, 0)
    x1, y1 = min(int(np.ceil(box.x_max)) + pad, w), min(int(np.ceil(box.y_max)) + pad, h)
    crop = bgr[y0:y1, x0:x1].astype(np.float32)
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    border = np.concatenate([crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
                             crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
    dist = np.linalg.norm(crop - np.median(border, axis=0), axis=2)
    if dist.max() < min_contrast:
        return None
    dist8 = np.clip(dist * (255.0 / dist.max()), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(otsu, min_contrast * 255.0 / dist.max())
    mask = (dist8 > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < 20:
        return None
    (cu, cv_), _, _ = rect = cv2.minAreaRect(blob)
    p = cv2.boxPoints(rect)  # 4 corners in order
    e1, e2 = p[1] - p[0], p[2] - p[1]
    long_e, short_e = (e1, e2) if np.hypot(*e1) >= np.hypot(*e2) else (e2, e1)
    angle = float(np.degrees(np.arctan2(long_e[1], long_e[0]))) % 180.0
    return PartAxis(cu + x0, cv_ + y0, angle, float(np.hypot(*long_e)), float(np.hypot(*short_e)))


def wrap_half_turn(deg: float) -> float:
    """An angle in degrees, wrapped into [-90, 90): a direction that repeats every 180 degrees."""
    return (deg + 90.0) % 180.0 - 90.0


def jaw_turn_deg(part_yaw_deg: float, jaw_yaw_deg: float) -> float:
    """How far to turn the jaws (world yaw, degrees, CCW from above) so they close across the
    part: jaw direction = part's long axis + 90. Both directions repeat every 180 degrees
    (the jaws are symmetric), so the answer is in [-90, 90): never more than a quarter turn.
    """
    return wrap_half_turn(part_yaw_deg + 90.0 - jaw_yaw_deg)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Overlap of two boxes, 0 (apart) to 1 (identical)."""
    w = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    h = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = w * h
    union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter
    return inter / union if union else 0.0


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
