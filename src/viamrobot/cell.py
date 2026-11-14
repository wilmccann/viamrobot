"""The armfarm01 work cell: arm, gripper, wrist camera, saved poses, and the pick itself.

Shared by tests/integration/test_pick_and_place.py and the demo loop (viamrobot.demo).
Everything here that moves hardware was proven on armfarm01 on 2026-11-14; the numbers
below are what made the first successful pick work.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.media.video import ViamImage
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from viamrobot.grasp import (
    Intrinsics,
    PartAxis,
    box_center,
    decode_depth,
    deproject,
    grasp_heights,
    jaw_turn_deg,
    part_axis,
    part_depth_mm,
    table_depth_mm,
    wrap_half_turn,
)
from viamrobot.pipeline import BoundingBox

ARM, GRIPPER, CAMERA, MOTION = "arm", "gripper", "cam", "builtin"
# sortit:vision:parts9 (a teammate's module): vision-7 (yolov8n-part-v2, NPU) finds parts,
# vision-8 (yolov8n-anomaly-v2, NPU) vetoes "other" boxes and, with the default
# decider "vision8", decides. One Detection per part: class_name is the verdict
# ("normal" / "anomalous" / "hold") and confidence is the anomaly score, not a detection
# confidence. Boxes in the bottom band of the image (the fingers) are already dropped.
PARTS = "vision-9"
NORMAL, ANOMALOUS, HOLD = "normal", "anomalous", "hold"
INSPECT_POSE, GOOD_POSE, BAD_POSE, UNSURE_POSE = "inspect-pose", "good-pose", "bad-pose", "unsure-pose"

# erh:vmodutils:arm-position-saver switch positions: 0 "idle", 1 "update config",
# 2 "go to". Position 1 OVERWRITES the saved pose with wherever the arm is now.
GO_TO = 2

GRIP_DEPTH_MM = 15.0  # fingertips this far below the part's top surface when closing
APPROACH_MM = 100.0  # come down on the part from this far above the grasp
# Fingertips never closer than this to the depth-measured table. The depth camera reads
# the table 1-2 mm low (touch probe, 2026-11-14), and at 2 mm every pick in a demo run
# dragged the fingertips on the table: they stalled 43-55 mm apart after 4-10 s and
# shoved the part instead of lifting it. 5 mm leaves ~3 mm above the real table.
TABLE_CLEARANCE_MM = 5.0
# A grip counts only if the fingers stop within this of the part's measured width (its
# short side, from the outline). Fingers stalled on the table stop far wider than that.
GRIP_SLACK_MM = 12.0
GRIPPER_UNITS_PER_MM = 10.0  # UFactory gripper position: ~0.1 mm per unit (840 = ~84 mm open)
# Fingertips hang this far below the `gripper` frame origin (which motion positions),
# measured on armfarm01. The "claws" collision box ends at 50 mm, which put the
# fingertips ~10 mm low: into the cardboard, where they couldn't close.
FINGERTIP_REACH_MM = 60.0

# The gripper is driven with viam:ufactory:gripper's DoCommand {"set": value}: it waits
# for the jaw to settle (up to ~10 s), where grab() returns before the fingers move, and
# is_holding_something reports True on any close. Positions: ~0 closed, ~810-840 open
# (~80 mm), ~0.1 mm per unit; read with {"get": true} -> "pos". If the fingers won't
# close at the part, check the fingertips aren't on the table: in the air they close
# in under a second.
GRIPPER_OPEN, GRIPPER_CLOSED = 840.0, 2.0
SET_TIMEOUT_S = 15.0
FINGERS_OPEN_AT_LEAST = 700.0
EMPTY_CLOSE_MAX = 30.0  # an empty close stops at -3..2; at or above this, something is between the fingers
FINGERS_STILL = 3.0  # two reads 0.25 s apart within this = stopped
FINGERS_TIMEOUT_S = 10.0
FINGERS_MIN_TRAVEL = 20.0  # a close has started once the fingers have moved this far in
CLOSE_START_TIMEOUT_S = 2.0  # grab() normally gets the fingers moving well within this

SAFE_Z = 150.0  # gripper origin below this = down near the table; lift straight up before a joint move
STRAIGHT_LINE = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=5)])
MOVE_TIMEOUT_S = 60.0

# Turning the jaws across long parts. A part whose outline is at least ELONGATED times
# longer than wide gets the jaws turned to close across it; anything rounder keeps the
# inspect pose's twist. The jaws close along the gripper frame's y axis (the "claws"
# collision box is 120 mm along y); JAW_AXIS is a point on that axis.
ELONGATED = 1.5
JAW_AXIS = (0.0, 100.0, 0.0)
# Jaws within this of square to the part are good enough: a bolt is ~8 mm thick and the
# jaws open ~80 mm. (7 deg off happened on the first try; the inspect pose isn't quite
# straight down, so the computed turn is a few degrees out.)
JAW_TOLERANCE_DEG = 10.0
# Which way a change in the orientation-vector theta turns the jaws in world yaw (CCW
# from above), with the gripper pointing straight down (o = 0,0,-1). Theta is a rotation
# about o, which points down, so +theta should turn the jaws clockwise: -1. pick()
# measures the result and flips this if it's wrong.
THETA_TO_YAW = -1.0

# Second look (see Cell.second_look): match the part within this of where the first look
# says it is; trust the refined position only if it moved less than SECOND_LOOK_MAX_MM;
# turn the jaws again only if the part's direction changed by more than RETURN_DEG.
SECOND_LOOK_MAX_PX = 120.0
SECOND_LOOK_MAX_MM = 40.0
RETURN_DEG = 10.0


class PickMissed(RuntimeError):
    """The fingers closed on nothing, or the part slipped. The gripper was opened."""


@dataclass(frozen=True)
class SeenPart:
    """One part vision-9 found: where it is in the image and its verdict."""

    box: BoundingBox
    verdict: str  # NORMAL / ANOMALOUS / HOLD
    score: float  # vision-9's anomaly score (0..1 with decider "vision8"; >0.5 = anomaly)


@dataclass(frozen=True)
class Snapshot:
    """One look from the wrist camera at the inspect pose, with every part in view."""

    color: ViamImage
    bgr: np.ndarray
    depth: np.ndarray
    intrinsics: Intrinsics
    parts: list[SeenPart]  # nearest the image centre first


@dataclass(frozen=True)
class PartLocation:
    x: float
    y: float
    top_z: float  # top surface of the part, world mm
    table_z: float
    yaw_deg: float | None = None  # long axis in the world (CCW from +x, seen from above); None = not long
    elongation: float = 1.0
    width_mm: float | None = None  # what the jaws should close to; None = unknown


def match_part(parts: list[SeenPart], expected_uv: tuple[float, float], max_px: float) -> SeenPart | None:
    """The part whose box centre is nearest `expected_uv`, if within `max_px`."""
    if not parts:
        return None
    best = min(parts, key=lambda p: math.dist(box_center(p.box), expected_uv))
    return best if math.dist(box_center(best.box), expected_uv) <= max_px else None


def yaw_between(a: Pose, b: Pose) -> float:
    """World yaw (degrees, CCW from +x seen from above) of the line from a to b."""
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


class Cell:
    """armfarm01's arm, gripper, wrist camera and services, from one RobotClient."""

    def __init__(self, robot: RobotClient, log: Callable[[str], None] = print):
        self.robot = robot
        self.log = log
        self.arm = Arm.from_robot(robot, ARM)
        self.gripper = Gripper.from_robot(robot, GRIPPER)
        self.camera = Camera.from_robot(robot, CAMERA)
        self.motion = MotionClient.from_robot(robot, MOTION)
        self.parts = VisionClient.from_robot(robot, PARTS)
        self.theta_to_yaw = THETA_TO_YAW  # learned per connection if the first turn comes out backwards
        self.theta_bias = 0.0  # learned: theta the first turn of each pick is short by

    # --- arm -----------------------------------------------------------------

    async def ping(self) -> None:
        """A cheap call that fails if the connection to the machine is gone."""
        await self.arm.get_joint_positions(timeout=5)

    async def wait_until_stopped(self, timeout_s: float = MOVE_TIMEOUT_S) -> None:
        async with asyncio.timeout(timeout_s):
            while await self.arm.is_moving():
                await asyncio.sleep(0.2)

    async def go_to(self, name: str) -> None:
        """Move to a saved pose through its arm-position-saver switch (a joint move)."""
        switch = Switch.from_robot(self.robot, name)
        _, labels = await switch.get_number_of_positions()
        if labels[GO_TO] != "go to":
            raise RuntimeError(f"{name}: unexpected switch positions {labels}")
        self.log(f"-> {name}")
        await switch.set_position(GO_TO, timeout=MOVE_TIMEOUT_S)
        await self.wait_until_stopped()

    async def joints(self) -> list[float]:
        return list((await self.arm.get_joint_positions()).values)

    async def in_world(self, frame: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
        """A point in `frame` (mm) as a pose in the world frame."""
        query = PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z, o_z=1))
        return (await self.robot.transform_pose(query, "world")).pose

    async def move_gripper(self, pose: Pose, straight: bool = False) -> None:
        self.log(f"-> gripper to x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f}" + (" (straight line)" if straight else ""))
        ok = await self.motion.move(
            GRIPPER,
            PoseInFrame(reference_frame="world", pose=pose),
            constraints=STRAIGHT_LINE if straight else None,
            timeout=MOVE_TIMEOUT_S,
        )
        if not ok:
            raise RuntimeError(f"motion service could not move the gripper to {pose}")

    async def lift_clear(self) -> None:
        """If the gripper is down near the table, lift straight up to SAFE_Z.

        A joint move to a saved pose from down there could drag the fingers across the table.
        """
        z = (await self.in_world(GRIPPER)).z
        if z >= SAFE_Z:
            return
        self.log(f"-> gripper is low (z={z:.0f}); lifting straight up to z={SAFE_Z:.0f}")
        flange = await self.arm.get_end_position()
        flange.z += SAFE_Z - z
        await self.arm.move_to_position(flange)
        await self.wait_until_stopped()

    # --- gripper -------------------------------------------------------------

    async def finger_pos(self) -> float:
        return float((await self.gripper.do_command({"get": True}))["pos"])

    async def set_fingers(self, value: float, label: str) -> float:
        """DoCommand {"set": value}, then wait until the fingers are still; return where they stopped."""
        start = time.perf_counter()
        await self.gripper.do_command({"set": value}, timeout=SET_TIMEOUT_S)
        set_s = time.perf_counter() - start
        async with asyncio.timeout(FINGERS_TIMEOUT_S):
            last = await self.finger_pos()
            while True:
                await asyncio.sleep(0.25)
                now = await self.finger_pos()
                if abs(now - last) <= FINGERS_STILL:
                    break
                last = now
        self.log(f"   {label}: set {value:.0f} returned after {set_s:.1f} s; fingers still at {now:.0f} "
                 f"after {time.perf_counter() - start:.1f} s")
        return now

    async def open_fingers(self) -> None:
        """Open the fingers fully; raise if they don't get there."""
        opened = await self.set_fingers(GRIPPER_OPEN, "open")
        if opened < FINGERS_OPEN_AT_LEAST:
            raise RuntimeError(f"gripper didn't open: fingers stopped at {opened:.0f}")

    async def close_fingers(self) -> float:
        """Close the fingers and return where they stopped (on the part, or ~0 on nothing).

        Uses grab(), which returns at once, and watches the fingers: started moving, then
        still. DoCommand "set" can't be used for this: when the fingers stop on a part they
        never reach "closed", and the driver polls for up to 10 s (it took 8-10.5 s on
        some picks, and one went over the 20 s part timeout). Falls back to "set" if
        grab() doesn't get the fingers moving.
        """
        start = time.perf_counter()
        before = await self.finger_pos()
        await self.gripper.grab(timeout=SET_TIMEOUT_S)
        try:
            async with asyncio.timeout(CLOSE_START_TIMEOUT_S):
                while (now := await self.finger_pos()) > before - FINGERS_MIN_TRAVEL:
                    await asyncio.sleep(0.05)
        except TimeoutError:
            self.log(f"   close: grab() didn't move the fingers in {CLOSE_START_TIMEOUT_S:.0f} s; using set")
            return await self.set_fingers(GRIPPER_CLOSED, "close")
        async with asyncio.timeout(FINGERS_TIMEOUT_S):
            last = now
            while True:
                await asyncio.sleep(0.2)
                now = await self.finger_pos()
                if abs(now - last) <= FINGERS_STILL:
                    break
                last = now
        self.log(f"   close: fingers still at {now:.0f} after {time.perf_counter() - start:.1f} s")
        return now

    @staticmethod
    def holding_part(position: float, width_mm: float | None = None) -> bool:
        """Fingers stopped on something: clear of an empty close and well short of open,
        and, when the part's width is known, no wider than it plus GRIP_SLACK_MM.

        The width bound is what tells a grip from fingers stalled on the table: those
        stop half-closed (~45 mm apart) whatever the part is.
        """
        widest = FINGERS_OPEN_AT_LEAST
        if width_mm is not None:
            widest = min(widest, (width_mm + GRIP_SLACK_MM) * GRIPPER_UNITS_PER_MM)
        return EMPTY_CLOSE_MAX <= position <= widest

    async def fingers_settled(self, timeout_s: float = 2.0) -> float:
        """Finger position once two reads 0.3 s apart agree (or after `timeout_s`).

        After a lift, fingers that were stalled on the table can carry on closing."""
        last = await self.finger_pos()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.3)
            now = await self.finger_pos()
            if abs(now - last) <= FINGERS_STILL:
                return now
            last = now
        return last

    # --- seeing ----------------------------------------------------------------

    async def look(self) -> Snapshot:
        """Color + depth from the wrist camera, and every part vision-9 finds in that colour frame."""
        images, _ = await self.camera.get_images()
        color = next((img for img in images if "dep" not in img.mime_type), None)
        depth = next((img for img in images if "dep" in img.mime_type), None)
        if color is None or depth is None:
            raise RuntimeError(f"need color + depth from {CAMERA}, got {[i.mime_type for i in images]}")
        bgr = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("could not decode the colour frame")
        intrinsics = Intrinsics.from_proto((await self.camera.get_properties()).intrinsic_parameters)
        parts = [
            SeenPart(BoundingBox(d.x_min, d.y_min, d.x_max, d.y_max, confidence=d.confidence, label=d.class_name),
                     d.class_name, d.confidence)
            for d in await self.parts.get_detections(color)
        ]
        # vision-9's confidence is an anomaly score, so it says nothing about which part to
        # do first: take the one nearest the image centre (straight below the camera).
        parts.sort(key=lambda p: math.dist(box_center(p.box), (intrinsics.cx, intrinsics.cy)))
        return Snapshot(color, bgr, decode_depth(depth.data), intrinsics, parts)

    async def locate(self, snap: Snapshot, box: BoundingBox) -> PartLocation:
        """Where the part in `box` is in the world, and which way it points, from the frames
        taken with it. Only valid while the arm is still where `snap` was taken.

        The centre and direction come from the part's outline in the colour image (falling
        back to the box centre); the height from the depth frame.
        """
        k = snap.intrinsics
        axis = part_axis(snap.bgr, box)
        u, v = (axis.u, axis.v) if axis else box_center(box)
        depth_mm = part_depth_mm(snap.depth, box)
        part = await self.in_world(CAMERA, *deproject(u, v, depth_mm, k))
        table = await self.in_world(CAMERA, *deproject(k.cx, k.cy, table_depth_mm(snap.depth), k))
        self.log(f"part top at x={part.x:.0f} y={part.y:.0f} z={part.z:.0f} mm; table at z={table.z:.0f} mm")
        if part.z <= table.z + 3:
            # Not a reason to skip it: vision-9 already decided it's a part, and a bolt lying
            # flat (~8 mm) is within depth noise (~2 mm) of this. The grasp stays safe:
            # grasp_heights keeps the fingertips TABLE_CLEARANCE_MM above the table.
            self.log(f"   part is only {part.z - table.z:.0f} mm above the table by depth; gripping just above the table")

        yaw, elongation = None, 1.0
        mm_per_px = depth_mm / k.fx
        if axis is None:
            width_mm = min(box.x_max - box.x_min, box.y_max - box.y_min) * mm_per_px
            self.log("   no outline found in the box; keeping the gripper's twist")
        else:
            elongation = axis.elongation
            # The jaws close across the short side for a long part; for a round one, the
            # long side is the widest it could be.
            width_mm = (axis.width_px if elongation >= ELONGATED else axis.length_px) * mm_per_px
            if elongation >= ELONGATED:
                yaw = await self._world_yaw(axis, depth_mm, k)
            self.log(f"   outline {axis.length_px:.0f}x{axis.width_px:.0f} px at {axis.angle_deg:.0f} deg in the image"
                     + (f" -> long part, world yaw {yaw:.0f} deg" if yaw is not None else " -> not long; keeping the twist")
                     + f"; grip width ~{width_mm:.0f} mm")
        return PartLocation(part.x, part.y, part.z, table.z, yaw, elongation, width_mm)

    async def _world_yaw(self, axis: PartAxis, depth_mm: float, k: Intrinsics) -> float:
        """The part's long axis in the world: both ends of it in the image, into the world."""
        (u0, v0), (u1, v1) = axis.ends()
        a = await self.in_world(CAMERA, *deproject(u0, v0, depth_mm, k))
        b = await self.in_world(CAMERA, *deproject(u1, v1, depth_mm, k))
        return yaw_between(a, b)

    async def jaw_yaw(self) -> float:
        """The direction the jaws close along, in the world (degrees, CCW from +x)."""
        return yaw_between(await self.in_world(GRIPPER), await self.in_world(GRIPPER, *JAW_AXIS))

    # --- doing -----------------------------------------------------------------

    async def _turn_across(self, part: PartLocation, over_part: Callable[[float, float], Pose], z: float,
                           theta0: float, from_inspect: bool = True) -> float:
        """Move above the part with the jaws turned across its long axis; return the theta used.

        The turn is computed from the jaws' current direction and applied as a change of
        theta, then measured. If it's off: correct by the measured error, then (if still
        off) by the yaw-per-theta measured from the two tries, which also settles which
        way theta turns the jaws for the rest of the session. All at the safe height.
        """
        want = part.yaw_deg + 90.0
        turn = jaw_turn_deg(part.yaw_deg, await self.jaw_yaw())
        self.log(f"   jaws to {wrap_half_turn(want):.0f} deg: turn {turn:+.0f} deg")
        tried: list[tuple[float, float]] = []  # (theta, measured jaw yaw)
        planned = theta0 + self.theta_to_yaw * turn
        # The learned bias corrects for starting at the inspect pose (not quite straight
        # down); a turn that starts straight down (after the second look) doesn't need it.
        theta = planned + (self.theta_bias if from_inspect else 0.0)
        for attempt in (1, 2, 3):
            await self.move_gripper(over_part(z, theta))
            yaw = await self.jaw_yaw()
            err = wrap_half_turn(want - yaw)
            self.log(f"   turn {attempt}: theta {theta:.1f} -> jaws {err:+.1f} deg off")
            if abs(err) <= JAW_TOLERANCE_DEG:
                # Remember the constant part of the error (the inspect pose isn't quite
                # straight down, ~10 deg on armfarm01) so the next part's first turn lands.
                if from_inspect:
                    self.theta_bias = theta + self.theta_to_yaw * err - planned
                return theta
            tried.append((theta, yaw))
            if len(tried) == 1:
                # Correct by the measured error, assuming the direction is right.
                theta = theta + self.theta_to_yaw * err
            else:
                # Two measurements give the real yaw change per degree of theta (sign and size).
                (t1, y1), (t2, y2) = tried[-2:]
                gain = wrap_half_turn(y2 - y1) / (t2 - t1) if abs(t2 - t1) > 0.5 else 0.0
                if abs(gain) < 0.3:  # no usable estimate: assume the other direction
                    gain = -self.theta_to_yaw
                self.theta_to_yaw = 1.0 if gain > 0 else -1.0
                self.log(f"   measured {gain:+.2f} deg of jaw yaw per deg of theta")
                theta = theta + err / gain
        raise RuntimeError(f"couldn't turn the jaws across the part ({err:+.0f} deg off)")

    async def second_look(self, part: PartLocation, theta: float, z: float) -> PartLocation:
        """Look at the part again from above it, and return a refined location (or `part`
        unchanged if the second look doesn't find it or disagrees too much).

        Seen from the inspect pose, a part near the edge of the image is located a few mm
        off (the camera's mounting in the config looks ~2 deg off: the flat table reads
        ~9 mm higher at one corner), enough to miss a small part. So: shift sideways by
        the camera's offset from the gripper (~83 mm) so the camera is straight above the
        part, at the same height, take a new frame, and locate the part from that. It's
        then near the image centre and closer, where those errors are small. (Straight
        above the gripper instead, the part would sit in the bottom band of the image,
        which vision-9 drops because the fingers show there.)

        Heights are kept from the first look: the clearance was tuned for that distance.
        """
        gripper = await self.in_world(GRIPPER)
        cam = await self.in_world(CAMERA)
        dx, dy = cam.x - gripper.x, cam.y - gripper.y
        self.log("-> second look")
        await self.move_gripper(Pose(x=part.x - dx, y=part.y - dy, z=z, o_x=0, o_y=0, o_z=-1, theta=theta))
        snap = await self.look()
        k = snap.intrinsics
        q = PoseInFrame(reference_frame="world", pose=Pose(x=part.x, y=part.y, z=part.top_z, o_z=1))
        p = (await self.robot.transform_pose(q, CAMERA)).pose
        expected = (k.fx * p.x / p.z + k.cx, k.fy * p.y / p.z + k.cy)
        seen = match_part(snap.parts, expected, SECOND_LOOK_MAX_PX)
        if seen is None:
            self.log(f"   second look: no part near ({expected[0]:.0f},{expected[1]:.0f}) px; keeping the first look")
            return part
        again = await self.locate(snap, seen.box)
        moved = math.hypot(again.x - part.x, again.y - part.y)
        if moved > SECOND_LOOK_MAX_MM:
            self.log(f"   second look puts it {moved:.0f} mm away; not trusting that, keeping the first look")
            return part
        self.log(f"   second look: part is {again.x - part.x:+.1f}, {again.y - part.y:+.1f} mm from the first look")
        return PartLocation(again.x, again.y, part.top_z, part.table_z,
                            again.yaw_deg if again.yaw_deg is not None else part.yaw_deg,
                            again.elongation, again.width_mm if again.width_mm is not None else part.width_mm)

    async def pick(self, part: PartLocation) -> float:
        """Open, come straight down on the part, close, confirm the grip, lift, confirm again.

        For a long part the jaws are first turned to close across it, at the safe height
        above it. The arm doesn't leave the part until the fingers have stopped on
        something. On a miss it lifts, opens and raises PickMissed. Returns where the
        fingers stopped.
        """
        theta0 = (await self.in_world(GRIPPER)).theta  # the inspect pose's twist
        above_z, grasp_z = grasp_heights(
            part.top_z, part.table_z, FINGERTIP_REACH_MM, GRIP_DEPTH_MM, APPROACH_MM, TABLE_CLEARANCE_MM
        )
        self.log(f"fingertips close at z={grasp_z - FINGERTIP_REACH_MM:.0f} mm")

        def at(z: float, theta: float) -> Pose:
            return Pose(x=part.x, y=part.y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta)

        await self.open_fingers()
        if part.yaw_deg is None:
            theta = theta0
            await self.move_gripper(at(above_z, theta))
        else:
            theta = await self._turn_across(part, at, above_z, theta0)

        # Second look, from closer and with the part near the middle of the image.
        first = part
        part = await self.second_look(part, theta, above_z)
        if part is not first:
            turned = first.yaw_deg is not None
            if part.yaw_deg is not None and (
                not turned or abs(wrap_half_turn(part.yaw_deg - first.yaw_deg)) > RETURN_DEG
            ):
                theta = await self._turn_across(part, at, above_z, theta, from_inspect=False)
            else:
                await self.move_gripper(at(above_z, theta))

        def over_part(z: float) -> Pose:
            return at(z, theta)

        await self.move_gripper(over_part(grasp_z), straight=True)

        closed = await self.close_fingers()
        if not self.holding_part(closed, part.width_mm):
            await self.move_gripper(over_part(above_z), straight=True)
            await self.open_fingers()
            expected = f", part ~{part.width_mm:.0f} mm wide" if part.width_mm is not None else ""
            raise PickMissed(f"fingers stopped at {closed:.0f} (~{closed / GRIPPER_UNITS_PER_MM:.0f} mm apart{expected})")
        self.log(f"holding: fingers stopped at {closed:.0f} (~{closed / GRIPPER_UNITS_PER_MM:.0f} mm apart)")

        await self.move_gripper(over_part(above_z), straight=True)
        lifted = await self.fingers_settled()
        self.log(f"after lift: fingers at {lifted:.0f}")
        if not self.holding_part(lifted, part.width_mm):
            await self.open_fingers()
            raise PickMissed(f"part slipped while lifting: fingers at {lifted:.0f}")
        return lifted

    async def place(self, pose_name: str) -> None:
        """Carry the held part to a saved pose and open the gripper there."""
        await self.go_to(pose_name)
        self.log("-> open gripper (drop)")
        await self.open_fingers()

    async def stop(self) -> None:
        """Best effort: stop the arm where it is."""
        try:
            await self.arm.stop(timeout=5)
        except Exception as exc:
            self.log(f"   arm stop failed: {type(exc).__name__}: {exc}")

    async def recover(self) -> None:
        """After a failure: stop the arm, open the gripper (drops anything held), lift clear
        if low, then go to the inspect pose.

        Opening is best effort, so a gripper fault doesn't keep the arm away from a safe pose.
        """
        self.log("-> recover: stop, open gripper, inspect pose")
        await self.stop()
        try:
            await self.open_fingers()
        except Exception as exc:
            self.log(f"   open failed: {type(exc).__name__}: {exc}")
        await self.lift_clear()
        await self.go_to(INSPECT_POSE)
