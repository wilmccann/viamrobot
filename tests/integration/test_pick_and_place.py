"""Pick one part off the table and drop it in the good box. MOVES THE ARM AND GRIPPER.

    1. arm to `inspect-pose`
    2. find a part with vision-6, locate it in 3D from the wrist depth camera, then
       open the gripper, come down on it from straight above and pick it up. The arm
       stays at the part until the fingers have stopped closing, and only leaves if they
       stopped on something (checked again after lifting); a miss lifts, opens the
       gripper and stops the test above the part
    3. arm (with the part) to `good-pose`, above the good box, and open the gripper to drop it
    4. arm back to `inspect-pose`, and check the part is gone from where it was

Clear the table except for the part(s), stand at the E-stop (red button on the xArm
controller), then run:

    VIAM_ALLOW_MOVES=1 uv run pytest tests/integration/test_pick_and_place.py -s

With several parts in view it takes the one vision-6 is most confident about.
If a step fails the arm stays where it stopped; nothing moves it back automatically.
"""

import asyncio
import time

import pytest
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from viamrobot.grasp import (
    Intrinsics,
    box_center,
    decode_depth,
    deproject,
    grasp_heights,
    part_depth_mm,
    table_depth_mm,
)
from viamrobot.pipeline import Frame
from viamrobot.stages.machine import MachineDetector

pytestmark = [pytest.mark.machine, pytest.mark.moves]

ARM, GRIPPER, CAMERA, DETECTOR, MOTION = "arm", "gripper", "cam", "vision-6", "builtin"
INSPECT_POSE, GOOD_POSE = "inspect-pose", "good-pose"

# erh:vmodutils:arm-position-saver switch positions: 0 "idle", 1 "update config",
# 2 "go to". Position 1 OVERWRITES the saved pose with wherever the arm is now.
GO_TO = 2

MIN_CONFIDENCE = 0.5  # vision-6 "part" boxes below this are ignored
GRIP_DEPTH_MM = 15.0  # fingertips this far below the part's top surface when closing
APPROACH_MM = 100.0  # come down on the part from this far above the grasp
# Fingertips never closer than this to the table (as measured by depth).
TABLE_CLEARANCE_MM = 2.0
# Fingertips hang this far below the `gripper` frame origin (which motion positions),
# measured on armfarm01 on 2026-11-14. The "claws" collision box ends at 50 mm, which
# put the fingertips ~10 mm low: into the cardboard, where they couldn't close.
FINGERTIP_REACH_MM = 60.0
# The gripper is driven with viam:ufactory:gripper's DoCommand {"set": value}, the path
# vla-controller uses: it waits for the jaw to settle (up to ~10 s), where grab()
# returns before the fingers move. is_holding_something reports True on any close.
# Positions: ~0 closed, ~810-840 open (~80 mm), ~0.1 mm per unit; read with
# {"get": true} -> "pos". If the fingers won't close at the part, check the fingertips
# aren't on the table: in the air they close in under a second.
GRIPPER_OPEN, GRIPPER_CLOSED = 840.0, 2.0  # as in the vla-controller config
SET_TIMEOUT_S = 15.0
FINGERS_OPEN_AT_LEAST = 700.0
EMPTY_CLOSE_MAX = 30.0  # an empty close stops at -3..2; at or above this, something is between the fingers
FINGERS_STILL = 3.0  # two reads 0.25 s apart within this = stopped
FINGERS_TIMEOUT_S = 10.0
SAFE_START_Z = 150.0  # gripper origin below this at the start = left down at a part
STRAIGHT_LINE = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=5)])
MOVE_TIMEOUT_S = 60.0
POSE_TOLERANCE_DEG = 1.0


async def wait_until_stopped(arm: Arm, timeout_s: float = MOVE_TIMEOUT_S) -> None:
    async with asyncio.timeout(timeout_s):
        while await arm.is_moving():
            await asyncio.sleep(0.2)


async def go_to_saved_pose(robot: RobotClient, arm: Arm, name: str) -> None:
    switch = Switch.from_robot(robot, name)
    _, labels = await switch.get_number_of_positions()
    assert labels[GO_TO] == "go to", f"{name}: unexpected switch positions {labels}"
    print(f"-> {name}")
    await switch.set_position(GO_TO, timeout=MOVE_TIMEOUT_S)
    await wait_until_stopped(arm)


async def move_gripper(motion: MotionClient, pose: Pose, straight: bool = False) -> None:
    print(f"-> gripper to x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f}" + (" (straight line)" if straight else ""))
    ok = await motion.move(
        GRIPPER,
        PoseInFrame(reference_frame="world", pose=pose),
        constraints=STRAIGHT_LINE if straight else None,
        timeout=MOVE_TIMEOUT_S,
    )
    assert ok, f"motion service could not move the gripper to {pose}"


async def finger_pos(gripper: Gripper) -> float:
    return float((await gripper.do_command({"get": True}))["pos"])


async def set_fingers(gripper: Gripper, value: float, label: str) -> float:
    """DoCommand {"set": value}, then wait until the fingers are still; return where they stopped."""
    start = time.perf_counter()
    await gripper.do_command({"set": value}, timeout=SET_TIMEOUT_S)
    set_s = time.perf_counter() - start
    async with asyncio.timeout(FINGERS_TIMEOUT_S):
        last = await finger_pos(gripper)
        while True:
            await asyncio.sleep(0.25)
            now = await finger_pos(gripper)
            if abs(now - last) <= FINGERS_STILL:
                break
            last = now
    print(f"   {label}: set {value:.0f} returned after {set_s:.1f} s; fingers still at {now:.0f} "
          f"after {time.perf_counter() - start:.1f} s")
    return now


async def close_and_wait(gripper: Gripper) -> float:
    """Close the fingers and return where they stopped (on the part, or ~0 on nothing)."""
    return await set_fingers(gripper, GRIPPER_CLOSED, "close")


def holding_part(position: float) -> bool:
    """Fingers stopped on something: clear of an empty close and well short of open."""
    return EMPTY_CLOSE_MAX <= position <= FINGERS_OPEN_AT_LEAST


async def open_fingers(gripper: Gripper) -> None:
    """Open the fingers fully; fail if they don't get there."""
    opened = await set_fingers(gripper, GRIPPER_OPEN, "open")
    if opened < FINGERS_OPEN_AT_LEAST:
        pytest.fail(f"gripper didn't open: fingers stopped at {opened:.0f}")


async def in_world(robot: RobotClient, frame: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
    """A point in `frame` (mm) as a pose in the world frame."""
    query = PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z, o_z=1))
    return (await robot.transform_pose(query, "world")).pose


def iou(a, b) -> float:
    """Overlap of two BoundingBoxes, 0 (apart) to 1 (identical)."""
    w = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    h = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = w * h
    union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter
    return inter / union if union else 0.0


async def test_pick_part_and_drop_in_good_box(robot):
    arm = Arm.from_robot(robot, ARM)
    gripper = Gripper.from_robot(robot, GRIPPER)
    camera = Camera.from_robot(robot, CAMERA)
    motion = MotionClient.from_robot(robot, MOTION)

    # 0. If a previous run stopped down at a part, lift straight up first: the joint move
    #    to inspect-pose could otherwise drag the fingers across the table. The fingers
    #    are opened at inspect-pose, before coming down on the part.
    start_z = (await in_world(robot, GRIPPER)).z
    if start_z < SAFE_START_Z:
        print(f"-> gripper starts low (z={start_z:.0f}); lifting straight up to z={SAFE_START_Z:.0f}")
        flange = await arm.get_end_position()
        flange.z += SAFE_START_Z - start_z
        await arm.move_to_position(flange)
        await wait_until_stopped(arm)

    # 1. Inspect pose.
    await go_to_saved_pose(robot, arm, INSPECT_POSE)
    inspect_joints = list((await arm.get_joint_positions()).values)

    # 2. Find a part, locate it in the world frame, and pick it up.
    images, _ = await camera.get_images()
    color = next((img for img in images if "dep" not in img.mime_type), None)
    depth_img = next((img for img in images if "dep" in img.mime_type), None)
    assert color is not None and depth_img is not None, f"need color + depth, got {[i.mime_type for i in images]}"

    detector = MachineDetector.from_robot(robot, DETECTOR, min_confidence=MIN_CONFIDENCE)
    boxes = await detector.find_items(Frame(color=color))
    assert boxes, f"{DETECTOR} sees no part at confidence >= {MIN_CONFIDENCE}"
    box = boxes[0]
    print(f"\n{len(boxes)} part(s) in view; picking {box.label} {box.confidence:.2f} at "
          f"({box.x_min:.0f},{box.y_min:.0f})-({box.x_max:.0f},{box.y_max:.0f}) px")

    intrinsics = Intrinsics.from_proto((await camera.get_properties()).intrinsic_parameters)
    depth = decode_depth(depth_img.data)
    u, v = box_center(box)
    part_cam = deproject(u, v, part_depth_mm(depth, box), intrinsics)
    table_cam = deproject(intrinsics.cx, intrinsics.cy, table_depth_mm(depth), intrinsics)
    part = await in_world(robot, CAMERA, *part_cam)
    table_z = (await in_world(robot, CAMERA, *table_cam)).z
    print(f"part top at x={part.x:.0f} y={part.y:.0f} z={part.z:.0f} mm; table at z={table_z:.0f} mm")
    assert part.z > table_z + 3, "the box has no height above the table; probably not a part"

    # Straight down, keeping the gripper's twist from the inspect pose.
    theta = (await in_world(robot, GRIPPER)).theta
    reach = FINGERTIP_REACH_MM
    above_z, grasp_z = grasp_heights(part.z, table_z, reach, GRIP_DEPTH_MM, APPROACH_MM, TABLE_CLEARANCE_MM)
    print(f"fingertips close at z={grasp_z - reach:.0f} mm (reach {reach:.0f} mm)")

    def over_part(z: float) -> Pose:
        return Pose(x=part.x, y=part.y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta)

    print("-> open gripper")
    await open_fingers(gripper)
    await move_gripper(motion, over_part(above_z))
    await move_gripper(motion, over_part(grasp_z), straight=True)

    # Don't leave the part until the fingers have finished closing on it, then check
    # where they stopped. Again after lifting, before going anywhere.
    print("-> close, waiting for the fingers to stop")
    closed = await close_and_wait(gripper)
    if not holding_part(closed):
        await move_gripper(motion, over_part(above_z), straight=True)
        await open_fingers(gripper)
        pytest.fail(f"pick missed: fingers stopped at {closed:.0f} (empty close < {EMPTY_CLOSE_MAX:.0f}); "
                    f"lifted {APPROACH_MM:.0f} mm above the part and opened")
    print(f"holding: fingers stopped at {closed:.0f}")

    await move_gripper(motion, over_part(above_z), straight=True)
    lifted = await finger_pos(gripper)
    print(f"after lift: fingers at {lifted:.0f}")
    if not holding_part(lifted):
        await open_fingers(gripper)
        pytest.fail(f"part slipped while lifting: fingers at {lifted:.0f}")

    # 3. Over the good box, then drop.
    await go_to_saved_pose(robot, arm, GOOD_POSE)
    print("-> open gripper (drop)")
    await open_fingers(gripper)

    # 4. Back to inspect.
    await go_to_saved_pose(robot, arm, INSPECT_POSE)
    final_joints = list((await arm.get_joint_positions()).values)
    off = max(abs(a - b) for a, b in zip(final_joints, inspect_joints))
    assert off <= POSE_TOLERANCE_DEG, f"not back at {INSPECT_POSE}: joints off by up to {off:.2f} deg"

    # The real proof of a pick: from the same pose, the part is gone from where it was.
    images, _ = await camera.get_images()
    color = next(img for img in images if "dep" not in img.mime_type)
    left = [b for b in await detector.find_items(Frame(color=color)) if iou(b, box) > 0.3]
    assert not left, f"the part is still on the table at {left[0]}: the pick missed"
