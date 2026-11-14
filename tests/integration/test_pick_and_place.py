"""Pick one part off the table and drop it in the good box. MOVES THE ARM AND GRIPPER.

    0. if a previous run left the gripper low, lift straight up
    1. arm to `inspect-pose`
    2. find a part with vision-9, locate it in 3D from the wrist depth camera (turning the
       jaws across it if it is long), then
       open the gripper, come down on it from straight above and pick it up. The arm
       stays at the part until the fingers have stopped closing, and only leaves if they
       stopped on something (checked again after lifting); a miss lifts, opens the
       gripper and stops the test above the part
    3. arm (with the part) to `good-pose`, above the good box, and open the gripper to drop it
    4. arm back to `inspect-pose`, and check the part is gone from where it was

The moves themselves live in viamrobot.cell, shared with the demo loop (viamrobot.demo).

Clear the table except for the part(s), stand at the E-stop (red button on the xArm
controller), then run:

    VIAM_ALLOW_MOVES=1 uv run pytest tests/integration/test_pick_and_place.py -s

With several parts in view it takes the one nearest the centre of the camera image.
If a step fails the arm stays where it stopped; nothing moves it back automatically.
"""

import pytest

from viamrobot.cell import GOOD_POSE, INSPECT_POSE, PARTS, Cell, PickMissed
from viamrobot.grasp import iou

pytestmark = [pytest.mark.machine, pytest.mark.moves]

POSE_TOLERANCE_DEG = 1.0


async def test_pick_part_and_drop_in_good_box(robot):
    cell = Cell(robot)

    # 0-1. Clear of the table, then the inspect pose.
    await cell.lift_clear()
    await cell.go_to(INSPECT_POSE)
    inspect_joints = await cell.joints()

    # 2. Find a part, locate it in the world frame, and pick it up.
    snap = await cell.look()
    assert snap.parts, f"{PARTS} sees no part"
    box = snap.parts[0].box
    print(f"\n{len(snap.parts)} part(s) in view; picking the one at "
          f"({box.x_min:.0f},{box.y_min:.0f})-({box.x_max:.0f},{box.y_max:.0f}) px ({snap.parts[0].verdict})")
    part = await cell.locate(snap, box)
    try:
        await cell.pick(part)
    except PickMissed as exc:
        pytest.fail(f"pick missed: {exc}; lifted above the part and opened")

    # 3. Over the good box, then drop.
    await cell.place(GOOD_POSE)

    # 4. Back to inspect.
    await cell.go_to(INSPECT_POSE)
    off = max(abs(a - b) for a, b in zip(await cell.joints(), inspect_joints))
    assert off <= POSE_TOLERANCE_DEG, f"not back at {INSPECT_POSE}: joints off by up to {off:.2f} deg"

    # The real proof of a pick: from the same pose, the part is gone from where it was.
    left = [p.box for p in (await cell.look()).parts if iou(p.box, box) > 0.3]
    assert not left, f"the part is still on the table at {left[0]}: the pick missed"
