"""Let the armfarm01-smolvla policy pick up a part. MOVES THE ARM AND GRIPPER.

    0. if a previous run left the gripper low, lift straight up (no gripper command:
       the UFactory gripper can stall while the arm sits still)
    1. arm to START_POSE
    2. start `vla-controller` (viam-labs:vla, policy `policy` = armfarm01-smolvla on the
       GPU) with its configured task, "pick up one object", and log `status` every 0.5 s
    3. stop it: when it stops or errors by itself, after VLA_MAX_S, or on any failure.
       `stop` is always sent. The arm stays wherever the policy left it.

The controller's own safety still applies (vla-controller config): it refuses to start
if the first action is > 15 deg from the current joints, and clamps joints to 45 deg/s.

Stand at the E-stop (red button on the xArm controller), then run:

    VIAM_ALLOW_MOVES=1 uv run pytest tests/integration/test_vla_pick.py -s
"""

import asyncio
import time

import pytest
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.generic import Generic

from viamrobot.cell import Cell

pytestmark = [pytest.mark.machine, pytest.mark.moves]

CONTROLLER = "vla-controller"
START_POSE = "inspect-pose"
VLA_MAX_S = 45.0
POLL_S = 0.5


async def status_line(controller: Generic, arm: Arm, gripper: Gripper, t: float) -> tuple[str, dict]:
    status = await controller.do_command({"command": "status"})
    pos = (await gripper.do_command({"get": True})).get("pos")
    flange = await arm.get_end_position()
    print(f"   t={t:5.1f}s state={status.get('state')} fps={status.get('measured_fps')} "
          f"clamps={status.get('clamp_counts')} starved={status.get('starved_ticks')} "
          f"flange=({flange.x:.0f},{flange.y:.0f},{flange.z:.0f}) gripper={pos} "
          f"err={status.get('last_error') or '-'}")
    return str(status.get("state")), status


async def test_vla_picks_up_a_part(robot):
    cell = Cell(robot)
    arm, gripper = cell.arm, cell.gripper
    controller = Generic.from_robot(robot, CONTROLLER)

    state, _ = await status_line(controller, arm, gripper, 0.0)
    assert state in ("idle", "stopped"), f"{CONTROLLER} is already {state}; not starting another run"

    # 0-1. Clear the table first if a previous run stopped down at a part, then the start pose.
    await cell.lift_clear()
    await cell.go_to(START_POSE)

    # 2-3. Run the policy, always stopping it.
    print(f"-> {CONTROLLER} start")
    result = await controller.do_command({"command": "start"})
    assert result.get("ok"), f"start refused: {result}"
    start = time.perf_counter()
    final = {}
    try:
        while (t := time.perf_counter() - start) < VLA_MAX_S:
            await asyncio.sleep(POLL_S)
            state, final = await status_line(controller, arm, gripper, t)
            if state in ("stopped", "error", "idle"):
                break
    finally:
        await controller.do_command({"command": "stop"})
        print(f"-> {CONTROLLER} stop")
        await status_line(controller, arm, gripper, time.perf_counter() - start)

    assert final.get("state") != "error", f"policy run errored: {final.get('last_error')}"
