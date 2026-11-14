"""Cell._turn_across against a simulated wrist: jaw yaw = gain * theta + offset (world degrees)."""

import pytest

from viamrobot.cell import JAW_TOLERANCE_DEG, Cell, PartLocation
from viamrobot.grasp import wrap_half_turn


class SimWrist(Cell):
    def __init__(self, gain: float, offset: float, start_theta: float, guess: float = -1.0):
        # No robot: only the parts of Cell that _turn_across uses.
        self.log = lambda msg: None
        self.theta_to_yaw = guess
        self.gain, self.offset, self.theta = gain, offset, start_theta
        self.moves = 0

    async def move_gripper(self, pose, straight=False):
        self.theta, self.moves = pose.theta, self.moves + 1

    async def jaw_yaw(self):
        return wrap_half_turn(self.gain * self.theta + self.offset)


def at(z, theta):
    from viam.proto.common import Pose

    return Pose(z=z, o_z=-1, theta=theta)


@pytest.mark.parametrize("gain, guess", [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)])
@pytest.mark.parametrize("offset", [0.0, 7.0, -12.0])
@pytest.mark.parametrize("part_yaw", [-91.7, 0.0, 30.0, 135.0])
async def test_turn_ends_square_to_the_part(gain, guess, offset, part_yaw):
    theta0 = -169.7
    wrist = SimWrist(gain, offset, theta0, guess)
    # The straight-down pose is a few degrees off the inspect pose the turn is computed from.
    wrist.offset = offset + 5.0
    part = PartLocation(0, 0, 10, 8, yaw_deg=part_yaw, elongation=6)

    theta = await wrist._turn_across(part, at, 170, theta0)

    assert abs(wrap_half_turn(part_yaw + 90 - (gain * theta + wrist.offset))) <= JAW_TOLERANCE_DEG
    assert wrist.moves <= 3


async def test_learns_which_way_theta_turns():
    wrist = SimWrist(gain=1.0, offset=0.0, start_theta=0.0, guess=-1.0)
    await wrist._turn_across(PartLocation(0, 0, 10, 8, yaw_deg=40.0), at, 170, 0.0)
    assert wrist.theta_to_yaw == 1.0
