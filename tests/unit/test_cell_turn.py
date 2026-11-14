"""Cell._turn_across against a simulated wrist: jaw yaw = gain * theta + offset (world degrees)."""

import pytest

from viamrobot.cell import JAW_TOLERANCE_DEG, Cell, PartLocation
from viamrobot.grasp import wrap_half_turn


class SimWrist(Cell):
    """jaw yaw = gain * theta + offset once the gripper points straight down (after a move);
    at the inspect pose it starts from, the jaws read `inspect_skew` degrees differently,
    as on armfarm01 (~10 deg), so a turn computed there comes out that far off."""

    def __init__(self, gain: float, offset: float, start_theta: float, guess: float = -1.0, inspect_skew: float = 0.0):
        # No robot: only the parts of Cell that _turn_across uses.
        self.log = lambda msg: None
        self.theta_to_yaw = guess
        self.theta_bias = 0.0
        self.gain, self.offset, self.theta, self.inspect_skew = gain, offset, start_theta, inspect_skew
        self.moves = 0
        self.at_inspect = True

    async def move_gripper(self, pose, straight=False):
        self.theta, self.moves, self.at_inspect = pose.theta, self.moves + 1, False

    async def jaw_yaw(self):
        skew = self.inspect_skew if self.at_inspect else 0.0
        return wrap_half_turn(self.gain * self.theta + self.offset + skew)


def at(z, theta):
    from viam.proto.common import Pose

    return Pose(z=z, o_z=-1, theta=theta)


@pytest.mark.parametrize("gain, guess", [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)])
@pytest.mark.parametrize("offset", [0.0, 7.0, -12.0])
@pytest.mark.parametrize("part_yaw", [-91.7, 0.0, 30.0, 135.0])
async def test_turn_ends_square_to_the_part(gain, guess, offset, part_yaw):
    theta0 = -169.7
    # The straight-down pose is ~10 deg off the inspect pose the turn is computed from.
    wrist = SimWrist(gain, offset, theta0, guess, inspect_skew=-10.5)
    part = PartLocation(0, 0, 10, 8, yaw_deg=part_yaw, elongation=6)

    theta = await wrist._turn_across(part, at, 170, theta0)

    assert abs(wrap_half_turn(part_yaw + 90 - (gain * theta + wrist.offset))) <= JAW_TOLERANCE_DEG
    assert wrist.moves <= 3


async def test_learned_offset_makes_the_next_first_turn_land():
    # As on armfarm01: the first turn of every pick came out +10.5 deg off.
    wrist = SimWrist(gain=-1.0, offset=0.0, start_theta=-170.0, guess=-1.0, inspect_skew=-10.5)
    await wrist._turn_across(PartLocation(0, 0, 10, 8, yaw_deg=175.0), at, 170, -170.0)
    assert wrist.moves == 2

    wrist.moves, wrist.theta, wrist.at_inspect = 0, -170.0, True  # back at the inspect pose
    theta = await wrist._turn_across(PartLocation(0, 0, 10, 8, yaw_deg=-98.0), at, 170, -170.0)
    assert wrist.moves == 1
    assert abs(wrap_half_turn(-98.0 + 90 - (wrist.gain * theta + wrist.offset))) < 1


@pytest.mark.parametrize(
    "position, width_mm, holding",
    [
        (90, 9.0, True),  # a bolt across the jaws
        (200, 9.0, True),  # within the 12 mm slack
        (464, 9.0, False),  # stalled on the table, as in the demo run
        (545, 14.0, False),
        (464, None, True),  # width unknown: only the open/closed bounds apply
        (0, 9.0, False),  # closed on nothing
    ],
)
def test_holding_part_checks_the_parts_width(position, width_mm, holding):
    assert Cell.holding_part(position, width_mm) is holding


def test_second_look_matches_the_nearest_part_within_reach():
    from viamrobot.cell import SeenPart, match_part
    from viamrobot.pipeline import BoundingBox

    near = SeenPart(BoundingBox(300, 220, 340, 260, confidence=0.9), "normal", 0.1)  # centre (320, 240)
    far = SeenPart(BoundingBox(500, 50, 540, 90, confidence=0.9), "anomalous", 0.9)  # centre (520, 70)
    assert match_part([far, near], (330, 250), 120) is near
    assert match_part([far], (330, 250), 120) is None  # ~260 px away: not the same part
    assert match_part([], (330, 250), 120) is None


async def test_a_turn_from_straight_down_ignores_and_keeps_the_inspect_bias():
    wrist = SimWrist(gain=-1.0, offset=0.0, start_theta=-170.0, guess=-1.0)
    wrist.theta_bias, wrist.at_inspect = 25.0, False  # learned at the inspect pose; we're straight down now
    theta = await wrist._turn_across(PartLocation(0, 0, 10, 8, yaw_deg=30.0), at, 170, -170.0, from_inspect=False)
    assert wrist.moves == 1
    assert abs(wrap_half_turn(30.0 + 90 - (wrist.gain * theta))) < 1
    assert wrist.theta_bias == 25.0


async def test_learns_which_way_theta_turns():
    wrist = SimWrist(gain=1.0, offset=0.0, start_theta=0.0, guess=-1.0)
    await wrist._turn_across(PartLocation(0, 0, 10, 8, yaw_deg=40.0), at, 170, 0.0)
    assert wrist.theta_to_yaw == 1.0
