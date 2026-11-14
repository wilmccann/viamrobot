"""The demo loop against a fake work cell: sorting, timeouts, dropped connections, failures."""

import asyncio

import pytest
from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.errors import ResourceNotFoundError, ViamError

from viamrobot.cell import BAD_POSE, GOOD_POSE, INSPECT_POSE, UNSURE_POSE, PartLocation, PickMissed, SeenPart, Snapshot
from viamrobot.decision import Destination
from viamrobot.demo import Demo, DemoStopped, Failure, choose_part, is_connection_error, route
from viamrobot.pipeline import BoundingBox

A = BoundingBox(100, 100, 150, 150, confidence=0.1, label="normal")
B = BoundingBox(300, 300, 350, 350, confidence=0.9, label="anomalous")


def seen(box, verdict="normal", score=0.1):
    return SeenPart(box, verdict, score)


class FakeRobot:
    async def close(self):
        pass


class World:
    """What's on the table and how each pick goes, shared across reconnects."""

    def __init__(self, parts, picks=()):
        self.parts = list(parts)  # SeenParts still on the table
        self.picks = list(picks)  # per pick: "ok", "miss", "hang", "drop"
        self.link_up = True
        self.calls: list[str] = []
        self.cells = 0
        self.on: dict = {}  # "miss" / "place": called once, the first time that happens
        self.fail_inspect = 0  # this many moves to the inspect pose fail

    def fire(self, event):
        if event in self.on:
            self.on.pop(event)()


class FakeCell:
    def __init__(self, world: World):
        self.w = world
        world.cells += 1

    def _check(self):
        if not self.w.link_up:
            raise ConnectionError("link down")

    async def ping(self):
        self._check()

    async def go_to(self, name):
        self._check()
        if name == INSPECT_POSE and self.w.fail_inspect:
            self.w.fail_inspect -= 1
            raise RuntimeError("motion failed")
        self.w.calls.append(f"go_to {name}")

    async def look(self):
        self._check()
        return Snapshot(color=None, bgr=None, depth=None, intrinsics=None, parts=list(self.w.parts))

    async def locate(self, snap, box):
        self._current = box
        return PartLocation(box.x_min, box.y_min, 15.0, 8.0)

    async def pick(self, part):
        self._check()
        how = self.w.picks.pop(0) if self.w.picks else "ok"
        self.w.calls.append(f"pick {how}")
        if how == "hang":
            await asyncio.sleep(10)
        if how == "miss":
            self.w.fire("miss")
            raise PickMissed("fingers stopped at 0")
        if how == "drop":
            self.w.link_up = False
            raise ConnectionError("link down")
        self.w.parts = [p for p in self.w.parts if p.box != self._current]
        return 150.0

    async def place(self, pose_name):
        self.w.calls.append(f"place {pose_name}")
        self.w.fire("place")

    async def recover(self):
        self._check()
        self.w.calls.append("recover")

    async def stop(self):
        self.w.calls.append("stop")


def make_demo(world, tmp_path, *, connects_fail=0, **kw):
    state = {"fails_left": connects_fail}

    async def connect():
        if state["fails_left"] and world.cells > 0:  # only fail reconnects, not the first connect
            state["fails_left"] -= 1
            raise ViamError("Unable to establish a connection")
        world.link_up = True
        return FakeRobot()

    options = dict(reconnect_wait_s=0, idle_poll_s=0.01, startup_wait_s=0, failure_backoff_s=0, log=lambda msg: None)
    options.update(kw)
    return Demo(connect=connect, make_cell=lambda robot: FakeCell(world), log_path=tmp_path / "judge.csv", **options)


def remaining(world):
    return [p.box for p in world.parts]


async def test_normal_goes_to_good_pose_and_anomalous_to_bad_pose(tmp_path):
    world = World([seen(A, "normal", 0.2), seen(B, "anomalous", 0.9)])
    results = await make_demo(world, tmp_path, max_parts=2).run()

    assert [(r.item, r.destination) for r in results] == [(1, Destination.BOX), (2, Destination.REJECT)]
    assert [c for c in world.calls if c.startswith("place")] == [f"place {GOOD_POSE}", f"place {BAD_POSE}"]
    lines = (tmp_path / "judge.csv").read_text().splitlines()
    assert [l.split(",")[:3] for l in lines] == [["1", "box", "0.800"], ["2", "reject", "0.900"]]


async def test_hold_goes_to_unsure_pose_by_default(tmp_path):
    world = World([seen(A, "hold", 0.5)])
    results = await make_demo(world, tmp_path, max_parts=1).run()
    assert f"place {UNSURE_POSE}" in world.calls and results[0].destination is Destination.HOLD
    assert (tmp_path / "judge.csv").read_text().split(",")[1] == "hold"


async def test_hold_can_be_sent_to_bad_pose(tmp_path):
    world = World([seen(A, "hold", 0.5)])
    results = await make_demo(world, tmp_path, max_parts=1, hold_pose=BAD_POSE).run()
    assert f"place {BAD_POSE}" in world.calls and results[0].destination is Destination.REJECT


async def test_each_part_starts_and_ends_at_inspect(tmp_path):
    world = World([seen(A)])
    await make_demo(world, tmp_path, max_parts=1).run()
    pick = world.calls.index("pick ok")
    assert world.calls[pick - 1] == f"go_to {INSPECT_POSE}"
    assert world.calls[pick + 2] == f"go_to {INSPECT_POSE}"  # after place


async def test_a_part_over_the_timeout_is_abandoned_and_the_next_part_is_done(tmp_path):
    world = World([seen(A), seen(B)], picks=["hang", "ok"])
    demo = make_demo(world, tmp_path, max_parts=1, part_timeout_s=0.2)
    results = await demo.run()

    assert world.calls.count("recover") == 2  # once at start, once after the timeout
    assert remaining(world) == [A]  # A was abandoned, B was sorted
    assert len(results) == 1
    assert demo.failures and demo.failures[0].box == A


async def test_a_dropped_connection_reconnects_recovers_and_carries_on(tmp_path):
    world = World([seen(A), seen(B, "anomalous", 0.9)], picks=["drop", "ok"])
    results = await make_demo(world, tmp_path, max_parts=1).run()

    assert world.cells == 2  # a fresh cell after reconnecting
    after_drop = world.calls[world.calls.index("pick drop") + 1:]
    assert after_drop[0] == "recover"  # open gripper + inspect, on the new connection
    assert results[0].destination is Destination.REJECT  # went on to B


async def test_reconnect_succeeds_on_the_second_attempt(tmp_path):
    world = World([seen(A)], picks=["drop", "ok"])
    results = await make_demo(world, tmp_path, max_parts=1, connects_fail=1).run()
    assert len(results) == 1


async def test_two_failed_reconnects_stop_the_demo(tmp_path):
    world = World([seen(A)], picks=["drop"])
    with pytest.raises(DemoStopped, match="reconnect"):
        await make_demo(world, tmp_path, max_parts=1, connects_fail=2).run()


async def test_any_other_failure_recovers_and_moves_on(tmp_path):
    world = World([seen(A), seen(B)], picks=["miss", "ok"])
    results = await make_demo(world, tmp_path, max_parts=1).run()
    assert world.calls.count("recover") == 2
    assert remaining(world) == [A] and len(results) == 1


async def test_a_retried_part_keeps_its_number_and_is_logged_once(tmp_path):
    # A misses, B is done, then A is retried and done: lines 1 (A) and 2 (B), nothing else.
    world = World([seen(A, "normal", 0.1), seen(B, "anomalous", 0.9)], picks=["miss", "ok", "ok"])
    results = await make_demo(world, tmp_path, max_parts=2).run()

    lines = (tmp_path / "judge.csv").read_text().splitlines()
    assert [l.split(",")[:2] for l in lines] == [["1", "box"], ["2", "reject"]]
    assert sorted(r.item for r in results) == [1, 2]
    assert {r.item: r.destination for r in results} == {1: Destination.BOX, 2: Destination.REJECT}


C = BoundingBox(500, 100, 550, 150, confidence=0.9, label="anomalous")
D = BoundingBox(500, 300, 550, 350, confidence=0.1, label="normal")


def log_rows(tmp_path):
    return [l.split(",")[:2] for l in (tmp_path / "judge.csv").read_text().splitlines()]


def places(world):
    return [c.split()[1] for c in world.calls if c.startswith("place")]


async def test_a_retry_goes_where_the_log_says_even_if_the_verdict_changes(tmp_path):
    world = World([seen(A, "normal", 0.1)], picks=["miss", "ok"])
    world.on["miss"] = lambda: setattr(world, "parts", [seen(A, "anomalous", 0.9)])
    results = await make_demo(world, tmp_path, max_parts=1).run()

    assert log_rows(tmp_path) == [["1", "box"]]
    assert places(world) == [GOOD_POSE]
    assert results[0].destination is Destination.BOX


async def test_a_failure_after_placing_doesnt_take_the_next_parts_log_line(tmp_path):
    # Part 1 is placed, the move back to inspect fails, and a new part sits where part 1 was.
    world = World([seen(A, "normal", 0.1)])

    def new_part_in_the_same_spot():
        world.parts = [seen(A, "anomalous", 0.9)]
        world.fail_inspect = 1

    world.on["place"] = new_part_in_the_same_spot
    results = await make_demo(world, tmp_path, max_parts=2).run()

    assert log_rows(tmp_path) == [["1", "box"], ["2", "reject"]]
    assert places(world) == [GOOD_POSE, BAD_POSE]
    assert [r.item for r in results] == [1, 2]


async def test_a_moved_part_keeps_its_number_and_destination(tmp_path):
    # Part 1 (A) misses and a judge moves it to C; B is done next, then the part at C as part 1.
    world = World([seen(A, "anomalous", 0.9), seen(B, "normal", 0.1)], picks=["miss", "ok", "ok"])
    world.on["miss"] = lambda: setattr(world, "parts", [seen(C, "normal", 0.2), seen(B, "normal", 0.1)])
    results = await make_demo(world, tmp_path, max_parts=2).run()

    assert log_rows(tmp_path) == [["1", "reject"], ["2", "box"]]
    assert {r.item: r.destination for r in results} == {1: Destination.REJECT, 2: Destination.BOX}
    assert places(world) == [GOOD_POSE, BAD_POSE]


async def test_a_moved_part_gets_a_new_line_when_its_ambiguous(tmp_path):
    # A moves to C and a new part D appears at the same time: either could be A, so both are new.
    world = World([seen(A, "anomalous", 0.9), seen(B, "normal", 0.1)], picks=["miss", "ok", "ok", "ok"])
    world.on["miss"] = lambda: setattr(world, "parts", [seen(B), seen(C, "anomalous", 0.9), seen(D)])
    await make_demo(world, tmp_path, max_parts=3).run()

    assert [row[0] for row in log_rows(tmp_path)] == ["1", "2", "3", "4"]


async def test_numbers_never_repeat_after_a_timeout(tmp_path):
    world = World([seen(A), seen(B)], picks=["hang", "ok"])
    await make_demo(world, tmp_path, max_parts=1, part_timeout_s=0.2).run()
    numbers = [l.split(",")[0] for l in (tmp_path / "judge.csv").read_text().splitlines()]
    assert numbers == ["1", "2"]


async def test_a_part_that_keeps_failing_is_dropped_after_max_attempts(tmp_path):
    world = World([seen(A)], picks=["miss", "miss", "miss"])
    demo = make_demo(world, tmp_path, max_parts=1, max_attempts=3)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(demo.run(), 0.5)  # idles: nothing left worth trying
    assert world.calls.count("pick miss") == 3


async def test_recovery_that_keeps_failing_stops_the_demo(tmp_path):
    world = World([seen(A)])
    demo = make_demo(world, tmp_path, max_parts=1)

    def broken_cell(robot):
        cell = FakeCell(world)

        async def recover():
            raise RuntimeError("arm faulted")

        cell.recover = recover
        return cell

    demo.make_cell = broken_cell
    with pytest.raises(DemoStopped, match="needs a person"):
        await demo.run()


def missing_cam_then(world, failures):
    """make_cell that fails like a machine whose RealSense is down, `failures` times."""
    state = {"left": failures}

    def make(robot):
        if state["left"]:
            state["left"] -= 1
            raise ResourceNotFoundError("camera", "cam")
        return FakeCell(world)

    return make


async def test_startup_waits_for_a_missing_camera(tmp_path):
    world = World([seen(A)])
    demo = make_demo(world, tmp_path, max_parts=1)
    demo.make_cell = missing_cam_then(world, 2)
    assert len(await demo.run()) == 1


async def test_startup_gives_up_with_a_clear_message(tmp_path):
    world = World([seen(A)])
    lines = []
    demo = make_demo(world, tmp_path, max_parts=1, startup_attempts=3, log=lines.append)
    demo.make_cell = missing_cam_then(world, 99)
    with pytest.raises(DemoStopped, match=r"'cam' \(camera\) isn't available.*RealSense"):
        await demo.run()
    assert sum("start attempt" in l for l in lines) == 3


async def test_failures_in_a_row_back_off(tmp_path):
    world = World([seen(A), seen(B)], picks=["miss", "miss", "ok"])
    lines = []
    demo = make_demo(world, tmp_path, max_parts=1, failure_backoff_s=0.01, log=lines.append)
    assert len(await demo.run()) == 1
    waits = [l for l in lines if "in a row" in l]
    assert [w.split()[0] for w in waits] == ["1", "2"]


# --- pure helpers ---------------------------------------------------------------


def test_choose_part_takes_them_in_order_then_least_failed():
    a, b = seen(A), seen(B)
    assert choose_part([a, b], [], 3) is a
    assert choose_part([a, b], [Failure(A)], 3) is b
    assert choose_part([a, b], [Failure(A, 2), Failure(B, 1)], 3) is b
    assert choose_part([a], [Failure(A, 3)], 3) is None
    assert choose_part([], [], 3) is None


@pytest.mark.parametrize(
    "verdict, score, hold_pose, expected",
    [
        ("normal", 0.1, BAD_POSE, (Destination.BOX, GOOD_POSE, 0.9)),
        ("anomalous", 0.8, BAD_POSE, (Destination.REJECT, BAD_POSE, 0.8)),
        ("hold", 0.55, BAD_POSE, (Destination.REJECT, BAD_POSE, 0.55)),
        ("hold", 0.45, UNSURE_POSE, (Destination.HOLD, UNSURE_POSE, 0.55)),
        ("normal", 41.0, BAD_POSE, (Destination.BOX, GOOD_POSE, 1.0)),  # decider "patchcore" scale
    ],
)
def test_route(verdict, score, hold_pose, expected):
    destination, pose, confidence = route(seen(A, verdict, score), hold_pose)
    assert (destination, pose) == expected[:2]
    assert confidence == pytest.approx(expected[2])


def test_route_rejects_an_unknown_verdict():
    with pytest.raises(ValueError):
        route(seen(A, "empty", 1.0))


def test_is_connection_error():
    assert is_connection_error(ConnectionError())
    assert is_connection_error(ViamError("Unable to establish a connection to x"))
    assert is_connection_error(GRPCError(Status.UNAVAILABLE, "gone"))
    assert not is_connection_error(TimeoutError())
    assert not is_connection_error(GRPCError(Status.CANCELLED, "context canceled"))
    assert not is_connection_error(PickMissed("x"))
