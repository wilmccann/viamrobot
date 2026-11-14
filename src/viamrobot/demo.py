"""Track 3 demo on armfarm01: sort every part on the table, one at a time. MOVES THE ARM.

    loop:
      inspect pose -> look (vision-9 finds every part and gives each a verdict)
      no part         -> wait, look again
      take one part   (nearest the image centre; long parts get the jaws turned across them)
        normal        -> log, pick it up, drop it at good-pose
        anomalous     -> log, pick it up, drop it at bad-pose
        hold          -> log, pick it up, drop it at unsure-pose (the hold area);
                         --hold-pose bad-pose treats it as anomalous instead
      back to the inspect pose, next part

Failures never stop the loop:
  - one part takes longer than --part-timeout (20 s): abandon it, recover, try the next part
  - the connection drops: reconnect (up to 2 attempts), then recover and carry on;
    if both attempts fail, stop
  - anything else (a missed pick, a motion error): recover and carry on
"Recover" = stop the arm, open the gripper (drops anything held), lift clear of the
table if low, go to the inspect pose. A part that failed is tried again only after every
other part, and dropped after --max-attempts failures. If recovery itself fails twice
in a row, the demo stops: the arm needs a person.

One judge-log line per decided part (number, decision, confidence, timestamp) is
appended to --log before the arm moves.

Stand at the E-stop (red button on the xArm controller), then:

    uv run viamrobot-demo                      # until Ctrl-C
    uv run viamrobot-demo --max-parts 6 --log judge.csv
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol

from dotenv import load_dotenv
from grpclib.exceptions import GRPCError, StreamTerminatedError
from grpclib.const import Status
from viam.errors import ResourceNotFoundError, ViamError
from viam.robot.client import RobotClient

from viamrobot.cell import (
    ANOMALOUS,
    BAD_POSE,
    GOOD_POSE,
    HOLD,
    INSPECT_POSE,
    NORMAL,
    UNSURE_POSE,
    PartLocation,
    SeenPart,
    Snapshot,
)
from viamrobot.decision import Destination, log_line
from viamrobot.grasp import iou
from viamrobot.pipeline import BoundingBox

PART_TIMEOUT_S = 20.0
RECONNECT_ATTEMPTS = 2
RECOVER_TIMEOUT_S = 45.0
RECOVER_ATTEMPTS = 2
MAX_ATTEMPTS_PER_PART = 3
IDLE_POLL_S = 1.0
SAME_PART_IOU = 0.3
STARTUP_ATTEMPTS = 6  # ~30 s for the machine's resources to come up
STARTUP_WAIT_S = 5.0
FAILURE_BACKOFF_S = 2.0  # wait 2, 4, 6 ... s after failures in a row
MAX_BACKOFF_S = 10.0

HoldPose = Literal["bad-pose", "unsure-pose"]


class CellLike(Protocol):
    """What the demo needs from viamrobot.cell.Cell (a fake stands in for it in unit tests)."""

    async def ping(self) -> None: ...
    async def go_to(self, name: str) -> None: ...
    async def look(self) -> Snapshot: ...
    async def locate(self, snap: Snapshot, box: BoundingBox) -> PartLocation: ...
    async def pick(self, part: PartLocation) -> float: ...
    async def place(self, pose_name: str) -> None: ...
    async def recover(self) -> None: ...
    async def stop(self) -> None: ...


class DemoStopped(RuntimeError):
    """The demo can't continue on its own: the arm needs a person."""


@dataclass
class Failure:
    box: BoundingBox  # where the part was, in the inspect-pose image
    count: int = 1


@dataclass
class Result:
    item: int
    destination: Destination
    score: float
    seconds: float


def choose_part(parts: list[SeenPart], failures: list[Failure], max_attempts: int) -> SeenPart | None:
    """The part to do next: the first one (in the order given) that hasn't failed; else the
    one that has failed least (fewer than `max_attempts` times); else None.

    A part is "the same part" as a failure when their boxes overlap by more than SAME_PART_IOU.
    """
    def failed(p: SeenPart) -> int:
        return max((f.count for f in failures if iou(f.box, p.box) > SAME_PART_IOU), default=0)

    tries = [(failed(p), i) for i, p in enumerate(parts)]
    tries = [t for t in tries if t[0] < max_attempts]
    return parts[min(tries)[1]] if tries else None


def route(seen: SeenPart, hold_pose: HoldPose = UNSURE_POSE) -> tuple[Destination, str, float]:
    """Where a part goes, from vision-9's verdict: (judge-log decision, saved pose, confidence).

    normal -> good-pose; anomalous -> bad-pose; hold -> `hold_pose` (unsure-pose, the
    hold area: logged as hold; bad-pose: treated as anomalous). Confidence is how far the score sits toward
    the decision when the score is a probability (decider "vision8"); 1.0 otherwise.
    """
    if seen.verdict == NORMAL:
        destination, pose = Destination.BOX, GOOD_POSE
    elif seen.verdict == ANOMALOUS or (seen.verdict == HOLD and hold_pose == BAD_POSE):
        destination, pose = Destination.REJECT, BAD_POSE
    elif seen.verdict == HOLD:
        destination, pose = Destination.HOLD, UNSURE_POSE
    else:
        raise ValueError(f"unexpected vision-9 verdict {seen.verdict!r}")
    s = seen.score
    if not 0.0 <= s <= 1.0:
        return destination, pose, 1.0
    confidence = {Destination.BOX: 1.0 - s, Destination.REJECT: s}.get(destination, max(s, 1.0 - s))
    return destination, pose, confidence


def is_connection_error(exc: BaseException) -> bool:
    """Errors that mean the link to the machine is gone (as opposed to a robot-side failure)."""
    if isinstance(exc, TimeoutError):  # an OSError subclass, but not a dropped link
        return False
    if isinstance(exc, (ConnectionError, StreamTerminatedError, OSError)):
        return True
    if isinstance(exc, GRPCError):
        return exc.status is Status.UNAVAILABLE
    if isinstance(exc, ViamError):
        return "connect" in str(exc).lower()
    return False


def describe(exc: BaseException | None) -> str:
    """One line for the log, with a hint where there's an obvious next step."""
    if isinstance(exc, ResourceNotFoundError):
        subtype, name = (exc.args + ("?", "?"))[:2]
        hint = " (the wrist RealSense: check its USB cable on the NUC)" if name == "cam" else ""
        return f"'{name}' ({subtype}) isn't available on the machine{hint}"
    return f"{type(exc).__name__}: {exc}"


async def connect_to_machine() -> RobotClient:
    load_dotenv()
    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


def say(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} {msg}", flush=True)


@dataclass
class Demo:
    connect: Callable[[], Awaitable[RobotClient]]
    make_cell: Callable[[RobotClient], CellLike]
    log_path: Path
    part_timeout_s: float = PART_TIMEOUT_S
    hold_pose: HoldPose = UNSURE_POSE
    max_parts: int | None = None
    max_attempts: int = MAX_ATTEMPTS_PER_PART
    reconnect_attempts: int = RECONNECT_ATTEMPTS
    reconnect_wait_s: float = 2.0
    recover_timeout_s: float = RECOVER_TIMEOUT_S
    idle_poll_s: float = IDLE_POLL_S
    startup_attempts: int = STARTUP_ATTEMPTS
    startup_wait_s: float = STARTUP_WAIT_S
    failure_backoff_s: float = FAILURE_BACKOFF_S
    max_backoff_s: float = MAX_BACKOFF_S
    last_reconnect_error: str = ""
    log: Callable[[str], None] = say
    results: list[Result] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    robot: RobotClient | None = None
    cell: CellLike | None = None
    _current: BoundingBox | None = None

    # --- one part ----------------------------------------------------------------

    async def process_one(self) -> Result | None:
        """Inspect, choose a part, log vision-9's verdict, pick it, drop it, back to inspect.

        Returns None when no part (that's still worth trying) is in view.
        """
        cell = self.cell
        start = time.perf_counter()
        await cell.go_to(INSPECT_POSE)
        snap = await cell.look()
        seen = choose_part(snap.parts, self.failures, self.max_attempts)
        if seen is None:
            return None
        box = self._current = seen.box
        item = len(self.results) + 1
        destination, pose, confidence = route(seen, self.hold_pose)
        self._write_log(log_line(item, destination, confidence, datetime.now(timezone.utc)))
        self.log(f"part {item}: {len(snap.parts)} in view; taking ({box.x_min:.0f},{box.y_min:.0f})-"
                 f"({box.x_max:.0f},{box.y_max:.0f}) px: {seen.verdict} (score {seen.score:.3f}) -> {pose}")

        part = await cell.locate(snap, box)
        await cell.pick(part)
        await cell.place(pose)
        await cell.go_to(INSPECT_POSE)

        self.failures = [f for f in self.failures if iou(f.box, box) <= SAME_PART_IOU]
        self._current = None
        return Result(item, destination, seen.score, time.perf_counter() - start)

    # --- failures ----------------------------------------------------------------

    def _note_failure(self) -> None:
        box, self._current = self._current, None
        if box is None:
            return
        for f in self.failures:
            if iou(f.box, box) > SAME_PART_IOU:
                f.count += 1
                f.box = box
                return
        self.failures.append(Failure(box))

    async def _connected(self) -> bool:
        try:
            await asyncio.wait_for(self.cell.ping(), 5)
            return True
        except Exception:
            return False

    async def _open(self) -> None:
        """A fresh connection and cell. Raises if the machine is missing a resource we need."""
        robot = await self.connect()
        try:
            cell = self.make_cell(robot)
        except Exception:
            try:
                await robot.close()
            except Exception:
                pass
            raise
        self.robot, self.cell = robot, cell

    async def start(self) -> None:
        """Connect, waiting for the machine's resources (up to `startup_attempts` tries).

        After a hardware hiccup (the RealSense dropping off USB, say) viam-server rebuilds
        the camera and everything that depends on it; a client that connects meanwhile
        doesn't see them. So retry with a fresh connection rather than failing at once.
        """
        last: Exception | None = None
        for attempt in range(1, self.startup_attempts + 1):
            try:
                await self._open()
                return
            except Exception as exc:
                last = exc
                self.log(f"start attempt {attempt}/{self.startup_attempts}: {describe(exc)}")
                if attempt < self.startup_attempts:
                    await asyncio.sleep(self.startup_wait_s)
        raise DemoStopped(f"couldn't start: {describe(last)}")

    async def _close(self) -> None:
        if self.robot is not None:
            try:
                await self.robot.close()
            except Exception:
                pass

    async def reconnect(self) -> bool:
        """Drop the old client and connect again, up to `reconnect_attempts` times."""
        await self._close()
        for attempt in range(1, self.reconnect_attempts + 1):
            await asyncio.sleep(self.reconnect_wait_s * attempt)
            self.log(f"reconnect attempt {attempt}/{self.reconnect_attempts}")
            try:
                await self._open()
                self.log("reconnected")
                return True
            except Exception as exc:
                self.last_reconnect_error = describe(exc)
                self.log(f"   reconnect failed: {self.last_reconnect_error}")
        return False

    async def recover(self, why: str) -> None:
        """Back to a known state after a failure: reconnect first if the link is gone,
        then stop, open the gripper and go to the inspect pose. Raises DemoStopped when
        that isn't possible.
        """
        self.log(f"RECOVER ({why})")
        for attempt in range(1, RECOVER_ATTEMPTS + 1):
            if not await self._connected():
                self.log("connection lost")
                if not await self.reconnect():
                    raise DemoStopped(f"could not reconnect after {self.reconnect_attempts} attempts "
                                      f"(last: {self.last_reconnect_error})")
            try:
                async with asyncio.timeout(self.recover_timeout_s):
                    await self.cell.recover()
                return
            except Exception as exc:
                self.log(f"   recovery attempt {attempt} failed: {type(exc).__name__}: {exc}")
        raise DemoStopped(f"recovery failed {RECOVER_ATTEMPTS} times in a row; the arm needs a person")

    # --- the loop ------------------------------------------------------------------

    async def run(self) -> list[Result]:
        await self.start()
        self.log(f"demo running: part timeout {self.part_timeout_s:.0f} s, verdicts from vision-9, "
                 f"hold -> {self.hold_pose}; judge log {self.log_path}")
        streak = 0  # failures in a row
        try:
            await self.recover("start")
            while self.max_parts is None or len(self.results) < self.max_parts:
                timer = asyncio.timeout(self.part_timeout_s)
                try:
                    async with timer:
                        result = await self.process_one()
                except Exception as exc:
                    self._note_failure()
                    if timer.expired():
                        why = f"part took longer than {self.part_timeout_s:.0f} s"
                    else:
                        kind = "connection" if is_connection_error(exc) else "failure"
                        why = f"{kind}: {describe(exc)}"
                    await self.recover(why)  # reconnects first if the link is gone
                    # A failure that keeps happening (a dead camera, say) shouldn't spin
                    # the arm through recovery back to back: wait a little longer each time.
                    streak += 1
                    wait = min(self.failure_backoff_s * streak, self.max_backoff_s)
                    if wait > 0:
                        self.log(f"{streak} failure(s) in a row; next try in {wait:.0f} s")
                        await asyncio.sleep(wait)
                    continue
                streak = 0
                if result is None:
                    await asyncio.sleep(self.idle_poll_s)
                    continue
                self.results.append(result)
                self.log(f"part {result.item}: done -> {result.destination.value} in {result.seconds:.1f} s")
        finally:
            if self.cell is not None:
                await self.cell.stop()
            await self._close()
        return self.results

    def _write_log(self, line: str) -> None:
        with self.log_path.open("a") as f:
            f.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part-timeout", type=float, default=PART_TIMEOUT_S, help="seconds allowed per part (default 20)")
    p.add_argument("--hold-pose", choices=(UNSURE_POSE, BAD_POSE), default=UNSURE_POSE,
                   help="where vision-9's 'hold' (unsure) parts go (default unsure-pose, the hold area; "
                        "bad-pose treats them as anomalous)")
    p.add_argument("--max-parts", type=int, default=None, help="stop after this many parts (default: run until Ctrl-C)")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_PER_PART, help="failures before a part is ignored")
    p.add_argument("--log", type=Path, default=Path("judge.csv"), help="judge log, one line per part (default judge.csv)")
    args = p.parse_args()

    from viamrobot.cell import Cell

    demo = Demo(
        connect=connect_to_machine,
        make_cell=lambda robot: Cell(robot, log=say),
        log_path=args.log,
        part_timeout_s=args.part_timeout,
        hold_pose=args.hold_pose,
        max_parts=args.max_parts,
        max_attempts=args.max_attempts,
    )
    try:
        asyncio.run(demo.run())
    except KeyboardInterrupt:
        say("stopped (Ctrl-C)")
    except DemoStopped as exc:
        say(f"DEMO STOPPED: {exc}")
        raise SystemExit(1)
    finally:
        n = {d: sum(r.destination is d for r in demo.results) for d in Destination}
        say(f"{len(demo.results)} part(s) sorted: {n[Destination.BOX]} good, {n[Destination.REJECT]} anomalous"
            + (f", {n[Destination.HOLD]} hold" if n[Destination.HOLD] else ""))


if __name__ == "__main__":
    main()
