"""Simulated Track 3 run: scripted items through the real Pipeline loop on fake stages.

    uv run viamrobot-sim                 # real time, about 30 s
    uv run viamrobot-sim --speed 10      # 10x faster; reported times stay in simulated seconds
    uv run viamrobot-sim --log judge.csv # also write the judge log to a file

Items arrive one at a time: the camera sees a few empty frames, then the item,
which stays in view until the arm places it. Stage latencies are placeholders;
replace LATENCY_S with numbers measured on the NUC.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from viamrobot.decision import Destination
from viamrobot.pipeline import (
    BoundingBox,
    CsvDecisionLog,
    Frame,
    ItemRecord,
    Pipeline,
    RunSummary,
    VlmVerdict,
)

# Nominal seconds per stage, +/-20% jitter. Guesses until measured on the NUC.
LATENCY_S = {"capture": 0.03, "detect": 0.015, "anomaly": 0.06, "vlm": 1.2, "move": 3.0}


@dataclass(frozen=True)
class SimItem:
    name: str
    truth: Destination  # where it should go
    anomaly_score: float
    vlm: VlmVerdict | Literal["timeout"] | None = None  # used only when the score is unsure
    empty_frames_before: int = 2


DEFAULT_SCENARIO: tuple[SimItem, ...] = (
    SimItem("good-1", Destination.BOX, 0.12),
    SimItem("dent-1", Destination.REJECT, 0.88),
    SimItem("good-borderline", Destination.BOX, 0.45, VlmVerdict(Destination.BOX, 0.85, "surface looks clean")),
    SimItem("unseen-defect", Destination.REJECT, 0.52, VlmVerdict(Destination.REJECT, 0.74, "discoloured patch")),
    SimItem("good-2", Destination.BOX, 0.08),
    SimItem("vlm-hangs", Destination.REJECT, 0.55, "timeout"),
)


class SimScene:
    """Shared state for the fake stages: what is in front of the camera right now."""

    def __init__(self, items: tuple[SimItem, ...], speed: float, rng: random.Random):
        self.pending = list(items)
        self.current: SimItem | None = None
        self.placed: list[tuple[SimItem, Destination]] = []
        self._empty_seen = 0
        self.speed, self.rng = speed, rng

    async def work(self, stage: str) -> None:
        await asyncio.sleep(LATENCY_S[stage] * self.rng.uniform(0.8, 1.2) / self.speed)

    def advance(self) -> None:
        if self.current is not None or not self.pending:
            return
        if self._empty_seen >= self.pending[0].empty_frames_before:
            self.current, self._empty_seen = self.pending.pop(0), 0
        else:
            self._empty_seen += 1


class SimCamera:
    def __init__(self, scene: SimScene):
        self.scene = scene

    async def capture(self) -> Frame:
        await self.scene.work("capture")
        self.scene.advance()
        return Frame(color=self.scene.current)


class SimDetector:
    def __init__(self, scene: SimScene):
        self.scene = scene

    async def find_item(self, frame: Frame) -> BoundingBox | None:
        await self.scene.work("detect")
        item = frame.color
        return None if item is None else BoundingBox(200, 150, 440, 330, confidence=0.93, label=item.name)


class SimScorer:
    def __init__(self, scene: SimScene):
        self.scene = scene

    async def score(self, frame: Frame, box: BoundingBox) -> float:
        await self.scene.work("anomaly")
        return frame.color.anomaly_score


class SimVlm:
    def __init__(self, scene: SimScene):
        self.scene = scene

    async def second_look(self, frame: Frame, box: BoundingBox, anomaly_score: float) -> VlmVerdict:
        item: SimItem = frame.color
        if item.vlm == "timeout":
            await asyncio.sleep(3600)  # cancelled by the pipeline's timeout
        await self.scene.work("vlm")
        if item.vlm is None:
            raise RuntimeError(f"no scripted VLM verdict for {item.name}")
        return item.vlm


class SimMover:
    def __init__(self, scene: SimScene):
        self.scene = scene

    async def pick_and_place(self, frame: Frame, box: BoundingBox, destination: Destination) -> None:
        await self.scene.work("move")
        self.scene.placed.append((frame.color, destination))
        self.scene.current = None


class SimLog:
    def __init__(self, path: str | Path | None):
        self.file = CsvDecisionLog(path) if path else None

    def write(self, record: ItemRecord) -> None:
        if self.file:
            self.file.write(record)


async def simulate(
    items: tuple[SimItem, ...] = DEFAULT_SCENARIO,
    speed: float = 1.0,
    seed: int = 0,
    vlm_timeout_s: float = 5.0,
    deadline_s: float = 300.0,
    log_path: str | Path | None = None,
    out: Callable[[str], None] = print,
) -> tuple[RunSummary, list[tuple[SimItem, ItemRecord]]]:
    """Run the scenario through Pipeline.run. Times are in simulated seconds."""
    scene = SimScene(items, speed, random.Random(seed))
    t0 = time.perf_counter()
    pipeline = Pipeline(
        camera=SimCamera(scene),
        detector=SimDetector(scene),
        scorer=SimScorer(scene),
        vlm=SimVlm(scene),
        mover=SimMover(scene),
        log=SimLog(log_path),
        vlm_timeout_s=vlm_timeout_s / speed,
        clock=lambda: (time.perf_counter() - t0) * speed,
    )
    results: list[tuple[SimItem, ItemRecord]] = []

    out(f"Simulated Track 3 run: {len(items)} items, speed {speed:g}x, VLM timeout {vlm_timeout_s:g} s")
    out(f"{'#':>3}  {'item':<16} {'score':>5}  {'decided by':<10}  {'dest':<7} {'conf':>5}  {'decide ms':>9}  truth")

    def on_record(record: ItemRecord) -> None:
        item = scene.placed[-1][0]
        results.append((item, record))
        verdict = "ok" if record.destination is item.truth else f"want {item.truth.value}"
        out(
            f"{record.item_number:>3}  {item.name:<16} {record.anomaly_score:>5.2f}  {record.decided_by:<10}"
            f"  {record.destination.value:<7} {record.confidence:>5.2f}  {sum(record.timings_ms.values()):>9.0f}"
            f"  {verdict}"
        )

    summary = await pipeline.run(
        max_items=len(items), deadline_s=deadline_s, idle_poll_s=0.2 / speed, on_record=on_record
    )

    correct = sum(record.destination is item.truth for item, record in results)
    held = sum(record.destination is Destination.HOLD for _, record in results)
    out("")
    out(f"Stopped: {summary.stopped_because} after {summary.elapsed_s:.1f} s. Correct {correct}/{len(results)}, held {held}.")
    if summary.fault:
        out(f"Fault: {summary.fault}")
    out("Stage latency, ms (mean / max):")
    for stage in LATENCY_S:
        values = [r.timings_ms[stage] for _, r in results if stage in r.timings_ms]
        if values:
            out(f"  {stage:<8} {statistics.mean(values):>7.0f} / {max(values):>7.0f}  (n={len(values)})")
    out("Judge log:")
    for _, record in results:
        out(f"  {record.csv()}")
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=1.0, help="time multiplier (default 1 = real time)")
    parser.add_argument("--seed", type=int, default=0, help="seed for latency jitter")
    parser.add_argument("--vlm-timeout", type=float, default=5.0, help="seconds before the VLM is abandoned")
    parser.add_argument("--deadline", type=float, default=300.0, help="run length in seconds (judged run is 5 min)")
    parser.add_argument("--log", type=Path, help="also append the judge log to this CSV file")
    args = parser.parse_args()
    asyncio.run(
        simulate(
            speed=args.speed,
            seed=args.seed,
            vlm_timeout_s=args.vlm_timeout,
            deadline_s=args.deadline,
            log_path=args.log,
        )
    )


if __name__ == "__main__":
    main()
