"""Simulated Track 3 run: the real Pipeline loop, with each stage either scripted or a real model.

    uv run viamrobot-sim                                   # scripted scenario, nothing real
    uv run viamrobot-sim --speed 10                        # 10x faster; times stay in simulated seconds
    uv run viamrobot-sim --fixtures fixtures/armfarm01     # captured images, vision-1's recorded scores
    uv run viamrobot-sim --fixtures fixtures/armfarm01 --scorer local
                                                           # captured images scored by PatchCore on this Mac
    uv run viamrobot-sim --log judge.csv                   # also write the judge log to a file

The run starts by printing which class handles each flowchart step and what it
actually runs, e.g. "LocalPatchCoreScorer  PatchCore models/... on this computer's CPU".
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Literal

from viamrobot.decision import Destination
from viamrobot.pipeline import AnomalyScorer, CsvDecisionLog, ItemRecord, Pipeline, RunSummary, VlmVerdict
from viamrobot.stages.scripted import (
    LATENCY_S,
    ScenarioItem,
    ScriptedAnomalyScorer,
    ScriptedCamera,
    ScriptedDetector,
    ScriptedMover,
    ScriptedScene,
    ScriptedVlm,
)

ScorerChoice = Literal["scripted", "local"]

DEFAULT_SCENARIO: tuple[ScenarioItem, ...] = (
    ScenarioItem("good-1", Destination.BOX, 0.12),
    ScenarioItem("dent-1", Destination.REJECT, 0.88),
    ScenarioItem("good-borderline", Destination.BOX, 0.45, VlmVerdict(Destination.BOX, 0.85, "surface looks clean")),
    ScenarioItem("unseen-defect", Destination.REJECT, 0.52, VlmVerdict(Destination.REJECT, 0.74, "discoloured patch")),
    ScenarioItem("good-2", Destination.BOX, 0.08),
    ScenarioItem("vlm-hangs", Destination.REJECT, 0.55, "timeout"),
)

FIXTURE_TRUTH = {"good": Destination.BOX, "defect": Destination.REJECT, "unlabeled": None}


def load_fixtures(root: str | Path) -> tuple[ScenarioItem, ...]:
    """Scenario from viamrobot-capture output: each capture's color image and vision-1's recorded score.

    Labels good/defect give the expected destination; unlabeled has none; empty
    captures (nothing in view) are skipped. No VLM verdicts, so unsure items fall back to hold.
    """
    root = Path(root)
    items = []
    with (root / "index.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] == "empty":
                continue
            items.append(
                ScenarioItem(
                    name=Path(row["dir"]).name,
                    truth=FIXTURE_TRUTH.get(row["label"]),
                    anomaly_score=float(row["anomaly_score"]) if row["anomaly_score"] else None,
                    image_path=root / row["dir"] / "cam_color.jpg",
                )
            )
    if not items:
        raise ValueError(f"no usable captures in {root / 'index.csv'}")
    return tuple(items)


class NoFileLog:
    runs = "not written to a file (pass --log to save it)"

    def write(self, record: ItemRecord) -> None:
        pass


def make_scorer(choice: ScorerChoice, scene: ScriptedScene, from_fixtures: bool, model: Path | None) -> AnomalyScorer:
    if choice == "local":
        if any(item.image_path is None for item in scene.pending):
            raise ValueError("--scorer local needs real images: add --fixtures <capture folder>")
        from viamrobot.stages.local import PATCHCORE_MODEL, LocalPatchCoreScorer  # loads OpenVINO only when used

        return LocalPatchCoreScorer(model or PATCHCORE_MODEL)
    if from_fixtures:
        return ScriptedAnomalyScorer(scene, "the vision-1 score recorded when each image was captured")
    return ScriptedAnomalyScorer(scene)


async def simulate(
    items: tuple[ScenarioItem, ...] = DEFAULT_SCENARIO,
    scorer: ScorerChoice = "scripted",
    from_fixtures: bool = False,
    model: Path | None = None,
    speed: float = 1.0,
    seed: int = 0,
    vlm_timeout_s: float = 5.0,
    deadline_s: float = 300.0,
    log_path: str | Path | None = None,
    out: Callable[[str], None] = print,
) -> tuple[RunSummary, list[tuple[ScenarioItem, ItemRecord]]]:
    """Run the items through Pipeline.run. Times are in simulated seconds."""
    scene = ScriptedScene(items, speed, random.Random(seed))
    t0 = time.perf_counter()
    pipeline = Pipeline(
        camera=ScriptedCamera(scene),
        detector=ScriptedDetector(scene),
        scorer=make_scorer(scorer, scene, from_fixtures, model),
        vlm=ScriptedVlm(scene),
        mover=ScriptedMover(scene),
        log=CsvDecisionLog(log_path) if log_path else NoFileLog(),
        vlm_timeout_s=vlm_timeout_s / speed,
        clock=lambda: (time.perf_counter() - t0) * speed,
    )
    results: list[tuple[ScenarioItem, ItemRecord]] = []

    out(f"Simulated Track 3 run: {len(items)} items, speed {speed:g}x, VLM timeout {vlm_timeout_s:g} s")
    out("Stages:")
    for step, cls, runs in pipeline.describe():
        out(f"  {step:<17} {cls:<23} {runs}")
    if scorer == "local" and speed != 1:
        out(f"  note: real model times are also multiplied by --speed {speed:g}")
    out("")
    out(f"{'#':>3}  {'item':<28} {'score':>5}  {'decided by':<10}  {'dest':<7} {'conf':>5}  {'decide ms':>9}  truth")

    def on_record(record: ItemRecord) -> None:
        item = scene.placed[-1][0]
        results.append((item, record))
        if item.truth is None:
            verdict = "?"
        else:
            verdict = "ok" if record.destination is item.truth else f"want {item.truth.value}"
        out(
            f"{record.item_number:>3}  {item.name:<28} {record.anomaly_score:>5.2f}  {record.decided_by:<10}"
            f"  {record.destination.value:<7} {record.confidence:>5.2f}  {sum(record.timings_ms.values()):>9.0f}"
            f"  {verdict}"
        )

    summary = await pipeline.run(
        max_items=len(items), deadline_s=deadline_s, idle_poll_s=0.2 / speed, on_record=on_record
    )

    known = [(item, r) for item, r in results if item.truth is not None]
    correct = sum(r.destination is item.truth for item, r in known)
    held = sum(r.destination is Destination.HOLD for _, r in results)
    out("")
    out(
        f"Stopped: {summary.stopped_because} after {summary.elapsed_s:.1f} s. "
        f"Correct {correct}/{len(known)} with known truth, held {held}."
    )
    if summary.fault:
        out(f"Fault: {summary.fault}")
    out("Stage latency, ms (mean / max):")
    for stage in ("capture", "detect", "anomaly", "vlm"):
        values = [r.timings_ms[stage] for _, r in results if stage in r.timings_ms]
        if values:
            out(f"  {stage:<8} {statistics.mean(values):>7.0f} / {max(values):>7.0f}  (n={len(values)})")
    out("Judge log:")
    for _, record in results:
        out(f"  {record.csv()}")
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, help="replay captures from this folder instead of the scripted scenario")
    parser.add_argument(
        "--scorer",
        choices=("scripted", "local"),
        default="scripted",
        help="scripted: no model, uses the given/recorded score; local: run PatchCore on this computer",
    )
    parser.add_argument("--model", type=Path, help="PatchCore model.xml for --scorer local")
    parser.add_argument("--speed", type=float, default=1.0, help="time multiplier (default 1 = real time)")
    parser.add_argument("--seed", type=int, default=0, help="seed for latency jitter")
    parser.add_argument("--vlm-timeout", type=float, default=5.0, help="seconds before the VLM is abandoned")
    parser.add_argument("--deadline", type=float, default=300.0, help="run length in seconds (judged run is 5 min)")
    parser.add_argument("--log", type=Path, help="also append the judge log to this CSV file")
    args = parser.parse_args()
    items = load_fixtures(args.fixtures) if args.fixtures else DEFAULT_SCENARIO
    asyncio.run(
        simulate(
            items=items,
            scorer=args.scorer,
            from_fixtures=args.fixtures is not None,
            model=args.model,
            speed=args.speed,
            seed=args.seed,
            vlm_timeout_s=args.vlm_timeout,
            deadline_s=args.deadline,
            log_path=args.log,
        )
    )


if __name__ == "__main__":
    main()
