"""Track 3 sorting pipeline, per the proposed design:

    item in view -> find the item -> anomaly score -> clear call?
        clear  -> log the decision -> pick and place (box / reject bin / hold area)
        unsure -> VLM second look -> log the decision -> pick and place

Every item gets a logged decision before the arm moves.

Each stage is a Protocol, so tests plug in fakes and the real pieces
(RealSense camera, yolo/rf-detr on the NPU, PatchCore on the GPU,
openvino-genai VLM, xArm motion) can be wired in later without touching
the control flow below.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Protocol

from viamrobot.decision import Destination, Thresholds, decide, log_line

DecidedBy = Literal["anomaly", "vlm", "fallback"]
StopReason = Literal["max_items", "deadline", "arm_fault"]


@dataclass(frozen=True)
class Frame:
    """One capture from the wrist RealSense. Image types are opaque to the pipeline."""

    color: object
    depth: object | None = None


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str = "item"


@dataclass(frozen=True)
class VlmVerdict:
    destination: Destination
    confidence: float
    reason: str = ""


# --- stage interfaces -------------------------------------------------------


class Camera(Protocol):
    """Item in view: RealSense on the wrist."""

    async def capture(self) -> Frame: ...


class Detector(Protocol):
    """Find the item: yolo or rf-detr on the NPU. None means nothing in view."""

    async def find_item(self, frame: Frame) -> BoundingBox | None: ...


class AnomalyScorer(Protocol):
    """Anomaly score in [0, 1], using normalized image variation as a baseline."""

    async def score(self, frame: Frame, box: BoundingBox) -> float:
        """Estimate anomaly from grayscale intensity variation in the item crop.

        This lightweight baseline works with nested pixel sequences and array-like
        images exposing ``tolist``; production deployments can provide PatchCore.
        """
        image = frame.color
        tolist = getattr(image, "tolist", None)
        if callable(tolist):
            image = tolist()

        try:
            height = len(image)
            width = len(image[0]) if height else 0
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError("frame.color must be a non-empty height-by-width image") from exc
        if height == 0 or width == 0:
            raise ValueError("frame.color must be a non-empty height-by-width image")

        x0 = max(0, min(width, box.x_min))
        y0 = max(0, min(height, box.y_min))
        x1 = max(x0, min(width, box.x_max + 1))
        y1 = max(y0, min(height, box.y_max + 1))
        intensities: list[float] = []
        for row in image[y0:y1]:
            for pixel in row[x0:x1]:
                if isinstance(pixel, (list, tuple)):
                    channels = [float(channel) for channel in pixel[:3]]
                    if not channels:
                        continue
                    intensities.append(sum(channels) / len(channels))
                else:
                    intensities.append(float(pixel))
        if not intensities:
            raise ValueError("bounding box does not overlap frame.color")

        mean = sum(intensities) / len(intensities)
        variance = sum((value - mean) ** 2 for value in intensities) / len(intensities)
        intensity_scale = 1.0 if max(intensities) <= 1.0 else 255.0
        return min(1.0, (variance**0.5) / (intensity_scale / 2.0))


class Vlm(Protocol):
    """VLM second look: openvino-genai on GPU or NPU."""

    async def second_look(self, frame: Frame, box: BoundingBox, anomaly_score: float) -> VlmVerdict: ...


class Mover(Protocol):
    """Pick and place: depth + motion on the CPU."""

    async def pick_and_place(self, frame: Frame, box: BoundingBox, destination: Destination) -> None: ...


class DecisionLog(Protocol):
    def write(self, record: ItemRecord) -> None: ...


# --- the decision record ----------------------------------------------------


@dataclass(frozen=True)
class ItemRecord:
    item_number: int
    destination: Destination
    confidence: float
    decided_by: DecidedBy
    anomaly_score: float
    decided_at: datetime
    timings_ms: dict[str, float] = field(default_factory=dict)
    vlm_reason: str = ""

    def csv(self) -> str:
        """The mandatory judge-log line: number, decision, confidence, timestamp."""
        return log_line(self.item_number, self.destination, self.confidence, self.decided_at)


class PickPlaceError(RuntimeError):
    """The decision was logged, but the arm failed to put the item where it was sent."""

    def __init__(self, record: ItemRecord, cause: BaseException):
        super().__init__(f"item {record.item_number} -> {record.destination.value}: pick and place failed: {cause}")
        self.record = record


@dataclass(frozen=True)
class RunSummary:
    records: list[ItemRecord]
    stopped_because: StopReason
    elapsed_s: float
    fault: str = ""


class CsvDecisionLog:
    """Appends one judge-log line per item and flushes, so a crash mid-run keeps prior lines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.runs = f"appends CSV lines to {self.path}"

    def write(self, record: ItemRecord) -> None:
        with self.path.open("a") as f:
            f.write(record.csv() + "\n")


# --- the pipeline -----------------------------------------------------------


@dataclass
class Pipeline:
    camera: Camera
    detector: Detector
    scorer: AnomalyScorer
    vlm: Vlm
    mover: Mover
    log: DecisionLog
    thresholds: Thresholds = field(default_factory=Thresholds)
    vlm_timeout_s: float = 5.0
    min_vlm_confidence: float = 0.5
    clock: Callable[[], float] = time.perf_counter
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def describe(self) -> list[tuple[str, str, str]]:
        """(flowchart step, class used, what it actually runs) for each stage."""
        stages = {
            "item in view": self.camera,
            "find the item": self.detector,
            "anomaly score": self.scorer,
            "VLM second look": self.vlm,
            "log the decision": self.log,
            "pick and place": self.mover,
        }
        return [(step, type(s).__name__, getattr(s, "runs", "")) for step, s in stages.items()]

    async def process(self, item_number: int) -> ItemRecord | None:
        """Run one item through the pipeline. Returns None when no item is in view."""
        timings: dict[str, float] = {}

        frame = await self._timed("capture", timings, self.camera.capture())
        box = await self._timed("detect", timings, self.detector.find_item(frame))
        if box is None:
            return None

        score = await self._timed("anomaly", timings, self.scorer.score(frame, box))
        destination = decide(score, self.thresholds)
        vlm_reason = ""

        if destination is not Destination.HOLD:
            # Clear call: confidence is how far the score sits toward that side.
            decided_by: DecidedBy = "anomaly"
            confidence = score if destination is Destination.REJECT else 1.0 - score
        else:
            # Unsure: ask the VLM. Any failure or weak answer goes to the hold area.
            try:
                verdict = await self._timed(
                    "vlm",
                    timings,
                    asyncio.wait_for(self.vlm.second_look(frame, box, score), self.vlm_timeout_s),
                )
            except Exception as exc:  # timeout, model crash: never stall the run
                decided_by, destination, confidence = "fallback", Destination.HOLD, 0.0
                vlm_reason = f"vlm failed: {type(exc).__name__}"
            else:
                decided_by, confidence, vlm_reason = "vlm", verdict.confidence, verdict.reason
                destination = verdict.destination
                if verdict.confidence < self.min_vlm_confidence:
                    destination = Destination.HOLD

        record = ItemRecord(
            item_number=item_number,
            destination=destination,
            confidence=confidence,
            decided_by=decided_by,
            anomaly_score=score,
            decided_at=self.now(),
            timings_ms=timings,
            vlm_reason=vlm_reason,
        )
        self.log.write(record)  # logged before the arm moves
        try:
            await self.mover.pick_and_place(frame, box, destination)
        except Exception as exc:
            raise PickPlaceError(record, exc) from exc
        return record

    async def run(
        self,
        max_items: int = 6,
        deadline_s: float = 300.0,
        idle_poll_s: float = 0.2,
        on_record: Callable[[ItemRecord], None] | None = None,
    ) -> RunSummary:
        """Loop back to "item in view" until max_items are decided or the deadline passes.

        Empty frames don't use up an item number. The deadline is checked before each
        attempt, so an item already in progress is allowed to finish. An arm fault stops
        the run: the item is already logged, and a faulted arm needs a person anyway.
        """
        start = self.clock()
        records: list[ItemRecord] = []

        def summary(reason: StopReason, fault: str = "") -> RunSummary:
            return RunSummary(records, reason, self.clock() - start, fault)

        while len(records) < max_items:
            if self.clock() - start >= deadline_s:
                return summary("deadline")
            try:
                record = await self.process(len(records) + 1)
            except PickPlaceError as exc:
                records.append(exc.record)
                if on_record:
                    on_record(exc.record)
                return summary("arm_fault", str(exc))
            if record is None:
                await asyncio.sleep(idle_poll_s)
                continue
            records.append(record)
            if on_record:
                on_record(record)
        return summary("max_items")

    async def _timed(self, stage: str, timings: dict[str, float], awaitable):
        start = self.clock()
        try:
            return await awaitable
        finally:
            timings[stage] = (self.clock() - start) * 1000.0
