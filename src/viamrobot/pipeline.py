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
    """Anomaly score in [0, 1]: PatchCore on the GPU, f32."""

    async def score(self, frame: Frame, box: BoundingBox) -> float: ...


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


class CsvDecisionLog:
    """Appends one judge-log line per item and flushes, so a crash mid-run keeps prior lines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

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
        await self.mover.pick_and_place(frame, box, destination)
        return record

    async def _timed(self, stage: str, timings: dict[str, float], awaitable):
        start = self.clock()
        try:
            return await awaitable
        finally:
            timings[stage] = (self.clock() - start) * 1000.0
