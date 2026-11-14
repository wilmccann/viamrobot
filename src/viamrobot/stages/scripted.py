"""Scripted stages: no models, no hardware. Used by the simulator (sim.py).

A ScriptedScene holds the items to show, one at a time: the camera sees a few
empty frames, then the item, which stays in view until the mover places it.
Each stage reads the item in view and answers from it, after a typical delay.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from viam.media.video import CameraMimeType, ViamImage

from viamrobot.decision import Destination
from viamrobot.pipeline import BoundingBox, Frame, VlmVerdict

# Typical seconds per stage, +/-20% jitter. Guesses until measured on the NUC.
LATENCY_S = {"capture": 0.03, "detect": 0.015, "anomaly": 0.06, "vlm": 1.2, "move": 3.0}


@dataclass(frozen=True)
class ScenarioItem:
    name: str
    truth: Destination | None  # where it should go; None if unknown
    anomaly_score: float | None = None  # what ScriptedAnomalyScorer returns
    vlm: VlmVerdict | Literal["timeout"] | None = None  # what ScriptedVlm returns, if asked
    image_path: Path | None = None  # a real captured image, for model-backed scorers
    empty_frames_before: int = 2


class ScriptedScene:
    """What is in front of the (pretend) camera, shared by all the scripted stages."""

    def __init__(self, items: tuple[ScenarioItem, ...], speed: float = 1.0, rng: random.Random | None = None):
        self.pending = list(items)
        self.current: ScenarioItem | None = None
        self.placed: list[tuple[ScenarioItem, Destination]] = []
        self._empty_seen = 0
        self.speed, self.rng = speed, rng or random.Random(0)

    async def work(self, stage: str) -> None:
        await asyncio.sleep(LATENCY_S[stage] * self.rng.uniform(0.8, 1.2) / self.speed)

    def advance(self) -> None:
        if self.current is not None or not self.pending:
            return
        if self._empty_seen >= self.pending[0].empty_frames_before:
            self.current, self._empty_seen = self.pending.pop(0), 0
        else:
            self._empty_seen += 1


class ScriptedCamera:
    """Item in view. No camera: shows scenario items, as their captured image when they have one."""

    runs = "no camera: shows scenario items one at a time (their captured image, if any)"

    def __init__(self, scene: ScriptedScene):
        self.scene = scene

    async def capture(self) -> Frame:
        await self.scene.work("capture")
        self.scene.advance()
        item = self.scene.current
        if item is not None and item.image_path is not None:
            return Frame(color=ViamImage(item.image_path.read_bytes(), CameraMimeType.JPEG))
        return Frame(color=None)


class ScriptedDetector:
    """Find the item. No model: reports an item whenever the scenario has one in view."""

    runs = "no model: reports an item whenever the scenario has one in view"

    def __init__(self, scene: ScriptedScene):
        self.scene = scene

    async def find_item(self, frame: Frame) -> BoundingBox | None:
        await self.scene.work("detect")
        item = self.scene.current
        return None if item is None else BoundingBox(200, 150, 440, 330, confidence=0.93, label=item.name)


class ScriptedAnomalyScorer:
    """Anomaly score. No model: returns the item's `anomaly_score`, wherever that came from."""

    def __init__(self, scene: ScriptedScene, source: str = "the score typed into the scenario"):
        self.scene = scene
        self.runs = f"no model: returns {source}"

    async def score(self, frame: Frame, box: BoundingBox) -> float:
        await self.scene.work("anomaly")
        item = self.scene.current
        if item is None or item.anomaly_score is None:
            raise ValueError(f"no scripted anomaly score for {item.name if item else 'empty view'}")
        return item.anomaly_score


class ScriptedVlm:
    """VLM second look. No model: returns the item's scripted verdict, or hangs if told to.

    An item with no verdict (e.g. a replayed capture) fails at once, so the run shows
    no made-up VLM time; the pipeline then sends the item to hold.
    """

    runs = "no model: returns the verdict typed into the scenario; fails at once if there is none"

    def __init__(self, scene: ScriptedScene):
        self.scene = scene

    async def second_look(self, frame: Frame, box: BoundingBox, anomaly_score: float) -> VlmVerdict:
        item = self.scene.current
        if item.vlm is None:
            raise RuntimeError(f"no scripted VLM verdict for {item.name}")
        if item.vlm == "timeout":
            await asyncio.sleep(3600)  # cancelled by the pipeline's timeout
        await self.scene.work("vlm")
        return item.vlm


class ScriptedMover:
    """Pick and place. No arm: waits a typical move time and takes the item out of view."""

    runs = "no arm: waits a typical move time, then removes the item from view"

    def __init__(self, scene: ScriptedScene):
        self.scene = scene

    async def pick_and_place(self, frame: Frame, box: BoundingBox, destination: Destination) -> None:
        await self.scene.work("move")
        self.scene.placed.append((self.scene.current, destination))
        self.scene.current = None
