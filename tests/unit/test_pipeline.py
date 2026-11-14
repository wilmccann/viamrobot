"""Mock tests for the Track 3 pipeline: every branch of the flowchart, with fake stages."""

import asyncio
from datetime import datetime, timezone

import pytest

from viamrobot.decision import Destination
from viamrobot.pipeline import BoundingBox, CsvDecisionLog, Frame, ItemRecord, Pipeline, VlmVerdict

FIXED_NOW = datetime(2026, 11, 14, 15, 30, tzinfo=timezone.utc)
BOX = BoundingBox(10, 10, 100, 100, confidence=0.9)


# --- fakes: each records what it was asked into a shared event list ---------


class FakeCamera:
    def __init__(self, events):
        self.events = events

    async def capture(self):
        self.events.append("capture")
        return Frame(color="rgb", depth="depth")


class FakeDetector:
    def __init__(self, events, box):
        self.events, self.box = events, box

    async def find_item(self, frame):
        self.events.append("detect")
        return self.box


class FakeScorer:
    def __init__(self, events, score):
        self.events, self.value = events, score

    async def score(self, frame, box):
        self.events.append("anomaly")
        return self.value


class FakeVlm:
    """`result` is a VlmVerdict to return, an exception to raise, or "hang" to sleep past the timeout."""

    def __init__(self, events, result):
        self.events, self.result = events, result

    async def second_look(self, frame, box, anomaly_score):
        self.events.append("vlm")
        if self.result == "hang":
            await asyncio.sleep(10)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeMover:
    def __init__(self, events, fail=False):
        self.events, self.fail, self.moves = events, fail, []

    async def pick_and_place(self, frame, box, destination):
        self.events.append(f"move:{destination.value}")
        self.moves.append(destination)
        if self.fail:
            raise RuntimeError("arm fault")


class FakeLog:
    def __init__(self, events):
        self.events, self.records = events, []

    def write(self, record):
        self.events.append(f"log:{record.destination.value}")
        self.records.append(record)


@pytest.fixture
def build():
    """build(score=..., box=..., vlm=..., mover_fails=...) -> (pipeline, events, fakes)."""

    def _build(score=0.1, box=BOX, vlm=None, mover_fails=False, **kwargs):
        events = []
        fakes = dict(
            camera=FakeCamera(events),
            detector=FakeDetector(events, box),
            scorer=FakeScorer(events, score),
            vlm=FakeVlm(events, vlm),
            mover=FakeMover(events, fail=mover_fails),
            log=FakeLog(events),
        )
        pipeline = Pipeline(**fakes, now=lambda: FIXED_NOW, **kwargs)
        return pipeline, events, fakes

    return _build


# --- clear calls: anomaly score decides, VLM never runs ---------------------


async def test_clear_good_goes_to_box_without_vlm(build):
    pipeline, events, _ = build(score=0.1)
    record = await pipeline.process(1)

    assert record.destination is Destination.BOX
    assert record.decided_by == "anomaly"
    assert record.confidence == pytest.approx(0.9)
    assert events == ["capture", "detect", "anomaly", "log:box", "move:box"]


async def test_clear_defect_goes_to_reject_without_vlm(build):
    pipeline, events, _ = build(score=0.85)
    record = await pipeline.process(2)

    assert record.destination is Destination.REJECT
    assert record.decided_by == "anomaly"
    assert record.confidence == pytest.approx(0.85)
    assert "vlm" not in events
    assert events[-2:] == ["log:reject", "move:reject"]


# --- unsure: VLM second look -------------------------------------------------


@pytest.mark.parametrize("vlm_dest", [Destination.BOX, Destination.REJECT, Destination.HOLD])
async def test_unsure_uses_vlm_verdict(build, vlm_dest):
    pipeline, events, _ = build(score=0.5, vlm=VlmVerdict(vlm_dest, 0.8, "scratch on lid"))
    record = await pipeline.process(3)

    assert record.destination is vlm_dest
    assert record.decided_by == "vlm"
    assert record.confidence == 0.8
    assert record.vlm_reason == "scratch on lid"
    assert record.anomaly_score == 0.5
    assert events == ["capture", "detect", "anomaly", "vlm", f"log:{vlm_dest.value}", f"move:{vlm_dest.value}"]


async def test_unsure_low_confidence_vlm_goes_to_hold(build):
    pipeline, _, fakes = build(score=0.5, vlm=VlmVerdict(Destination.REJECT, 0.3))
    record = await pipeline.process(4)

    assert record.destination is Destination.HOLD
    assert record.decided_by == "vlm"
    assert fakes["mover"].moves == [Destination.HOLD]


async def test_vlm_crash_falls_back_to_hold(build):
    pipeline, _, fakes = build(score=0.5, vlm=RuntimeError("GPU out of memory"))
    record = await pipeline.process(5)

    assert record.destination is Destination.HOLD
    assert record.decided_by == "fallback"
    assert record.confidence == 0.0
    assert "RuntimeError" in record.vlm_reason
    assert fakes["mover"].moves == [Destination.HOLD]


async def test_vlm_timeout_falls_back_to_hold(build):
    pipeline, _, fakes = build(score=0.5, vlm="hang", vlm_timeout_s=0.01)
    record = await pipeline.process(6)

    assert record.destination is Destination.HOLD
    assert record.decided_by == "fallback"
    assert "TimeoutError" in record.vlm_reason
    assert fakes["mover"].moves == [Destination.HOLD]


# --- invariants ---------------------------------------------------------------


async def test_no_item_in_view_does_nothing(build):
    pipeline, events, fakes = build(box=None)

    assert await pipeline.process(7) is None
    assert events == ["capture", "detect"]
    assert fakes["log"].records == []
    assert fakes["mover"].moves == []


@pytest.mark.parametrize(
    ("score", "vlm"),
    [(0.1, None), (0.9, None), (0.5, VlmVerdict(Destination.BOX, 0.9)), (0.5, RuntimeError("boom"))],
)
async def test_every_item_logged_before_arm_moves(build, score, vlm):
    pipeline, events, _ = build(score=score, vlm=vlm)
    await pipeline.process(8)

    log_i = next(i for i, e in enumerate(events) if e.startswith("log:"))
    move_i = next(i for i, e in enumerate(events) if e.startswith("move:"))
    assert log_i < move_i
    assert events[log_i].split(":")[1] == events[move_i].split(":")[1]  # same destination


async def test_arm_failure_still_leaves_log_entry(build):
    pipeline, _, fakes = build(score=0.9, mover_fails=True)

    with pytest.raises(RuntimeError, match="arm fault"):
        await pipeline.process(9)
    assert [r.item_number for r in fakes["log"].records] == [9]


async def test_stage_timings_recorded(build):
    ticks = iter(range(0, 100))
    pipeline, _, _ = build(score=0.5, vlm=VlmVerdict(Destination.BOX, 0.9), clock=lambda: next(ticks) / 1000)
    record = await pipeline.process(10)

    assert record.timings_ms.keys() == {"capture", "detect", "anomaly", "vlm"}
    assert all(ms == pytest.approx(1.0) for ms in record.timings_ms.values())


# --- the judge log file ---------------------------------------------------------


def test_csv_decision_log_appends_one_line_per_item(tmp_path):
    path = tmp_path / "judge_log.csv"
    log = CsvDecisionLog(path)
    for n, dest, conf in [(1, Destination.BOX, 0.91), (2, Destination.HOLD, 0.0)]:
        log.write(ItemRecord(n, dest, conf, "anomaly", 0.1, FIXED_NOW))

    assert path.read_text().splitlines() == [
        "1,box,0.910,2026-11-14T15:30:00+00:00",
        "2,hold,0.000,2026-11-14T15:30:00+00:00",
    ]
