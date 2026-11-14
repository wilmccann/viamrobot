"""Sorting decisions and the per-item judge log. Pure logic: no robot connection needed."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Destination(str, Enum):
    BOX = "box"  # good item
    REJECT = "reject"  # defective item
    HOLD = "hold"  # not sure


@dataclass(frozen=True)
class Thresholds:
    good_below: float = 0.4
    defect_above: float = 0.6


def decide(anomaly_score: float, t: Thresholds = Thresholds()) -> Destination:
    """Map an anomaly score in [0, 1] to where the item should go."""
    if not 0.0 <= anomaly_score <= 1.0:
        raise ValueError(f"anomaly score out of range: {anomaly_score}")
    if anomaly_score < t.good_below:
        return Destination.BOX
    if anomaly_score > t.defect_above:
        return Destination.REJECT
    return Destination.HOLD


def log_line(item_number: int, destination: Destination, confidence: float, when: datetime | None = None) -> str:
    """One judge-log line per item: number, decision, confidence, timestamp (CSV)."""
    when = when or datetime.now(timezone.utc)
    return f"{item_number},{destination.value},{confidence:.3f},{when.isoformat()}"
