from datetime import datetime, timezone

import pytest

from viamrobot.decision import Destination, Thresholds, decide, log_line


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, Destination.BOX),
        (0.39, Destination.BOX),
        (0.4, Destination.HOLD),
        (0.6, Destination.HOLD),
        (0.61, Destination.REJECT),
        (1.0, Destination.REJECT),
    ],
)
def test_decide_default_thresholds(score, expected):
    assert decide(score) == expected


def test_decide_custom_thresholds():
    assert decide(0.5, Thresholds(good_below=0.55, defect_above=0.8)) == Destination.BOX


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_decide_rejects_bad_scores(score):
    with pytest.raises(ValueError):
        decide(score)


def test_log_line_format():
    when = datetime(2026, 11, 14, 15, 30, tzinfo=timezone.utc)
    assert log_line(3, Destination.REJECT, 0.87654, when) == "3,reject,0.877,2026-11-14T15:30:00+00:00"
