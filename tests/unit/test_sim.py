"""The simulated run end to end, sped up so it takes well under a second."""

from viamrobot.decision import Destination
from viamrobot.sim import DEFAULT_SCENARIO, simulate


async def test_default_scenario_end_to_end(tmp_path):
    log_path = tmp_path / "judge.csv"
    lines = []

    summary, results = await simulate(speed=200, log_path=log_path, out=lines.append)

    assert summary.stopped_because == "max_items"
    assert [item.name for item, _ in results] == [item.name for item in DEFAULT_SCENARIO]
    assert [r.destination for _, r in results] == [
        Destination.BOX,  # clear good
        Destination.REJECT,  # clear defect
        Destination.BOX,  # unsure, VLM says box
        Destination.REJECT,  # unseen defect, VLM catches it
        Destination.BOX,  # clear good
        Destination.HOLD,  # VLM times out
    ]
    assert [r.decided_by for _, r in results] == ["anomaly", "anomaly", "vlm", "vlm", "anomaly", "fallback"]
    assert len(log_path.read_text().splitlines()) == 6
    assert any("Correct 5/6, held 1" in line for line in lines)


async def test_deadline_cuts_the_run_short():
    summary, results = await simulate(speed=200, deadline_s=5, out=lambda _: None)

    assert summary.stopped_because == "deadline"
    assert 0 < len(results) < len(DEFAULT_SCENARIO)
