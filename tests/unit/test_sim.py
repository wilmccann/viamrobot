"""The simulated run end to end, sped up so it takes well under a second."""

import json

import pytest

from viamrobot.decision import Destination
from viamrobot.sim import DEFAULT_SCENARIO, load_fixtures, simulate


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
    assert any("Correct 5/6 with known truth, held 1" in line for line in lines)


async def test_deadline_cuts_the_run_short():
    summary, results = await simulate(speed=200, deadline_s=5, out=lambda _: None)

    assert summary.stopped_because == "deadline"
    assert 0 < len(results) < len(DEFAULT_SCENARIO)


async def test_prints_which_class_runs_each_step():
    lines = []
    await simulate(speed=200, out=lines.append)
    stage_lines = "\n".join(lines[2:8])

    assert "anomaly score     ScriptedAnomalyScorer   no model: returns the score typed into the scenario" in stage_lines
    assert "VLM second look   ScriptedVlm" in stage_lines


def write_capture(root, rel, label, score):
    d = root / rel
    d.mkdir(parents=True)
    (d / "cam_color.jpg").write_bytes(b"jpeg")
    (d / "meta.json").write_text(json.dumps({"anomaly_score": score}))
    return f"{rel},{label},x,{score},normal=0.9,2026-11-14T00:00:00+00:00,\n"


def test_load_fixtures_maps_labels_and_skips_empty(tmp_path):
    rows = "dir,label,name,anomaly_score,vision2_top,captured_at,errors\n"
    rows += write_capture(tmp_path, "good/g_1", "good", 0.1)
    rows += write_capture(tmp_path, "defect/d_1", "defect", 0.9)
    rows += write_capture(tmp_path, "empty/e_1", "empty", 0.5)
    rows += write_capture(tmp_path, "unlabeled/u_1", "unlabeled", 0.5)
    (tmp_path / "index.csv").write_text(rows)

    items = load_fixtures(tmp_path)

    assert [(i.name, i.truth, i.anomaly_score) for i in items] == [
        ("g_1", Destination.BOX, 0.1),
        ("d_1", Destination.REJECT, 0.9),
        ("u_1", None, 0.5),
    ]
    assert items[0].image_path == tmp_path / "good/g_1/cam_color.jpg"


async def test_fixture_replay_uses_recorded_scores(tmp_path):
    rows = "dir,label,name,anomaly_score,vision2_top,captured_at,errors\n"
    rows += write_capture(tmp_path, "good/g_1", "good", 0.1)
    rows += write_capture(tmp_path, "defect/d_1", "defect", 0.9)
    (tmp_path / "index.csv").write_text(rows)
    lines = []

    _, results = await simulate(load_fixtures(tmp_path), from_fixtures=True, speed=200, out=lines.append)

    assert [r.destination for _, r in results] == [Destination.BOX, Destination.REJECT]
    assert any("recorded when each image was captured" in line for line in lines)


async def test_local_scorer_needs_images():
    with pytest.raises(ValueError, match="--fixtures"):
        await simulate(scorer="local", speed=200, out=lambda _: None)
