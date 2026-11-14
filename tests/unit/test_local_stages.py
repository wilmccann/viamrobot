"""LocalPatchCoreScorer with the downloaded model. Skipped when models/ or fixtures/ aren't on this computer."""

import json
from pathlib import Path

import pytest
from viam.media.video import CameraMimeType, ViamImage

from viamrobot.pipeline import BoundingBox, Frame
from viamrobot.stages.local import PATCHCORE_MODEL, LocalPatchCoreScorer

FIXTURES = Path("fixtures/armfarm01")
BOX = BoundingBox(0, 0, 1, 1, confidence=1.0)

needs_model = pytest.mark.skipif(not PATCHCORE_MODEL.exists(), reason=f"{PATCHCORE_MODEL} not downloaded")


@pytest.fixture(scope="module")
def scorer():
    return LocalPatchCoreScorer()


def captures_with_scores():
    if not FIXTURES.exists():
        return []
    return sorted(p.parent for p in FIXTURES.glob("*/*/meta.json") if "anomaly_score" in json.loads(p.read_text()))


def test_missing_model_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="models/README.md"):
        LocalPatchCoreScorer(tmp_path / "nope.xml")


@needs_model
def test_describes_what_it_runs(scorer):
    assert "PatchCore" in scorer.runs and "CPU" in scorer.runs
    assert (scorer.h, scorer.w) == (256, 256)


@needs_model
async def test_needs_a_viam_image(scorer):
    with pytest.raises(TypeError, match="ViamImage"):
        await scorer.score(Frame(color=None), BOX)


@needs_model
@pytest.mark.parametrize("capture", captures_with_scores(), ids=lambda p: p.name)
async def test_matches_vision1_score_recorded_on_armfarm01(scorer, capture):
    """Same model + same preprocessing should reproduce what the NUC's GPU said for this image."""
    recorded = json.loads((capture / "meta.json").read_text())["anomaly_score"]
    image = ViamImage((capture / "cam_color.jpg").read_bytes(), CameraMimeType.JPEG)

    local = await scorer.score(Frame(color=image), BOX)

    print(f"\n{capture.name}: local {local:.4f}, armfarm01 {recorded:.4f}")
    assert local == pytest.approx(recorded, abs=0.02)
