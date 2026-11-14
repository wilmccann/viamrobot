"""Local stages: the downloaded model files in models/, run on this computer with OpenVINO."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from viam.media.video import ViamImage

from viamrobot.pipeline import BoundingBox, Frame

# Version 1 is what armfarm01 runs (its ov service loads sortit-patchcore-1); latest is a later retrain.
PATCHCORE_MODEL = Path("models/sortit-patchcore/1/model.xml")


def decode_image(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB HxWx3 uint8. Same as viam-labs:anomalib src/decode.py."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image; expected JPEG or PNG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_input(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """RGB uint8 -> float32 [1, 3, h, w] in [0, 1]. Same as viam-labs:anomalib src/decode.py."""
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


class LocalPatchCoreScorer:
    """Anomaly score: the downloaded PatchCore model (sortit-patchcore), run here with OpenVINO.

    The same model files and the same preprocessing as vision-1 on armfarm01, so scores
    should match the machine's closely. Runs in f32, as on the NUC GPU.
    """

    def __init__(self, model_path: str | Path = PATCHCORE_MODEL, device: str = "CPU"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found; see models/README.md to download it")
        self.compiled = ov.Core().compile_model(str(model_path), device, {"INFERENCE_PRECISION_HINT": "f32"})
        _, _, self.h, self.w = self.compiled.input(0).get_shape()
        self.input_name = self.compiled.input(0).get_any_name()
        self.runs = f"PatchCore {model_path} on this computer's {device} (OpenVINO, f32)"

    def score_bytes(self, data: bytes) -> float:
        """Raw PatchCore score for one JPEG/PNG image, clamped to [0, 1]."""
        out = self.compiled({self.input_name: to_input(decode_image(data), self.h, self.w)})
        score = float(np.asarray(out["pred_score"]).reshape(-1)[0])
        return min(max(score, 0.0), 1.0)

    async def score(self, frame: Frame, box: BoundingBox) -> float:
        if not isinstance(frame.color, ViamImage):
            raise TypeError(f"LocalPatchCoreScorer needs a ViamImage frame, got {type(frame.color).__name__}")
        return await asyncio.to_thread(self.score_bytes, frame.color.data)
