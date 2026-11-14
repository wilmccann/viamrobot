"""Save replayable fixtures from armfarm01: camera frames plus what the models said about them.

Read-only: never moves the arm or gripper. Place the item yourself, then run:

    uv run viamrobot-capture --label good --name bottle-1
    uv run viamrobot-capture --label defect --name bottle-dent --count 3
    uv run viamrobot-capture --label empty --name tray

Each capture goes to fixtures/armfarm01/<label>/<name>_<time>_<n>/ with
    cam_color.jpg, cam_depth.dep  wrist RealSense (depth is Viam's raw depth format)
    overview.jpg                  overview webcam
    meta.json                     vision-1 anomaly result and score, vision-2 classes,
                                  arm pose and joints, camera intrinsics, call timings
and one row is appended to fixtures/armfarm01/index.csv. Failures of any optional
part are recorded in meta.json instead of aborting the capture.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable

from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import ViamImage
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

from viamrobot.stages.machine import anomaly_score

CAMERA, OVERVIEW, ARM, ANOMALY, CLASSIFIER = "cam", "overview-cam", "arm", "vision-1", "vision-2"
LABELS = ("good", "defect", "empty", "unlabeled")
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/vnd.viam.dep": ".dep"}


def _ext(mime: str) -> str:
    return EXT.get(mime, ".bin")


def _is_depth(img: ViamImage) -> bool:
    return "dep" in img.mime_type


def _to_dict(value: Any) -> Any:
    if hasattr(value, "DESCRIPTOR"):
        return MessageToDict(value, preserving_proto_field_name=True)
    return value


async def _timed(meta: dict, key: str, awaitable: Awaitable) -> Any:
    """Await, recording ms in meta["timings_ms"]; on failure record the error and return None."""
    start = time.perf_counter()
    try:
        return await awaitable
    except Exception as exc:
        meta.setdefault("errors", {})[key] = f"{type(exc).__name__}: {exc}"
        return None
    finally:
        meta.setdefault("timings_ms", {})[key] = round((time.perf_counter() - start) * 1000, 1)


async def connect() -> RobotClient:
    load_dotenv()
    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


async def camera_intrinsics(robot: RobotClient) -> dict:
    props = await Camera.from_robot(robot, CAMERA).get_properties()
    return {
        "intrinsic_parameters": _to_dict(props.intrinsic_parameters),
        "distortion_parameters": _to_dict(props.distortion_parameters),
    }


async def capture_one(robot: RobotClient, out_dir: Path, label: str, name: str, intrinsics: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    meta: dict[str, Any] = {
        "label": label,
        "name": name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "camera_intrinsics": intrinsics,
        "files": {},
    }

    result = await _timed(meta, "cam_get_images", Camera.from_robot(robot, CAMERA).get_images())
    images = result[0] if result else []
    color = next((img for img in images if not _is_depth(img)), None)
    for img in images:
        fname = f"cam_{'depth' if _is_depth(img) else 'color'}{_ext(img.mime_type)}"
        (out_dir / fname).write_bytes(img.data)
        meta["files"][fname] = img.mime_type

    overview = await _timed(meta, "overview_get_images", Camera.from_robot(robot, OVERVIEW).get_images())
    for img in (overview[0] if overview else [])[:1]:
        fname = f"overview{_ext(img.mime_type)}"
        (out_dir / fname).write_bytes(img.data)
        meta["files"][fname] = img.mime_type

    if color is not None:
        anomaly = await _timed(meta, ANOMALY, VisionClient.from_robot(robot, ANOMALY).get_classifications(color, 1))
        if anomaly is not None:
            meta[ANOMALY] = [{"class_name": c.class_name, "confidence": c.confidence} for c in anomaly]
            meta["anomaly_score"] = anomaly_score(anomaly)
        classes = await _timed(meta, CLASSIFIER, VisionClient.from_robot(robot, CLASSIFIER).get_classifications(color, 5))
        if classes is not None:
            meta[CLASSIFIER] = [{"class_name": c.class_name, "confidence": c.confidence} for c in classes]

    arm = Arm.from_robot(robot, ARM)
    pose = await _timed(meta, "arm_pose", arm.get_end_position())
    joints = await _timed(meta, "arm_joints", arm.get_joint_positions())
    meta["arm_pose"] = _to_dict(pose) if pose is not None else None
    meta["arm_joints_deg"] = _to_dict(joints) if joints is not None else None

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def append_index(index: Path, out_dir: Path, meta: dict) -> None:
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dir", "label", "name", "anomaly_score", "vision2_top", "captured_at", "errors"])
        top = (meta.get(CLASSIFIER) or [{}])[0]
        w.writerow([
            out_dir.relative_to(index.parent).as_posix(),
            meta["label"],
            meta["name"],
            f"{meta['anomaly_score']:.4f}" if "anomaly_score" in meta else "",
            f"{top.get('class_name', '')}={top.get('confidence', 0):.3f}" if top else "",
            meta["captured_at"],
            ";".join(sorted(meta.get("errors", {}))),
        ])


async def run(label: str, name: str, count: int, interval_s: float, root: Path) -> None:
    robot = await connect()
    try:
        intrinsics = await camera_intrinsics(robot)
        for i in range(1, count + 1):
            stamp = datetime.now().strftime("%H%M%S")
            out_dir = root / label / f"{name}_{stamp}_{i}"
            meta = await capture_one(robot, out_dir, label, name, intrinsics)
            append_index(root / "index.csv", out_dir, meta)
            score = meta.get("anomaly_score")
            top = (meta.get(CLASSIFIER) or [{}])[0]
            print(
                f"{out_dir}: anomaly {score:.3f}" if score is not None else f"{out_dir}: anomaly n/a",
                f"| vision-2 {top.get('class_name')}={top.get('confidence', 0):.3f}" if top else "",
                f"| files {', '.join(meta['files'])}",
                f"| errors {meta['errors']}" if meta.get("errors") else "",
            )
            if i < count:
                await asyncio.sleep(interval_s)
    finally:
        await robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, choices=LABELS, help="what is in view")
    parser.add_argument("--name", required=True, help="short item name, e.g. bottle-dent")
    parser.add_argument("--count", type=int, default=1, help="frames to capture (default 1)")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    parser.add_argument("--out", type=Path, default=Path("fixtures/armfarm01"), help="fixture root")
    args = parser.parse_args()
    asyncio.run(run(args.label, args.name, args.count, args.interval, args.out))


if __name__ == "__main__":
    main()
