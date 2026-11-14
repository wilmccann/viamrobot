"""Read-only checks against the real armfarm01. Nothing here moves the arm or gripper.

Run with:  uv run pytest -m machine -s
Anything that moves hardware must also be marked `moves` and run deliberately, with someone at the E-stop.
"""

import pytest
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.vision import VisionClient

pytestmark = pytest.mark.machine

ARM, CAMERA, ANOMALY_VISION = "arm", "cam", "vision-1"
EXPECTED = {ARM, "gripper", CAMERA, "overview-cam", ANOMALY_VISION}

# Work-cell limits from the obstacles fragment (mm, arm base frame): table top and ceiling.
TABLE_Z, CEILING_Z = -123, 1050


async def test_expected_resources(robot):
    names = {r.name for r in robot.resource_names}
    missing = EXPECTED - names
    assert not missing, f"missing on machine: {sorted(missing)}"


async def test_arm_pose_inside_work_cell(robot):
    arm = Arm.from_robot(robot, ARM)
    pose = await arm.get_end_position()
    print(f"\narm end pose: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} mm")
    assert TABLE_Z < pose.z < CEILING_Z


async def test_camera_returns_image(robot):
    images, _ = await Camera.from_robot(robot, CAMERA).get_images()
    assert images, "camera returned no images"
    for img in images:
        print(f"\n{img.name}: {img.mime_type}, {len(img.data)} bytes")
        assert img.data


async def test_anomaly_vision_classifies(robot):
    """The anomaly service labels the current camera view (e.g. "normal" + confidence).

    Uses get_classifications_from_camera: on 2026-11-14 capture_all_from_camera on vision-1
    crashed server-side with a nil pointer error.
    """
    vision = VisionClient.from_robot(robot, ANOMALY_VISION)
    props = await vision.get_properties()
    assert props.classifications_supported
    results = await vision.get_classifications_from_camera(CAMERA, count=3)
    print(f"\n{ANOMALY_VISION}: " + ", ".join(f"{c.class_name}={c.confidence:.3f}" for c in results))
    assert results, "no classification returned"
    assert all(0.0 <= c.confidence <= 1.0 for c in results)
