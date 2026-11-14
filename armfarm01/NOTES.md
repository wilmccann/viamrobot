# armfarm01 — hardware probe (2026-11-13)

Machine `f98e9633-107d-4ece-b19f-b8ab78ec95ef`, part `f08ae88f-5fa8-4775-a3bf-0f4fda96940f` (`armfarm1-main`), host `intel-NUC16GDKX76`.

## Compute
- Intel Core Ultra X7 358H, 16 cores, 62 GiB RAM, 824 GB free disk, Ubuntu 24.04, kernel 7.0
- OpenVINO 2026.4.1 sees: `CPU`, `GPU` = Arc B390 iGPU, `NPU` = Intel AI Boost (`xe` and `intel_vpu` drivers loaded)
- No system-wide OpenVINO for python3; venvs with it: `/root/act-bench/venv`, `/root/smolvla-bench/venv`
- Organizer benchmarks in `/root`: `act-bench`, `smolvla-bench`, `pi05-bench`

## Cameras
| Device | What | Notes |
|---|---|---|
| `/dev/video0-5` | RealSense D435 (USB serial 201723022077) | used by `cam` via `viam:camera:realsense` |
| `/dev/video6` | USB webcam, color stream | **overview-cam config points at `/dev/video38` — wrong.** Stable path: `/dev/v4l/by-id/usb-USB_Camera_USB_Camera_20250807114148-video-index0` |
| `/dev/video7` | USB webcam metadata node | |

## Track 3 starter: `viam-labs:anomalib` (https://github.com/viam-labs/viam-anomalib-service)
- Vision service `viam-labs:anomalib:vision` on top of mlmodel `erh:openvino:mlmodel`
- Training script `hackathons:anomalib-train` v0.4.0, container `9accd6e7-2992-47c9-9662-2ca5298fa1f1`, PatchCore by default, exports OpenVINO IR
- Needs ≥10 `normal` images in a dataset; optional ≥4 `anomalous`
- PatchCore latency per the README at 256×256: GPU f32 31 ms, CPU 77 ms, NPU 158 ms (but the event reference sheet says PatchCore does not compile on the station NPU). **Use `device: GPU`, `inference_precision: f32`** (f16 zeroes the anomaly map)
- Practice dataset in org: `mvtec-bottle` `6ae0082a78f241c0f6d45593`

## Track 3 rules (from docs/ reference sheet + overview deck)
- Judged run: **one 5-minute run, 6 judge items** (good, defective, and at least one defect type we have not seen), then a **5-minute design explanation**
- Destinations: **box** (good), **reject bin** (defective), **hold area** (third option, e.g. unsure)
- Scoring: decision precision/recall from our log 30%, items ending in the right place 30%, latency 10% (item enters view → arm starts responding; correct decisions only), device placement with a measured reason shown through diagnostics 10%, design explanation 20%. Deck also lists recovery: keep working when a judge moves an item mid-run
- **Log one line per item: number, decision, confidence, timestamp. No log = no Decision or Latency points**
- Starting point: `viam:viam-openvino` 0.0.6 (models `mlmodel`, `yolo`, `rf-detr`, `diagnostics`; formerly `erh:openvino`), `viam-labs:anomalib` 0.1.5, reference package `fastener-patchcore` v1, dataset `fastener-anomaly` `6af5edb82deac4224da1b4cd` (169 normal / 83 anomalous, from another station's camera)
- Setup: add `viam-openvino/diagnostics` block named `ov-diag`, `list_devices` should show CPU/GPU/NPU; one diagnostics service per model to measure
- **PatchCore does not compile on the NPU** (explicit NPU never starts; AUTO falls back). armfarm04: 23 ms GPU, 75 ms CPU. GPU needs `inference_precision: f32`
- NPU needs static input shapes: no `resize` in preprocess; set `input_shape`
- Decision/motion is ours: motion service + 3D position from depth, a Track 1 style policy, or a VLM via `viam:openvino-genai` (`vlm` block)
- Helpful modules: `viam-labs:grasp-pose`, `viam:detections-to-segments`, `viam:obstacles-pointcloud`, `viam:apriltag`, `viam:camera-calibration`
- Gotchas: UFactory gripper `set` blocks up to ~10 s per gripper change (fix not merged: viam-modules/viam-ufactory-xarm#145); arm's own moves run at 60 deg/s; E-stop is the red button on the xArm controller
- Training: `viam train submit custom from-registry --org-id=6d6c7293-bc67-43e4-9c8b-31c76aac27d3 --dataset-id=<id> --model-name=<m> --model-version=1 --script-name=hackathons:anomalib-train --version=0.4.0 --container-id=9accd6e7-2992-47c9-9662-2ca5298fa1f1 --args=model=patchcore,image_size=256` (~10 min; new `--model-version` each retrain)
- Discord: https://discord.gg/YnEMfZCRCf · Viam MCP server: https://app.viam.com/mcp

## Remote shell
`viam machines part shell` needs a TTY; scripted via `script -q` + `viam machines part cp` (root shell).
