# Judging-day runbook

One 5-minute run with 6 judge items (good, defective, and at least one defect type we have not seen), then the
5-minute design explanation ([DESIGN.md](DESIGN.md)).

## Pre-flight (15 min before)

1. **Nobody else drives the arm.** Teleop (`teleop`, `teleop-virtual`), `sorter`, `sorter-9` and VLA controllers
   all command the same arm. Say it in the team chat; check `sorter` / `sorter-9` `{"command": "status"}` says
   `"running": false`.
2. **Scene matches training:** lamp on, cardboard taped flat, gripper open, nothing on the table but judge items.
3. **Pre-warm the models** (first call compiles them): on the CONTROL tab open `vision-9`, or run
   `viam machines part run --part=<part> --data '{"name":"vision-9","command":{"command":"objects"}}' viam.service.vision.v1.VisionService.DoCommand`.
4. **Diagnostics for the device-placement score**, on the `ov-diag` test card:
   - `{"command": "list_devices"}` → CPU, GPU, NPU
   - `{"command": "benchmark", "mlmodel_name": "mlmodel-7"}` → NPU, ~2.6 ms p50
   - `{"command": "benchmark", "mlmodel_name": "mlmodel-8"}` → NPU, ~2.4 ms
   - `{"command": "benchmark", "mlmodel_name": "mlmodel-9"}` → GPU, ~21 ms (and `ov` → GPU, ~18 ms)
5. **Offline tests:** `uv run pytest tests/unit -q`.
6. **Someone at the E-stop** (red button on the xArm controller) for the whole run.

## The run

```bash
uv run viamrobot-demo --max-parts 6 --log runs/judge-$(date +%H%M%S).csv
```

- Parts vision-9 is unsure about are treated as anomalous: reject bin, logged `reject`.
- A fresh `--log` file per run: the log is opened in append mode.
- Ctrl-C stops the loop; recovery (stop, open gripper, lift, inspect pose) runs on any failure.

Optional monitor from a laptop, in the companion repo: `uv run python tools/arm_console.py` →
http://127.0.0.1:8900. It shows what `vision-9` sees and which bin each part is heading to, plus a STOP button and
Home / Inspect poses.

## The judge log

Written by `Demo._write_log` (`src/viamrobot/demo.py`) **before the arm moves**, one line per decided part, no header:

| Column | Meaning |
|-|-|
| `item` | 1, 2, 3 … in decision order |
| `decision` | `box` (good) or `reject` (anomalous, or unsure) |
| `confidence` | How far vision-9's score sits toward its decision (1.0 = certain) |
| `timestamp` | UTC ISO-8601, taken just before the arm starts its response |

Scoring from the log: match each line to the judge's truth for that item, then
precision = correct `reject` / all `reject`, recall = correct `reject` / all truly defective items.
Latency = log `timestamp` minus the moment the item entered view (the judge's clock).

Hand in the run's file (`runs/judge-<time>.csv`). If the run restarts, start a new file and say which one counts.

## If something goes wrong

| Symptom | Do |
|-|-|
| Arm behaves oddly | E-stop, then Ctrl-C the demo |
| vision-9 sees nothing | Check the arm is at `inspect-pose` and the camera isn't covered; re-query `objects` |
| Machine shows offline | Wait 30 s and re-run the command; too many camera previews can make it look offline, so close extra CONTROL tabs |
| A config change broke something | Every change made from the companion repo was backed up first: `uv run python tools/viam_config.py restore runs/config-backups/<file>.json` |

## Where things live on the machine (armfarm01, part `f08ae88f-5fa8-4775-a3bf-0f4fda96940f`)

| Service | Model / files |
|-|-|
| `vision-9` | `sortit:vision:parts9`, local module `/root/viam_module` (companion repo `viam_module/`) |
| `mlmodel-7` / `vision-7` | `/root/yolov8n-part-v2/yolov8n-part.onnx`, NPU |
| `mlmodel-8` / `vision-8` | `/root/yolov8n-anomaly-v2/yolov8n-anomaly.onnx` (3 classes), NPU |
| `mlmodel-9` | `/root/sortit-patchcore-crops-v2/model.xml`, GPU f32 |
| `ov` / `vision-1` | `sortit-patchcore` v1 package, GPU f32 |
| Poses | `inspect-pose`, `good-pose`, `bad-pose`, `home-pose` (switch components) |
