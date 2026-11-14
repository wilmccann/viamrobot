# Design: Spot it, sort it

The five-minute explanation. Rules we are scored on: detection precision and recall from our own log (30%), items in
the right place (30%), latency from an item entering view to the arm starting its response (10%), device placement
with a measured reason shown through diagnostics (10%), this explanation (20%). The judges include a defect type we
have not seen.

Visual walkthrough to present alongside this doc: [design explainer](https://claude.ai/artifact/A2VmX8REEjd1baSrMwH3R3#bc50ee5a-07ae.m42mf6dkzaw.48).

## What happens to one part

```
inspect pose ─▶ wrist cam frame (RealSense, colour + depth)
   │
   ▼  vision-9  (sortit:vision:parts9, our module on the NUC)
   ├─ 1. Is it a part?      vision-7  YOLOv8n, one class "part"              NPU   2.6 ms
   │       no  → ignored. Boxes in the bottom band (rig fixture, gripper fingers) are dropped.
   ├─ 2. Is it ours?        vision-8  YOLOv8n, normal / anomaly / other      NPU   2.4 ms
   │       "other" (a block, not a fastener) → ignored, never picked
   └─ 3. Is it defective?   vision-8's class for the matching box (decider "vision8")
           anomaly → anomalous · normal → normal · no matching box → unsure
   │
   ▼  viamrobot-demo
   log line (item, decision, confidence, timestamp)  ── written BEFORE the arm moves
   pick: depth at the box centre → 3D point → motion service, jaws turned across long parts
   drop: normal → good-pose · anomalous → bad-pose · unsure → unsure-pose (the hold area)
   back to inspect pose, next part
```

The decision tree is the one we set out to build: **if it's a part, check it for defects; if it isn't, leave it be.**

## Models and what they learned from

| Service | Model | Trained on | Measured |
|-|-|-|-|
| `vision-7` | YOLOv8n "part" (`yolov8n-part-v2`) | team's part-box dataset | live: 0.79–0.94 confidence on every part in view |
| `vision-8` | YOLOv8n, 3 classes | `sortit-armfarm01-part-boxes-v2-anomaly` (466 normal / 140 anomaly / 8 other boxes) | scenes it never saw: 29/29 parts found, 29/29 classified right |
| `mlmodel-9` | crop-PatchCore (Anomalib), 256 px crops | the 466 normal crops only (it learns "good") | held-out AUROC 0.70–1.00 depending on the split |
| `ov` / `vision-1` | full-frame PatchCore `sortit-patchcore` v1 | 61 good / 12 defective frames from our camera | AUROC 0.955 on photos taken after training |

**How the labels were made.** The 376-frame part-box dataset had every box labelled "part". A colour rule relabelled
each box: a large blob of strongly-saturated blue = **anomaly** (the defect marking), yellow-green blocks = **other**,
everything else = **normal**. Thresholds are per camera (the two stations render zinc differently), and every boundary
case was checked by eye. The relabelled copy was published as a new dataset, so the original stayed untouched.

## Device placement

Measured with the diagnostics service `ov-diag` (`benchmark`, 100 iterations, 2026-11-14).

| Model | Device | p50 | p90 | Why this device |
|-|-|-|-|-|
| `mlmodel-7` part detector | NPU | 2.6 ms | 3.3 ms | Small static-shape CNN suits the NPU and leaves the GPU free |
| `mlmodel-8` normal/anomaly/other | NPU | 2.4 ms | 3.2 ms | Same; both YOLOs together cost ~5 ms |
| `mlmodel-9` crop-PatchCore | GPU f32 | 21.0 ms | 21.7 ms | PatchCore does not compile on the NPU; f16 zeroes its anomaly map |
| `ov` full-frame PatchCore | GPU f32 | 18.3 ms | 18.9 ms | Same; CPU measured 75 ms on armfarm04 (reference sheet) |
| Camera handling, crops, logging, arm control | CPU | — | — | Glue work; no model |

Reproduce on the `ov-diag` test card: `{"command": "benchmark", "mlmodel_name": "mlmodel-7"}` (and `-8`, `-9`, `ov`);
`{"command": "list_devices"}` shows CPU, GPU (Arc B390) and NPU.

## Why each choice

| Choice | Why | Evidence |
|-|-|-|
| Separate "is it a part" from "is it defective" | A detector that also judges defects ties two errors together; splitting them lets the best part finder (`vision-7`) and the best defect judge (`vision-8`) each do one job | `vision-7` beat `vision-6` on confidence; `vision-8` matched `vision-6` on finding parts while also classifying |
| A third class, "other" | Non-parts (blocks) should not be sorted at all; labelling them "normal" taught the model blocks are good parts | Blocks now come out `other` at 0.95 |
| Unsure goes to hold | A part we can't vouch for never goes in the good box, and the hold area is one of the three destinations the track names | A part with no `vision-8` match is held, never boxed; `--hold-pose bad-pose` rejects it instead |
| One log line per physical part | The log is scored per item; a second line for the same part would count as a second decision | A retry, or a part a judge moved (one part gone, one appeared), keeps its first line and goes where that line says |
| Log before motion | No log, no decision or latency points; writing first means a failed pick still has its decision recorded | `demo.py`: `_write_log` runs before the pick |
| Ignore the bottom band | The black rig fixture sits in every frame and drew false boxes | All 5 held-out false positives were on the fixture, at 0.29–0.39 |
| Burst-aware evaluation | Frames came in bursts of the same scene, so random splits leaked | Half the original val set had a twin in train; we re-split by scene for honest numbers |

## Unseen defects — where we stand

`vision-8` is supervised: it knows the blue marking well and will likely miss a defect that looks nothing like it.
PatchCore is the model that flags *anything* unlike a good part, and it is deployed as `mlmodel-9`, but we turned it
off as the decider after it called an unfamiliar **good** zinc hex bolt anomalous (it had too few examples of that part
type). Switching back is one config attribute on `vision-9` (`"decider": "patchcore"`, thresholds `box_lo 58` /
`reject_hi 68` in raw PatchCore score). With more good examples of every part type, PatchCore as a second opinion is
the fix for unseen defects; see [STORYLINE.md](STORYLINE.md) for how we got here.

## What failed and what we learned

- A model trained on another station's camera scored 1.00 on everything here: anomaly models learn the whole scene.
- Evening light tripled colour saturation and collapsed the gap between good and bad parts; the fix is physical
  (fixed lamp, taped cardboard).
- Whole-frame PatchCore judged lighting and cardboard as much as parts; cropping to the part fixed the scene problem
  but exposed a data problem (too few good examples per part type).
- The crop model's boolean outputs (`pred_label`, `pred_mask`) broke the Viam SDK; re-exported with numeric outputs only.
- Our first validation scored 100%, because the frames leaked; the honest number came from holding out whole scenes.

## Known limits

- The judge log is appended to: give each run its own `--log` file.
- The log has no separate entered-view / arm-start timestamps; its timestamp is taken just before the arm moves.
- `place()` opens the gripper without re-checking it still holds the part.
- A moved part keeps its line only when it's unambiguous: if a new part appears at the same moment, both get new lines.
