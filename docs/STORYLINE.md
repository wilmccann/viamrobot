# Storyline: how we arrived at the current system

Two repos built this together: **[namitkapoor/spot-it-sort-it](https://github.com/namitkapoor/spot-it-sort-it)**
(vision, datasets, the `sortit` Viam module, the teammate `robot-sort` module) and **this repo** (the demo loop, the
pick, tests). Times are local, 2026-11-13 to 2026-11-14. Each iteration says what we tried, what broke, and what
we changed because of it.

## Day 1 — 2026-11-13

**1. Scaffold** (spot-it-sort-it `c9510aa`, 11:00). Plan: Anomalib for "is it defective", a VLM through OpenVINO GenAI
for the decision, the arm last. Synthetic smoke test, no hardware.

**2. The borrowed model fails.** A PatchCore trained on another station's camera scored 1.00 on every frame here.
*Lesson:* an anomaly model learns the whole scene (camera pose, light, background, gripper), not the part. Normal has
to come from our own camera.

**3. Our own data, and the first traps.** v0 trained on 16 good frames only was degenerate (AUROC 0.5); mixed
backgrounds (white table vs cardboard) and gripper states (open vs closed) taught the wrong cue. We standardised on
cardboard + open gripper and built a capture page so the team could tag shots from any laptop. `sortit-patchcore` v1
(61 good / 12 defective) and v2 (83 / 46).

**4. Per-object sorting** (spot-it-sort-it `af05ecc`, 20:38). A frame-level "anomalous / normal" can't say *which* part
to reject, and it flickered across 0.5 as evening light changed (0.38 → 0.60 on the same scene). New `objects`
service: find parts by colour difference from the cardboard, score each part as heat on its pixels minus heat on the
empty surface, three outcomes with a **hold** band, and a per-item log written on the NUC before the arm moves.
Also: a supervised YOLO classifier learned a shortcut ("small dark thing = defect", 2/7 on a blind test vs
PatchCore's 6/7), so it stayed out of the decision.

## Day 2 — 2026-11-14 morning

**5. Handoff and a robot-side repo.** Live-machine state, rollback and a judging checklist written down
(spot-it-sort-it `4275829`, 10:29). In parallel this repo started: a pipeline with mock stages and tests (`e12f4d3`,
08:15), a simulation (`43ebd85`), a local PatchCore and a capture tool (`afa218c`).

**6. A trained part detector replaces colour subtraction.** Teammates trained `yolov8n-part` (`vision-6`) and the
`robot-sort` module (spot-it-sort-it `32551e5`, 11:35): YOLO boxes scored by PatchCore's heatmap, plus a picker that
turns a box + depth into a grasp. This repo added `MachineDetector` for `vision-6` boxes (`105f294`, 11:34).
Colour subtraction had merged touching parts; a detector doesn't.

## Day 2 — midday: one model for "where" and "what"

**7. Relabelling for defects** (~12:00). The part-box dataset (376 frames, 614 boxes) only said "part". A colour rule
relabelled each box: strongly-saturated blue blob = **anomaly** (the defect marking). Per-camera thresholds were
needed because the armfarm01 camera makes bare zinc nuts look cyan; every borderline box was checked by eye.
Published as a new dataset (`sortit-armfarm01-part-boxes-v2-anomaly`) by re-uploading copies, so the original
dataset a teammate trains on was never touched.

**8. `vision-8`, 2 classes** (~12:15). YOLOv8n trained on normal / anomaly. Validation said 100%, but half the val
frames had a near-twin in train (burst captures of the same scene). Re-split by scene: still 29/29 found and
classified, with 5 false boxes, all on the black rig fixture at the bottom of every frame. *Changes:* ignore the
bottom band; an honest-evaluation habit for the rest of the day.

**9. Not everything on the table is a part** (~12:45). Yellow and green blocks were labelled "normal", which taught the
model that blocks are good parts, and the sorter would have spent picks on them. Added a third class, **other**
(skip it). `vision-8` re-trained and swapped in on the NPU; blocks now come out `other` at 0.95.

**10. `vision-7`.** Teammates' `yolov8n-part-v2` found parts with 0.79–0.94 confidence, clearly better than
`vision-6`. Decision: *if it's a part (`vision-7`), check it for defects; if not, leave it be.*

## Day 2 — afternoon: `vision-9`

**11. Crop-PatchCore** (~13:00). Whole-frame PatchCore judged lighting and cardboard as much as the part, so we trained
PatchCore on 256 px crops around each good part (466 crops), keeping the anomaly crops for calibration. Held-out
AUROC ranged 0.70–1.00 depending on which scenes were held out, so the thresholds got a wide hold band (58 / 68).

**12. `vision-9` deployed** (~13:20) as one service: `vision-7` finds parts → `vision-8` vetoes "other" → crop-PatchCore
scores each part. Two problems surfaced in the first live minutes:
- The crop model's boolean outputs (`pred_label`, `pred_mask`) broke the Viam SDK's tensor decoding. Re-exported with
  only `pred_score` and `anomaly_map`.
- Every defective part landed in **hold** (scores ~62, inside the 58–68 band), and a good zinc hex bolt scored
  **anomalous**: the model had seen too few good examples of that part type.

**13. `vision-8` decides** (~13:26). `vision-8` was the most reliable judge on everything live, so it became
`vision-9`'s decider: its normal/anomaly class for the matching box decides, no match means hold, and crop-PatchCore
stays loaded behind a config switch (`decider`). Saved frames: 6/6 correct; the zinc bolt went from anomalous to
normal (0.06).

**14. The robot side converges on `vision-9`.** This repo's demo loop (`8638c3b`, 14:20) takes every verdict from
`vision-9`, logs before motion, picks with depth + the motion service, turns the jaws across long parts, and drops at
good / bad / unsure poses. A laptop arm console (companion repo `tools/arm_console.py`) shows which bin each part is
heading to, with STOP and Home / Inspect buttons; `sorter-9` mirrors the teammates' sorter on `vision-9`.

**15. Device placement measured** (`ov-diag benchmark`): YOLOs on the NPU at 2.4–2.6 ms, PatchCore on the GPU at f32
18–21 ms. PatchCore can't compile on the NPU, and f16 blanks its heatmap.

## Where that leaves us

| Asked for | Where it stands |
|-|-|
| Find parts, ignore non-parts | `vision-7` + `vision-8` "other" + bottom-band filter |
| Defective or not | `vision-8` (supervised on the blue marking) |
| Unseen defect type | PatchCore deployed but switched off after false alarms on unfamiliar good parts; more good examples per part type would let it vote again |
| Log before motion | Yes, one line per part |
| Device placement with a reason | Measured, see [DESIGN.md](DESIGN.md#device-placement) |

## Lessons we'd carry forward

1. Anomaly models learn the scene: capture "normal" on the exact station, pose and light you'll be judged on.
2. Burst-captured data leaks across random splits; split by scene before you believe a number.
3. Give "not a part" its own class, or the model learns that distractors are good parts.
4. Write the decision down before the arm moves.
5. Measure every model on every device you might use; the diagnostics service makes it one command.
