# Retrospective: the Track 3 demo

What the robot-side demo (`viamrobot-demo`) does well, what it doesn't yet, the design choices behind it, and what to
do next to score better on the Track 3 rubric. The evidence is the full rehearsal on armfarm01 on 2026-11-14
(`runs/rehearsal-172511.csv`, 17:28–17:30 local) plus the earlier demo runs described in
[STORYLINE.md](STORYLINE.md) and PR #5.

The rubric, from the reference sheet and the kickoff deck: decision precision and recall from our own log (30%), items
ending up where we decided (30%), latency from an item entering view to the arm starting its response (10%), device
placement with a measured reason shown through diagnostics (10%), the five-minute design explanation (20%). The deck
adds recovery: keep working when a judge moves an item mid-run. The judges include a defect type we have not seen.

## The latest run in numbers

| | Result |
|-|-|
| Parts sorted | 6 of 6: 2 `reject`, 4 `box`; one log line per part, numbered 1–6 |
| Time per part | 8.4–10.1 s from the inspect pose to the drop (median 9.2 s), plus ~1.5 s back to inspect |
| Log timestamp → first arm move | 0.39–0.50 s on every part |
| Missed picks | 1 of 7 attempts (part 4, first try); retried and boxed under its original log line |
| Whole run once connected | 92 s, including 14 s with nothing on the table and ~14 s lost to the miss |
| Connecting | 3 min 16 s: three attempts timed out before the fourth connected |

The run was not scored against the judges' truth: we know where each part went, not whether each verdict was right.

## What worked well

**Logging before the arm moves.** Every decision was on disk ~0.45 s before the first arm move, which is the
latency the deck measures (timestamp to first arm motion). A failed pick can't lose a decision.

**One log line per physical part, even through failures.** The first pick of part 4 stopped with the fingers ~29 mm
apart on an ~11 mm part, pushing the part ~300 px across the image. The demo saw one logged part vanish and one part
appear, printed `part 4 moved ... keeping its log line`, sorted part 5, then went back and boxed part 4 under line 4.
No duplicate line, no lost decision. This is the deck's recovery criterion happening on its own.

**The grip check tells a grip from a stall.** Comparing where the fingers stop with the part's measured width caught
the part 4 miss immediately (`fingers stopped at 286, ~29 mm apart, part ~11 mm wide`). In the first demo run, before
this check, every close stalled on the table and was still logged as sorted.

**The second look made picks reliable.** Moving the camera straight above the part and locating it again shifted the
grasp point by up to 31 mm (median ~16 mm, mostly in −x). Parts are 10–20 mm wide, so most of those picks would have
missed without it. It costs ~0.6–1.0 s per part.

**The wrist turn learns its offset.** The first turn of the run landed 10.5° off (the inspect pose isn't quite
straight down); every turn after that landed within 0.2° on the first try.

**Closing is fast.** Watching the fingers instead of waiting for the gripper's `set` call to finish brought every close
down to 1.0–1.2 s (2.4 s on the miss). It used to take up to 10.5 s, and once ran past the 20 s part timeout.

**Recovery and retries hold up.** Recover (stop, open, lift, inspect pose), retry with back-off, a failed part tried
again only after the others, and reconnecting if the link drops. In the rehearsal it turned a miss into a correct
sort ~23 s later.

**Plenty of time.** Six parts take about a minute of sorting at ~10 s each, well inside the five-minute run, and no
part came near the 20 s part timeout.

**Device placement is measured, not asserted.** The `ov-diag` benchmarks (YOLOs on the NPU at 2.4–2.6 ms, PatchCore on
the GPU at f32, 18–21 ms) and the reasons behind them are in [DESIGN.md](DESIGN.md#device-placement).

**Offline tests catch the loop's logic.** 180 unit tests run in ~7 s against a fake work cell. They found a real bug
(a placed part not counting toward `--max-parts`, by hanging) before it reached the arm.

## What needs improvement

In order of how much rubric weight each one puts at risk.

| # | Gap | Rubric at risk | Evidence |
|-|-|-|-|
| 1 | **Unseen defects will likely be boxed.** `vision-8` decides, and it learned one defect: the blue marking. PatchCore, which flags anything unlike a good part, is loaded but switched off after it called an unfamiliar good bolt anomalous. | Decision 30%, Execution 30% | DESIGN.md "Unseen defects"; the rehearsal had no novel defect |
| 2 | **The first look is off by ~16 mm**, mostly in −x. The camera's position in the machine config looks ~2° off, so the second look corrects for it on every part. A part near the image edge is worse: part 4 started 30 mm off and was the one miss. | Execution 30% | Second-look shifts of −11.7, −31.0, +4.4, +18.5, −16.7, −18.1, −16.3 mm in x |
| 3 | **Connecting can take minutes.** Each startup attempt makes three 20 s dial tries; three attempts failed before the fourth worked. The machine was online and the CLI connected in 2.7 s, so this was a transient WebRTC problem, not our code. | All of it, if the run starts late | 17:25:12 launch → 17:28:28 connected |
| 4 | **We don't know when an item entered view.** The log has one timestamp (the decision). With nothing to sort, the loop looks every ~1.3 s, so a new item can wait that long before it's seen. | Latency 10% | 10 idle looks between parts 4 and 6, 1.3–1.5 s apart |
| 5 | **The hold area may score as a wrong decision.** Unsure parts go to `unsure-pose`, logged `hold`. For a defective part that's not a `reject`, so it counts against recall. | Decision 30% | JUDGING.md |
| 6 | **Drops aren't verified.** `place()` opens the gripper at the bin without checking the part is still held, so a part lost on the way is still reported as sorted. | Execution 30% | DESIGN.md "Known limits" |
| 7 | **Per-part overhead that doesn't need to be there.** Opening an already-open gripper before every pick (~0.33 s, part of the 0.45 s latency); waiting ~1.3 s for the gripper to open fully at the bin; a second-look move on every part. | Latency 10%, time | `open: set 840 returned after 1.3 s` at every drop |
| 8 | **Station-specific tuning lives in code.** `cell.py` has 25 hand-tuned constants (fingertip reach, table clearance, theta-to-yaw sign, second-look limits) measured on armfarm01. Moving to another station means re-measuring them. | Risk if the station changes | `cell.py` header |
| 9 | **Unused code on the side.** `pipeline.py`, `sim.py`, `stages/` and `decision.decide()` are the original design (anomaly score → VLM second look), never wired into the demo. | Design explanation clarity | `viamrobot-demo` imports none of them except `log_line` |

## Design choices and trade-offs

| Choice | What it bought | What it cost |
|-|-|-|
| `vision-8` (supervised YOLO) decides; PatchCore stays loaded but off | Reliable on the known defect, 2.4 ms on the NPU, no false alarms on good parts | Probably blind to the unseen defect |
| Separate "is it a part?" (`vision-7`) from "is it defective?" (`vision-8`), plus an "other" class | Blocks and the rig fixture are never picked; each model does one job | Boxes from two models must be matched; no match becomes `hold` |
| Unsure parts go to the hold area (`--hold-pose bad-pose` rejects them) | Uses the third destination the track names; never boxes a part we can't vouch for | A held defective part may count against recall |
| Log before any motion, one line per physical part | Decision points survive failed picks; latency is ~0.45 s | Lines can't be corrected later, so part identity has to be tracked carefully |
| Our own pick: depth → 3D point → motion service, jaws turned across long parts, second look | Full control; each failure seen on the arm was fixed within the hour | Many station-tuned constants; the second look works around a calibration error |
| The grip check uses the part's measured width | Tells a grip from a stall on the table | Depends on the outline being right (part 6's first estimate was 40 mm, the second 14 mm) |
| One part at a time, back to `inspect-pose` each time | Simple; every look matches the view the models were trained on | ~1.5 s back to inspect per part; can't plan the next pick while carrying |
| Retry a failed part only after every other part, up to 3 times | One stubborn part can't stall the run | A part that keeps failing is left on the table |
| Logic tested offline against a fake cell; motion tuned live | Fast, reliable tests of the loop | The pick itself is only tested by running the arm |

## Learnings

1. **Watch the hardware, don't trust the call.** The gripper's `set` returns when it gives up, not when the jaw is
   on the part, and depth reads the table 1–2 mm low. Measuring what the fingers did (where they stopped, compared with
   the part's width) is what made picks trustworthy.
2. **Look again, from closer.** A first look from the inspect pose is good enough to decide, not to grasp. A second
   frame with the part near the image centre fixed most misses for under a second per part.
3. **Make identity explicit.** "Which physical part is this?" came up in retries, in failures after a drop, and when a
   part moved. Deciding it in one place (box overlap, plus "one vanished, one appeared") kept the judge log right.
4. **Check what you depend on before the arm moves.** The log folder, the connection and the camera are all checked or
   retried at startup now. A missing folder used to fail only at the first part, inside the loop.
5. **Rehearse the whole run, including a judge moving a part.** The one full rehearsal surfaced the slow connect, the
   edge-of-view miss and the moved-part case in two minutes. Weeks of single-part testing would not have.
6. **Anomaly models learn the scene** (from STORYLINE.md): capture "normal" on the station, pose and light you'll be
   judged on, and split train and validation by scene before believing a number.

## Recommended next steps

In order of expected points per hour of work.

**1. Catch unseen defects: PatchCore as a second opinion (Decision 30%, Execution 30%).**
When `vision-8` says normal but crop-PatchCore scores high, send the part to hold (or reject). That keeps `vision-8`
in charge of the defect it knows, and PatchCore's false alarms cost a hold instead of a wrong box. To make PatchCore
trustworthy, capture 30+ good crops of every part type on armfarm01 under the judging lamp. Test it with a defect
nobody trained on (a scratch, a bent or broken part, a different-coloured marking) and check how many go to hold or
reject rather than the box.

**2. Calibrate the wrist camera to the arm (Execution 30%, time).**
Run `viam:camera-calibration` with a ChArUco board and put the measured camera pose in the frame config. That removes
the ~16 mm first-look error at the source, makes edge-of-view parts as safe as central ones, and may let the second
look become a fallback for small parts only (~1 s saved per part).

**3. Make startup robust (protects every criterion).**
Start the demo a few minutes before the run, or keep a warm connection: connect once, check the camera and `vision-9`,
then wait for Enter before the first part. Shorten each dial attempt and retry more often, and print a clear "connected
in N s" line.

**4. Cut latency (Latency 10%).**
Look every ~0.3 s instead of every ~1.3 s while the table is empty, and skip the inspect-pose move when the arm is
already there. Skip opening an already-open gripper before a pick (~0.3 s off the 0.45 s). Add a "first seen"
timestamp to the console output so we can show the judges the full entered-view → response time.

**5. Settle the hold rule with data (Decision 30%).**
Ask the judges whether a defective item in the hold area counts as correct. If not, run with `--hold-pose bad-pose`.
Either way, write the choice in the design explanation.

**6. Verify drops (Execution 30%).**
Check the fingers are still on the part just before opening at the bin. If not, log it and look for the part on the
table, instead of reporting it sorted.

**7. Small speedups.** Don't wait for the gripper to open fully at the bin; plan the move back to inspect while the
gripper opens.

**8. Tidy for the design explanation (20%).**
Move the unused `pipeline.py` / `sim.py` / `stages/` design out of the package or label it as the first design. Add
this rehearsal's numbers (≈9 s per part, 0.45 s to first motion, 1 recovered miss) to [DESIGN.md](DESIGN.md).
Move the station-tuned constants in `cell.py` into one documented config block.

**Before any judged run:** do one full rehearsal with 6 judge-like items, including a novel defect and a part moved
mid-run. Score the log against the truth by hand (precision, recall, items in the right place) so we know the numbers
before the judges do.
