# viamrobot — Track 3 "Spot it, sort it" on armfarm01

A wrist camera looks at parts on the table. Each part gets a verdict, which is written to a judge log **before** the
arm moves, and then the arm drops the part in the **good** bin or the **reject** bin.

```
cam ─▶ vision-9 (sortit:vision:parts9)
         ├─ vision-7  YOLOv8n "part" detector          NPU  ── is it a part?    (no → leave it)
         ├─ vision-8  YOLOv8n normal/anomaly/other     NPU  ── is it defective? ("other" → leave it)
         └─ mlmodel-9 crop-PatchCore                   GPU  ── loaded; decider switch for unseen defects
      ─▶ viamrobot-demo: log line → pick with depth + motion service → good-pose (normal) / bad-pose (anomalous)
```

| Doc | What |
|-|-|
| [docs/DESIGN.md](docs/DESIGN.md) | The five-minute design explanation: pipeline, models, device placement with measurements, trade-offs |
| [docs/JUDGING.md](docs/JUDGING.md) | Judging-day runbook: pre-flight, diagnostics, the run, the judge log, rollback |
| [docs/STORYLINE.md](docs/STORYLINE.md) | How we got here: every iteration, what failed, what each change fixed |
| [armfarm01/NOTES.md](armfarm01/NOTES.md) | Hardware probe and the Track 3 rules as we read them |
| [models/README.md](models/README.md) | Model packages pulled from the org, with checksums |

The vision side (the `sortit` Viam module that serves `vision-9`, the dataset tools and the arm console) lives in the
companion repo **[namitkapoor/spot-it-sort-it](https://github.com/namitkapoor/spot-it-sort-it)** (`viam_module/`,
`tools/`). This repo is the robot side: the demo loop, the pick and the tests.

## Run

```bash
uv sync
uv run pytest tests/unit -q                  # offline, no robot (158 pass, 4 skipped on 2026-11-14)

# The judged run. MOVES THE ARM: stand at the E-stop.
uv run viamrobot-demo --max-parts 6 --log runs/judge-$(date +%H%M%S).csv
```

A part vision-9 is unsure about is treated as anomalous: it goes to the reject bin and is logged `reject`. A new
`--log` file name per run keeps earlier runs out of the judged log, because the log is appended to.

Connection settings come from `.env` (gitignored); see [docs/python project setup.md](docs/python%20project%20setup.md).

## The judge log

One CSV line per decided part, written before the arm starts moving (`src/viamrobot/demo.py`, `decision.log_line`).
No header row; the columns are `item,decision,confidence,timestamp` (UTC). Illustrative lines:

```
1,reject,0.972,2026-11-14T19:31:02.114502+00:00
2,box,0.949,2026-11-14T19:31:21.870113+00:00
```

`decision` is `box` (good) or `reject` (anomalous or unsure). Details and how to score it are in
[docs/JUDGING.md](docs/JUDGING.md#the-judge-log).
