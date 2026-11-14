"""Demo dashboard: rubric, live run, docs, diagnostics and arm controls, served from THIS laptop.

    ./scripts/run-demo.sh --max-parts 6                    # starts this, then the demo (recommended)
    VIAMROBOT_DIR=$PWD PUBLIC_PORT=8901 python3 tools/demo_dashboard.py
    -> http://127.0.0.1:8900  full: Rubric, Run, Docs, Diagnostics, Controls (STOP / Home / Inspect / sorter)
    -> http://127.0.0.1:8901  read-only: Rubric, Run, Docs, Diagnostics; refuses every robot call (safe to tunnel)

Run tracking reads <VIAMROBOT_DIR>/runs/judge-*.csv (one file per viamrobot-demo run). Robot calls go through the
Viam CLI login (`viam machines part run`), no API key here, and are rate-limited. Listens on 127.0.0.1 only.
Controls: one motion at a time; STOP is never blocked. Sorter motion needs that sorter's dry_run to be false.
Stdlib only; also lives in namitkapoor/spot-it-sort-it as tools/arm_console.py.
"""
from __future__ import annotations

import base64
import csv
import glob
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PART = os.environ.get("VIAM_PART", "f08ae88f-5fa8-4775-a3bf-0f4fda96940f")
SORTER = os.environ.get("SORTER", "sorter-9")
VISION = os.environ.get("VISION", "vision-9")
ARM = os.environ.get("ARM", "arm")
PORT = int(os.environ.get("PORT", "8900"))
POSES = {"home": "home-pose", "inspect": "inspect-pose"}
SNAP_MIN_AGE = 2.0  # seconds; never poll cameras faster (too many previews once made the NUC look offline)

motion = threading.Lock()
pool = ThreadPoolExecutor(4)
events: list[str] = []
_snap: dict[str, tuple[float, bytes]] = {}
_snap_lock = threading.Lock()


def log(msg: str) -> None:
    events.append(f"{time.strftime('%H:%M:%S')}  {msg}")
    del events[:-40]
    print(events[-1], flush=True)


def viam_run(method: str, data: dict, timeout: int = 120) -> dict:
    out = subprocess.run(["viam", "machines", "part", "run", f"--part={PART}", f"--method={method}", "-d", json.dumps(data)],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode:
        raise RuntimeError((out.stderr or out.stdout).strip()[-400:])
    return json.loads(out.stdout)


def sorter(cmd: dict, timeout: int = 120) -> dict:
    return viam_run("viam.service.generic.v1.GenericService.DoCommand", {"name": SORTER, "command": cmd}, timeout).get("result", {})


def stop_all() -> dict:
    """Arm stop and sorter stop in parallel; report both."""
    futs = {"arm": pool.submit(viam_run, "viam.component.arm.v1.ArmService.Stop", {"name": ARM}, 30),
            "sorter": pool.submit(sorter, {"command": "stop"}, 30)}
    res = {}
    for k, f in futs.items():
        try:
            f.result()
            res[k] = "stopped"
        except Exception as e:  # noqa: BLE001
            res[k] = f"error: {e}"
    log(f"STOP -> {res}")
    return res


def run_motion(name: str, fn) -> dict:
    if not motion.acquire(blocking=False):
        raise RuntimeError("another motion is in progress — wait, or press STOP")
    try:
        st = sorter({"command": "status"}, 30)
        if st.get("running"):
            raise RuntimeError("the sorter run is still going — press STOP first")
        log(f"{name} ...")
        out = fn()
        log(f"{name} done: {json.dumps(out)[:300]}")
        return out
    finally:
        motion.release()


def action(name: str, max_items: int) -> dict:
    if name == "stop":
        return stop_all()
    if name in POSES:
        return run_motion(f"go {name}", lambda: viam_run("viam.component.switch.v1.SwitchService.SetPosition",
                                                         {"name": POSES[name], "position": 2}))
    if name == "plan":
        return run_motion("plan", lambda: sorter({"command": "plan"}))
    if name == "sort_one":
        return run_motion("sort one", lambda: sorter({"command": "sort_one"}, 300))
    if name == "start":
        return run_motion(f"start x{max_items}", lambda: sorter({"command": "start", "max_items": max_items}))
    raise ValueError(f"unknown action {name}")


def snapshot(cam: str) -> bytes:
    with _snap_lock:
        hit = _snap.get(cam)
        if hit and time.time() - hit[0] < SNAP_MIN_AGE:
            return hit[1]
        r = viam_run("viam.component.camera.v1.CameraService.GetImages", {"name": cam}, 30)
        img = next(base64.b64decode(i["image"]) for i in r.get("images", []) if "jpeg" in (i.get("mimeType") or ""))
        _snap[cam] = (time.time(), img)
        return img


# ---------------------------------------------------------------- demo dashboard (Run / Docs / Diagnostics tabs)
VR = os.path.expanduser(os.environ.get("VIAMROBOT_DIR", "~/Documents/personal/code/viamrobot"))
DOCS = ("README.md", "docs/DESIGN.md", "docs/JUDGING.md", "docs/STORYLINE.md", "docs/VLA-EXPERIMENTS.md", "armfarm01/NOTES.md")
MODELS = ("mlmodel-7", "mlmodel-8", "mlmodel-9", "ov")
VLOG_MIN_AGE = 10.0  # seconds between vision-9 log reads (each is a robot call)
LIVE_WINDOW = 60.0   # newest CSV touched within this many seconds => run in progress
_vlog: dict = {"at": 0.0, "data": None, "error": None, "busy": False}
_vlog_lock = threading.Lock()
_bench: dict[str, dict] = {}
_bench_lock = threading.Lock()


def _iso(ts: str) -> float | None:
    try:
        d = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def run_files() -> list[dict]:
    """All judge CSVs (runs/judge-*.csv and older judge*.csv at the root), newest run first."""
    out = []
    for rel_dir, pat in (("runs", "judge-*.csv"), ("", "judge*.csv")):
        for p in glob.glob(os.path.join(VR, rel_dir, pat)):
            st = os.stat(p)
            name = os.path.basename(p)
            m = re.fullmatch(r"judge-(\d{8}-\d{6})\.csv", name)
            if m:  # runs/judge-<UTC time>.csv
                start = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            else:  # older judge*.csv: first item's timestamp, else the file's mtime
                start = next((i["t"] for i in read_run(p) if i["t"]), None) or st.st_mtime
            out.append({"rel": f"{rel_dir}/{name}" if rel_dir else name, "path": p, "mtime": st.st_mtime, "start": start})
    return sorted(out, key=lambda f: (f["start"], f["mtime"]), reverse=True)


def read_run(path: str) -> list[dict]:
    items = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 4 or not row[0].strip():
                continue
            items.append({"item": row[0].strip(), "decision": row[1].strip(), "confidence": row[2].strip(),
                          "timestamp": row[3].strip(), "t": _iso(row[3])})
    prev = None
    for it in items:
        it["gap"] = round(it["t"] - prev, 1) if it["t"] is not None and prev is not None else None
        prev = it["t"] if it["t"] is not None else prev
    return items


def _refresh_vlog() -> None:
    try:
        r = viam_run("viam.service.vision.v1.VisionService.DoCommand", {"name": VISION, "command": {"command": "log"}}, 30)
        res = r.get("result", {})
        data, err = {"run": res.get("run"), "log_path": res.get("log_path"), "items": res.get("items")}, None
    except Exception as e:  # noqa: BLE001
        data, err = None, str(e)
    with _vlog_lock:
        _vlog.update(at=time.time(), busy=False, error=err, **({"data": data} if data else {}))


def vision_log(refresh: bool) -> dict:
    with _vlog_lock:
        if refresh and not _vlog["busy"] and time.time() - _vlog["at"] >= VLOG_MIN_AGE:
            _vlog["busy"] = True
            pool.submit(_refresh_vlog)
        return {"at": _vlog["at"] or None, "data": _vlog["data"], "error": _vlog["error"]}


def run_state(refresh_vlog: bool) -> dict:
    now = time.time()
    files = run_files()
    hist = []
    for f in files:
        items = read_run(f["path"])
        hist.append({"rel": f["rel"], "start": f["start"], "mtime": f["mtime"], "n": len(items),
                     "box": sum(i["decision"] == "box" for i in items), "reject": sum(i["decision"] == "reject" for i in items)})
    cur = None
    if files:
        f, items = files[0], read_run(files[0]["path"])
        live = now - f["mtime"] < LIVE_WINDOW
        end = now if live else max([i["t"] for i in items if i["t"]] + [f["mtime"]])
        cur = {**hist[0], "live": live, "elapsed": round(end - f["start"], 1), "items": items[::-1]}
    return {"now": now, "dir": VR, "current": cur, "history": hist, "vision_log": vision_log(refresh_vlog)}


def docs_index() -> list[dict]:
    out = []
    for rel in DOCS:
        p = os.path.join(VR, rel)
        out.append({"path": rel, "exists": os.path.isfile(p), "mtime": os.path.getmtime(p) if os.path.isfile(p) else None})
    return out


def benchmark(model: str) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    if not _bench_lock.acquire(blocking=False):
        raise RuntimeError("a benchmark is already running — wait for it")
    try:
        t0 = time.time()
        r = viam_run("viam.service.generic.v1.GenericService.DoCommand",
                     {"name": "ov-diag", "command": {"command": "benchmark", "mlmodel_name": model}}, 180)
        _bench[model] = {"at": time.time(), "wall_s": round(time.time() - t0, 1), "result": r.get("result", r)}
        log(f"benchmark {model} done")
        return _bench[model]
    finally:
        _bench_lock.release()


ORG = os.environ.get("VIAM_ORG", "6d6c7293-bc67-43e4-9c8b-31c76aac27d3")
LOGS_MIN_AGE = 15.0     # seconds between `viam machines part logs` calls
STATUS_MIN_AGE = 3.0    # sorter status is shared by all viewers (incl. the public port); never faster than this
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LOG_KEEP = ("sortit", "robot-sort", "robotsort", "vision", "motion", "arm", "gripper")
_logs: dict = {"at": 0.0, "lines": [], "error": None, "busy": False}
_logs_lock = threading.Lock()
_status: dict = {"at": 0.0, "data": None}
_status_lock = threading.Lock()


def sorter_status() -> dict:
    with _status_lock:
        if _status["data"] is None or time.time() - _status["at"] >= STATUS_MIN_AGE:
            _status.update(data=sorter({"command": "status"}, 30), at=time.time())
        return _status["data"]


def parse_logs(text: str) -> list[dict]:
    out = []
    for line in text.splitlines()[1:]:  # first line is the "org -> location -> machine" header
        f = line.split("\t")
        if len(f) < 4:
            continue
        ts, level, logger, msg = (x.strip() for x in f[:4])
        lg, lv = logger.lower(), level.lower()
        if lg.startswith(("rdk.networking", "viam-agent")):
            continue
        if not (lv in ("warn", "warning", "error") or (lv == "info" and any(k in lg for k in LOG_KEEP))):
            continue
        if len(f) == 4:
            msg = re.sub(r"\s*\{.*\}\s*$", "", msg)
        out.append({"t": _iso(ts), "level": lv, "logger": IPV4.sub("x.x.x.x", ".".join(logger.split(".")[-2:])),
                    "msg": IPV4.sub("x.x.x.x", msg)[:500]})
    out.sort(key=lambda r: r["t"] or 0, reverse=True)
    return out[:50]


def _refresh_logs() -> None:
    try:
        r = subprocess.run(["viam", "machines", "part", "logs", f"--part={PART}", f"--organization={ORG}", "--count", "200"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
        lines, err = parse_logs(r.stdout), None
    except Exception as e:  # noqa: BLE001
        lines, err = None, IPV4.sub("x.x.x.x", str(e))
    with _logs_lock:
        _logs.update(at=time.time(), busy=False, error=err, **({"lines": lines} if lines is not None else {}))


def machine_logs(refresh: bool) -> dict:
    with _logs_lock:
        if refresh and not _logs["busy"] and time.time() - _logs["at"] >= LOGS_MIN_AGE:
            _logs["busy"] = True
            pool.submit(_refresh_logs)
        return {"at": _logs["at"] or None, "lines": _logs["lines"], "error": _logs["error"]}


def health() -> dict:
    out = {}
    for name, fn in (("vision objects", lambda: viam_run("viam.service.vision.v1.VisionService.DoCommand",
                                                         {"name": VISION, "command": {"command": "objects"}}, 30)),
                     ("sorter status", lambda: sorter({"command": "status"}, 30))):
        t0 = time.time()
        try:
            fn()
            out[name] = {"ok": True, "ms": round((time.time() - t0) * 1000)}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "ms": round((time.time() - t0) * 1000), "error": str(e)[-200:]}
    return {"at": time.time(), "checks": out}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arm Console</title><style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e1db;--go:#1f7a4d;--stop:#c62828;--acc:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a9993;--card:#1e1e1c;--line:#33322f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:0 0 10px;color:var(--mut);font-weight:600}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:10px 14px;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#stop{background:var(--stop);color:#fff;border:0;font-weight:700;font-size:20px;width:100%;padding:18px;margin-bottom:12px}
.go{background:var(--go);color:#fff;border:0}.pill{padding:2px 8px;border-radius:99px;border:1px solid var(--line);font-size:13px}
.warn{color:var(--stop);font-weight:600}img{width:100%;border-radius:8px;background:#000;aspect-ratio:4/3;object-fit:contain}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,monospace;max-height:260px;overflow:auto;margin:0}
.pick{margin:14px 0 4px;padding:14px;border-radius:10px;border:2px solid var(--line);font-size:18px;font-weight:700}
.pick.normal{border-color:var(--go);color:var(--go)}.pick.anomalous{border-color:var(--stop);color:var(--stop)}
.pick.hold{border-color:#b7791f;color:#b7791f}
input{width:56px;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
.hwrap{max-width:1100px;margin:0 auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
nav{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}nav button{padding:6px 12px}nav button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.tab[hidden]{display:none!important}.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px}
.live{background:var(--stop);color:#fff;border-color:var(--stop);font-weight:700}
.banner{padding:12px 14px;border-radius:10px;background:var(--acc);color:#fff;font-weight:700;font-size:18px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:13px}.tw{overflow-x:auto}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}.stat{flex:1;min-width:90px;border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.stat b{display:block;font-size:24px}.stat span{color:var(--mut);font-size:13px}.box{color:var(--go);font-weight:600}.reject{color:var(--stop);font-weight:600}
.docs{display:grid;grid-template-columns:240px 1fr;gap:16px}@media (max-width:800px){.docs{grid-template-columns:1fr}}
.dl button{display:block;width:100%;text-align:left;margin-bottom:6px;padding:8px 10px}.dl button.on{border-color:var(--acc)}
.dl small{display:block;color:var(--mut);font-size:12px}.md{overflow-wrap:anywhere}.md img{max-width:100%}
.md pre{background:var(--bg);padding:10px;border-radius:8px;max-height:none}.md code{font:13px ui-monospace,monospace}
.md table{margin:8px 0}a{color:var(--acc)}.mut{color:var(--mut);font-size:13px}
.rub{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}.rub h3{margin:0 0 8px;font-size:22px}.rub h3 em{color:var(--mut);font-style:normal;font-weight:500}
.big{font-size:40px;font-weight:700;line-height:1.15;overflow-wrap:anywhere}.sub{font-size:19px;color:var(--mut);margin-top:6px;overflow-wrap:anywhere}
.rub table{font-size:18px}.links a{display:block;font-size:26px;font-weight:600;margin:6px 0}
.lv{display:inline-block;padding:1px 7px;border-radius:6px;font-size:12px;font-weight:700;color:#fff;background:var(--mut)}
.lv.error{background:var(--stop)}.lv.warn,.lv.warning{background:#b7791f}.lv.info{background:var(--acc)}
</style></head><body>
<header><div class="hwrap"><b>Spot it, sort it</b><span class="pill" id="hpart">part …</span>
<span class="pill" id="hrun">idle</span><span class="pill" id="hdry">…</span>
<nav><button data-t="rubric" onclick="tab('rubric')">Rubric</button><button data-t="run" onclick="tab('run')">Run</button>
<button data-t="docs" onclick="tab('docs')">Docs</button><button data-t="diag" onclick="tab('diag')">Diagnostics</button>
<!--PRIV--><button data-t="controls" onclick="tab('controls')">Controls</button><!--/PRIV--></nav></div></header>
<div class="tab wrap rub" id="t-rubric">
<section class="card"><h3>Decision log <em>30%</em></h3><div id="rb-log" class="big">—</div><div id="rb-log2" class="sub"></div></section>
<section class="card"><h3>Right place <em>30%</em></h3><div class="big"><span class="box" id="rb-box">0</span> box &middot; <span class="reject" id="rb-rej">0</span> reject</div>
<div id="rb-place" class="sub"></div></section>
<section class="card"><h3>Latency <em>10%</em></h3><div id="rb-lat" class="big">—</div><div id="rb-lat2" class="sub"></div></section>
<section class="card"><h3>Device placement <em>10%</em></h3><div id="rb-dev"></div>
<div class="sub"><a href="#" onclick="tab('diag');return false">Measured on the robot &rarr; Diagnostics</a></div></section>
<section class="card"><h3>Design explanation <em>20%</em></h3><div class="links">
<a href="#" onclick="openDoc('docs/DESIGN.md');return false">Design</a><a href="#" onclick="openDoc('docs/STORYLINE.md');return false">Storyline</a>
<a href="#" onclick="openDoc('docs/VLA-EXPERIMENTS.md');return false">VLA experiments</a></div></section></div>
<div class="tab wrap" id="t-run" hidden>
<div id="newrun" class="banner" hidden onclick="this.hidden=true" title="click to dismiss"></div>
<section class="card"><h2>Current run</h2><div id="rcur">Loading…</div><p id="rerr" class="warn"></p></section>
<section class="card"><h2>Machine logs <span class="mut" id="lgmeta"></span></h2><div class="tw" id="lgs">Loading…</div></section>
<section class="card"><h2>Run history</h2><div id="rhist" class="tw"></div>
<p class="mut" id="vlog">vision-9 log: open this tab to read it (refreshed at most every 10 s)</p></section></div>
<div class="tab wrap" id="t-docs" hidden><section class="card docs"><div class="dl" id="dlist">Loading…</div>
<div><div class="row"><span class="mut" id="dmeta"></span><button onclick="docsLoaded=false;loadDocs()" style="margin-left:auto;padding:4px 10px">Reload</button></div>
<div class="md" id="dbody"></div></div></section></div>
<div class="tab wrap" id="t-diag" hidden>
<section class="card"><h2>Device placement (measured)</h2>
<p class="mut">Measured 2026-11-14 with the ov-diag benchmark, 100 iterations. Live runs call ov-diag on the robot, one at a time.</p>
<div class="tw"><table id="btab"></table></div><p id="berr" class="warn"></p></section>
<!--PRIV--><section class="card"><h2>Devices and service health</h2><div class="row"><button onclick="devices()">List devices</button>
<button onclick="health()">Check service health</button></div><div class="tw" id="hres" style="margin-top:10px"></div>
<pre id="dres" style="margin-top:10px"></pre></section><!--/PRIV--></div>
<!--PRIV--><main class="tab" id="t-controls" hidden>
<section class="card"><h1>Arm Console</h1><div class="row" style="margin-bottom:12px">
<span class="pill" id="who"></span><span class="pill" id="dry"></span><span class="pill" id="run"></span></div>
<button id="stop" onclick="act('stop')">STOP</button>
<h2>Poses</h2><div class="row" style="margin-bottom:12px"><button data-m onclick="act('home')">Home</button>
<button data-m onclick="act('inspect')">Inspect</button></div>
<h2>Sorting</h2><div class="row"><button onclick="preview()">Preview (no motion)</button>
<button data-m onclick="act('plan')">Plan (inspect only)</button>
<button data-m onclick="ok('Pick ONE item now? Hand on the e-stop.')&&act('sort_one')">Sort one</button>
<button data-m class="go" onclick="ok('Start a run of '+n.value+' items? Hand on the e-stop.')&&act('start')">Start</button>
<label>items <input id="n" type="number" min="1" max="10" value="3"></label></div>
<div id="pick" class="pick">No target yet — press Plan (moves to inspect only) or Preview (no motion).</div>
<p id="msg" class="warn"></p><h2>Last result</h2><pre id="out">—</pre></section>
<section class="card"><h2>Camera <select id="cam" onchange="snap()"><option>cam</option><option>overview-cam</option></select>
<label><input type="checkbox" id="live"> refresh</label> <button onclick="snap()">Snapshot</button>
<button onclick="see()">What does __VISION__ see?</button></h2><img id="img" alt="camera"><pre id="see"></pre>
<h2 style="margin-top:12px">Events</h2><pre id="ev"></pre></section></main><!--/PRIV-->
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script>
const $=id=>document.getElementById(id);let busy=false;const ok=m=>confirm(m);
const BIN={box:['normal','NORMAL part','good bin (good-pose)'],reject:['anomalous','ANOMALOUS part','reject bin (bad-pose)'],
 hold:['hold','UNSURE part','hold bin (unsure-pose)']};
function banner(dest,prefix,extra){const b=BIN[dest];if(!b)return;$('pick').className='pick '+b[0];
 $('pick').textContent=prefix+': '+b[1]+' \\u2192 '+b[2]+(extra?'  \\u00b7 '+extra:'')}
function showPick(j){if(!j)return;if(j.destination)banner(j.destination,j.status==='sorted'?'Sorted item #'+j.item:'Picking item #'+j.item,
 j.status==='sorted'?('held: '+j.held):'');else if(j.status==='empty'){$('pick').className='pick';$('pick').textContent='Nothing to pick in view'}}
async function preview(){$('pick').className='pick';$('pick').textContent='Looking…';
 const j=await (await fetch('/api/preview')).json();if(j.error){$('pick').textContent=j.error;return}
 if(j.next)banner(j.next.destination,'Next in queue: item #'+j.next.item,'score '+j.next.score);
 else if(j.objects&&j.objects.length){const o=j.objects[0];banner({normal:'box',anomalous:'reject',hold:'hold'}[o.label],
  'In view (not queued yet)',j.objects.length+' part(s): '+j.objects.map(o=>o.label).join(', '))}
 else{$('pick').textContent='Nothing in view'}}
function setBusy(b){busy=b;document.querySelectorAll('[data-m]').forEach(x=>x.disabled=b)}
async function act(a){if(a!=='stop'&&busy)return;if(a!=='stop')setBusy(true);$('msg').textContent='';
 try{const r=await fetch('/api/action',{method:'POST',body:JSON.stringify({action:a,max_items:+$('n').value})});
  const j=await r.json();$('out').textContent=JSON.stringify(j,null,1);if(!r.ok)$('msg').textContent=j.error;else showPick(j)}
 catch(e){$('msg').textContent=e}finally{if(a!=='stop')setBusy(false);poll()}}
async function poll(){try{const j=await (await fetch('/api/status')).json();
 sRunning=!!j.running;$('hpart').textContent='part '+j.part;$('hdry').textContent=j.dry_run?'dry run (no motion)':'MOTION ENABLED';
 $('hdry').className='pill'+(j.dry_run?'':' warn');hdr();if(PUBLIC)return;
 $('who').textContent=j.sorter+' · '+j.vision;$('dry').textContent=j.dry_run?'dry run (no motion)':'MOTION ENABLED';
 $('dry').className='pill'+(j.dry_run?'':' warn');$('run').textContent=j.running?'running':'idle';
 $('ev').textContent=j.events.join('\\n');if(j.error)$('msg').textContent='sorter error: '+j.error;
 if(j.running&&j.last)showPick(j.last)}catch(e){$('hdry').textContent='status unavailable'}}
function snap(){$('img').src='/api/snapshot?cam='+$('cam').value+'&t='+Date.now()}
async function see(){$('see').textContent='…';const j=await (await fetch('/api/vision')).json();
 $('see').textContent=(j.objects||[]).map(o=>o.label+'  '+o.score+'  centre '+o.center).join('\\n')||JSON.stringify(j)}
const PUBLIC=__PUBLIC__;
setInterval(poll,3000);poll();if(!PUBLIC){setInterval(()=>{if($('live').checked)snap()},3000);snap()}
// ---- demo dashboard
let sRunning=false,csvLive=false,lastRun=null,docsLoaded=false,curDoc=null,benchBusy=false;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hms=t=>t?new Date(t*1000).toLocaleTimeString([],{hour12:false}):'—';
const when=t=>t?new Date(t*1000).toLocaleString([],{hour12:false}):'—';
const dur=s=>{s=Math.max(0,Math.round(s||0));return Math.floor(s/60)+'m '+String(s%60).padStart(2,'0')+'s'};
function hdr(){const on=sRunning||csvLive;$('hrun').textContent=on?'RUN IN PROGRESS':'idle';$('hrun').className='pill'+(on?' live':'')}
function tab(n){document.querySelectorAll('.tab').forEach(x=>x.hidden=x.id!=='t-'+n);
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===n));
 try{history.replaceState(null,'','#'+n)}catch(e){}if(n==='docs'&&!docsLoaded)loadDocs();if(n==='diag')loadBench();if(n==='run')runPoll()}
const stat=(v,l,c)=>'<div class="stat"><b class="'+(c||'')+'">'+esc(v)+'</b><span>'+esc(l)+'</span></div>';
async function runPoll(){try{const vis=!$('t-run').hidden;const j=await (await fetch('/api/run?vlog='+(vis?1:0))).json();
 if(j.error)throw j.error;$('rerr').textContent='';const c=j.current;csvLive=!!(c&&c.live);hdr();
 if(c&&lastRun!==null&&c.rel!==lastRun){$('newrun').hidden=false;$('newrun').textContent='NEW RUN STARTED '+hms(c.start)+'  \\u00b7  '+c.rel}
 lastRun=c?c.rel:'';rubric(c);if(vis)logsPoll();
 if(!c)$('rcur').innerHTML='<p>No judge CSVs yet in '+esc(j.dir)+'/runs</p>';
 else{const gaps=c.items.map(i=>i.gap).filter(g=>g!=null);const avg=gaps.length?(gaps.reduce((a,b)=>a+b,0)/gaps.length).toFixed(1)+' s':'—';
  $('rcur').innerHTML='<div class="row"><b>'+esc(c.rel)+'</b><span class="pill'+(c.live?' live':'')+'">'+(c.live?'IN PROGRESS':'finished')+
   '</span><span class="mut">started '+esc(when(c.start))+' \\u00b7 last write '+esc(hms(c.mtime))+'</span></div>'+
   '<div class="stats">'+stat(dur(c.elapsed),'elapsed')+stat(c.n,'items')+stat(c.box,'box','box')+stat(c.reject,'reject','reject')+stat(avg,'avg between items')+'</div>'+
   '<div class="tw"><table><tr><th>item</th><th>decision</th><th>confidence</th><th>time</th><th>since previous</th></tr>'+
   (c.items.map(i=>'<tr><td>'+esc(i.item)+'</td><td class="'+(i.decision==='box'?'box':i.decision==='reject'?'reject':'')+'">'+esc(i.decision)+
    '</td><td>'+esc(i.confidence)+'</td><td title="'+esc(i.timestamp)+'">'+esc(i.t?hms(i.t):i.timestamp)+'</td><td>'+(i.gap==null?'—':esc(i.gap)+' s')+'</td></tr>').join('')||
    '<tr><td colspan="5" class="mut">no items yet</td></tr>')+'</table></div>'}
 $('rhist').innerHTML='<table><tr><th>run</th><th>started</th><th>items</th><th>box / reject</th><th></th></tr>'+j.history.map(h=>
  '<tr><td>'+esc(h.rel)+'</td><td>'+esc(when(h.start))+'</td><td>'+esc(h.n)+'</td><td><span class="box">'+esc(h.box)+'</span> / <span class="reject">'+esc(h.reject)+
  '</span></td><td><a href="/api/runcsv?name='+encodeURIComponent(h.rel)+'">download CSV</a></td></tr>').join('')+'</table>';
 const v=j.vision_log;if(vis&&v){const d=v.data;$('vlog').textContent='__VISION__ log: '+(d?('run '+(d.run??'—')+' \\u00b7 items '+
  (Array.isArray(d.items)?d.items.length:(d.items??'—'))+' \\u00b7 '+(d.log_path??'')):'not read yet')+(v.error?' \\u00b7 error: '+v.error:'')+(v.at?' \\u00b7 as of '+hms(v.at):'')}
}catch(e){$('rerr').textContent='run poll failed: '+e}}
function rubric(c){if(!c){$('rb-log').textContent='No run yet';$('rb-log2').textContent='waiting for runs/judge-*.csv';return}
 const it=c.items.slice().reverse(),last=c.items[0];
 $('rb-log').textContent=c.n+' part'+(c.n===1?'':'s')+' logged';
 $('rb-log2').textContent=c.rel+(c.live?' \\u00b7 LIVE':'')+(last?' \\u00b7 last: '+[last.item,last.decision,last.confidence,last.timestamp].join(','):'');
 $('rb-box').textContent=c.box;$('rb-rej').textContent=c.reject;
 $('rb-place').textContent=c.n?(c.n-c.box-c.reject?(c.n-c.box-c.reject)+' other \\u00b7 ':'')+'from the judge CSV':'no parts yet';
 const gaps=it.map(i=>i.gap).filter(g=>g!=null),first=it.find(i=>i.t);
 $('rb-lat').textContent=gaps.length?(gaps.reduce((a,b)=>a+b,0)/gaps.length).toFixed(1)+' s / part':'—';
 $('rb-lat2').textContent=(first?'start \\u2192 first part '+Math.max(0,first.t-c.start).toFixed(1)+' s':'no part logged yet')+
  (gaps.length?' \\u00b7 last gaps '+gaps.slice(-5).map(g=>g+'s').join(' '):'')}
async function logsPoll(){try{const j=await (await fetch('/api/logs?refresh=1')).json();
 $('lgmeta').textContent=(j.at?'as of '+hms(j.at):'loading…')+(j.error?' \\u00b7 error: '+j.error:'');
 const L=j.lines||[];$('lgs').innerHTML=L.length?'<table>'+L.map(r=>'<tr><td>'+esc(hms(r.t))+'</td><td><span class="lv '+esc(r.level)+'">'+
  esc(r.level)+'</span></td><td class="mut">'+esc(r.logger)+'</td><td>'+esc(r.msg)+'</td></tr>').join('')+'</table>':'<p class="mut">no matching log lines yet</p>'}
 catch(e){$('lgmeta').textContent='logs failed: '+e}}
function openDoc(p){curDoc=p;tab('docs');if(docsLoaded)showDoc(p)}
async function loadDocs(){try{const j=await (await fetch('/api/docs')).json();docsLoaded=true;
 $('dlist').innerHTML=j.docs.map(d=>'<button data-p="'+esc(d.path)+'"'+(d.exists?'':' disabled')+'>'+esc(d.path)+'<small>'+
  (d.exists?'As of '+esc(when(d.mtime)):'not found')+'</small></button>').join('');
 $('dlist').querySelectorAll('button').forEach(b=>b.onclick=()=>showDoc(b.dataset.p));
 const f=j.docs.find(d=>d.exists&&d.path===curDoc)||j.docs.find(d=>d.exists);if(f)showDoc(f.path);else $('dbody').textContent='No docs found in '+j.dir}
 catch(e){$('dlist').textContent='docs failed: '+e}}
async function showDoc(p){curDoc=p;$('dlist').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.p===p));
 const r=await fetch('/api/doc?path='+encodeURIComponent(p));const t=await r.text();
 if(!r.ok){$('dmeta').textContent=p;$('dbody').textContent='not found';return}
 $('dmeta').textContent=p+' \\u00b7 As of '+when(+r.headers.get('X-Doc-Mtime'));
 $('dbody').innerHTML=window.marked?marked.parse(t):'<pre>'+esc(t)+'</pre>';$('dbody').scrollIntoView({block:'nearest'})}
const ST=[['mlmodel-7','part detector','NPU','2.6'],['mlmodel-8','normal / anomaly / other','NPU','2.4'],
 ['mlmodel-9','crop PatchCore','GPU f32','21.0'],['ov','full-frame PatchCore','GPU f32','18.3']];
const pk=(o,...ks)=>{for(const k of ks)if(o&&o[k]!=null)return o[k];return null};
const num=x=>x==null?'—':(typeof x==='number'?x.toFixed(x<10?2:1):x);
let BENCH={};
function liveCell(b){if(!b)return '<span class="mut">not run</span>';const r=b.result||{};const ex=pk(r,'execution_devices','exec_devices');
 return esc(pk(r,'device')??'—')+' \\u00b7 exec '+esc(Array.isArray(ex)?ex.join(', '):(ex??'—'))+'<br>p50 '+esc(num(pk(r,'p50_ms','p50')))+' \\u00b7 p90 '+
  esc(num(pk(r,'p90_ms','p90')))+' \\u00b7 mean '+esc(num(pk(r,'mean_ms','mean')))+' ms \\u00b7 '+esc(num(pk(r,'throughput_fps','fps')))+' fps'+
  '<br><span class="mut">at '+esc(hms(b.at))+'</span><details><summary class="mut">raw</summary><pre>'+esc(JSON.stringify(r,null,1))+'</pre></details>'}
function drawBench(){$('btab').innerHTML='<tr><th>model</th><th>role</th><th>device</th><th>p50 ms</th><th>'+(PUBLIC?'last live benchmark':'live benchmark')+'</th>'+(PUBLIC?'':'<th></th>')+'</tr>'+
 ST.map(s=>'<tr><td>'+s[0]+'</td><td>'+s[1]+'</td><td><b>'+s[2]+'</b></td><td>'+s[3]+'</td><td>'+liveCell(BENCH[s[0]])+'</td>'+
  (PUBLIC?'':'<td><button data-b="'+s[0]+'"'+(benchBusy?' disabled':'')+'>Run live benchmark</button></td>')+'</tr>').join('');
 $('btab').querySelectorAll('button[data-b]').forEach(b=>b.onclick=()=>bench(b.dataset.b))}
async function loadBench(){try{BENCH=await (await fetch('/api/diag/bench')).json()}catch(e){}drawBench()}
async function bench(m){if(benchBusy)return;benchBusy=true;$('berr').textContent='Benchmarking '+m+' on the robot…';drawBench();
 try{const r=await fetch('/api/diag/bench?run=1&model='+encodeURIComponent(m));const j=await r.json();
  if(!r.ok)throw j.error;BENCH[m]=j;$('berr').textContent=''}catch(e){$('berr').textContent='benchmark failed: '+e}
 finally{benchBusy=false;drawBench()}}
async function devices(){$('dres').textContent='Listing devices…';try{const j=await (await fetch('/api/diag/devices')).json();
 $('dres').textContent=(j.error?'error: '+j.error:'as of '+hms(j.at)+'  '+JSON.stringify(j.result,null,1))}catch(e){$('dres').textContent=e}}
async function health(){$('hres').textContent='Timing one call each…';try{const j=await (await fetch('/api/diag/health')).json();
 if(j.error)throw j.error;$('hres').innerHTML='<table><tr><th>service call</th><th>result</th><th>latency</th></tr>'+Object.entries(j.checks).map(([k,v])=>
  '<tr><td>'+esc(k)+'</td><td class="'+(v.ok?'box':'reject')+'">'+(v.ok?'ok':'error: '+esc(v.error))+'</td><td>'+esc(v.ms)+' ms</td></tr>').join('')+
  '</table><p class="mut">as of '+esc(hms(j.at))+'</p>'}catch(e){$('hres').textContent='health failed: '+e}}
$('rb-dev').innerHTML='<table>'+ST.map(s=>'<tr><td>'+s[0]+'</td><td><b>'+s[2]+'</b></td><td>'+s[3]+' ms</td></tr>').join('')+'</table>';
const TABS=PUBLIC?['rubric','run','docs','diag']:['rubric','run','docs','diag','controls'];
const t0=(location.hash||'').slice(1);tab(TABS.includes(t0)?t0:'rubric');  // always lands on Rubric unless #tab is given
setInterval(runPoll,2000);
</script></body></html>"""


PRIVATE_PATHS = {"/api/action", "/api/snapshot", "/api/preview", "/api/vision", "/api/diag/devices", "/api/diag/health"}


def render_page(public: bool) -> str:
    p = PAGE.replace("__VISION__", VISION).replace("__PUBLIC__", "true" if public else "false")
    return re.sub(r"<!--PRIV-->.*?<!--/PRIV-->", "", p, flags=re.S) if public else p


class Handler(BaseHTTPRequestHandler):
    public = False  # PublicHandler: read-only page and API for sharing through a tunnel

    def log_message(self, *a) -> None:  # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if self.public and (u.path in PRIVATE_PATHS or (u.path == "/api/diag/bench" and q.get("run", ["0"])[0] == "1")):
            self._json(403, {"error": "read-only view"})
            return
        try:
            if u.path == "/":
                body = render_page(self.public).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/status":
                st = sorter_status()
                self._json(200, {"part": PART, "sorter": SORTER, "vision": VISION, "dry_run": st.get("dry_run"), "running": st.get("running"),
                                 "error": st.get("error"), "events": events[-15:][::-1],
                                 "last": (st.get("results") or [None])[-1]})
            elif u.path == "/api/preview":  # no motion: queue head from the vision service's log + what is in view now
                vis = lambda c: viam_run("viam.service.vision.v1.VisionService.DoCommand",  # noqa: E731
                                         {"name": VISION, "command": {"command": c}}, 30).get("result", {})
                nxt, objs = pool.submit(vis, "next"), pool.submit(vis, "objects")
                self._json(200, {"next": nxt.result().get("item"), "objects": objs.result().get("objects", [])})
            elif u.path == "/api/snapshot":
                img = snapshot(parse_qs(u.query).get("cam", ["cam"])[0])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(img)
            elif u.path == "/api/vision":
                r = viam_run("viam.service.vision.v1.VisionService.DoCommand", {"name": VISION, "command": {"command": "objects"}}, 30)
                self._json(200, r.get("result", {}))
            elif u.path == "/api/run":  # local files; ?vlog=1 also refreshes vision-9's log (cached, >=10 s apart)
                self._json(200, run_state(q.get("vlog", ["0"])[0] == "1"))
            elif u.path == "/api/runcsv":
                rel = q.get("name", [""])[0]
                f = next((f for f in run_files() if f["rel"] == rel), None)
                if not f:
                    self._json(404, {"error": "no such run file"})
                    return
                self._send(200, open(f["path"], "rb").read(), "text/csv; charset=utf-8",
                           {"Content-Disposition": f'attachment; filename="{os.path.basename(f["path"])}"'})
            elif u.path == "/api/logs":  # ?refresh=1 re-reads machine logs (cached, >=15 s apart)
                self._json(200, machine_logs(q.get("refresh", ["0"])[0] == "1"))
            elif u.path == "/api/docs":
                self._json(200, {"dir": VR, "docs": docs_index()})
            elif u.path == "/api/doc":  # fixed allowlist only; no user path ever reaches the filesystem
                rel = q.get("path", [""])[0]
                if rel not in DOCS:
                    self._json(403, {"error": "not an allowed doc"})
                    return
                p = os.path.join(VR, rel)
                if not os.path.isfile(p):
                    self._json(404, {"error": "not found"})
                    return
                self._send(200, open(p, "rb").read(), "text/markdown; charset=utf-8",
                           {"X-Doc-Mtime": str(os.path.getmtime(p))})
            elif u.path == "/api/diag/bench":  # ?model=..&run=1 runs one (one at a time); without run=1 returns the cache
                m = q.get("model", [""])[0]
                if q.get("run", ["0"])[0] == "1":
                    try:
                        self._json(200, {"model": m, **benchmark(m)})
                    except (RuntimeError, ValueError) as e:
                        self._json(409, {"error": str(e)})
                else:
                    self._json(200, _bench)
            elif u.path == "/api/diag/devices":
                r = viam_run("viam.service.generic.v1.GenericService.DoCommand",
                             {"name": "ov-diag", "command": {"command": "list_devices"}}, 60)
                self._json(200, {"at": time.time(), "result": r.get("result", r)})
            elif u.path == "/api/diag/health":
                self._json(200, health())
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        if self.public:
            self._json(403, {"error": "read-only view"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(200, action(str(req.get("action")), max(1, min(10, int(req.get("max_items", 3))))))
        except Exception as e:  # noqa: BLE001
            log(f"error: {e}")
            self._json(409, {"error": str(e)})


class PublicHandler(Handler):
    public = True


if __name__ == "__main__":
    log(f"arm console for {SORTER} / {VISION} on http://127.0.0.1:{PORT}")
    if os.environ.get("PUBLIC_PORT"):
        pub = ThreadingHTTPServer(("127.0.0.1", int(os.environ["PUBLIC_PORT"])), PublicHandler)
        threading.Thread(target=pub.serve_forever, daemon=True).start()
        log(f"read-only public view on http://127.0.0.1:{os.environ['PUBLIC_PORT']}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
