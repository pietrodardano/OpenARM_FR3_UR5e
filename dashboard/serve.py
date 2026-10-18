"""Read-only dashboard server for RoboTokens runs (standalone: touches no operative code).

It only READS files: `runs/<time>/report.json`, `runs/detect/{detections,frame}.png`, `configs/robotokens.yaml`
and the figures in `docs/figures/`. It never opens the camera, the CAN bus or the robot, so it can run
beside a live experiment (also on another machine, if `runs/` is synced or mounted).

    .venv/bin/python dashboard/serve.py                 # watch ./runs on http://127.0.0.1:8765
    .venv/bin/python dashboard/serve.py --runs /path    # another runs directory
    .venv/bin/python dashboard/serve.py --demo          # synthetic data, clearly labelled DEMO

    # live joint torques, on the robot PC (passive CAN listener: receive-only, never transmits)
    .venv/bin/python dashboard/serve.py --can           # interface, IDs and motor types from the config
    .venv/bin/python dashboard/serve.py --live-demo     # synthetic torque feed, to preview the live panel

Note: a run's report is written when the run finishes, and the camera panel shows the last saved
detection image. The dashboard shows what the robot saved; it is not a live video feed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = ROOT / "docs/figures"
FILES = {  # whitelist: URL name -> file, resolved per request (no path traversal)
    "detections.png": "detect/detections.png",
    "frame.png": "detect/frame.png",
}
FIGURE_NAMES = {"exploration.gif", "cross_embodiment_results.png", "cross_embodiment_scene.png", "logo.png"}
MAX_RUNS = 40


def clean(v):
    """JSON-safe: NaN / inf (Python's json writes them, browsers cannot parse them) become null."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def read_runs(runs: Path) -> list[dict]:
    out = []
    for p in sorted(runs.glob("*/report.json"), reverse=True)[:MAX_RUNS]:
        try:
            out.append({"id": p.parent.name, "mtime": p.stat().st_mtime, "report": clean(json.loads(p.read_text()))})
        except (OSError, ValueError):
            continue  # being written, or corrupt: skip, it will show up on the next poll
    return out


def read_image(runs: Path, name: str) -> dict | None:
    p = runs / FILES[name]
    return {"url": f"/files/{name}", "mtime": p.stat().st_mtime, "size": p.stat().st_size} if p.exists() else None


def workcell(config: Path) -> dict:
    cfg = yaml.safe_load(config.read_text())
    w, v, t = cfg["workcell"], cfg["vision"], cfg["task"]
    return {"table": w["table"], "bins": w["bins"], "camera": w["camera"], "workspace": v["workspace"],
            "weigh_pose": _find(cfg, "weigh", {}).get("pose"), "min_z": _find(cfg, "min_z", 3.0), "manual_objects": t.get("manual_objects", []),
            "table_top_m": w["table"]["top_height_m"]}


def _find(d, key, default):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for x in d.values():
            r = _find(x, key, None)
            if r is not None:
                return r
    return default


class Handler(BaseHTTPRequestHandler):
    runs: Path
    config: Path
    demo: bool
    tap = None  # can_tap.Tap or None

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            state = {"now": time.time(), "demo": self.demo, "runs_dir": str(self.runs),
                     "runs": read_runs(self.runs), "workcell": workcell(self.config),
                     "images": {n: read_image(self.runs, n) for n in FILES},
                     "figures": sorted(n for n in FIGURE_NAMES if (FIGURES / n).exists())}
            return self._send(200, json.dumps(clean(state)).encode(), "application/json")
        if path == "/api/live":
            if self.tap is None:
                return self._send(200, b'{"enabled": false}', "application/json")
            since = int((parse_qs(urlparse(self.path).query).get("since") or ["0"])[0])
            return self._send(200, json.dumps(clean(self.tap.snapshot(since))).encode(), "application/json")
        if path.startswith("/files/"):
            name = path[len("/files/"):]
            if name in FILES and (self.runs / FILES[name]).exists():
                return self._send(200, (self.runs / FILES[name]).read_bytes(), "image/png")
            return self._send(404, b"no such file", "text/plain")
        if path.startswith("/figures/"):
            name = path[len("/figures/"):]
            if name in FIGURE_NAMES and (FIGURES / name).exists():
                ctype = "image/gif" if name.endswith(".gif") else "image/png"
                return self._send(200, (FIGURES / name).read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory to watch (read-only)")
    ap.add_argument("--config", default=str(ROOT / "configs/robotokens.yaml"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--demo", action="store_true", help="serve synthetic DEMO data from dashboard/demo_data")
    ap.add_argument("--can", nargs="?", const="config", metavar="IFACE",
                    help="live torques from a passive, receive-only CAN listener (default: interface from the config)")
    ap.add_argument("--live-demo", action="store_true", help="synthetic torque feed to preview the live panel")
    a = ap.parse_args()
    runs = Path(a.runs)
    if a.demo:
        import sys

        sys.path.insert(0, str(HERE))
        import demo

        runs = demo.make(HERE / "demo_data")
    Handler.runs, Handler.config, Handler.demo = runs, Path(a.config), a.demo
    if a.can or a.live_demo:
        import os
        import sys

        sys.path.insert(0, str(HERE))
        import can_tap

        iface, fd, channels = can_tap.channels_from_config(Path(a.config))
        if a.live_demo:
            Handler.tap = can_tap.synthetic(channels)
        else:
            if hasattr(os, "nice"):
                os.nice(10)  # never compete with the robot's servo thread for CPU
            Handler.tap = can_tap.Tap(channels).start(a.can if a.can != "config" else iface, fd)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dashboard on http://{a.host}:{a.port}  (watching {runs}{'  [DEMO DATA]' if a.demo else ''}; read-only)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
