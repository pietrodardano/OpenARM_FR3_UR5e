"""Synthetic DEMO runs for previewing the dashboard (never mixed with real data, clearly labelled).

Writes the same schema as `Report.to_dict()` in src/robotokens/task/demo.py, a `DEMO` marker, and a
drawn stand-in for the camera image.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _est(label, kg, se):
    return {"label": label, "mass_kg": kg, "se_kg": se, "cycle_mass": [kg - se, kg, kg + se], "joints": [1, 3]}


def _phases(*items):
    return [[p, m] for p, m in items]


def make(root: Path) -> Path:
    runs = root / "runs"
    shutil.rmtree(runs, ignore_errors=True)  # always a clean, labelled demo set
    (runs / "detect").mkdir(parents=True, exist_ok=True)
    (root / "DEMO").write_text("synthetic data for previewing the dashboard\n")
    now = time.time()
    sort_text = "Sort the two cans: put the lighter one in the left bin."
    base_intent = {"mode": "sort", "bin_for": {"light": "left", "heavy": "right"}, "handover_class": "heavy",
                   "object_prompt": "aluminum can . box", "text": sort_text}
    obj = lambda n, x, y, m, cls: {"name": n, "x": x, "y": y, "confidence": 0.5, "mass": m, "weight_class": cls}
    sorted_run = {
        "intent": base_intent, "outcome": "done", "error": None, "handover_confirmed": None,
        "objects": [obj("object_1", 0.21, 0.13, _est("object_1", 0.101, 0.004), "light"),
                    obj("object_2", 0.21, 0.27, _est("object_2", 0.349, 0.005), "heavy")],
        "comparison": {"heavier": "object_2", "lighter": "object_1", "diff_kg": 0.248, "z": 38.8, "conclusive": True},
        "placements": {"object_1": "left", "object_2": "right"},
        "phase_log": _phases(("detect", "2 objects"), ("approach", "object_1"), ("grasp", "object_1"),
                             ("lift", "object_1"), ("weigh", "object_1: 101 +- 4 g"), ("place_back", "object_1"),
                             ("approach", "object_2"), ("grasp", "object_2"), ("lift", "object_2"),
                             ("weigh", "object_2: 349 +- 5 g"), ("place_back", "object_2"),
                             ("compare", "z = 38.8, heavier object_2"), ("sort", "object_1 -> left"),
                             ("sort", "object_2 -> right"), ("home", ""), ("done", "")),
    }
    handover = {
        "intent": {**base_intent, "mode": "handover", "text": "give me the heavy can"},
        "outcome": "done", "error": None, "handover_confirmed": True,
        "objects": [obj("object_1", 0.20, 0.14, _est("object_1", 0.352, 0.006), "heavy"),
                    obj("object_2", 0.22, 0.27, _est("object_2", 0.097, 0.005), "light")],
        "comparison": {"heavier": "object_1", "lighter": "object_2", "diff_kg": 0.255, "z": 32.7, "conclusive": True},
        "placements": {"object_1": "user", "object_2": "left"},
        "phase_log": _phases(("detect", "2 objects"), ("weigh", "object_1: 352 +- 6 g"), ("weigh", "object_2: 97 +- 5 g"),
                             ("compare", "z = 32.7"), ("handover", "object_1"), ("wait_contact", "baseline 1.0 s"),
                             ("release", "contact felt"), ("sort", "object_2 -> left"), ("home", ""), ("done", "")),
    }
    inconclusive = {
        "intent": base_intent, "outcome": "inconclusive", "error": None, "handover_confirmed": None,
        "objects": [obj("object_1", 0.21, 0.13, _est("object_1", 0.205, 0.045), None),
                    obj("object_2", 0.21, 0.27, _est("object_2", 0.236, 0.045), None)],
        "comparison": {"heavier": "object_2", "lighter": "object_1", "diff_kg": 0.031, "z": 0.49, "conclusive": False},
        "placements": {},
        "phase_log": _phases(("detect", "2 objects"), ("weigh", "object_1"), ("weigh", "object_2"),
                             ("compare", "z = 0.49 < 3: re-measure"), ("weigh", "object_2 again"),
                             ("compare", "still inconclusive"), ("inconclusive", "object put back, arm home")),
    }
    aborted = {
        "intent": base_intent, "outcome": "aborted", "objects": [], "comparison": None, "placements": {},
        "handover_confirmed": None, "error": "TaskAborted: expected 2 objects, found 1",
        "phase_log": _phases(("detect", "1 object found"), ("aborted", "TaskAborted: expected 2 objects, found 1")),
    }
    for k, rep in enumerate((aborted, inconclusive, handover, sorted_run)):  # oldest first
        t = now - (len((aborted, inconclusive, handover, sorted_run)) - k) * 420
        d = runs / time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(rep, indent=2))
        os.utime(d / "report.json", (t, t))
    _camera(runs / "detect")
    os.utime(runs / "detect" / "detections.png", (now - 400, now - 400))
    return runs


def _camera(out: Path) -> None:
    """A drawn stand-in for the HBV frame: table, two red cans, GroundingDINO-style boxes."""
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (92, 98, 110))
    d = ImageDraw.Draw(img)
    d.polygon([(120, 720), (1160, 720), (1010, 170), (270, 170)], fill=(184, 135, 92))
    f = ImageFont.load_default(size=26) if hasattr(ImageFont, "load_default") else None
    boxes = []
    for cx, cy in ((520, 470), (790, 420)):
        d.rectangle([cx - 46, cy - 100, cx + 46, cy + 60], fill=(205, 38, 44))
        d.rectangle([cx - 46, cy - 100, cx + 46, cy - 86], fill=(205, 208, 214))
        d.rectangle([cx - 46, cy + 46, cx + 46, cy + 60], fill=(205, 208, 214))
        d.rectangle([cx - 46, cy - 30, cx + 46, cy - 12], fill=(245, 245, 245))
        boxes.append((cx - 56, cy - 110, cx + 56, cy + 70))
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=(0, 255, 0), width=3)
        d.text((b[0], b[1] - 32), f"aluminum can {0.52 - 0.04 * i:.2f}", fill=(0, 255, 0), font=f)
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        d.line([cx - 10, cy, cx + 10, cy], fill=(255, 0, 0), width=3)
        d.line([cx, cy - 10, cx, cy + 10], fill=(255, 0, 0), width=3)
    d.text((20, 20), "DEMO IMAGE (drawn, not a camera frame)", fill=(255, 255, 255), font=f)
    img.save(out / "detections.png")
