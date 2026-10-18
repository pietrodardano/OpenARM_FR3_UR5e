"""Rebuild the real case in MuJoCo: the workcell with the cans/boxes where vision found them.

Standalone and read-only. It imports `robotokens.config.load_config` and `robotokens.robot.model.build_scene`
(the exact workcell the project's own twin uses: official OpenArm model, torso, table, bins, handover marker)
and changes nothing in them. It adds nothing to the robot: it only reads a run report (or explicit positions),
places one object per detection at its believed (x, y) in the robot base frame, and renders a PNG offscreen.

    .venv/bin/python dashboard/twin3d.py --report runs/<time>/report.json --out twin.png
    .venv/bin/python dashboard/twin3d.py --xy "0.21,0.13;0.21,0.27" --out twin.png

Objects are drawn as red cans (or boxes when the prompt names a box and no can). Once a run has weighed them,
the band on each can is colored: orange = heavy, teal = light. Before that, both bands are white: they look alike.
The arm is shown at the configured home pose, hand forward over the table.

Rendering runs in a subprocess started by serve.py, so a GL problem can never take the dashboard down.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotokens.config import load_config  # noqa: E402
from robotokens.robot.model import OBJECT_HALF_HEIGHT_M, build_scene  # noqa: E402

RED, SILVER, WHITE = (0.82, 0.05, 0.07, 1), (0.82, 0.84, 0.88, 1), (0.97, 0.97, 0.97, 1)
CLASS_RGBA = {"heavy": (0.94, 0.66, 0.24, 1), "light": (0.31, 0.76, 0.74, 1)}
W, H = 1280, 720


def find_model(cfg):
    """The config points at third_party/openarm_mujoco; some checkouts nest it under third_party/openarm/."""
    p = ROOT / cfg.robot.model_path
    if p.exists():
        return cfg
    alt = ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml"
    if alt.exists():
        return dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, model_path=str(alt.relative_to(ROOT))))
    raise FileNotFoundError(f"OpenArm model not found at {p} (run ./setup.sh)")


def render(objects: list[dict], kind: str, out: Path, caption: str | None = None) -> None:
    cfg = find_model(load_config())
    spec = build_scene(cfg, n_objects=2)
    for i in range(2):
        body = spec.body(f"object_{i + 1}")
        for g in list(body.geoms):
            spec.delete(g)
        band = CLASS_RGBA.get((objects[i].get("weight_class") if i < len(objects) else None) or "", WHITE)
        add = lambda **k: body.add_geom(contype=0, conaffinity=0, **k)  # noqa: E731
        if kind == "box":
            add(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.035, OBJECT_HALF_HEIGHT_M], rgba=[0.74, 0.58, 0.38, 1])
            add(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.046, 0.036, 0.008], pos=[0, 0, 0.0], rgba=list(band))
        else:
            r, hh = 0.033, OBJECT_HALF_HEIGHT_M
            add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, hh - 0.015, 0], rgba=list(RED))
            for z in (hh - 0.0075, -hh + 0.0075):
                add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r - 0.002, 0.0075, 0], pos=[0, 0, z], rgba=list(SILVER))
            add(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r + 0.0005, 0.007, 0], pos=[0, 0, 0.0], rgba=list(band))
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.75
    data = mujoco.MjData(model)
    side = cfg.robot.side
    for i, q in enumerate(cfg.robot.home_qpos):  # arm at the configured home pose
        data.qpos[model.jnt_qposadr[model.joint(f"openarm_{side}_joint{i + 1}").id]] = q
    for k in (1, 2):
        data.qpos[model.jnt_qposadr[model.joint(f"openarm_{side}_finger_joint{k}").id]] = 0.044
    for i in range(2):
        mid = int(model.body_mocapid[model.body(f"object_{i + 1}").id])
        if i < len(objects):
            data.mocap_pos[mid] = [objects[i]["x"], objects[i]["y"], cfg.table_z + OBJECT_HALF_HEIGHT_M]
        else:
            data.mocap_pos[mid] = [0.0, 0.0, -10.0]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.26, 0.19, cfg.table_z + 0.07]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 140, -30
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera=cam)
        img = Image.fromarray(r.render())
    if caption:
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.load_default(size=22)
        except TypeError:
            f = ImageFont.load_default()
        d.rectangle([0, H - 40, W, H], fill=(14, 18, 28))
        d.text((16, H - 33), caption, font=f, fill=(230, 235, 245))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def objects_from_report(path: Path) -> tuple[list[dict], str, str]:
    rep = json.loads(path.read_text())
    objs = [{"name": o["name"], "x": float(o["x"]), "y": float(o["y"]), "weight_class": o.get("weight_class"),
             "mass": o.get("mass")} for o in rep.get("objects", [])]
    prompt = (rep.get("intent", {}).get("object_prompt") or "can").lower()
    kind = "box" if "box" in prompt and "can" not in prompt else "can"
    cap = "Rebuilt from vision: " + "   ".join(
        f"{o['name']} ({o['x']:+.2f}, {o['y']:+.2f}) m" + (f" {o['mass']['mass_kg'] * 1000:.0f} g" if o.get("mass") else "")
        for o in objs)
    return objs, kind, cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="runs/<time>/report.json")
    ap.add_argument("--xy", help='explicit positions, e.g. "0.21,0.13;0.21,0.27"')
    ap.add_argument("--kind", default="can", choices=["can", "box"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.report:
        objs, kind, cap = objects_from_report(Path(a.report))
    elif a.xy:
        objs = [{"name": f"object_{i + 1}", "x": float(p.split(",")[0]), "y": float(p.split(",")[1])}
                for i, p in enumerate(a.xy.split(";"))]
        kind, cap = a.kind, "Positions given on the command line"
    else:
        ap.error("give --report or --xy")
    render(objs, kind, Path(a.out), cap)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
