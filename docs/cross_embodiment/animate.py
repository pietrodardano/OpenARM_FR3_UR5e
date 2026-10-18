"""GIF of the three arms (OpenArm v1, UR5e, Franka FR3) exploring two look-alike cans in MuJoCo.

Standalone and kinematic: no robotokens code is touched and no physics is stepped. Each arm reaches a can,
lifts it to a fixed weigh pose, dithers its joints (the motion the torque estimator uses), puts the can
back, and repeats for the second can. Arm joint angles come from a small numerical IK.

The masses shown in the captions are the simulated estimates from `simulate.py` (empty-gripper tare),
not hardware data. The GIF plays at 2x speed.

Usage:  .venv/bin/python -I docs/cross_embodiment/animate.py [--still | --preview] [--out docs/figures/exploration.gif]
Needs:  third_party/mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml, Pillow, numpy, mujoco.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "third_party/mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml"

SPEED = 2.0  # playback speed relative to the simulated motion
FPS = 10  # GIF frames per second
DT = SPEED / FPS  # simulated seconds per GIF frame
PANEL = (400, 400)
CAN_R, CAN_H = 0.033, 0.12
EST_G = (100, 350)  # simulated estimates (g) of can 1 and can 2, from simulate.py (with tare)

CAN_RED, CAN_SILVER, CAN_WHITE = (0.82, 0.05, 0.07, 1), (0.82, 0.84, 0.88, 1), (0.97, 0.97, 0.97, 1)


@dataclass
class Arm:
    name: str
    tool_site: str
    joints: list[str]
    cans: list[str]  # free-body names of the two cans
    cam: dict  # lookat / distance / azimuth / elevation of this arm's panel
    tilt_deg: float  # tool approach angle below horizontal
    color: tuple


ARMS = [
    Arm("OpenArm v1", "left_tcp", [f"openarm_left_joint{i}" for i in range(1, 8)], ["pick_cube", "openarm_can_2"],
        dict(lookat=(0.28, 0.15, 1.02), distance=1.75, azimuth=140, elevation=-22), 60, (42, 111, 219)),
    Arm("UR5e", "gripper_ur5e_left_pinch",
        ["ur5e_left_" + n + "_joint" for n in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")],
        ["ur5e_can_1", "ur5e_can_2"], dict(lookat=(0.40, 1.52, 1.05), distance=1.85, azimuth=140, elevation=-22), 75,
        (224, 138, 30)),
    Arm("Franka FR3", "gripper_fr3_left_pinch", [f"fr3_left_fr3_joint{i}" for i in range(1, 8)],
        ["fr3_can_1", "fr3_can_2"], dict(lookat=(0.40, -1.15, 1.05), distance=1.85, azimuth=140, elevation=-22), 75,
        (42, 157, 91)),
]


def build_model() -> mujoco.MjModel:
    """Scene with can-like cans (red body, silver rims, white band) instead of the plain grey ones."""
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.body("openarm_left_hand_tcp").add_site(name="left_tcp", size=[0.01, 0, 0], group=4)

    def can_geoms(body):
        for g in list(body.geoms):
            spec.delete(g)
        parts = ((CAN_RED, CAN_R, 0.045, 0.0), (CAN_SILVER, CAN_R - 0.002, 0.0075, 0.0525),
                 (CAN_SILVER, CAN_R - 0.002, 0.0075, -0.0525), (CAN_WHITE, CAN_R + 0.0004, 0.006, 0.005),
                 (CAN_WHITE, CAN_R + 0.0004, 0.006, -0.014))
        for k, (rgba, r, hh, z) in enumerate(parts):
            body.add_geom(name=f"{body.name}_part{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, hh, 0],
                          pos=[0, 0, z], rgba=list(rgba), contype=0, conaffinity=0, mass=0.002 if k == 0 else 0)

    for arm in ARMS:
        for cname in arm.cans:
            if arm.name == "OpenArm v1":
                continue
            can_geoms(spec.body(cname))
    # OpenArm: turn the cube into can 1, add can 2
    cube = spec.body("pick_cube")  # the name is referenced by equality constraints, so keep it
    cube.pos = [0.30, 0.10, 0.81]
    can_geoms(cube)
    second = spec.worldbody.add_body(name="openarm_can_2", pos=[0.30, 0.26, 0.81])
    second.add_freejoint()
    second.mass, second.inertia, second.explicitinertial = 0.01, [1e-5] * 3, 1
    can_geoms(second)
    return spec.compile()


def tool_pose(model, data, site_id):
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


class Solver:
    """Damped least-squares IK on one arm: tool position + approach axis (site z) pointing down-ish."""

    def __init__(self, model: mujoco.MjModel, arm: Arm):
        self.m, self.arm = model, arm
        self.d = mujoco.MjData(model)
        jt = [model.joint(j) for j in arm.joints]
        self.qadr = np.array([j.qposadr[0] for j in jt])
        self.dadr = np.array([j.dofadr[0] for j in jt])
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in arm.joints])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in arm.joints])
        self.sid = model.site(arm.tool_site).id
        self.base = self.d.qpos.copy()
        t = np.radians(arm.tilt_deg)
        self.axis = np.array([np.cos(t), 0.0, -np.sin(t)])  # forward (+x) and down

    def fk(self, q):
        self.d.qpos[:] = self.base
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def err(self, q, target):
        p, R = self.fk(q)
        return np.concatenate([target - p, 0.35 * np.cross(R[:, 2], self.axis)])

    def solve(self, target, q0, base_qpos, iters=60):
        self.base = base_qpos
        q = q0.copy()
        for _ in range(iters):
            e = self.err(q, target)
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            J = np.empty((6, len(q)))
            for i in range(len(q)):
                dq = np.zeros(len(q))
                dq[i] = 1e-5
                J[:, i] = (self.err(q + dq, target) - e) / 1e-5  # d(err)/dq
            step = -J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(6), e)
            q = np.clip(q + np.clip(step, -0.15, 0.15), self.lo, self.hi)
        return q, self.err(q, target)


def smooth(u):
    return u * u * u * (u * (6 * u - 15) + 10)


def build_timeline(can_pos, home_tool):
    """List of (duration_s, from_pos, to_pos, phase, can_idx_attached, dither, can_pos_idx)."""
    segs = []
    cz = can_pos[0][2]
    weigh = np.array([can_pos[0][0], 0.5 * (can_pos[0][1] + can_pos[1][1]), cz + 0.24])
    cur = home_tool.copy()

    def go(to, dur, phase, att=None, dither=False):
        nonlocal cur
        segs.append(dict(dur=dur, a=cur.copy(), b=np.asarray(to, float), phase=phase, att=att, dither=dither))
        cur = np.asarray(to, float)

    go(cur, 1.5, "Two identical-looking cans")
    for k in range(2):
        c = np.array(can_pos[k])
        up = c + [0, 0, 0.16]
        go(up, 2.5, f"Reach can {k + 1}")
        go(c, 1.6, f"Grasp can {k + 1}")
        go(c, 0.7, f"Grasp can {k + 1}", att=k)
        go(weigh, 2.2, f"Lift can {k + 1}", att=k)
        go(weigh, 6.0, f"Weigh can {k + 1} (joint torques)", att=k, dither=True)
        go(weigh, 0.8, f"Can {k + 1}: about {EST_G[k]} g", att=k)
        go(c, 2.2, f"Put can {k + 1} back", att=k)
        go(up, 1.2, f"Can {k + 1}: about {EST_G[k]} g")
    go(up, 1.0, "Can 2 is the heavier one", dither=False)
    go(up, 2.0, "Can 2 is the heavier one")
    return segs


def font(size):
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def render(args) -> None:
    panel = (640, 520) if args.still else PANEL
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    solvers = [Solver(model, a) for a in ARMS]
    can_ids = [[model.body(c) for c in a.cans] for a in ARMS]
    can_qadr = [[model.joint(model.body(c).jntadr[0]).qposadr[0] for c in a.cans] for a in ARMS]
    can_pos = []
    for ids in can_ids:
        can_pos.append([data.xpos[b.id].copy() + [0, 0, 0.0] for b in ids])

    # per-arm trajectory of joint angles, from the IK along the timeline
    trajs, rows = [], []
    for s, a, cp in zip(solvers, ARMS, can_pos):
        d0 = mujoco.MjData(model)
        d0.qpos[:] = base_qpos
        q = base_qpos[s.qadr].copy()
        s.base = base_qpos
        home_tool = s.fk(q)[0]
        segs = build_timeline(cp, home_tool)
        qs, meta = [], []
        t = 0.0
        total = sum(g["dur"] for g in segs)
        n = int(np.ceil(total / DT))
        worst = 0.0
        for f in range(n):
            t = f * DT
            acc = 0.0
            for g in segs:
                if t < acc + g["dur"] or g is segs[-1]:
                    u = min(max((t - acc) / g["dur"], 0.0), 1.0)
                    pos = g["a"] + (g["b"] - g["a"]) * smooth(u)
                    break
                acc += g["dur"]
            q, e = s.solve(pos, q, base_qpos)
            worst = max(worst, np.linalg.norm(e[:3]))
            dith = np.zeros(len(q))
            if g["dither"]:
                w = np.sin(2 * np.pi * (t - acc) / 2.0) * min(1.0, (t - acc) / 0.6, (acc + g["dur"] - t) / 0.6)
                dith[: min(4, len(q))] = np.radians(3.0) * w
            qs.append(np.clip(q + dith, s.lo, s.hi))
            meta.append((g["phase"], g["att"]))
        print(f"{a.name}: {n} frames, worst IK position error {worst * 1000:.1f} mm")
        trajs.append(np.array(qs))
        rows.append(meta)
    n = max(len(t) for t in trajs)
    for i, t in enumerate(trajs):
        if len(t) < n:
            trajs[i] = np.vstack([t, np.repeat(t[-1:], n - len(t), axis=0)])
            rows[i] += [rows[i][-1]] * (n - len(rows[i]))

    cams = []
    for a in ARMS:
        c = mujoco.MjvCamera()
        c.lookat[:] = a.cam["lookat"]
        c.distance, c.azimuth, c.elevation = a.cam["distance"], a.cam["azimuth"], a.cam["elevation"]
        cams.append(c)
    model.vis.global_.offwidth, model.vis.global_.offheight = panel
    if args.still:  # mid-weigh of can 2, all three arms holding a can
        frames_idx = [next(f for f in range(n) if rows[0][f][0].startswith("Weigh can 2")) + 8]
    else:
        frames_idx = [n // 3] if args.preview else range(n)
    out_frames = []
    f_title, f_cap, f_small = font(17 if not args.still else 20), font(15 if not args.still else 18), font(12)
    with mujoco.Renderer(model, panel[1], panel[0]) as r:
        for f in frames_idx:
            data.qpos[:] = base_qpos
            for i, (s, a) in enumerate(zip(solvers, ARMS)):
                data.qpos[s.qadr] = trajs[i][f]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            for i, (s, a) in enumerate(zip(solvers, ARMS)):
                att = rows[i][f][1]
                if att is not None:
                    p = data.site_xpos[s.sid]
                    qa = can_qadr[i][att]
                    data.qpos[qa:qa + 3] = p
                    data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            panels = []
            for i, a in enumerate(ARMS):
                r.update_scene(data, camera=cams[i])
                img = Image.fromarray(r.render())
                dr = ImageDraw.Draw(img)
                dr.rectangle([0, 0, panel[0], 30], fill=a.color)
                dr.text((10, 5), a.name, font=f_title, fill="white")
                dr.rectangle([0, panel[1] - 30, panel[0], panel[1]], fill=(25, 25, 30))
                dr.text((10, panel[1] - 25), rows[i][f][0], font=f_cap, fill="white")
                panels.append(img)
            strip = Image.new("RGB", (panel[0] * 3 + 8, panel[1]), (20, 20, 20))
            for i, p in enumerate(panels):
                strip.paste(p, (i * (panel[0] + 4), 0))
            if not args.still:
                ImageDraw.Draw(strip).text((strip.width - 215, panel[1] - 48),
                                           f"simulation, {SPEED:.0f}x speed", font=f_small, fill=(200, 200, 200))
            out_frames.append(strip)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still:
        out_frames[0].save(ROOT / "docs/figures/cross_embodiment_scene.png")
        print("still -> docs/figures/cross_embodiment_scene.png")
        return
    if args.preview:
        out_frames[0].save(out.with_suffix(".png"))
        print("preview ->", out.with_suffix(".png"))
        return
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in out_frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(pal)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="write one PNG frame instead of the GIF")
    ap.add_argument("--still", action="store_true", help="write docs/figures/cross_embodiment_scene.png (large panels)")
    ap.add_argument("--out", default=str(ROOT / "docs/figures/exploration.gif"))
    render(ap.parse_args())


if __name__ == "__main__":
    main()
