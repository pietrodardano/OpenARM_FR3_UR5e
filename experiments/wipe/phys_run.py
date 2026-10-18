"""Full physical pipeline: hard-coded grasp -> force-controlled contact -> wipe controller.

    .venv/bin/python experiments/wipe/phys_run.py              # eval (10 seeds) + video
    .venv/bin/python experiments/wipe/phys_run.py --video-only

Wipe controllers get the same interface as in run.py: given the sponge xy (and the dirt map
for the goal), return the next xy target at 12 Hz. Everything below that is physics.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import torch

import run
from physim import F_DES, PhysWipe, dirt_pattern
from sanity import Scrub

RATE_HZ = 12.0
WIPE_S = 25.0
SQUEEZE_M = 0.002  # finger set-point this far inside the contact opening (~11 N squeeze)


class Zigzag:
    """Hard-coded: boustrophedon rows over the whole dirt area (pad lanes 3 cm apart, pad 4 cm wide)."""

    def __init__(self, sim) -> None:
        pts, ys = [], np.arange(sim.dirt_y[0] + 0.015, sim.dirt_y[1], 0.03)
        for i, y in enumerate(ys):
            xs = (sim.dirt_x[0] + 0.01, sim.dirt_x[1] - 0.01)
            xs = xs if i % 2 == 0 else xs[::-1]
            pts += [(xs[0], y), (xs[1], y)]
        self.pts, self.i = np.array(pts), 0

    def __call__(self, sim, xy, rng):
        tgt = self.pts[self.i % len(self.pts)]
        d = tgt - xy
        if np.linalg.norm(d) < run.STEP:
            self.i += 1
            return tgt
        return xy + d / np.linalg.norm(d) * run.STEP

    def observe(self, d) -> None:
        pass


CONTROLLERS = {"zigzag": Zigzag, "diffusion": lambda sim: run.Learned("diffusion"),
               "scrub": lambda sim: Scrub(1.0, 0.05, 0.35),  # best of 18 settings tuned on physics seeds 0-4
               "greedy": lambda sim: run.Greedy(),
               "bc": lambda sim: run.Learned("bc"), "random": lambda sim: run.RandomWalk()}


def episode(sim: PhysWipe, name: str, seed: int, frames: list | None = None, fps: float = 30.0,
            stubborn: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sim.reset(dirt_pattern(sim, seed, stubborn))
    log = {"phase": [], "force": [], "slip_mm": [], "tilt_deg": []}
    next_frame = [0.0]

    def tick(ms: int, f_des: float = F_DES) -> None:
        for _ in range(ms // 10):
            sim.step(10, f_des)
            if frames is not None and sim.t >= next_frame[0]:
                frames.append((sim.render(), sim.t, sim.cleaned(), sim.contact_force(), phase[0]))
                next_frame[0] += 1 / fps

    def move(p, T: float) -> None:
        p0, n = sim.p_des.copy(), max(1, int(T * 20))
        for k in range(1, n + 1):
            a = k / n
            sim.set_target(p0 + (np.asarray(p) - p0) * a * a * (3 - 2 * a))
            tick(int(T * 1000 / n))

    phase = ["grasp (hard-coded)"]
    sp, _ = sim.sponge_pose()
    grasp = sim.tcp_for_pad(sp[:2])
    move(grasp + sim.u * 0.06, 1.0)  # pre-grasp back along the handle axis
    move(grasp, 1.2)  # slide the open jaws onto the handle
    sim.gripper(False, 0.0)
    tick(300)
    q_contact = float(sim.data.qpos[sim.arm.finger_qadr[0]])
    sim.gripper(False, max(0.0, q_contact - SQUEEZE_M))
    tick(500)
    p_tcp, R_tcp = sim.tcp()
    sp, Rs = sim.sponge_pose()
    rel0 = R_tcp.T @ (sp - p_tcp)  # sponge in the hand frame right after the grasp

    phase[0] = "transfer"
    move(grasp + [0, 0, 0.06], 0.8)
    lifted = sim.sponge_pose()[0][2] - sp[2]  # did the sponge come up with the hand?
    centre = np.array([np.mean(sim.dirt_x), np.mean(sim.dirt_y)])
    move(sim.tcp_for_pad(centre, 0.04), 1.5)
    phase[0] = "touch down"
    felt = 0
    while felt < 4 and sim.t < 30:  # descend at 3 cm/s until the pad feels the board for 40 ms
        sim.set_target(sim.p_des - [0, 0, 0.03 / 100], update_posture=False)
        tick(10)
        felt = felt + 1 if sim.contact_force() > 2.0 else 0
    sim.z_mode = "force"
    tick(500)

    phase[0] = f"wipe: {name}"
    ctrl = CONTROLLERS[name](sim)
    cap = run.MAX_STEP_FACTOR * run.STEP
    lo, hi = [sim.dirt_x[0], sim.dirt_y[0]], [sim.dirt_x[1], sim.dirt_y[1]]
    # Controllers act on the commanded pad position (not the measured one, which lags under
    # friction): the impedance + force loop below turn the command into motion.
    xy = sim.p_des[:2] + sim.pad_offset_xy
    t_wipe0, path0 = sim.t, sim.wiped_path
    curve = []
    for k in range(int(WIPE_S * RATE_HZ)):
        d = ctrl(sim, xy, rng) - xy
        n = np.linalg.norm(d)
        tgt = np.clip(xy + (d if n <= cap else d / n * cap), lo, hi)
        p0 = sim.p_des.copy()
        t_tcp = tgt - sim.pad_offset_xy
        sim.set_target(np.array([t_tcp[0], t_tcp[1], p0[2]]))
        for j in range(1, 9):  # 83 ms between policy steps, set-point ramped
            sim.p_des[:2] = p0[:2] + (t_tcp - p0[:2]) * j / 8
            tick(10)
        p_tcp, R_tcp = sim.tcp()
        sp, Rs = sim.sponge_pose()
        log["force"].append(sim.contact_force())
        log["slip_mm"].append(float(np.linalg.norm(R_tcp.T @ (sp - p_tcp) - rel0) * 1000))
        log["tilt_deg"].append(float(np.degrees(np.arccos(np.clip(Rs[2, 2], -1, 1)))))
        new = tgt.copy()
        ctrl.observe(new - xy)
        xy = new
        curve.append(sim.cleaned())

    phase[0] = "lift"
    sim.z_mode = "pos"
    sim.p_des = sim.tcp()[0]
    move(sim.p_des + [0, 0, 0.06], 1.0)
    f = np.array(log["force"])
    return {"cleaned": curve[-1], "t90_s": next((i / RATE_HZ for i, c in enumerate(curve) if c >= 0.9), None),
            "force_mean": float(f.mean()), "force_std": float(f.std()), "force_min": float(f.min()),
            "frac_in_contact": float((f > 1.0).mean()), "slip_mm_max": float(max(log["slip_mm"])),
            "tilt_deg_max": float(max(log["tilt_deg"])), "sponge_path_m": float(sim.wiped_path - path0),
            "squeeze_q": q_contact, "lifted_m": float(lifted), "torque_sat_frac": sim.sat_steps / max(1, sim.steps),
            "tau_peak": sim.tau_peak.round(1).tolist(), "curve": curve}


def overlay(img, t, cleaned, force, phase) -> np.ndarray:
    img = np.ascontiguousarray(img[:, :, ::-1])
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(img, phase, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{cleaned * 100:4.0f}% clean   contact {force:5.1f} N   t={t:4.1f}s",
                (img.shape[1] - 345, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 120), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--video-seed", type=int, default=3)
    a = ap.parse_args()
    sim = PhysWipe()
    if not a.video_only:
        res = {}
        for name in CONTROLLERS:
            r = [episode(sim, name, s) for s in range(a.seeds)]
            c = np.array([x["cleaned"] for x in r])
            t90 = [x["t90_s"] or np.inf for x in r]
            res[name] = {k: float(np.mean([x[k] for x in r])) for k in
                         ("cleaned", "force_mean", "force_std", "frac_in_contact", "slip_mm_max", "tilt_deg_max", "sponge_path_m")}
            res[name]["cleaned_std"] = float(c.std())
            res[name]["t90_median_s"] = float(np.median(t90))
            res[name]["reached90"] = int(np.isfinite(t90).sum())
            res[name]["tau_peak"] = np.max([x["tau_peak"] for x in r], 0).tolist()
            v = res[name]
            print(f"{name:10s} cleaned {v['cleaned'] * 100:5.1f} ± {v['cleaned_std'] * 100:4.1f}%  90% in {v['reached90']}/{a.seeds}"
                  f" (med {v['t90_median_s']:.1f}s)  force {v['force_mean']:.1f}±{v['force_std']:.1f} N  contact {v['frac_in_contact'] * 100:.0f}%"
                  f"  slip≤{v['slip_mm_max']:.1f} mm  tilt≤{v['tilt_deg_max']:.1f}°  sponge path {v['sponge_path_m']:.2f} m", flush=True)
        (run.OUT / "phys_results.json").write_text(json.dumps(res, indent=1))
    vw = None
    for name in ("diffusion", "scrub", "zigzag"):
        frames: list = []
        episode(sim, name, a.video_seed, frames)
        if vw is None:
            vw = cv2.VideoWriter(str(run.OUT / "phys_wipe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (sim.width, sim.height))
        for f in frames:
            vw.write(overlay(*f))
        cv2.imwrite(str(run.OUT / f"phys_{name}_last.png"), overlay(*frames[-30]))
    vw.release()
    sim.close()


if __name__ == "__main__":
    main()
