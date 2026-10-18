"""Evaluate wiping controllers in the sim and render a side-by-side video.

Controllers (same tool speed, same dirt, same 40 s):
  zigzag     hard-coded boustrophedon over the whole board
  random     random walk
  bc         MLP behaviour cloning on the Eidon strokes, goal = towards the dirt
  diffusion  diffusion policy on the Eidon strokes, goal = towards the dirt
The goal rule (where the dirt is) is hand-coded; how to move the sponge is learned.
"""

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import torch

from policy import ACT_H, OBS_H, OUT, RETARGET_SCALE, load
from wipesim import BOARD_X, BOARD_Y, RATE_HZ, WipeSim, dirt_pattern

SECONDS = 25
MAX_STEP_FACTOR = 1.5  # every controller: per-step tool motion capped at 1.5x the mean human step
EXEC = 8  # chunk steps executed before replanning
segs = np.load(OUT / "wipe_segments.npz")["segs"] * RETARGET_SCALE
STEP = float(np.linalg.norm(np.diff(segs, axis=1), axis=2).mean())  # mean human step (m)
GOAL_MAX = 0.06


def goal_to_dirt(sim: WipeSim, xy: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    d2 = np.sum((sim.cells - xy) ** 2, 1)
    w = sim.dirt * np.exp(-d2 / (2 * sigma ** 2))
    if w.sum() < 0.5:  # nothing dirty nearby: head for the nearest dirt anywhere
        w = sim.dirt * np.exp(-d2 / (2 * 0.15 ** 2))
    if w.sum() < 1e-6:
        return np.zeros(2)
    g = (w[:, None] * sim.cells).sum(0) / w.sum() - xy
    n = np.linalg.norm(g)
    return g if n < GOAL_MAX else g / n * GOAL_MAX


class Learned:
    def __init__(self, name: str) -> None:
        self.model = load(name)
        self.hist, self.plan = [np.zeros(2)] * OBS_H, []

    def __call__(self, sim: WipeSim, xy: np.ndarray, rng) -> np.ndarray:
        if not self.plan:
            obs = torch.tensor(np.array(self.hist[-OBS_H:]), dtype=torch.float32)[None]
            goal = torch.tensor(goal_to_dirt(sim, xy), dtype=torch.float32)[None]
            self.plan = list(self.model.sample(obs, goal)[0].numpy()[:EXEC])
        return xy + self.plan.pop(0)

    def observe(self, delta: np.ndarray) -> None:
        self.hist.append(delta)


class Zigzag:
    def __init__(self) -> None:
        pts, ys = [], np.arange(BOARD_Y[0] + 0.02, BOARD_Y[1], 0.04)
        for i, y in enumerate(ys):
            xs = BOARD_X if i % 2 == 0 else BOARD_X[::-1]
            pts += [(xs[0] + 0.01 * np.sign(xs[1] - xs[0]), y), (xs[1] - 0.01 * np.sign(xs[1] - xs[0]), y)]
        self.pts, self.i = np.array(pts), 0

    def __call__(self, sim, xy, rng) -> np.ndarray:
        tgt = self.pts[self.i % len(self.pts)]
        d = tgt - xy
        if np.linalg.norm(d) < STEP:
            self.i += 1
            return tgt
        return xy + d / np.linalg.norm(d) * STEP

    def observe(self, delta) -> None:
        pass


class RandomWalk:
    def __init__(self) -> None:
        self.v = np.zeros(2)

    def __call__(self, sim, xy, rng) -> np.ndarray:
        self.v = 0.8 * self.v + 0.2 * rng.normal(size=2)
        return xy + self.v / (np.linalg.norm(self.v) + 1e-9) * STEP

    def observe(self, delta) -> None:
        pass


class Greedy:
    """Ablation: the same hand-coded dirt goal, followed in a straight line (no learned motion)."""

    def __call__(self, sim, xy, rng) -> np.ndarray:
        g = goal_to_dirt(sim, xy)
        n = np.linalg.norm(g)
        return xy if n < 1e-9 else xy + g / n * STEP

    def observe(self, delta) -> None:
        pass


CONTROLLERS = {"zigzag": Zigzag, "random": RandomWalk, "greedy": Greedy, "bc": lambda: Learned("bc"),
               "diffusion": lambda: Learned("diffusion")}


def episode(sim: WipeSim, name: str, seed: int, frames: list | None = None) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ctrl = CONTROLLERS[name]()
    xy = sim.reset(dirt_pattern(sim, seed))
    curve, path = [], [xy.copy()]
    for k in range(int(SECONDS * RATE_HZ)):
        d = ctrl(sim, xy, rng) - xy
        n = np.linalg.norm(d)
        new = sim.step(xy + (d if n <= MAX_STEP_FACTOR * STEP else d / n * MAX_STEP_FACTOR * STEP))
        ctrl.observe(new - xy)
        xy = new
        path.append(xy.copy())
        curve.append(sim.cleaned())
        if frames is not None:
            frames.append(sim.render())
    return {"cleaned": curve[-1], "t90_s": next((i / RATE_HZ for i, c in enumerate(curve) if c >= 0.9), None),
            "curve": curve, "ik_fail": sim.ik_fail, "max_qd": sim.max_qd.tolist(),
            "path_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())}


def label(img: np.ndarray, title: str, pct: float, t: float) -> np.ndarray:
    img = np.ascontiguousarray(img[:, :, ::-1])
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(img, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{pct * 100:4.0f}% clean  t={t:4.1f}s", (img.shape[1] - 205, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 120), 1, cv2.LINE_AA)
    return img


def main() -> None:
    sim = WipeSim()
    sim.model.site_rgba[sim.model.site("handover").id, 3] = 0.0
    print(f"tool step {STEP * 1000:.1f} mm @ {RATE_HZ:.0f} Hz = {STEP * RATE_HZ * 100:.1f} cm/s")
    results = {}
    for name in CONTROLLERS:
        r = [episode(sim, name, s) for s in range(20)]
        c = np.array([x["cleaned"] for x in r])
        results[name] = {"cleaned_mean": float(c.mean()), "cleaned_std": float(c.std()),
                         "t90_s": [x["t90_s"] for x in r], "ik_fail": int(sum(x["ik_fail"] for x in r)),
                         "max_qd_rad_s": np.max([x["max_qd"] for x in r], 0).round(2).tolist(),
                         "path_m": float(np.mean([x["path_m"] for x in r]))}
        t90 = [x["t90_s"] for x in r]
        print(f"{name:10s} cleaned {c.mean() * 100:5.1f} ± {c.std() * 100:4.1f} %   reached 90% in {sum(t is not None for t in t90)}/20"
              f" (median {np.median([t if t is not None else np.inf for t in t90]):.1f} s)   path {results[name]['path_m']:.2f} m"
              f"   ik_fail {results[name]['ik_fail']}   max qd {results[name]['max_qd_rad_s']}")
    (OUT / "wipe_results.json").write_text(json.dumps(results, indent=1))

    seed, titles = 3, {"zigzag": "hard-coded zigzag", "random": "random walk",
                       "bc": "BC on Eidon (learned)", "diffusion": "diffusion on Eidon (learned)"}
    panels = {}
    for name in titles:
        frames: list = []
        r = episode(sim, name, seed, frames)
        panels[name] = [label(f, titles[name], c, i / RATE_HZ) for i, (f, c) in enumerate(zip(frames, r["curve"]))]
    h, w = sim.height, sim.width
    vw = cv2.VideoWriter(str(OUT / "wipe_compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), RATE_HZ * 2, (2 * w, 2 * h))
    for i in range(len(panels["zigzag"])):
        top = np.hstack([panels["zigzag"][i], panels["random"][i]])
        bot = np.hstack([panels["bc"][i], panels["diffusion"][i]])
        vw.write(np.vstack([top, bot]))
    vw.release()
    cv2.imwrite(str(OUT / "wipe_compare_last.png"), np.vstack([np.hstack([panels["zigzag"][-1], panels["random"][-1]]),
                                                            np.hstack([panels["bc"][-1], panels["diffusion"][-1]])]))
    sim.close()


if __name__ == "__main__":
    main()
