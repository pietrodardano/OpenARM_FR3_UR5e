"""Watch one controller wipe live in the MuJoCo viewer (sim only).

    .venv/bin/python experiments/wipe/live.py diffusion --seed 3
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer
import numpy as np
import torch

import run
from wipesim import RATE_HZ, WipeSim, dirt_pattern


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("controller", nargs="?", default="diffusion", choices=list(run.CONTROLLERS))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    a = ap.parse_args()

    sim = WipeSim()
    sim.model.site_rgba[sim.model.site("handover").id, 3] = 0.0
    with mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:], v.cam.distance = sim.cam.lookat, sim.cam.distance
        v.cam.azimuth, v.cam.elevation = sim.cam.azimuth, sim.cam.elevation
        seed = a.seed
        while v.is_running():  # loops over new dirt layouts until the window is closed
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            ctrl = run.CONTROLLERS[a.controller]()
            xy = sim.reset(dirt_pattern(sim, seed))
            cap = run.MAX_STEP_FACTOR * run.STEP
            for k in range(int(run.SECONDS * RATE_HZ)):
                if not v.is_running():
                    break
                t0 = time.monotonic()
                d = ctrl(sim, xy, rng) - xy
                n = np.linalg.norm(d)
                with v.lock():
                    new = sim.step(xy + (d if n <= cap else d / n * cap))
                    sim.model.geom_rgba[sim.dirt_gid, 3] = np.clip(sim.dirt, 0, 1) * 0.95
                ctrl.observe(new - xy)
                xy = new
                v.sync()
                if k % 12 == 0:
                    print(f"\r{a.controller} seed {seed}  t={k / RATE_HZ:4.1f}s  clean {sim.cleaned() * 100:5.1f}%", end="")
                time.sleep(max(0.0, 1 / (RATE_HZ * a.speed) - (time.monotonic() - t0)))
            print()
            time.sleep(1.5)
            seed += 1


if __name__ == "__main__":
    main()
