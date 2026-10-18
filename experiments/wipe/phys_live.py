"""Watch the physical pipeline live in the MuJoCo viewer: hard-coded grasp of the sponge, then a
force-controlled wipe by the chosen controller (sim only, full contact physics).

    .venv/bin/python experiments/wipe/phys_live.py diffusion
    .venv/bin/python experiments/wipe/phys_live.py zigzag --seed 7 --stubborn
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer
import numpy as np

import phys_run
from physim import PhysWipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("controller", nargs="?", default="diffusion", choices=list(phys_run.CONTROLLERS))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--stubborn", action="store_true", help="small stains that need 3x the scrubbing")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = ap.parse_args()

    sim = PhysWipe()
    step = sim.step
    with mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:], v.cam.distance = sim.cam.lookat, sim.cam.distance
        v.cam.azimuth, v.cam.elevation = sim.cam.azimuth, sim.cam.elevation
        wall0, sim_t0, last = time.monotonic(), 0.0, [0.0]

        def paced_step(n: int = 1, f_des: float = phys_run.F_DES) -> None:
            if not v.is_running():
                raise SystemExit
            with v.lock():
                step(n, f_des)
                sim.model.geom_rgba[sim.dirt_gid, 3] = np.clip(sim.dirt, 0, 1) * 0.95
            if sim.t - last[0] >= 1 / 60:
                v.sync()
                last[0] = sim.t
                lag = (sim.t - sim_t0) / a.speed - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if int(sim.t * 1000) % 500 == 0:
                print(f"\rt={sim.t:5.1f}s  contact {sim.contact_force():5.1f} N  clean {sim.cleaned() * 100:5.1f}%", end="")

        sim.step = paced_step
        r = phys_run.episode(sim, a.controller, a.seed, stubborn=a.stubborn)
        print(f"\n{a.controller}: {r['cleaned'] * 100:.1f}% cleaned, contact force {r['force_mean']:.1f} N, "
              f"sponge slid {r['sponge_path_m']:.2f} m. Close the window to exit.")
        while v.is_running():
            v.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
