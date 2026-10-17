#!/usr/bin/env python3
"""
Slowly move every arm joint of both arms to 0 rad (arms hanging), then stop.

    python scripts/go_zero.py --backend real              # asks before moving
    python scripts/go_zero.py --backend real --speed 3    # even slower (deg/s)
    python scripts/go_zero.py --backend sim --render --yes  # rehearse in MuJoCo

Sequence: enable and hold the measured pose -> minimum-jerk move to zero at a
peak speed of --speed deg/s (grippers keep their current target) -> report the
remaining error -> stop (move to rest_qpos, then disable the motors).

Joints whose limits exclude 0 (joint 4: [0, 2.44] rad) stop at the limit minus
the safety margin, so they end ~0.02 rad from zero. Ctrl+C stops safely.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import ARM_DOF, SIDES  # noqa: E402
from scripts.jog_joints import confirm, make_robot  # noqa: E402

# Peak speed of a minimum-jerk profile is 1.875x its average speed.
MIN_JERK_PEAK_FACTOR = 1.875


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--speed", type=float, default=5.0, help="peak joint speed (deg/s)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="never move faster than this (s)")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    robot = make_robot(cfg, args.backend, args.render)
    robot.start()
    try:
        st = robot.get_state()
        zero = {s: np.zeros(ARM_DOF) for s in SIDES}
        dist = max(float(np.max(np.abs(st.arms[s].q))) for s in SIDES)
        duration = max(args.min_duration, MIN_JERK_PEAK_FACTOR * np.degrees(dist) / args.speed)
        for s in SIDES:
            print(f"{s:>5} now: " + " ".join(f"{q:+.3f}" for q in st.arms[s].q))
        if not confirm(f"Move to zero (largest offset {np.degrees(dist):.1f} deg) over {duration:.1f} s?", args.yes):
            return 0

        robot.move_joints(zero, duration=duration)
        robot.advance(1.0)  # let the motors settle

        st = robot.get_state()
        for s in SIDES:
            print(f"{s:>5} end: " + " ".join(f"{q:+.3f}" for q in st.arms[s].q)
                  + f"   max |q| {np.degrees(np.max(np.abs(st.arms[s].q))):.1f} deg")
        if st.fault:
            print(f"FAULT: {st.fault}")
            return 1
    finally:
        robot.stop()  # rest_qpos is all zeros: no further motion, then disable
    return 0


if __name__ == "__main__":
    sys.exit(main())
