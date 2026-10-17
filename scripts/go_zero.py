#!/usr/bin/env python3
"""
Slowly move every arm joint of both arms to 0 rad (arms hanging), then stop.

    python scripts/go_zero.py --backend real              # asks before moving
    python scripts/go_zero.py --backend real --speed 3    # even slower (deg/s)
    python scripts/go_zero.py --backend sim --render --yes  # rehearse in MuJoCo

Sequence: enable and hold the measured pose -> minimum-jerk move to zero at a
peak speed of --speed deg/s (grippers close) -> report the remaining error ->
disable the motors (scripts/safe_zero.py; Ctrl+C keeps going to zero).

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

from src.control.robot_interface import SIDES  # noqa: E402
from scripts.jog_joints import confirm, make_robot  # noqa: E402
from scripts.safe_zero import run_with_zero  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--speed", type=float, default=5.0, help="peak joint speed (deg/s)")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    robot = make_robot(cfg, args.backend, args.render)

    def report() -> None:
        st = robot.get_state()
        for s in SIDES:
            print(f"{s:>5} end: " + " ".join(f"{q:+.3f}" for q in st.arms[s].q)
                  + f"   max |q| {np.degrees(np.max(np.abs(st.arms[s].q))):.1f} deg")

    # run_with_zero: confirm -> go to zero -> report -> (already at zero) -> disable.
    return run_with_zero(robot, cfg, report, args.speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
