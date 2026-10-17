#!/usr/bin/env python3
"""
OpenArm bring-up check. Reads the active arms (hardware.active_sides) WITHOUT enabling or moving anything.

Prints every joint's measured position next to the official limits, the
gripper opening and the motor status codes. Run this first after power-on and
after any zero-position calibration:

    python scripts/check_robot.py
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import ARM_DOF, arm_slice  # noqa: E402
from src.hardware import OpenArmBimanual  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    robot = OpenArmBimanual.from_config(cfg)
    state = robot.read_passive()
    tol = cfg["hardware"]["startup_limit_tolerance_rad"]
    ok = True
    for side in robot.active_sides:
        arm = state.arms[side]
        lo, hi = robot.limits.lower[arm_slice(side)], robot.limits.upper[arm_slice(side)]
        print(f"\n{side} arm ({cfg['hardware']['arms'][side]['can_interface']})")
        for i in range(ARM_DOF):
            inside = lo[i] - tol <= arm.q[i] <= hi[i] + tol
            ok &= inside
            print(f"  joint{i + 1}: {arm.q[i]:+.3f} rad  limits [{lo[i]:+.3f}, {hi[i]:+.3f}]  "
                  f"status {arm.motor_errors[i]:#x}  {'' if inside else '<-- OUTSIDE LIMITS'}")
        print(f"  gripper: {arm.gripper * 1000:.1f} mm open  status {arm.motor_errors[ARM_DOF]:#x}")
        faults = [c for c in arm.motor_errors if c >= 0x8]
        ok &= not faults
    print("\nOK: ready to stream." if ok else "\nNOT OK: fix motor faults / zero calibration before streaming.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
