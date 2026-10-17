#!/usr/bin/env python3
"""
Manual grip-strength ramp at the lifted pose: each Enter adds one torque step. Mark where the
object stops slipping down and where it gives way.

    python scripts/crush_test.py --backend real                         # 0.1 Nm steps up to 1.0 Nm
    python scripts/crush_test.py --backend real --start 0.3 --step 0.1 --max-torque 2.5 --label empty_can
    python scripts/crush_test.py --backend sim --yes                    # rehearse (no object in sim)

Squeeze force ~ 12 N per Nm of gripper motor torque (ideal linkage, friction makes it lower).
The driver cap gripper.max_torque_nm (1.0) is raised for this run only by --max-torque
(at most 3.0 Nm, about the DM4310 rated torque). Keep fingers away from the jaws.

Sequence (scripts/safe_zero.py: Ctrl+C at any point = release, go to zero, then disable):
1. Go to zero, then to the lift pose of compare_weights.py (robot.home_qpos: elbow 90 deg,
   forearm forward, fingers horizontal): gravity pulls along the object's vertical axis, so
   a weak grip shows up as vertical slip. Gripper open.
2. Hold the object between the fingers, Enter: fast approach, then grip at --start Nm.
   Let go of it to watch for slip (keep a hand under it at low torque).
3. Prompt per step:
     Enter = +step   d = -step
     s = it slips at this torque (recorded; the minimum holding torque is above the last one)
     b = it broke (records the torque)   q = stop
   After each change the opening settles and is printed with how far the fingers closed
   since contact (a jump there = the wall buckled).
4. Support the object, Enter: release, open. The run is appended to --history.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import GRIPPER_OPEN_M, Side  # noqa: E402
from scripts.grip_torque_test import grip, move_to_pose, release, squeeze_n, wait_gripper_settled  # noqa: E402
from scripts.jog_joints import active_sides, confirm, make_robot  # noqa: E402
from scripts.safe_zero import run_with_zero  # noqa: E402

HARD_MAX_TORQUE_NM = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--label", default="object", help="name stored in the history")
    parser.add_argument("--start", type=float, default=0.1, help="first grip torque (Nm)")
    parser.add_argument("--step", type=float, default=0.1, help="torque added per Enter (Nm)")
    parser.add_argument("--max-torque", type=float, default=None,
                        help=f"torque cap for this run (Nm, default gripper.max_torque_nm, at most {HARD_MAX_TORQUE_NM})")
    parser.add_argument("--settle", type=float, default=0.5, help="minimum wait after each step (s)")
    parser.add_argument("--open-mm", type=float, default=44.0, help="opening before the grasp (mm per finger)")
    parser.add_argument("--joint-speed", type=float, default=15.0, help="peak joint speed (deg/s)")
    parser.add_argument("--history", default="logs/crush_history.jsonl")
    parser.add_argument("--yes", action="store_true", help="do not ask: 5 steps, no break (sim rehearsal)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    g_cfg = cfg["hardware"]["gripper"]
    if args.max_torque is not None:
        g_cfg["max_torque_nm"] = float(min(args.max_torque, HARD_MAX_TORQUE_NM))
    cap = g_cfg.get("max_torque_nm", 1.0)
    robot = make_robot(cfg, args.backend, args.render)
    side: Side = active_sides(robot, args.side)[0]
    lift = np.asarray(cfg["robot"]["home_qpos"][side], dtype=np.float64)  # elbow 90 deg, forearm forward
    open_m = float(np.clip(args.open_mm / 1e3, 0.0, GRIPPER_OPEN_M))
    history = Path(args.history)
    steps = []
    result = {"broke_at_nm": None, "last_intact_nm": None, "slipped_at_nm": []}

    def settle_and_print(tau: float, contact: float) -> dict:
        robot.advance(args.settle)
        g = wait_gripper_settled(robot, side, timeout_s=3.0)
        arm = robot.get_state().arms[side]
        row = {"torque_nm": tau, "measured_nm": arm.gripper_tau, "opening_mm": g * 1e3,
               "closed_since_contact_mm": (contact - g) * 1e3}
        print(f"  {tau:6.2f} Nm (~{squeeze_n(tau, cfg):4.0f} N)  measured {arm.gripper_tau:+.3f} Nm  "
              f"opening {g * 1e3:5.1f} mm  closed since contact {row['closed_since_contact_mm']:+5.2f} mm")
        return row

    def test() -> None:
        move_to_pose(robot, side, lift, args.joint_speed, gripper_m=open_m)
        print(f"\nTorque cap for this run: {cap:.2f} Nm (~{squeeze_n(cap, cfg):.0f} N ideal). Keep fingers clear of the jaws.")
        if not confirm(f"Hold the {args.label} between the {side} fingers. Grip at {args.start:.2f} Nm?", args.yes):
            return
        tau = float(min(args.start, cap))
        contact = grip(robot, side, tau, cfg)
        steps.append(settle_and_print(tau, contact))
        n_auto = 0
        while True:
            if args.yes:
                ans = "" if n_auto < 5 else "q"
                n_auto += 1
            else:
                ans = input(f"  [Enter] +{args.step:.2f} Nm   [d] -{args.step:.2f}   [s] slips   [b] broke   [q] stop  > ").strip().lower()
            if ans == "s":
                result["slipped_at_nm"].append(tau)
                steps[-1]["slipped"] = True
                print(f"  slips at {tau:.2f} Nm (~{squeeze_n(tau, cfg):.0f} N ideal)")
                continue
            if ans == "b":
                result["broke_at_nm"] = tau
                result["last_intact_nm"] = steps[-2]["torque_nm"] if len(steps) > 1 else None
                print(f"  BROKE at {tau:.2f} Nm (~{squeeze_n(tau, cfg):.0f} N ideal); last intact step "
                      f"{result['last_intact_nm'] if result['last_intact_nm'] is not None else 'none'} Nm")
                break
            if ans == "q":
                break
            new = tau - args.step if ans == "d" else tau + args.step
            new = float(np.clip(round(new, 4), 0.0, cap))
            if new == tau:
                print(f"  at the {'cap' if ans != 'd' else 'minimum'} ({tau:.2f} Nm)" + (" (raise --max-torque)" if ans != "d" else ""))
                continue
            tau = new
            robot.set_gripper_torque({side: tau})
            steps.append(settle_and_print(tau, contact))
        if result["slipped_at_nm"]:
            print(f"  highest torque that slipped: {max(result['slipped_at_nm']):.2f} Nm -> hold needs more than that")
        confirm(f"\nSupport the {args.label}. Release and open?", args.yes)
        release(robot, side, open_m)
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": args.label, "side": side,
                                "contact_opening_mm": contact * 1e3, **result, "steps": steps}) + "\n")
        print(f"  registered in {history}")
        print("\nTest done.")

    return run_with_zero(robot, cfg, test, args.joint_speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
