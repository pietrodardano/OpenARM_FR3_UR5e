#!/usr/bin/env python3
"""
Gripper-only test: close both grippers, then open/close left, then open/close right.
The arms hold their current pose the whole time; left and right never move together.

    python scripts/gripper_test.py --backend real              # asks before each step
    python scripts/gripper_test.py --backend real --open-mm 20 # partial stroke
    python scripts/gripper_test.py --backend sim --render --yes

How a gripper action is defined (src/control/robot_interface.py, src/hardware/openarm_bimanual.py):
  - ABSOLUTE position, in metres: the slide of one finger (MJCF openarm_<side>_finger_joint1).
    0.0 = closed, 0.044 = fully open. Both fingers move this much (mirrored), so the jaw gap
    changes by about 2x the command.
  - Not a percentage, not a force. percent = cmd / 0.044 * 100 if you want one.
  - On the motor (DM4310, id 0x08) it is a position too: motor_rad = cmd / 0.044 * (-1.0472),
    i.e. 0 rad = closed, -1.0472 rad (-60 deg) = open, tracked by MIT impedance
    (kp 5, kd 0.1 from configs/task_config.yaml). Low kp: the gripper is compliant and squeezes
    an object with a force roughly proportional to (cmd - actual opening).
  - The safety filter clips to [0, 0.044] and rate-limits to 0.022 m/s (full stroke in 2 s).

Sequence (scripts/safe_zero.py: Ctrl+C at any point = go to zero, then disable):
1. Enable, hold the measured pose, move the arms to zero (grippers untouched).
2. Left gripper -> 0, then right gripper -> 0.
3. Left: open to --open-mm, hold, close.   4. Right: open, hold, close.
5. Go to zero, disable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.joint_position_controller import JointPositionController  # noqa: E402
from src.control.robot_interface import GRIPPER_OPEN_M, RobotBackend, Side  # noqa: E402
from scripts.jog_joints import confirm, make_robot  # noqa: E402
from scripts.safe_zero import run_with_zero  # noqa: E402

RATE_HZ = 50.0
PRINT_EVERY_S = 0.25


def move_gripper(robot: RobotBackend, jc: JointPositionController, side: Side, goal_m: float,
                 v_max: float, motor_open_rad: float, settle_s: float) -> None:
    """Commands one gripper (step target; the safety filter ramps it) and prints a trace."""
    other: Side = "right" if side == "left" else "left"
    st = robot.get_state()
    start_m = st.arms[side].gripper
    other0 = st.arms[other].gripper
    duration = abs(goal_m - st.arms[side].gripper_setpoint) / v_max + settle_s
    print(f"\n  {side} gripper: {start_m * 1e3:.1f} -> {goal_m * 1e3:.1f} mm "
          f"({goal_m / GRIPPER_OPEN_M * 100:.0f} %), {duration:.1f} s")
    print("      t[s]  target[mm]  setpoint[mm]  measured[mm]  measured[%]  motor[rad]")
    jc.command(gripper_m={side: goal_m})
    t0 = time.monotonic()
    next_print = 0.0
    reached_at = None
    other_dev = 0.0
    for _ in range(int(duration * RATE_HZ)):
        robot.advance(1.0 / RATE_HZ)
        st = robot.get_state()
        if st.fault:
            raise RuntimeError(st.fault)
        t = time.monotonic() - t0
        g, sp = st.arms[side].gripper, st.arms[side].gripper_setpoint
        other_dev = max(other_dev, abs(st.arms[other].gripper - other0))
        if reached_at is None and abs(g - goal_m) < 1e-3:
            reached_at = t
        if t >= next_print:
            next_print += PRINT_EVERY_S
            print(f"    {t:6.2f}  {goal_m * 1e3:10.1f}  {sp * 1e3:12.1f}  {g * 1e3:12.1f}  "
                  f"{g / GRIPPER_OPEN_M * 100:10.0f}  {g / GRIPPER_OPEN_M * motor_open_rad:+10.3f}")
    err = st.arms[side].gripper - goal_m
    print(f"    -> final {st.arms[side].gripper * 1e3:.1f} mm, error {err * 1e3:+.1f} mm, "
          f"within 1 mm after {'%.2f s' % reached_at if reached_at is not None else 'NEVER'}; "
          f"{other} gripper moved {other_dev * 1e3:.1f} mm meanwhile, "
          f"motor status {hex(st.arms[side].motor_errors[-1]) if st.arms[side].motor_errors else 'n/a'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--open-mm", type=float, default=44.0, help="opening to test, 0..44 mm per finger")
    parser.add_argument("--hold", type=float, default=1.0, help="hold time at each end (s)")
    parser.add_argument("--joint-speed", type=float, default=10.0, help="peak joint speed for the zero moves (deg/s)")
    parser.add_argument("--yes", action="store_true", help="do not ask before each step")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    hw = cfg["hardware"]
    v_max = hw["max_gripper_velocity_m_s"]
    motor_open_rad = hw["gripper"]["motor_open_rad"]
    open_m = float(np.clip(args.open_mm / 1e3, 0.0, GRIPPER_OPEN_M))

    robot = make_robot(cfg, args.backend, args.render)

    def test() -> None:
        st = robot.get_state()
        print("Gripper command = absolute finger slide in m (0 = closed, 0.044 = open), "
              f"rate limit {v_max * 1e3:.0f} mm/s, motor 0 .. {motor_open_rad:+.4f} rad")
        for s in robot.active_sides:
            print(f"  {s:>5}: measured {st.arms[s].gripper * 1e3:5.1f} mm "
                  f"({st.arms[s].gripper / GRIPPER_OPEN_M * 100:.0f} %)")
        jc = JointPositionController(robot)
        jc.reset()  # arms keep their held targets: only gripper entries change below
        steps = [
            ("[1/3] Close both grippers (left, then right)", [("left", 0.0), ("right", 0.0)]),
            (f"[2/3] Left gripper: open to {open_m * 1e3:.0f} mm, then close", [("left", open_m), ("left", 0.0)]),
            (f"[3/3] Right gripper: open to {open_m * 1e3:.0f} mm, then close", [("right", open_m), ("right", 0.0)]),
        ]
        for title, moves in steps:
            moves = [(side, goal) for side, goal in moves if side in robot.active_sides]
            if not moves or not confirm(f"\n{title}?", args.yes):
                continue
            for side, goal in moves:
                move_gripper(robot, jc, side, goal, v_max, motor_open_rad, settle_s=args.hold)
        print("\nTest done.")

    # Arms go to zero first, grippers stay put: step 1 closes them one at a time.
    return run_with_zero(robot, cfg, test, args.joint_speed, zero_grippers_first=False,
                         confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
