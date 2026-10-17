#!/usr/bin/env python3
"""
Find a grip torque that holds an object without crushing it (gripper torque mode).

    python scripts/grip_torque_test.py --backend real                       # left arm, 0.1 -> 1.0 Nm
    python scripts/grip_torque_test.py --backend real --start 0.05 --stop 0.6 --step 0.05
    python scripts/grip_torque_test.py --backend sim --render --yes          # rehearse (no object in sim)

Torque mode = MIT with kp = 0: the gripper motor's current loop outputs the commanded
torque (Nm, + closes). Rough squeeze force on the object, ideal linkage without friction:
    F [N] ~ torque / (2 * 0.042 m/rad) ~ 12 N per Nm
(both fingers move 0.044 m per 1.047 rad of motor). Linkage friction makes the real force
lower and the measured torque the better number to report.

Sequence (scripts/safe_zero.py: Ctrl+C at any point = release, go to zero, then disable):
1. Go to zero, then to the low pose of compare_weights.py (elbow --low-elbow-deg), gripper open.
2. You hold the object between the fingers, press Enter: the gripper closes at --start Nm
   and the contact opening is recorded.
3. The torque steps up to --stop. At each step: measured torque, opening, and how far the
   fingers closed since contact. Closing more than --crush-mm => the object is deforming:
   the torque goes back one step and the sweep stops.
4. You let go: the elbow lifts to 90 deg (robot.home_qpos), holds for --hold-s (watch for
   slip), and comes back down.
5. Release, gripper opens.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import GRIPPER_OPEN_M, RobotBackend, Side, arm_slice  # noqa: E402
from scripts.jog_joints import active_sides, confirm, make_robot  # noqa: E402
from scripts.safe_zero import MIN_JERK_PEAK_FACTOR, run_with_zero  # noqa: E402

RATE_HZ = 50.0


def squeeze_n(torque_nm: float, cfg: dict) -> float:
    """Ideal squeeze force (N) for a gripper motor torque: no linkage friction."""
    m_per_rad = GRIPPER_OPEN_M / abs(cfg["hardware"]["gripper"]["motor_open_rad"])
    return torque_nm / (2 * m_per_rad)


def move_to_pose(robot: RobotBackend, side: Side, q: np.ndarray, speed_deg_s: float,
                 gripper_m: Optional[float] = None) -> None:
    """Minimum-jerk joint move of one arm at ``speed_deg_s`` peak (optionally sets the gripper too)."""
    q0 = robot.get_joint_targets()[arm_slice(side)]
    dist = float(np.degrees(np.max(np.abs(q - q0))))
    duration = max(3.0, MIN_JERK_PEAK_FACTOR * dist / speed_deg_s)
    print(f"Moving {side} arm ({dist:.0f} deg max, {duration:.1f} s)...")
    robot.move_joints({side: q}, gripper=None if gripper_m is None else {side: gripper_m}, duration=duration)
    robot.advance(0.5)


def wait_gripper_settled(robot: RobotBackend, side: Side, timeout_s: float = 6.0,
                         still_mm_s: float = 0.5, still_s: float = 0.3) -> float:
    """Waits until the opening stops changing; returns it (m)."""
    t_end = time.monotonic() + timeout_s
    prev = robot.get_state().arms[side].gripper
    still = 0.0
    while time.monotonic() < t_end:
        robot.advance(1.0 / RATE_HZ)
        st = robot.get_state()
        g = st.arms[side].gripper
        if abs(g - prev) * RATE_HZ * 1e3 < still_mm_s:
            still += 1.0 / RATE_HZ
            if still >= still_s:
                return g
        else:
            still = 0.0
        prev = g
    return robot.get_state().arms[side].gripper


def approach(robot: RobotBackend, side: Side, cfg: dict, timeout_s: float = 8.0) -> str:
    """Closes the fingers in torque mode until they stop on something.

    Linkage friction makes a gentle grip torque close very slowly, so the torque ramps up
    until the fingers close at ``approach_speed_mm_s`` and is then regulated around that
    speed (capped at ``approach_max_torque_nm``). Returns "contact", "closed" (nothing
    between the fingers) or "timeout". The torque is left at the approach value.
    """
    g_cfg = cfg["hardware"]["gripper"]
    v_ref = g_cfg.get("approach_speed_mm_s", 15.0) / 1e3
    tau_max = g_cfg.get("approach_max_torque_nm", 0.6)
    ramp = g_cfg.get("approach_ramp_nm_s", 1.5) / RATE_HZ
    dt = 1.0 / RATE_HZ
    g0 = g_prev = robot.get_state().arms[side].gripper
    tau, v, slow_s, stuck_s, moved = 0.0, 0.0, 0.0, 0.0, False
    robot.set_gripper_torque({side: tau})
    for _ in range(int(timeout_s * RATE_HZ)):
        robot.advance(dt)
        st = robot.get_state()
        if st.fault:
            raise RuntimeError(st.fault)
        g = st.arms[side].gripper
        v = 0.7 * v + 0.3 * (g_prev - g) / dt  # closing speed, m/s (filtered)
        g_prev = g
        if g < 0.5e-3:
            return "closed"
        moved = moved or g0 - g > 2e-3
        # Contact: was moving, now (nearly) stopped. Or never got going at full approach torque
        # (the object was already touching the fingers).
        slow_s = slow_s + dt if moved and v < 0.3 * v_ref else 0.0
        stuck_s = stuck_s + dt if not moved and tau >= tau_max else 0.0
        if slow_s >= 0.1 or stuck_s >= 0.3:
            return "contact"
        tau = min(tau_max, tau + ramp) if v < v_ref else max(0.0, tau - ramp)
        robot.set_gripper_torque({side: tau})
    return "timeout"


def grip(robot: RobotBackend, side: Side, torque_nm: float, cfg: dict) -> float:
    """Closes on an object (fast approach, then ``torque_nm``); returns the settled opening (m)."""
    t0 = time.monotonic()
    result = approach(robot, side, cfg)
    print(f"  approach: {result} after {time.monotonic() - t0:.1f} s at "
          f"{robot.get_state().arms[side].gripper * 1e3:.1f} mm")
    robot.set_gripper_torque({side: torque_nm})
    g = wait_gripper_settled(robot, side)
    arm = robot.get_state().arms[side]
    print(f"  gripping at {torque_nm:.2f} Nm (~{squeeze_n(torque_nm, cfg):.0f} N ideal): opening {g * 1e3:.1f} mm "
          f"(jaw ~{2 * g * 1e3:.0f} mm), measured torque {arm.gripper_tau:+.3f} Nm")
    if g < 1e-3:
        print("  WARNING: gripper is (almost) fully closed: no object between the fingers?")
    return g


def release(robot: RobotBackend, side: Side, open_m: float) -> None:
    """Torque mode off (holds the measured opening, no squeeze), then opens to ``open_m``."""
    robot.set_gripper_torque({side: None})
    robot.advance(0.1)
    robot.set_joint_targets(gripper={side: open_m})
    robot.advance(abs(open_m - robot.get_state().arms[side].gripper) / 0.022 + 0.5)


def hold_report(robot: RobotBackend, side: Side, seconds: float, ref_m: float) -> float:
    """Holds and returns how far (m) the fingers closed beyond ``ref_m`` (creep / crushing / slip)."""
    worst = 0.0
    for _ in range(int(seconds * RATE_HZ)):
        robot.advance(1.0 / RATE_HZ)
        st = robot.get_state()
        if st.fault:
            raise RuntimeError(st.fault)
        worst = max(worst, ref_m - st.arms[side].gripper)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--start", type=float, default=0.1, help="first grip torque (Nm)")
    parser.add_argument("--stop", type=float, default=1.0, help="last grip torque (Nm), capped by gripper.max_torque_nm")
    parser.add_argument("--step", type=float, default=0.05, help="torque step (Nm)")
    parser.add_argument("--step-s", type=float, default=1.5, help="hold per step (s)")
    parser.add_argument("--crush-mm", type=float, default=1.0, help="closing beyond contact that counts as deforming (mm)")
    parser.add_argument("--hold-s", type=float, default=5.0, help="hold after you let go (s)")
    parser.add_argument("--open-mm", type=float, default=44.0, help="opening before the grasp (mm per finger)")
    parser.add_argument("--low-elbow-deg", type=float, default=30.0, help="elbow angle of the low pose")
    parser.add_argument("--joint-speed", type=float, default=15.0, help="peak joint speed (deg/s)")
    parser.add_argument("--yes", action="store_true", help="do not ask (sim rehearsal)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    robot = make_robot(cfg, args.backend, args.render)
    side: Side = active_sides(robot, args.side)[0]
    lift = np.asarray(cfg["robot"]["home_qpos"][side], dtype=np.float64)  # elbow 90 deg, forearm forward
    low = np.zeros(7)
    low[3] = np.radians(args.low_elbow_deg)
    open_m = float(np.clip(args.open_mm / 1e3, 0.0, GRIPPER_OPEN_M))
    stop = min(args.stop, cfg["hardware"]["gripper"].get("max_torque_nm", 1.0))
    torques = np.round(np.arange(args.start, stop + 1e-9, args.step), 4)

    def test() -> None:
        move_to_pose(robot, side, low, args.joint_speed, gripper_m=open_m)
        if not confirm(f"\nHold the object between the {side} fingers. Close at {args.start:.2f} Nm?", args.yes):
            return
        contact = grip(robot, side, float(torques[0]), cfg)
        print("\n  torque[Nm]  ~force[N]  measured[Nm]  opening[mm]  closed since contact[mm]")
        last_ok = float(torques[0])
        for tau in torques:
            robot.set_gripper_torque({side: float(tau)})
            closed = hold_report(robot, side, args.step_s, contact)
            arm = robot.get_state().arms[side]
            flag = "  <-- DEFORMING" if closed * 1e3 > args.crush_mm else ""
            print(f"  {tau:9.2f}  {squeeze_n(tau, cfg):9.1f}  {arm.gripper_tau:+12.3f}  {arm.gripper * 1e3:11.1f}  "
                  f"{closed * 1e3:+24.2f}{flag}")
            if flag:
                print(f"  Deformation onset between {last_ok:.2f} and {tau:.2f} Nm: back to {last_ok:.2f} Nm.")
                robot.set_gripper_torque({side: last_ok})
                break
            last_ok = float(tau)
        if confirm(f"\nLet go of the object (gripper holds it at {last_ok:.2f} Nm). Lift to elbow 90 deg?", args.yes):
            ref = robot.get_state().arms[side].gripper
            move_to_pose(robot, side, lift, args.joint_speed)
            closed = hold_report(robot, side, args.hold_s, ref)
            print(f"  held {args.hold_s:.0f} s at {last_ok:.2f} Nm: fingers closed a further {closed * 1e3:.2f} mm "
                  "(check visually for slip)")
            move_to_pose(robot, side, low, args.joint_speed)
        release(robot, side, open_m)
        print("\nTest done.")

    return run_with_zero(robot, cfg, test, args.joint_speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
