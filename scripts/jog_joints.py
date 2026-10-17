#!/usr/bin/env python3
"""
Gentle first-motion test: jog each joint by a few degrees, one at a time.

Run scripts/check_robot.py (read-only) first. Then:

    python scripts/jog_joints.py --backend real              # +-3 deg at 2 deg/s, Enter before each joint
    python scripts/jog_joints.py --backend real --side right --joints 5,6,7
    python scripts/jog_joints.py --backend sim --render --yes  # rehearse in MuJoCo

Sequence (scripts/safe_zero.py: Ctrl+C at any point = go to zero, then disable):
0. Enable, hold the measured pose, go to zero (all joints 0, grippers closed).
1. Hold zero for 3 s (no motion). Report hold error and motor status.
2. Plan every (arm, joint, +/-) jog from zero. A direction is skipped if it
   - leaves the official joint limits (minus the safety margin), or
   - brings the arm closer to the torso or the other arm (official collision meshes),
     e.g. joint 2 inward from the hanging pose.
3. For each planned jog: minimum-jerk out, short hold, back, short hold. Report the
   measured motion, the tracking error, and whether the joint moved in the commanded
   direction (this checks the sign convention against the model).
4. Optionally open/close the gripper a few mm.
5. Go to zero, disable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.joint_position_controller import JointPositionController  # noqa: E402
from src.control.robot_interface import (  # noqa: E402
    GRIPPER_OPEN_M,
    SIDES,
    RobotBackend,
    Side,
    arm_slice,
    gripper_index,
    minimum_jerk,
)
from src.planning.openarm_kinematics import OpenArmKinematics  # noqa: E402
from scripts.safe_zero import run_with_zero  # noqa: E402

RATE_HZ = 100.0
Jog = Tuple[Side, int, int]  # (side, joint index 0..6, direction +1/-1)


def make_robot(cfg: dict, backend: str, render: bool) -> RobotBackend:
    if backend in ("real", "fake"):
        from src.hardware import FakeOpenArmBus, OpenArmBimanual, OpenArmBus

        dt = 1.0 / cfg["hardware"]["control_rate_hz"]
        bus = OpenArmBus if backend == "real" else (lambda _cfg: FakeOpenArmBus(dt=dt))
        return OpenArmBimanual.from_config(cfg, bus_factory=bus)
    from src.simulation.openarm_sim import OpenArmSim

    return OpenArmSim.from_config(cfg, realtime=True, render=render)


def active_sides(robot: RobotBackend, choice: str) -> List[Side]:
    """``--side`` choice ("left" / "right" / "both") restricted to the arms the backend drives."""
    wanted = list(SIDES) if choice == "both" else [choice]
    sides = [s for s in wanted if s in robot.active_sides]
    if not sides:
        raise SystemExit(f"--side {choice}: arm not active (hardware.active_sides = {list(robot.active_sides)})")
    return sides


def confirm(msg: str, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    return input(f"{msg} [Enter = yes, n = no] ").strip().lower() in ("", "y", "yes")


def hold_check(robot: RobotBackend, seconds: float) -> None:
    """Holds the current pose and reports the worst setpoint-vs-measured error."""
    worst = {s: 0.0 for s in robot.active_sides}
    for _ in range(int(seconds * RATE_HZ)):
        robot.advance(1.0 / RATE_HZ)
        st = robot.get_state()
        for s in robot.active_sides:
            worst[s] = max(worst[s], float(np.max(np.abs(st.arms[s].q_setpoint - st.arms[s].q))))
        if st.fault:
            raise RuntimeError(st.fault)
    st = robot.get_state()
    for s in robot.active_sides:
        print(f"  {s}: max hold error {np.degrees(worst[s]):.2f} deg, motor status {[hex(c) for c in st.arms[s].motor_errors]}")


def plan_jogs(
    kin: OpenArmKinematics, q_start: Dict[Side, np.ndarray], sides: List[Side], joints: List[int],
    delta: float, margin: float, clearance_tol: float, min_clearance: float,
) -> List[Jog]:
    plan: List[Jog] = []
    print(f"\nPlan (+-{np.degrees(delta):.1f} deg from the current pose):")
    for side in sides:
        lo, hi = kin.limits.lower[arm_slice(side)], kin.limits.upper[arm_slice(side)]
        base = kin.clearance(side, q_start)
        print(f"  {side} arm: clearance to torso/other arm {base * 1000:.0f} mm")
        for j in joints:
            notes = []
            for sign in (+1, -1):
                q = {s: v.copy() for s, v in q_start.items()}
                q[side][j] += sign * delta
                tag = f"{'+' if sign > 0 else '-'}"
                if not (lo[j] + margin <= q[side][j] <= hi[j] - margin):
                    notes.append(f"{tag} skip (joint limit)")
                    continue
                c = kin.clearance(side, q)
                if c < base - clearance_tol or c < min_clearance:
                    notes.append(f"{tag} skip (inward: clearance {base * 1000:.0f} -> {c * 1000:.0f} mm)")
                    continue
                notes.append(f"{tag} ok")
                plan.append((side, j, sign))
            print(f"    joint{j + 1}: " + ", ".join(notes))
    return plan


def run_jog(robot: RobotBackend, jc: JointPositionController, jog: Jog, delta: float, speed: float) -> None:
    side, j, sign = jog
    q0 = jc.q_cmd[arm_slice(side)].copy()
    meas0 = robot.get_state().arms[side].q[j]
    t_move = max(0.5, delta / speed)
    trace_meas, trace_err = [], []

    def go(frm: float, to: float, duration: float) -> None:
        steps = int(duration * RATE_HZ)
        for k in range(1, steps + 1):
            q = q0.copy()
            q[j] = frm + minimum_jerk(k / steps) * (to - frm)
            jc.command({side: q})
            robot.advance(1.0 / RATE_HZ)
            st = robot.get_state()
            if st.fault:
                raise RuntimeError(st.fault)
            trace_meas.append(st.arms[side].q[j] - meas0)
            trace_err.append(abs(st.arms[side].q_setpoint[j] - st.arms[side].q[j]))

    def hold(duration: float) -> None:
        go(jc.q_cmd[arm_slice(side)][j], jc.q_cmd[arm_slice(side)][j], duration)

    go(q0[j], q0[j] + sign * delta, t_move)
    hold(0.5)
    go(q0[j] + sign * delta, q0[j], t_move)
    hold(0.5)

    meas = np.array(trace_meas)
    peak = meas[np.argmax(np.abs(meas))]
    ok = np.sign(peak) == sign and abs(peak) > 0.5 * delta
    print(f"    {side} joint{j + 1} {'+' if sign > 0 else '-'}{np.degrees(delta):.1f} deg: "
          f"measured peak {np.degrees(peak):+.2f} deg, max tracking error {np.degrees(max(trace_err)):.2f} deg, "
          f"return error {np.degrees(meas[-1]):+.2f} deg  "
          f"{'OK' if ok else '<-- CHECK: direction/amplitude does not match the command'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--deg", type=float, default=3.0, help="jog amplitude per joint (deg)")
    parser.add_argument("--speed", type=float, default=2.0, help="jog speed (deg/s)")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--joints", default="1,2,3,4,5,6,7", help="comma-separated joint numbers")
    parser.add_argument("--zero-speed", type=float, default=10.0, help="peak joint speed for the zero moves (deg/s)")
    parser.add_argument("--gripper-mm", type=float, default=5.0, help="gripper open/close test (mm), 0 = skip")
    parser.add_argument("--clearance-tol-mm", type=float, default=1.0, help="max allowed clearance decrease")
    parser.add_argument("--min-clearance-mm", type=float, default=20.0, help="never plan below this clearance")
    parser.add_argument("--yes", action="store_true", help="do not ask before each joint")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    joints = [int(j) - 1 for j in args.joints.split(",")]
    delta, speed = np.radians(args.deg), np.radians(args.speed)

    kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
    robot = make_robot(cfg, args.backend, args.render)
    sides: List[Side] = active_sides(robot, args.side)

    def test() -> None:
        print("\n[1/3] Holding the zero pose for 3 s (no motion)...")
        hold_check(robot, 3.0)

        jc = JointPositionController(robot)
        jc.reset()
        q_start = {s: jc.q_cmd[arm_slice(s)].copy() for s in SIDES}
        plan = plan_jogs(kin, q_start, sides, joints, delta, cfg["hardware"]["joint_limit_margin_rad"],
                         args.clearance_tol_mm / 1000, args.min_clearance_mm / 1000)
        if not plan or not confirm(f"\n[2/3] Run {len(plan)} jogs at {args.speed:.1f} deg/s?", args.yes):
            return
        for jog in plan:
            side, j, sign = jog
            if not args.yes:
                ans = input(f"  next: {side} joint{j + 1} {'+' if sign > 0 else '-'}{args.deg:.1f} deg "
                            "[Enter = go, s = skip, q = quit] ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
            run_jog(robot, jc, jog, delta, speed)

        if args.gripper_mm > 0 and confirm(f"\n[3/3] Gripper test: move {args.gripper_mm:.0f} mm and back?", args.yes):
            for side in sides:
                g0 = float(jc.q_cmd[gripper_index(side)])
                step = args.gripper_mm / 1000 * (1 if g0 < GRIPPER_OPEN_M / 2 else -1)  # towards mid-stroke
                for g in (g0 + step, g0):
                    jc.command(gripper_m={side: g})
                    robot.advance(args.gripper_mm / 1000 / cfg["hardware"]["max_gripper_velocity_m_s"] + 0.5)
                    print(f"    {side} gripper cmd {g * 1000:.1f} mm, measured {robot.get_state().arms[side].gripper * 1000:.1f} mm")
        print("\nTest done.")

    sys.exit(run_with_zero(robot, cfg, test, args.zero_speed, confirm_first=lambda m: confirm(m, args.yes)))


if __name__ == "__main__":
    main()
