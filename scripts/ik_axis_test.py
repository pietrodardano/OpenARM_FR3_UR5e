#!/usr/bin/env python3
"""
Small Cartesian (IK) test: straight-line TCP moves along each base axis, then stacked.

    python scripts/ik_axis_test.py --backend sim --render --yes     # rehearse first
    python scripts/ik_axis_test.py --backend real                   # 5 cm moves, asks before each group
    python scripts/ik_axis_test.py --backend real --side left --dist-cm 2

Base frame: +X forward, +Y left, +Z up. TCP orientation is held fixed at the start orientation.
One arm moves at a time; the other holds its pose.

Why it starts from home (j4 = 90 deg) and not from the zero pose:
  At zero the arm hangs fully stretched (elbow j4 = 0, its lower limit). The position
  Jacobian loses Z: sigma_min = 0 (check with scripts/read_ee_pose.py --q-left 0,0,0,0,0,0,0).
    - down: unreachable, the arm is already at full length.
    - up:   only by bending the elbow, and dz ~ (dj4)^2 there: 3 cm up costs ~42 deg on j4
            plus ~21 deg on j1 and j7 (vs ~8 deg at home). Big, fast joint motion for a tiny
            Cartesian move.
    - the zero-pose TCP (x 0.0, z 0.07) is also outside the IK workspace box
      (x >= 0.15, z >= 0.15), so IKController would clamp the first target and yank the TCP
      up to 15 cm towards the box.
  At home the TCPs are at (0.41, +-0.15, 0.48), sigma_min 0.16, and every 3 cm axis move
  needs < 10 deg per joint.

Sequence (scripts/safe_zero.py: Ctrl+C at any point = go to zero, then disable):
0. Go to zero (all joints 0, grippers closed).
1. Move to home in joint space (--joint-speed), re-anchor IK. Then per arm:
2. Single axes: forward/back (X), up/down (Z), left/right (Y): +d, back, -d, back.
3. Stacked: +X, then +Z on top, then +Y outward on top (a 3D staircase to the corner of a
   d-cube), then undo in reverse order, then the cube diagonal straight out and back.
4. Back to home, go to zero, disable.

Per segment it prints the commanded vs measured TCP displacement, the measured direction
(cosine with the command; a sign/frame problem shows up here), the IK error, the
measured-vs-commanded TCP offset (gravity sag on hardware) and the largest joint motion.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.ik_controller import IKController  # noqa: E402
from src.control.robot_interface import SIDES, RobotBackend, Side, arm_slice  # noqa: E402
from scripts.jog_joints import confirm, make_robot  # noqa: E402
from scripts.safe_zero import MIN_JERK_PEAK_FACTOR, run_with_zero  # noqa: E402

X, Y, Z = np.eye(3)
Segment = Tuple[str, np.ndarray]  # (label, absolute offset from the start TCP position, m)


def axis_segments(d: float) -> List[Tuple[str, List[Segment]]]:
    groups = []
    for name, axis, pos, neg in (("X", X, "forward", "back"), ("Z", Z, "up", "down"), ("Y", Y, "left", "right")):
        groups.append((f"{name} axis ({pos}/{neg})", [
            (f"{pos} {d * 100:.0f} cm", d * axis), ("return", 0 * axis),
            (f"{neg} {d * 100:.0f} cm", -d * axis), ("return", 0 * axis),
        ]))
    return groups


def stacked_segments(d: float, side: Side) -> List[Tuple[str, List[Segment]]]:
    out = Y if side == "left" else -Y  # away from the other arm
    out_name = "left" if side == "left" else "right"
    c = d * 100
    return [
        ("stacked staircase", [
            (f"forward {c:.0f} cm", d * X),
            (f"+ up {c:.0f} cm", d * (X + Z)),
            (f"+ {out_name} {c:.0f} cm", d * (X + Z + out)),
            (f"- {out_name}", d * (X + Z)),
            ("- up", d * X),
            ("- forward", 0 * X),
        ]),
        ("stacked diagonal", [
            (f"diagonal forward+up+{out_name}", d * (X + Z + out)),
            ("diagonal back", 0 * X),
        ]),
    ]


def run_segment(robot: RobotBackend, ik: IKController, side: Side, p_start: np.ndarray, R_start: np.ndarray,
                label: str, offset: np.ndarray, speed: float, hold: float) -> None:
    p_cmd0 = ik.pose_cmd[side][0].copy()
    goal = p_start + offset
    dist = float(np.linalg.norm(goal - p_cmd0))
    duration = max(1.0, MIN_JERK_PEAK_FACTOR * dist / speed)
    meas0 = ik.get_tcp_pose()[side][0]
    q0 = robot.get_state().arms[side].q.copy()

    errs = ik.move_l({side: (goal, R_start)}, duration=duration)
    robot.advance(hold)

    st = robot.get_state()
    if st.fault:
        raise RuntimeError(st.fault)
    p_meas, R_meas = ik.get_tcp_pose()[side]
    cmd_d = goal - p_cmd0
    meas_d = p_meas - meas0
    cos = float(cmd_d @ meas_d / (np.linalg.norm(cmd_d) * np.linalg.norm(meas_d) + 1e-12)) if dist > 1e-6 else 1.0
    dq = np.degrees(np.abs(st.arms[side].q - q0))
    rot_drift = np.degrees(np.linalg.norm((Rotation.from_matrix(R_meas) * Rotation.from_matrix(R_start).inv()).as_rotvec()))
    ok = dist < 1e-6 or (cos > 0.9 and 0.5 < np.linalg.norm(meas_d) / dist < 1.5)
    print(f"    {label:<28} cmd {np.round(cmd_d * 1e3, 1)} mm  meas {np.round(meas_d * 1e3, 1)} mm  "
          f"cos {cos:+.2f}  IK err {errs.get(side, 0) * 1e3:.1f} mm  "
          f"|meas-cmd| {np.linalg.norm(p_meas - goal) * 1e3:.1f} mm  rot drift {rot_drift:.1f} deg  "
          f"max dq {dq.max():.1f} deg (j{int(dq.argmax()) + 1})  {'OK' if ok else '<-- CHECK'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="both = one after the other")
    parser.add_argument("--dist-cm", type=float, default=5.0, help="move length per axis (cm)")
    parser.add_argument("--speed-cm", type=float, default=2.0, help="peak TCP speed (cm/s)")
    parser.add_argument("--joint-speed", type=float, default=10.0, help="peak joint speed for zero / home moves (deg/s)")
    parser.add_argument("--hold", type=float, default=0.5, help="hold after each segment (s)")
    parser.add_argument("--skip-stacked", action="store_true")
    parser.add_argument("--yes", action="store_true", help="do not ask before each group")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    sides: List[Side] = list(SIDES) if args.side == "both" else [args.side]
    d, speed = args.dist_cm / 100, args.speed_cm / 100
    home = {s: np.asarray(cfg["robot"]["home_qpos"][s]) for s in SIDES}
    robot = make_robot(cfg, args.backend, args.render)
    ik = IKController.from_config(cfg, robot)

    def move_home() -> None:
        q_now = robot.get_joint_targets()
        dist = max(np.degrees(np.max(np.abs(q_now[arm_slice(s)] - home[s]))) for s in SIDES)
        duration = max(3.0, MIN_JERK_PEAK_FACTOR * dist / args.joint_speed)
        print(f"\nMoving to home, the IK working pose ({dist:.0f} deg max, {duration:.1f} s)...")
        robot.move_joints(home, duration=duration)
        robot.advance(1.0)

    def test() -> None:
        if not confirm(f"Move both arms to home {np.round(home['left'], 2).tolist()}?", args.yes):
            return
        move_home()
        for side in sides:
            ik.reset()
            p_start, R_start = (v.copy() for v in ik.pose_cmd[side])
            print(f"\n=== {side} arm: start TCP {np.round(p_start, 3)} m, moves of {d * 100:.0f} cm "
                  f"at <= {speed * 100:.0f} cm/s ===")
            groups = axis_segments(d) + ([] if args.skip_stacked else stacked_segments(d, side))
            for title, segments in groups:
                if not confirm(f"  {side}: {title}?", args.yes):
                    continue
                print(f"  {title}")
                for label, offset in segments:
                    run_segment(robot, ik, side, p_start, R_start, label, offset, speed, args.hold)
            ik.move_l({side: (p_start, R_start)}, duration=2.0)  # exactly back to start (no-op if already there)
        move_home()
        print("\nTest done.")

    return run_with_zero(robot, cfg, test, args.joint_speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
