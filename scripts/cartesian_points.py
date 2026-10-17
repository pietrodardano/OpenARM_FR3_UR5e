#!/usr/bin/env python3
"""
Cartesian (absolute TCP pose) control test: from the rest pose, visit 4 fixed points with the
hand facing forward, and come back to rest, using only Cartesian commands.

    python scripts/cartesian_points.py --backend sim --render --yes         # rehearse in MuJoCo
    python scripts/cartesian_points.py --backend real                      # POINTS below, left arm
    python scripts/cartesian_points.py --backend real --speed-cm 4          # slower (default 8 cm/s peak)
    python scripts/cartesian_points.py --backend real --points "0.40,0.15,0.55;0.45,0.20,0.45"

Path (base frame: +X forward, +Y left, +Z up, origin on the floor under the torso):
    rest  (0.00, 0.15, 0.07)  arm hanging, fingers pointing down   <- start
    P1    (0.40, 0.25, 0.55)  out to the left and up               (straight line from rest; the hand
                                                                    turns from down to forward on the way)
    P2    (0.50, 0.15, 0.50)  straight ahead, further out
    P3    (0.40, 0.08, 0.42)  in toward the body's midline, down
    P4    (0.32, 0.20, 0.45)  pulled back toward the body
    rest                                                            <- end (fingers down again)
At P1..P4 the hand has its elbow-90 orientation (robot.home_qpos): fingers pointing forward (+X).

Interface: IKController.move_l({side: (p, R)}), i.e. absolute TCP poses streamed through
IKController.command_poses -> 6-DoF IK (position AND orientation) -> joint targets. No joint-space
moves are used to reach a configuration. Gravity is compensated by the driver at every setpoint
(hardware.gravity_compensation_scale), like any other motion.

Notes:
- The rest TCP is outside controllers.ik.workspace (x >= 0.15, z >= 0.15), which would clamp the
  start of the line. This script extends the box just enough to contain the path and prints it:
  the box's table guard (z_min) does NOT protect the first part of rest -> P1. Check the line is clear.
- Coming back, the arm straightens towards a singular pose, so IK ends a few degrees from exact zero.
  The final "go to zero" of scripts/safe_zero.py (a joint move) closes that gap before disabling;
  it is also what runs first if the arm is not already at rest when the script starts.
- Before moving, every straight-line path is checked in the model (IK error < 3 mm / 2 deg,
  clearance to torso / other arm > 3 cm, no joint jumps).

Ctrl+C at any point: go to zero (joint move), then disable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.ik_controller import IKController, WorkspaceBox  # noqa: E402
from src.control.robot_interface import ARM_DOF, Side, minimum_jerk  # noqa: E402
from src.planning.openarm_ik import BimanualIK  # noqa: E402
from src.planning.openarm_kinematics import OpenArmKinematics  # noqa: E402
from scripts.jog_joints import active_sides, confirm, make_robot  # noqa: E402
from scripts.safe_zero import MIN_JERK_PEAK_FACTOR, run_with_zero  # noqa: E402

# Fixed targets (base frame, m), all with the forward hand orientation of home (elbow 90 deg).
POINTS = [
    ((0.40, 0.25, 0.55), "out to the left and up"),
    ((0.50, 0.15, 0.50), "straight ahead, further out"),
    ((0.40, 0.08, 0.42), "in toward the midline, down"),
    ((0.32, 0.20, 0.45), "pulled back toward the body"),
]
MAX_POS_ERR = 3e-3
MAX_ROT_ERR = np.radians(2.0)
MIN_CLEARANCE = 0.03
MAX_JOINT_STEP = 0.15  # rad between consecutive path samples: more = IK branch flip
ROT_SPEED_DEG_S = 40.0  # peak hand rotation speed (rest <-> forward turns the hand 90 deg)
WORKSPACE_MARGIN = 0.02  # m added around the path when the IK box has to be extended

Pose = Tuple[np.ndarray, np.ndarray]


def rot_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm((Rotation.from_matrix(R_a) * Rotation.from_matrix(R_b).inv()).as_rotvec())))


def check_path(kin: OpenArmKinematics, ik: BimanualIK, side: Side, q_start: np.ndarray, a: Pose, b: Pose,
               ik_iters: int) -> Optional[Tuple[np.ndarray, float]]:
    """Straight line a -> b (orientation slerped), solved like move_l: each step seeded by the last.

    Returns (joints at b, minimum clearance), or None if a step fails.
    """
    other: Side = "right" if side == "left" else "left"
    q_other = np.zeros(ARM_DOF)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a[1], b[1]])))
    samples = max(25, int(np.linalg.norm(b[0] - a[0]) / 0.005) + int(rot_deg(a[1], b[1])))
    q, min_clear = q_start.copy(), np.inf
    for k in range(1, samples + 1):
        s = minimum_jerk(k / samples)
        r = ik.solve({side: (a[0] + s * (b[0] - a[0]), slerp(s).as_matrix())}, {side: q, other: q_other},
                     max_iters=ik_iters)
        q_new = r.q_arm[side]
        clear = kin.clearance(side, {side: q_new, other: q_other})
        if (r.pos_error[side] > MAX_POS_ERR or r.rot_error[side] > MAX_ROT_ERR
                or np.max(np.abs(q_new - q)) > MAX_JOINT_STEP or clear < MIN_CLEARANCE):
            return None
        q, min_clear = q_new, min(min_clear, clear)
    return q, min_clear


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--points", help='other points "x,y,z;x,y,z;..." (m, base frame) instead of POINTS')
    parser.add_argument("--speed-cm", type=float, default=8.0, help="peak TCP speed (cm/s)")
    parser.add_argument("--hold", type=float, default=1.0, help="hold at each point before measuring (s)")
    parser.add_argument("--zero-speed", type=float, default=10.0, help="peak joint speed of the go-to-zero moves (deg/s)")
    parser.add_argument("--yes", action="store_true", help="do not ask before each move")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(open(args.config))
    kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
    side: Side = args.side
    other: Side = "right" if side == "left" else "left"
    q_rest = np.asarray(cfg["hardware"]["rest_qpos"][side], dtype=np.float64)
    q_home = np.asarray(cfg["robot"]["home_qpos"][side], dtype=np.float64)
    rest: Pose = kin.fk({side: q_rest, other: np.zeros(ARM_DOF)})[side]
    R_fwd = kin.fk({side: q_home, other: np.zeros(ARM_DOF)})[side][1]
    if args.points:
        points = [(np.array([float(v) for v in pt.split(",")]), "custom") for pt in args.points.split(";")]
    else:
        points = [(np.array(p, dtype=np.float64), desc) for p, desc in POINTS]

    # Waypoints: rest -> P1..PN (forward hand) -> rest.
    waypoints: List[Tuple[str, Pose, str]] = [("rest", rest, "arm hanging, fingers down")]
    waypoints += [(f"P{i}", (p, R_fwd), desc) for i, (p, desc) in enumerate(points, 1)]
    waypoints += [("rest", rest, "back down, fingers down")]

    # IK workspace: the configured box, extended just enough to contain the path.
    ws = cfg["controllers"]["ik"]["workspace"][side]
    path_pts = np.array([w[1][0] for w in waypoints])
    low = np.minimum(ws["low"], path_pts.min(axis=0) - WORKSPACE_MARGIN)
    high = np.maximum(ws["high"], path_pts.max(axis=0) + WORKSPACE_MARGIN)
    ik_iters = cfg["controllers"]["ik"]["ik_iters"]

    print(f"Hand orientation at the points: approach (z) axis {np.round(R_fwd[:, 2], 3)} (fingers forward)")
    print("Checking the straight-line paths in the model...")
    ik_check = BimanualIK(kin)
    q = q_rest
    print(f"\n  {'':6} {'x':>6} {'y':>6} {'z':>6}   {'':30} min clearance   joints at target [deg]")
    print(f"  {'rest':6} {rest[0][0]:6.3f} {rest[0][1]:6.3f} {rest[0][2]:6.3f}   {waypoints[0][2]:30}")
    for (name_a, pose_a, _), (name_b, pose_b, desc) in zip(waypoints[:-1], waypoints[1:]):
        res = check_path(kin, ik_check, side, q, pose_a, pose_b, ik_iters)
        if res is None:
            raise SystemExit(f"{name_a} -> {name_b} {np.round(pose_b[0], 3).tolist()}: not reachable on a straight line "
                             "with this orientation (or too close to the torso)")
        q, clear = res
        p = pose_b[0]
        print(f"  {name_b:6} {p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}   {desc:30} {clear * 100:7.1f} cm      "
              + " ".join(f"{v:+6.1f}" for v in np.degrees(q)))
    if np.any(low < ws["low"]) or np.any(high > ws["high"]):
        print(f"\nNOTE: the path leaves controllers.ik.workspace.{side} (low {ws['low']}, high {ws['high']}).")
        print(f"      Extended for this run to low {np.round(low, 3).tolist()}, high {np.round(high, 3).tolist()}:")
        print("      the table guard (z_min) does not protect the start of rest -> P1. Make sure that line is clear.")

    robot = make_robot(cfg, args.backend, args.render)
    active_sides(robot, side)  # refuse an inactive arm before moving anything
    ik = IKController.from_config(cfg, robot, kin)
    ik.cfg.workspace[side] = WorkspaceBox(tuple(low), tuple(high))
    v_lin = args.speed_cm / 100

    def go(name: str, target: Pose) -> None:
        p0, R0 = ik.pose_cmd[side]
        dist, ang = float(np.linalg.norm(target[0] - p0)), rot_deg(target[1], R0)
        duration = max(2.0, MIN_JERK_PEAK_FACTOR * max(dist / v_lin, ang / ROT_SPEED_DEG_S))
        print(f"\n{name}: {dist * 100:.1f} cm, hand turns {ang:.0f} deg, {duration:.1f} s")
        errs = ik.move_l({side: target}, duration=duration)
        robot.advance(args.hold)
        p_meas, R_meas = ik.get_tcp_pose()[side]
        print(f"  target {np.round(target[0], 3)}  measured {np.round(p_meas, 3)}  "
              f"error {np.linalg.norm(p_meas - target[0]) * 1e3:.1f} mm "
              f"(xyz {np.round((p_meas - target[0]) * 1e3, 1)} mm), orientation {rot_deg(R_meas, target[1]):.1f} deg, "
              f"IK {errs.get(side, 0) * 1e3:.1f} mm")
        print("  q[deg] " + " ".join(f"{v:+6.1f}" for v in np.degrees(robot.get_state().arms[side].q)))

    def test() -> None:
        ik.reset()  # commanded pose = rest (run_with_zero just put the arm there)
        for name, target, desc in waypoints[1:]:
            if not confirm(f"\n{name} {np.round(target[0], 3).tolist()} ({desc})?", args.yes):
                if name != "rest":
                    continue
                return
            go(name, target)
        print("\nTest done.")

    return run_with_zero(robot, cfg, test, args.zero_speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
