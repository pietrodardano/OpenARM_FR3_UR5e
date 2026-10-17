#!/usr/bin/env python3
"""
Print both TCP (end-effector) poses. Nothing is enabled and nothing moves.

    python scripts/read_ee_pose.py                    # real robot, one passive read (motors stay disabled)
    python scripts/read_ee_pose.py --watch 2          # keep reading at 2 Hz; move the arms by hand
    python scripts/read_ee_pose.py --q-left 0,0,0,90,0,0,0 --deg   # forward kinematics only, no robot

Pose = TCP site of the official model (point between the fingertips), base frame:
origin on the floor under the torso, +X forward, +Y left, +Z up.

Also printed per arm, from the 3x7 position Jacobian:
  - sigma_min: how well the TCP can still translate in its weakest direction (m/rad).
    0 = singular. The zero pose (arms hanging straight) is exactly singular in Z.
  - weakest direction: the base-frame direction that needs the largest joint motion.
  - in IK box: whether IKController would accept this position (configs/task_config.yaml,
    controllers.ik.workspace). Outside the box, IK targets get clamped into it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import ARM_DOF, SIDES, Side  # noqa: E402
from src.planning.openarm_kinematics import OpenArmKinematics  # noqa: E402

SINGULAR_WARN = 0.05  # m/rad


def parse_q(text: Optional[str], deg: bool) -> Optional[np.ndarray]:
    if text is None:
        return None
    q = np.array([float(v) for v in text.split(",")])
    if q.shape != (ARM_DOF,):
        raise SystemExit(f"expected {ARM_DOF} comma-separated joint values, got {text!r}")
    return np.radians(q) if deg else q


def position_jacobian(kin: OpenArmKinematics, side: Side) -> np.ndarray:
    """3x7 TCP position Jacobian at the configuration last set by kin.fk()."""
    mujoco.mj_comPos(kin.model, kin.data)
    jp = np.zeros((3, kin.model.nv))
    mujoco.mj_jacSite(kin.model, kin.data, jp, None, kin.site_id[side])
    return jp[:, kin.dof_adr[side]]


def report(kin: OpenArmKinematics, cfg: dict, q: Dict[Side, np.ndarray], gripper: Optional[Dict[Side, float]]) -> None:
    poses = kin.fk(q)
    for side in SIDES:
        p, R = poses[side]
        kin.fk(q)  # leave both arms set for the Jacobian
        U, S, _ = np.linalg.svd(position_jacobian(kin, side))
        rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
        quat = Rotation.from_matrix(R).as_quat()  # x, y, z, w
        box = cfg["controllers"]["ik"]["workspace"][side]
        inside = bool(np.all(p >= box["low"]) and np.all(p <= box["high"]))
        print(f"{side} arm")
        print(f"  q [rad] : " + " ".join(f"{v:+.3f}" for v in q[side]))
        print(f"  q [deg] : " + " ".join(f"{v:+7.1f}" for v in np.degrees(q[side])))
        if gripper is not None:
            print(f"  gripper : {gripper[side] * 1e3:.1f} mm")
        print(f"  pos [m] : x {p[0]:+.4f}  y {p[1]:+.4f}  z {p[2]:+.4f}")
        print(f"  rpy[deg]: roll {rpy[0]:+7.1f}  pitch {rpy[1]:+7.1f}  yaw {rpy[2]:+7.1f}   (extrinsic xyz)")
        print(f"  quat    : w {quat[3]:+.4f}  x {quat[0]:+.4f}  y {quat[1]:+.4f}  z {quat[2]:+.4f}")
        print("  R       : " + "\n            ".join(" ".join(f"{v:+.3f}" for v in row) for row in R))
        print(f"  approach: {np.round(R[:, 2], 3)} (TCP z-axis in base frame)")
        warn = "  <-- NEAR SINGULAR" if S[-1] < SINGULAR_WARN else ""
        print(f"  sigma   : {np.round(S, 4)}  min {S[-1]:.4f}{warn}")
        print(f"  weakest : {np.round(U[:, -1], 2)}")
        print(f"  in IK box: {'yes' if inside else 'NO'}  (low {box['low']}, high {box['high']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q-left", help="7 joint values: FK only, no robot")
    parser.add_argument("--q-right", help="7 joint values: FK only, no robot")
    parser.add_argument("--deg", action="store_true", help="--q-* values are in degrees")
    parser.add_argument("--watch", type=float, default=0.0, help="keep reading the real robot at this rate (Hz)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])

    q_left, q_right = parse_q(args.q_left, args.deg), parse_q(args.q_right, args.deg)
    if q_left is not None or q_right is not None:
        q = {"left": q_left if q_left is not None else np.zeros(ARM_DOF),
             "right": q_right if q_right is not None else np.zeros(ARM_DOF)}
        report(kin, cfg, q, None)
        return 0

    from src.hardware import OpenArmBimanual

    robot = OpenArmBimanual.from_config(cfg)
    try:
        while True:
            st = robot.read_passive()
            print(f"\n=== {time.strftime('%H:%M:%S')} (motors not enabled) ===")
            report(kin, cfg, {s: st.arms[s].q for s in SIDES}, {s: st.arms[s].gripper for s in SIDES})
            if args.watch <= 0:
                break
            time.sleep(1.0 / args.watch)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
