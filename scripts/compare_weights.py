#!/usr/bin/env python3
"""
Lift objects one at a time, measure their load from the arm's joint torques, register each
lift in a history file, and decide which object is heavier.

    python scripts/compare_weights.py --backend real                      # left arm, objects A, B
    python scripts/compare_weights.py --backend real --labels full empty --grip-torque 0.3
    python scripts/compare_weights.py --decide                            # re-decide from the history only
    python scripts/compare_weights.py --backend sim --yes                 # rehearse the flow (no objects in sim)

Per object (scripts/safe_zero.py: Ctrl+C at any point = release, go to zero, disable):
1. Low pose (elbow --low-elbow-deg, other joints 0: hand ~20 cm forward, pointing down-forward),
   gripper open. You put the object in the hand -> Enter: gripper closes in torque mode.
2. Lift: the elbow rotates to 90 deg (robot.home_qpos, forearm forward). Only the elbow moves.
3. Measure (src/estimation/payload.py): joints 1, 4, 7 dither +-1 deg for 3 cycles of 2 s,
   torques averaged over whole cycles (friction cancels), the model's arm gravity subtracted,
   residual projected on the vertical TCP Jacobian -> "equivalent mass". Its absolute value is
   rough; the DIFFERENCE between objects measured the same way is what decides
   (300 g adds ~1.2 Nm on joints 1/4, ~0.56 Nm on joint 7).
4. Back to the low pose, release, open. The lift is appended to --history (JSON lines).
After >= 2 lifts in this session: ranking, heavier vs lighter of the last two, and
z = difference / noise (> 3 = clear call).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import GRIPPER_OPEN_M, Side  # noqa: E402
from src.estimation.payload import LoadSample, compare, measure_load  # noqa: E402
from scripts.grip_torque_test import grip, move_to_pose, release  # noqa: E402
from scripts.jog_joints import active_sides, confirm, make_robot  # noqa: E402
from scripts.safe_zero import run_with_zero  # noqa: E402

DEFAULT_HISTORY = "logs/weigh_history.jsonl"
LOW_ELBOW_DEG = 30.0  # low pose: elbow only; hand ~0.20 m forward, ~0.13 m above the base origin


def sample_to_record(s: LoadSample, session: str, grip_torque: float, low_elbow_deg: float) -> dict:
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session": session,
        "label": s.label,
        "side": s.side,
        "mass_eq_kg": s.mass,
        "mass_se_kg": s.mass_se,
        "cycle_mass_kg": s.cycle_mass.tolist(),
        "per_joint_mass_kg": {str(j): m for j, m in s.per_joint_mass().items()},
        "joints": [j + 1 for j in s.joints],
        "q": s.q.tolist(),
        "tau": s.tau.tolist(),
        "tau_model": s.tau_model.tolist(),
        "jz": s.jz.tolist(),
        "gripper_m": s.gripper_m,
        "grip_torque_nm": grip_torque,
        "low_elbow_deg": low_elbow_deg,
    }


def record_to_sample(r: dict) -> LoadSample:
    return LoadSample(label=r["label"], side=r["side"], q=np.asarray(r["q"]), tau=np.asarray(r["tau"]),
                      tau_model=np.asarray(r["tau_model"]), jz=np.asarray(r["jz"]),
                      joints=[j - 1 for j in r["joints"]], cycle_mass=np.asarray(r["cycle_mass_kg"]),
                      gripper_m=r["gripper_m"])


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_sample(s: LoadSample) -> None:
    per_joint = "  ".join(f"j{j}: {m * 1e3:+5.0f} g" for j, m in s.per_joint_mass().items())
    print(f"  [{s.label}] equivalent mass {s.mass * 1e3:+6.0f} +- {s.mass_se * 1e3:.0f} g   ({per_joint})")
    print("      residual torque (measured - model) [Nm]: "
          + " ".join(f"j{j + 1} {s.tau[j] - s.tau_model[j]:+.3f}" for j in s.joints)
          + f"   gripper {s.gripper_m * 1e3:.1f} mm")


def decide(samples: List[LoadSample]) -> None:
    """Prints the ranking and the heavier/lighter call for the last two samples."""
    if len(samples) < 2:
        print("\nNeed two lifts to compare.")
        return
    ranked = sorted(samples, key=lambda s: s.mass, reverse=True)
    print("\nRanking (heaviest first): " + " > ".join(f"{s.label} ({s.mass * 1e3:+.0f} g)" for s in ranked))
    c = compare(samples[-2], samples[-1])
    verdict = "clear" if c.z > 3 else ("weak" if c.z > 1.5 else "NOT RELIABLE: repeat the lifts / grip closer to the TCP")
    print(f"HEAVIER: {c.heavier.label}   LIGHTER: {c.lighter.label}   "
          f"difference ~{c.diff_kg * 1e3:.0f} g, z = {c.z:.1f} ({verdict})")
    if c.pose_diff_deg > 3.0:
        print(f"  note: lift poses differ by {c.pose_diff_deg:.1f} deg (model subtraction compensates, roughly)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["real", "sim", "fake"], default="real")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--labels", nargs="+", default=["A", "B"], help="objects, lifted in this order")
    parser.add_argument("--grip-torque", type=float, default=0.3,
                        help="gripper torque (Nm, ~12 N/Nm ideal squeeze); find it with grip_torque_test.py")
    parser.add_argument("--low-elbow-deg", type=float, default=LOW_ELBOW_DEG, help="elbow angle of the low pose")
    parser.add_argument("--dither-deg", type=float, default=1.0)
    parser.add_argument("--period", type=float, default=2.0, help="dither period (s)")
    parser.add_argument("--cycles", type=int, default=3, help="measured dither cycles")
    parser.add_argument("--settle", type=float, default=1.0, help="wait after the lift before measuring (s)")
    parser.add_argument("--open-mm", type=float, default=44.0, help="opening before each grasp (mm per finger)")
    parser.add_argument("--joint-speed", type=float, default=15.0, help="peak joint speed (deg/s)")
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="JSON-lines file every lift is appended to")
    parser.add_argument("--decide", action="store_true", help="no robot: decide from the last session in --history")
    parser.add_argument("--yes", action="store_true", help="do not ask (sim rehearsal)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")

    history = Path(args.history)
    if args.decide:
        records = load_history(history)
        if not records:
            print(f"No lifts in {history}.")
            return 1
        last = [r for r in records if r["session"] == records[-1]["session"]]
        print(f"Session {last[0]['session']} ({last[0]['time']}): {len(last)} lifts")
        samples = [record_to_sample(r) for r in last]
        for s in samples:
            print_sample(s)
        decide(samples)
        return 0

    from src.planning.openarm_kinematics import OpenArmKinematics

    cfg = yaml.safe_load(open(args.config))
    kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
    robot = make_robot(cfg, args.backend, args.render)
    side: Side = active_sides(robot, args.side)[0]
    lift = np.asarray(cfg["robot"]["home_qpos"][side], dtype=np.float64)  # elbow 90 deg, forearm forward
    low = np.zeros(7)
    low[3] = np.radians(args.low_elbow_deg)
    open_m = float(np.clip(args.open_mm / 1e3, 0.0, GRIPPER_OPEN_M))
    session = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    samples: List[LoadSample] = []
    history.parent.mkdir(parents=True, exist_ok=True)

    def lift_and_measure(label: str) -> None:
        move_to_pose(robot, side, low, args.joint_speed, gripper_m=open_m)
        if not confirm(f"\n[{label}] Put it in the {side} hand. Grip at {args.grip_torque:.2f} Nm?", args.yes):
            return
        grip(robot, side, args.grip_torque, cfg)
        if not confirm(f"[{label}] Let go of it. Lift and measure?", args.yes):
            release(robot, side, open_m)
            return
        move_to_pose(robot, side, lift, args.joint_speed)
        robot.advance(args.settle)
        print(f"  measuring ({(args.cycles + 2) * args.period:.0f} s, +-{args.dither_deg:.1f} deg dither)...")
        s = measure_load(robot, kin, side, label, amplitude_deg=args.dither_deg, period_s=args.period, cycles=args.cycles)
        print_sample(s)
        move_to_pose(robot, side, low, args.joint_speed)
        release(robot, side, open_m)
        with history.open("a") as f:
            f.write(json.dumps(sample_to_record(s, session, args.grip_torque, args.low_elbow_deg)) + "\n")
        print(f"  registered in {history} (session {session})")
        samples.append(s)

    def test() -> None:
        for label in args.labels:
            lift_and_measure(label)
            if len(samples) >= 2:
                decide(samples)
        print("\nTest done.")

    return run_with_zero(robot, cfg, test, args.joint_speed, confirm_first=lambda m: confirm(m, args.yes))


if __name__ == "__main__":
    sys.exit(main())
