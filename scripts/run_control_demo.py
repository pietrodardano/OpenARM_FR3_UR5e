#!/usr/bin/env python3
"""
Exercise the joint position controller or the IK controller, in MuJoCo or on the real robot.

    # Dry run in the official MuJoCo model (viewer opens)
    python scripts/run_control_demo.py --backend sim --controller ik --render
    # Real robot, small motions first
    python scripts/run_control_demo.py --backend real --controller joint --amplitude 0.5
    # Real driver code path (threaded loop, safety, faults) against a simulated CAN bus
    python scripts/run_control_demo.py --backend fake --controller ik

Sequence: start (hold measured pose) -> move_joints to home -> controller demo
-> gripper close/open -> back home -> stop (move to rest, disable motors).
Ctrl+C at any time triggers the same safe stop.
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

from src.control.ik_controller import IKController  # noqa: E402
from src.control.joint_position_controller import JointPositionController  # noqa: E402
from src.control.robot_interface import GRIPPER_OPEN_M, SIDES, RobotBackend, arm_slice  # noqa: E402

RATE_HZ = 50.0


def joint_demo(robot: RobotBackend, amplitude: float) -> None:
    """Sinusoid on joints 1 and 7 of both arms around home (absolute commands)."""
    jc = JointPositionController(robot)
    jc.reset()
    q0 = {s: jc.q_cmd[arm_slice(s)].copy() for s in SIDES}
    t0 = time.monotonic()
    while (t := time.monotonic() - t0) < 6.0:
        w = 0.2 * amplitude * np.sin(2 * np.pi * t / 3.0)
        q = {}
        for s in SIDES:
            q[s] = q0[s].copy()
            q[s][0] += w
            q[s][6] += 2 * w
        jc.command(q)
        robot.advance(1.0 / RATE_HZ)
    jc.command(q0)
    robot.advance(1.0)
    gripper_cycle(robot, lambda g: jc.command(gripper_m={s: g for s in SIDES}))


def ik_demo(robot: RobotBackend, ik: IKController, amplitude: float) -> None:
    """move_l down/forward, then stream a circle in the y-z plane with both TCPs."""
    ik.reset()
    start = dict(ik.pose_cmd)
    offset = amplitude * np.array([0.05, 0.0, -0.05])
    ik.move_l({s: (start[s][0] + offset, start[s][1]) for s in SIDES}, duration=2.0)
    center = {s: ik.pose_cmd[s][0].copy() for s in SIDES}
    r = 0.04 * amplitude
    t0 = time.monotonic()
    while (t := time.monotonic() - t0) < 8.0:
        a = 2 * np.pi * t / 4.0
        circle = np.array([0.0, r * np.sin(a), r * (np.cos(a) - 1.0)])
        err = ik.command_poses({s: (center[s] + circle, start[s][1]) for s in SIDES})
        robot.advance(1.0 / RATE_HZ)
    print(f"  last IK position error: {', '.join(f'{s} {e * 1e3:.2f} mm' for s, e in err.items())}")
    meas = ik.get_tcp_pose()
    print("  measured TCP - commanded: " + ", ".join(
        f"{s} {np.linalg.norm(meas[s][0] - ik.pose_cmd[s][0]) * 1e3:.1f} mm" for s in SIDES))
    ik.move_l(start, duration=2.0)
    gripper_cycle(robot, lambda g: ik.joints.command(gripper_m={s: g for s in SIDES}))


def gripper_cycle(robot: RobotBackend, send) -> None:
    for g in (0.0, GRIPPER_OPEN_M):
        send(g)
        robot.advance(2.5)  # full stroke takes 2 s at max_gripper_velocity
        print(f"  gripper -> {g * 1000:.0f} mm, measured " + ", ".join(
            f"{s} {robot.get_state().arms[s].gripper * 1000:.1f} mm" for s in SIDES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["sim", "real", "fake"], default="sim",
                        help="sim: MuJoCo; real: OpenArm over CAN; fake: real driver on a simulated CAN bus")
    parser.add_argument("--controller", choices=["joint", "ik"], default="ik")
    parser.add_argument("--amplitude", type=float, default=1.0, help="scale of the demo motions (start small on hardware)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (sim only)")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(args.config))

    if args.backend in ("real", "fake"):
        from src.hardware import FakeOpenArmBus, OpenArmBimanual, OpenArmBus

        dt = 1.0 / cfg["hardware"]["control_rate_hz"]
        bus = OpenArmBus if args.backend == "real" else (lambda _cfg: FakeOpenArmBus(dt=dt))
        robot = OpenArmBimanual.from_config(cfg, bus_factory=bus)
    else:
        from src.simulation.openarm_sim import OpenArmSim

        robot = OpenArmSim.from_config(cfg, realtime=True, render=args.render)
    ik = IKController.from_config(cfg, robot) if args.controller == "ik" else None
    home = {s: np.asarray(cfg["robot"]["home_qpos"][s]) for s in SIDES}

    robot.start()
    try:
        print("Moving to home...")
        robot.move_joints(home, gripper={s: GRIPPER_OPEN_M for s in SIDES}, duration=4.0)
        print(f"{args.controller} controller demo...")
        if ik is not None:
            ik_demo(robot, ik, args.amplitude)
        else:
            joint_demo(robot, args.amplitude)
        robot.move_joints(home, duration=2.0)
        stats = robot.get_state().stats
        print(f"Done. control loop stats: {stats}")
    except KeyboardInterrupt:
        print("\nInterrupted: stopping safely...")
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
