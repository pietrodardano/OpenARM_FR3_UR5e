#!/usr/bin/env python3
"""
Scripted demo in the sandbox: one arm reaches the dispenser lever tip and presses it down.

    python scripts/run_lever_demo.py                     # headless, prints the result
    python scripts/run_lever_demo.py --render            # MuJoCo viewer, real time
    python scripts/run_lever_demo.py --snapshots outputs/lever   # PNG per phase (headless)
    python scripts/run_lever_demo.py --side right

Uses the same stack as the real robot: OpenArmSim (same API as the hardware driver, safety
filter in the loop) + IKController (move_l on the official model). The lever tip position
is read from the scene (site ``dispenser_lever_tip``) and converted into the robot base
frame used by the IK (origin on the floor under the official model's torso).

Sequence: start at home (forearms forward) -> move above the tip (gripper closed) ->
press down -> hold -> lift (spring returns the lever) -> back home.

The sim starts directly at home: with the table just under the elbows, swinging the
forearms up from the hanging rest pose would go through the tabletop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_sandbox_scene import SANDBOX_MODEL, build_sandbox  # noqa: E402
from view_sandbox import write_png  # noqa: E402
from src.control.ik_controller import IKController  # noqa: E402
from src.control.robot_interface import SIDES  # noqa: E402
from src.simulation.openarm_sim import OpenArmSim  # noqa: E402

LEVER_DISPENSING_RAD = 0.15
ABOVE_M = 0.05  # approach height above the lever tip
PRESS_M = 0.025  # how far below the resting tip the TCP is commanded (the lever stops it at ~1.4 cm)
TIP_BACK_M = 0.01  # TCP slightly towards the jug from the tip, so the fingertips land on the lever


def lever_contacts(m: mujoco.MjModel, d: mujoco.MjData) -> list:
    """Names of the geoms currently touching the lever."""
    lever = m.body("dispenser_lever_body").id
    out = []
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if lever in (b1, b2):
            other = c.geom2 if b1 == lever else c.geom1
            out.append(m.geom(other).name or m.body(m.geom_bodyid[other]).name)
    return sorted(set(out))


class Snapshots:
    """Offscreen renders of the sandbox camera, one per phase."""

    def __init__(self, robot: OpenArmSim, out_dir: Optional[Path]) -> None:
        self.robot, self.out_dir, self.renderer = robot, out_dir, None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.renderer = mujoco.Renderer(robot.model, height=480, width=640)

    def __call__(self, name: str) -> None:
        if self.renderer is None:
            return
        self.renderer.update_scene(self.robot.data, camera="sandbox_overview")
        write_png(self.out_dir / f"{name}.png", self.renderer.render())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (real time)")
    parser.add_argument("--snapshots", type=Path, help="directory for one PNG per phase")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the sandbox scene first")
    parser.add_argument("--config", default="configs/task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")

    if args.rebuild or not SANDBOX_MODEL.is_file():
        build_sandbox(["dispenser", "cups"])
    cfg = yaml.safe_load(open(args.config))
    hw = cfg["hardware"]
    home = {s: np.asarray(cfg["robot"]["home_qpos"][s]) for s in SIDES}
    robot = OpenArmSim(
        mjcf_path=str(SANDBOX_MODEL),
        realtime=args.render,
        render=args.render,
        control_rate_hz=hw["control_rate_hz"],
        max_joint_velocity=hw["max_joint_velocity_rad_s"],
        max_gripper_velocity=hw["max_gripper_velocity_m_s"],
        joint_limit_margin=hw["joint_limit_margin_rad"],
        initial_qpos=home,
        gravity_compensation_scale=cfg["robot"].get("sim_gravity_compensation_scale", 1.0),
    )
    ik = IKController.from_config(cfg, robot)
    m, d = robot.model, robot.data
    lever_adr = m.jnt_qposadr[m.joint("dispenser_lever").id]
    side = args.side

    robot.start()
    snap = Snapshots(robot, args.snapshots)
    try:
        robot.advance(0.5)
        ik.reset()
        # Scene world -> IK base frame: the sandbox lifts the official robot root (no rotation).
        base = d.body("openarm_body_link0").xpos.copy()
        tip = d.site("dispenser_lever_tip").xpos.copy() - base
        p_home, R_home = ik.get_tcp_pose()[side]
        print(f"{side} TCP at home (base frame): {np.round(p_home, 3)}")
        print(f"lever tip (base frame):          {np.round(tip, 3)}   lever {d.qpos[lever_adr]:+.3f} rad")
        snap("0_home")

        target = tip + np.array([TIP_BACK_M, 0.0, 0.0])
        above = target + np.array([0.0, 0.0, ABOVE_M])
        press = target - np.array([0.0, 0.0, PRESS_M])
        err = ik.move_l({side: (above, R_home)}, duration=3.0, gripper_m={side: 0.0})
        reached = ik.get_tcp_pose()[side][0]
        print(f"above the tip: IK error {err[side] * 1e3:.1f} mm, measured TCP error {np.linalg.norm(reached - above) * 1e3:.1f} mm")
        snap("1_above")

        ik.move_l({side: (press, R_home)}, duration=2.0)
        robot.advance(0.5)
        pressed = float(d.qpos[lever_adr])
        reached = ik.get_tcp_pose()[side][0]
        print(f"pressed: lever {pressed:+.3f} rad ({np.degrees(pressed):.1f} deg), TCP z {reached[2]:.3f} (tip was {tip[2]:.3f})")
        print(f"         touching the lever: {lever_contacts(m, d)}")
        snap("2_pressed")

        ik.move_l({side: (above, R_home)}, duration=1.5)
        robot.advance(1.0)
        released = float(d.qpos[lever_adr])
        print(f"released: lever {released:+.3f} rad")
        snap("3_released")

        ik.move_l({side: (p_home, R_home)}, duration=3.0)
        snap("4_home")
        ok = pressed > LEVER_DISPENSING_RAD and released < 0.03
        print("RESULT:", "lever pressed and released" if ok else "lever NOT actuated as expected")
        return 0 if ok else 1
    finally:
        robot.stop()


if __name__ == "__main__":
    sys.exit(main())
