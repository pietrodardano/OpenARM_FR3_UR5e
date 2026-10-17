#!/usr/bin/env python3
"""
Main Execution Entrypoint: Bimanual OpenArms Teleoperation Pipeline.

Connects DaiDai Lab egocentric hand tracking stream (or mock stream),
maps human gestures to robot base frame, plans collision-free motion via
RobotBlockSet (RBS) and OMPL, and steps the MuJoCo simulation environment.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import logging
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.vision.hand_tracker_listener import MockHandTrackerStream, EgocentricToRobotMapper
from src.planning.ompl_planner import BimanualOMPLPlanner
from src.simulation.mujoco_env import BimanualMuJoCoEnv
from src.control.bimanual_controller import BimanualController


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found at {config_path}. Using internal defaults.")
        return {}
    if not HAS_YAML:
        logging.info("PyYAML not installed in Python environment. Using defaults with config path recorded.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DaiDai Lab Egocentric Vision -> RBS / MuJoCo OpenArms Pipeline")
    parser.add_argument("--config", type=str, default="configs/task_config.yaml", help="Path to YAML task config")
    parser.add_argument("--mock-stream", action="store_true", default=True, help="Use mock hand tracker stream (default)")
    parser.add_argument("--steps", type=int, default=150, help="Number of control cycles to execute")
    parser.add_argument("--render", action="store_true", default=False, help="Launch interactive MuJoCo passive viewer")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("RoboTokens.Main")
    logger.info("Starting DaiDai Lab + RBS + MuJoCo OpenArms Pipeline...")

    cfg = load_config(args.config)
    vis_cfg = cfg.get("vision", {})
    mapping_cfg = vis_cfg.get("mapping", {})
    plan_cfg = cfg.get("planning", {})
    safety_cfg = cfg.get("safety", {})
    sim_cfg = cfg.get("simulation", {})

    # 1. Initialize Simulation Environment
    mjcf_file = sim_cfg.get("mjcf_path", "configs/legacy/bimanual_openarms.xml")
    env = BimanualMuJoCoEnv(
        xml_path=mjcf_file,
        timestep=sim_cfg.get("sim_timestep", 0.002),
        render=args.render,
    )

    # 2. Initialize Egocentric Coordinate Mapper
    bounds = safety_cfg.get("cartesian_bounds", None)
    mapper = EgocentricToRobotMapper(
        scale_factor=mapping_cfg.get("scale_factor", 1.2),
        alpha=vis_cfg.get("smoothing", {}).get("alpha", 0.75),
        axis_permutation=tuple(mapping_cfg.get("axis_permutation", [2, 0, 1])),
        axis_signs=tuple(mapping_cfg.get("axis_signs", [1.0, -1.0, -1.0])),
        left_offset=mapping_cfg.get("left_nominal_offset", [0.35, 0.18, 0.55]),
        right_offset=mapping_cfg.get("right_nominal_offset", [0.35, -0.18, 0.55]),
        workspace_bounds=bounds,
    )

    # 3. Initialize RBS & OMPL Planner
    planner = BimanualOMPLPlanner(
        planner_type=plan_cfg.get("planner_type", "RRTConnect"),
        max_planning_time=plan_cfg.get("max_planning_time_sec", 0.15),
        clearance_margin=plan_cfg.get("collision_clearance", 0.015),
        interpolation_steps=plan_cfg.get("interpolation_steps", 25),
        max_joint_velocity=safety_cfg.get("max_joint_velocity_rad_s", 2.5),
    )

    # 4. Initialize Bimanual Coordination Controller
    controller = BimanualController(
        env=env,
        planner=planner,
        mapper=mapper,
        replan_delta_threshold=plan_cfg.get("min_cartesian_delta_for_replan", 0.03),
        min_ee_distance=safety_cfg.get("min_bimanual_ee_distance", 0.08),
    )

    # 5. Connect Vision Stream
    stream = MockHandTrackerStream(fps=vis_cfg.get("fps", 30))
    stream.connect()
    controller.start()

    logger.info(f"Pipeline running for {args.steps} iterations...")
    loop_period = 1.0 / vis_cfg.get("fps", 30)

    try:
        for step in range(args.steps):
            cycle_start = time.time()

            # Poll latest hand frame
            hand_frame = stream.get_latest_frame()

            # Step control loop
            robot_state = controller.update(hand_frame)

            # Telemetry logging every 25 steps
            if step % 25 == 0:
                l_pos = robot_state.left.ee_position
                r_pos = robot_state.right.ee_position
                l_grip = robot_state.left.gripper_position
                r_grip = robot_state.right.gripper_position
                logger.info(
                    f"[Step {step:03d}/{args.steps}] "
                    f"Left EE: [{l_pos[0]:.2f}, {l_pos[1]:.2f}, {l_pos[2]:.2f}] | Grip: {l_grip:.3f} || "
                    f"Right EE: [{r_pos[0]:.2f}, {r_pos[1]:.2f}, {r_pos[2]:.2f}] | Grip: {r_grip:.3f} | "
                    f"Collision: {robot_state.in_collision}"
                )

            # Maintain frequency
            elapsed = time.time() - cycle_start
            sleep_t = loop_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by operator.")
    finally:
        stream.disconnect()
        env.close()
        logger.info("Pipeline closed safely. Ready for hackathon evaluation.")


if __name__ == "__main__":
    main()
