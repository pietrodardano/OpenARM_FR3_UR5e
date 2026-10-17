#!/usr/bin/env python3
"""Plan and visualize a collision-checked OpenArm v1 top-down pick-and-place.

This is a simulation demonstration only. OMPL queries MuJoCo contacts to reject
left-arm table and right-arm collisions. The object is kinematically attached
only during visualization; it is not a validated grasp or hardware controller.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"
VENV_PYTHON = VENV_BIN / "python"
VENV_MJPYTHON = VENV_BIN / "mjpython"


def _ensure_project_interpreter(show_viewer: bool) -> None:
    """Relaunch under this repository's venv, using mjpython for macOS GUI.

    Activating a virtual environment only changes a parent shell's environment,
    which a child script cannot do. Re-executing makes direct editor or system
    Python launches use the reproducible project interpreter instead.
    """
    target = VENV_MJPYTHON if show_viewer and sys.platform == "darwin" else VENV_PYTHON
    if not target.is_file():
        raise FileNotFoundError(
            f"Project interpreter not found: {target}. Create it with `uv sync --extra rbs --extra dev`."
        )
    launched_with_mjpython = os.environ.get("ROBOTTOKENS_MJPYTHON") == "1"
    already_in_venv = Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    if (show_viewer and sys.platform == "darwin" and not launched_with_mjpython) or (
        not show_viewer and not already_in_venv
    ):
        environment = os.environ.copy()
        if target == VENV_MJPYTHON:
            environment["ROBOTTOKENS_MJPYTHON"] = "1"
        os.execve(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap before importing packages that may only exist in the project venv.
_ensure_project_interpreter("--show" in sys.argv[1:])

import mujoco
import numpy as np

from src.planning.ompl_planner import BimanualOMPLPlanner
from src.simulation.mujoco_env import BimanualMuJoCoEnv


SCENE_PATH = PROJECT_ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml"
PICK_CENTER = np.array([0.259, 0.194, 0.800])
PLACE_CENTER = np.array([0.200, 0.250, 0.750])
TOP_CLEARANCE_M = 0.150
LATERAL_RAISE_TCP = np.array([0.250, 0.500, 1.050])
# Verified reachable TCP pose for a lateral parallel-link grasp. The cube is
# centered between the official finger collision meshes at this posture.
LATERAL_GRASP_TCP = np.array([0.305, 0.183, 0.820])
VISUAL_GRIPPER_CLOSED_POSITION_M = 0.020


def _solve_or_raise(env: BimanualMuJoCoEnv, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
    solution = env.solve_position_ik("left", target, seed)
    if solution is None:
        raise RuntimeError(f"No position IK solution for target {target}.")
    return solution


def _plan_segment(
    planner: BimanualOMPLPlanner,
    env: BimanualMuJoCoEnv,
    start: np.ndarray,
    goal: np.ndarray,
    label: str,
    planning_time_sec: float = 3.0,
    interpolation_count: int = 80,
    planner_range: float = 0.20,
) -> np.ndarray:
    path = planner.plan_single_arm_ompl(
        env,
        "left",
        start,
        goal,
        planning_time_sec=planning_time_sec,
        interpolation_count=interpolation_count,
        planner_range=planner_range,
    )
    if path is None:
        raise RuntimeError(f"OMPL could not find a collision-free path for {label}.")
    print(f"{label}: {len(path)} collision-checked waypoints")
    return path


def build_task_plan(
    env: BimanualMuJoCoEnv,
    planning_time_sec: float = 3.0,
    interpolation_count: int = 80,
    planner_range: float = 0.20,
) -> list[tuple[str, np.ndarray, bool]]:
    """Create lateral raise, top approach, grasp, transfer, place, and retreat."""
    planner = BimanualOMPLPlanner()
    start = env.get_bimanual_state().left.joint_positions
    pick_approach = LATERAL_GRASP_TCP + np.array([0.0, 0.0, TOP_CLEARANCE_M])
    pick_grasp = LATERAL_GRASP_TCP
    place_approach = PLACE_CENTER + np.array([0.0, 0.0, TOP_CLEARANCE_M])
    place_release = PLACE_CENTER + np.array([0.0, 0.0, 0.070])

    q_lateral_raise = _solve_or_raise(env, LATERAL_RAISE_TCP, start)
    q_pick_approach = _solve_or_raise(env, pick_approach, q_lateral_raise)
    q_pick_grasp = _solve_or_raise(env, pick_grasp, q_pick_approach)
    q_place_approach = _solve_or_raise(env, place_approach, q_pick_grasp)
    q_place_release = _solve_or_raise(env, place_release, q_place_approach)

    return [
        ("lateral raise", _plan_segment(planner, env, start, q_lateral_raise, "lateral raise", planning_time_sec, interpolation_count, planner_range), False),
        ("move over table", _plan_segment(planner, env, q_lateral_raise, q_pick_approach, "move over table", planning_time_sec, interpolation_count, planner_range), False),
        ("vertical descend", _plan_segment(planner, env, q_pick_approach, q_pick_grasp, "vertical descend", planning_time_sec, interpolation_count, planner_range), False),
        ("lift", _plan_segment(planner, env, q_pick_grasp, q_pick_approach, "lift", planning_time_sec, interpolation_count, planner_range), True),
        ("transfer", _plan_segment(planner, env, q_pick_approach, q_place_approach, "transfer", planning_time_sec, interpolation_count, planner_range), True),
        ("vertical place", _plan_segment(planner, env, q_place_approach, q_place_release, "vertical place", planning_time_sec, interpolation_count, planner_range), True),
        ("retreat", _plan_segment(planner, env, q_place_release, q_place_approach, "retreat", planning_time_sec, interpolation_count, planner_range), False),
    ]


def run_robustness_trials(trials: int, base_seed: int) -> None:
    """Restart and replan the complete task under distinct OMPL random seeds."""
    from ompl import util as ou

    # OMPL's process-wide RNG can only be seeded before it creates samplers.
    # It then advances naturally between trials; each fresh environment below
    # deliberately varies the RRTConnect configuration as well.
    ou.RNG.setSeed(base_seed)
    planner_ranges = (0.12, 0.16, 0.20, 0.28, 0.35)
    planning_budgets = (1.0, 1.5, 2.0, 2.5, 3.0)
    interpolation_counts = (50, 60, 80, 100, 120)
    successes = 0
    total_waypoints = 0
    durations: list[float] = []
    for trial_index in range(trials):
        planner_range = planner_ranges[trial_index % len(planner_ranges)]
        planning_budget = planning_budgets[trial_index % len(planning_budgets)]
        interpolation_count = interpolation_counts[trial_index % len(interpolation_counts)]
        env = BimanualMuJoCoEnv(xml_path=str(SCENE_PATH))
        started_at = perf_counter()
        try:
            task_plan = build_task_plan(env, planning_budget, interpolation_count, planner_range)
            elapsed = perf_counter() - started_at
            waypoint_count = sum(len(path) for _, path, _ in task_plan)
            successes += 1
            total_waypoints += waypoint_count
            durations.append(elapsed)
            print(
                f"trial {trial_index + 1}/{trials}: PASS "
                f"range={planner_range:.2f} budget={planning_budget:.1f}s "
                f"interpolation={interpolation_count} waypoints={waypoint_count} planning={elapsed:.3f}s"
            )
        except RuntimeError as error:
            elapsed = perf_counter() - started_at
            print(f"trial {trial_index + 1}/{trials}: FAIL after {elapsed:.3f}s: {error}")
        finally:
            env.close()

    if successes != trials:
        raise RuntimeError(f"Robustness run failed: {successes}/{trials} trials completed.")
    mean_duration = float(np.mean(durations)) if durations else 0.0
    print(
        f"Robustness passed: {successes}/{trials} clean restarts; "
        f"{total_waypoints} total waypoints; mean planning {mean_duration:.3f}s."
    )


def visualize_task(env: BimanualMuJoCoEnv, task_plan: list[tuple[str, np.ndarray, bool]]) -> None:
    """Show the task, attaching only after physical finger-to-cube contact."""
    assert env.model is not None and env.data is not None
    cube_joint = env.model.joint("pick_cube_freejoint")
    cube_qpos_address = env.model.jnt_qposadr[cube_joint.id]
    left_finger_qpos_indices = [
        env.model.jnt_qposadr[env.model.joint(name).id]
        for name in ("openarm_left_finger_joint1", "openarm_left_finger_joint2")
    ]
    grasp_offset_in_tcp: np.ndarray | None = None
    grasp_orientation_in_tcp: np.ndarray | None = None
    grasp_confirmed = False

    def has_finger_cube_contact() -> bool:
        """Return whether either official left finger contacts the cube geometry."""
        for contact_index in range(env.data.ncon):
            contact = env.data.contact[contact_index]
            first = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            second = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            names = {first, second}
            if "pick_cube_geom" in names and any(
                name in names
                for name in (
                    "openarm_left_left_finger_collision",
                    "openarm_left_right_finger_collision",
                )
            ):
                return True
        return False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for label, path, cube_attached in task_plan:
            print(f"visualizing {label}")
            for waypoint_index, joint_target in enumerate(path):
                env.data.qpos[env.left_qpos_indices] = joint_target
                mujoco.mj_forward(env.model, env.data)
                if label == "vertical descend" and waypoint_index == len(path) - 1:
                    # Close at the reached grasp posture, then query MuJoCo
                    # contacts. No contact means no simulated grasp.
                    env.data.qpos[left_finger_qpos_indices] = VISUAL_GRIPPER_CLOSED_POSITION_M
                    mujoco.mj_forward(env.model, env.data)
                    grasp_confirmed = has_finger_cube_contact()
                    print("grasp contact confirmed" if grasp_confirmed else "grasp failed: fingers do not contact cube")

                if cube_attached and grasp_confirmed:
                    tcp_position = env.data.xpos[env.left_tcp_body_id].copy()
                    tcp_rotation = env.data.xmat[env.left_tcp_body_id].reshape(3, 3).copy()
                    if grasp_offset_in_tcp is None or grasp_orientation_in_tcp is None:
                        # The lift begins at the exact grasp posture. Capture the
                        # cube where it physically is, relative to the TCP, so
                        # the object does not jump up into the gripper.
                        cube_position = env.data.qpos[cube_qpos_address:cube_qpos_address + 3].copy()
                        cube_quaternion = env.data.qpos[cube_qpos_address + 3:cube_qpos_address + 7].copy()
                        cube_rotation = np.empty(9)
                        mujoco.mju_quat2Mat(cube_rotation, cube_quaternion)
                        cube_rotation = cube_rotation.reshape(3, 3)
                        grasp_offset_in_tcp = tcp_rotation.T @ (cube_position - tcp_position)
                        grasp_orientation_in_tcp = tcp_rotation.T @ cube_rotation
                        print("grasp established at the cube's physical tabletop pose")

                    cube_position = tcp_position + tcp_rotation @ grasp_offset_in_tcp
                    cube_rotation = tcp_rotation @ grasp_orientation_in_tcp
                    cube_quaternion = np.empty(4)
                    mujoco.mju_mat2Quat(cube_quaternion, cube_rotation.ravel())
                    env.data.qpos[cube_qpos_address:cube_qpos_address + 3] = cube_position
                    env.data.qpos[cube_qpos_address + 3:cube_qpos_address + 7] = cube_quaternion
                    env.data.qvel[env.model.jnt_dofadr[cube_joint.id]:env.model.jnt_dofadr[cube_joint.id] + 6] = 0.0
                    mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                time.sleep(0.025)
            if label == "vertical place":
                # The next (retreat) segment leaves the cube at this reached
                # release pose while the hand moves away and opens.
                env.data.qpos[left_finger_qpos_indices] = 0.0
                mujoco.mj_forward(env.model, env.data)
        viewer.sync()
        time.sleep(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open MuJoCo and animate the planned task.")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Restart and replan this many times with varied RRTConnect configurations (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=20261017, help="OMPL random seed used once before all trials.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1.")
    if args.show and args.trials != 1:
        parser.error("--show supports exactly one trial; run robustness mode without --show.")
    if not SCENE_PATH.exists():
        raise FileNotFoundError("Generate the physical scene first with scripts/setup/setup_openarm_v1_scene.py.")
    if args.trials > 1:
        run_robustness_trials(args.trials, args.seed)
        return
    env = BimanualMuJoCoEnv(xml_path=str(SCENE_PATH))
    try:
        task_plan = build_task_plan(env)
        total_waypoints = sum(len(path) for _, path, _ in task_plan)
        print(f"Planned top-down pick-and-place: {total_waypoints} collision-checked waypoints.")
        if args.show:
            visualize_task(env, task_plan)
    finally:
        env.close()


if __name__ == "__main__":
    main()