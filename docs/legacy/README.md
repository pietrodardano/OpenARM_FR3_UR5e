# Legacy: what we used to get started, and what is stale

> **Status: legacy.** Kept for reference; not part of the final demo.

One register for everything that no longer serves the final demo
([demo_task.md](../final_task/demo_task.md)) or whose documentation claims are stale.

**What was moved, and what was not.** Documents, scripts and the placeholder model were moved into `legacy/` folders:
`docs/legacy/`, `scripts/legacy/` and `configs/legacy/`. The source code under `src/` and the tests under `tests/`
were **not** moved (that would rewrite imports and touch files the other branches also edit), so the deprecated modules
below still sit in `src/`. They are listed here and flagged in the README. They get moved or removed after the branches are
integrated ([branches_and_integration.md](../final_task/branches_and_integration.md)).
Do not add features or new callers to anything listed here.

## Legacy files by folder

| Folder | Contents |
|---|---|
| `docs/legacy/` | This register |
| `scripts/legacy/` | `run_pipeline.py`, `calibrate_camera.py` (hand-tracking teleop), `run_pick_place_demo.py` (older sim demo) |
| `configs/legacy/` | `bimanual_openarms.xml` (5-DoF placeholder model) |
| `src/` (not moved) | `src/vision/hand_tracker_listener.py`, `src/control/bimanual_controller.py`, the older parts of `src/planning/ompl_planner.py` and `src/simulation/mujoco_env.py` |
| `tests/` (not moved) | `tests/test_pipeline.py` |

## Egocentric hand-tracking teleoperation

The original project mapped a wearable camera's hand tracking to robot targets. The final demo uses an RGB camera
with SAM2 and Grounding DINO instead, and the robot moves on vision targets, not on hand motion.

| Item | Where | Why deprecated | Replaced by |
|---|---|---|---|
| Hand tracker listener, mock stream, camera-to-robot mapper | `src/vision/hand_tracker_listener.py` (`HandTrackerListener`, `MockHandTrackerStream`, `EgocentricToRobotMapper`, `BimanualHandFrame`) | Hand tracking is not part of the demo | The vision branch (segmentation and object pose) |
| Hand-driven bimanual controller | `src/control/bimanual_controller.py` (`BimanualController`) | Built on hand targets, the old local IK and the heuristic planner | The deploy branch controller plus a task state machine |
| Teleop entry point | `scripts/legacy/run_pipeline.py` | Runs the hand-tracking pipeline | None needed |
| Hand-tracker calibration | `scripts/legacy/calibrate_camera.py` | Despite its name it zeroes the hand tracker offsets; it is **not** camera-intrinsics or hand-eye calibration | A real camera and hand-eye calibration (see [interfaces.md](../final_task/interfaces.md)) |
| Hand-tracking tests | `tests/test_pipeline.py`: `TestHandTrackerAndMapper`, `TestBimanualController` | Test the deprecated modules | Tests for the new vision and task code |
| Hand-tracking configuration | `configs/task_config.yaml`: `task` (teleop name), `vision` (stream, smoothing, mapping) | Parameters of the deprecated pipeline | A `demo` section when the demo parameters are measured |

## Superseded planning and IK code

| Item | Where | Why deprecated | Replaced by |
|---|---|---|---|
| Local position-only DLS IK | `BimanualMuJoCoEnv.solve_position_ik`, `solve_top_grasp_ik`, `compute_ik_step` in `src/simulation/mujoco_env.py` | Local minima, no limits inside the solve, no collision awareness; only the sim demo calls it | `mink` IK on the deploy branch |
| Heuristic bimanual planner | `BimanualOMPLPlanner.plan_bimanual_joint_path`, `plan_cartesian_goals`, `_is_bimanual_state_valid` in `src/planning/ompl_planner.py` | Not a real collision check; `plan_cartesian_goals` re-solves against unchanged state | `plan_single_arm_ompl` for single-arm planning (left arm) |
| Sim pick-and-place demo | `scripts/legacy/run_pick_place_demo.py` | Uses the deprecated IK, attaches the cube kinematically, left arm only; not the final demo | The task state machine; kept as a sim reference until then |
| Dead RBS planner import | `src/planning/ompl_planner.py` (`from robotblockset.planning import JointPathPlanner`) | That module does not exist | Remove when the file is next touched |
| 5-DoF placeholder model | `configs/legacy/bimanual_openarms.xml` | Not the 7-DoF OpenArm v1 | Official `openarm_mujoco/v1` model via `scripts/setup/setup_openarm_v1_scene.py` |

## Stale claims in documents and config

| Claim | Where | Reality |
|---|---|---|
| "14-DOF bimanual OMPL planning" | `README.md`, `agents.md` | Not implemented. The demo plans one arm (7 joints); the right arm is disabled |
| "6-DOF" and "10-DOF" arms | old `docs/final_task/architecture.md` | OpenArm v1 has 7 joints per arm |
| "RBS Jacobian DLS" IK and `JointPathPlanner` | old `docs/final_task/architecture.md`, `README.md` | RBS provides neither for the OpenArm; IK is custom (`main`) or `mink` (deploy) |
| "Dynamixel / serial bridge" | old `docs/final_task/hackathon_roadmap.md` | The arm uses Damiao motors over CAN through `openarm_can` |
| "Build: Passing" badge | `README.md` | A static image, not tied to any build |
| "Automatic contact E-Stop in simulation and hardware" | `README.md`, old `docs/final_task/architecture.md` | The hardware code on the deploy branch has no contact e-stop, only joint limits, velocity limits and latched faults |
| `safety.cartesian_bounds` z from 0.38 to 0.75 m | `configs/task_config.yaml` | Predates the 1.33 m shoulder mount; excludes the working volume |

## Moved back out of legacy

The cross-embodiment scene is part of the final task again: the simulated FR3 and UR5e now run the demo next to the
real OpenArm ([cross_embodiment.md](../final_task/cross_embodiment.md)). Its scripts live in `scripts/final_task/`.
`scripts/setup/generate_charuco_board.py`, `docs/setup/camera_specifications.md` and `docs/setup/physical_workcell.md`
stay in use.
