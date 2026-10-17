# Branches and Integration

> **Status: final task.** Describes the demo the project is building toward.

## Policy: nothing is merged yet

Several lines of work run in parallel. **None of them is merged into another yet**, and `main` does not take code
from them until the team decides how. This page records what each line is, where they overlap, and what must
happen before they meet. Facts about `origin/deploy/openarm-real` come from reading it at tip `814dcfc`;
check again if it has moved.

## Branch map

| Line | Purpose | Status | Owner |
|---|---|---|---|
| `main` | Documentation and the older simulation pipeline (egocentric hand-tracking teleop, now deprecated: see [legacy register](../legacy/README.md)) | Docs being aligned with the final demo | TBD |
| `origin/deploy/openarm-real` | **The real-robot controller.** `IKController` (`mink` IK), a 250 Hz driver on the official `openarm_can` binding, a safety filter, Docker, a bring-up checklist in `docs/real_robot_deployment.md`. Canonical for hardware | Reported working and robust by the team. Not validated by us. No recorded results in the branch | TBD |
| Vision branch | Segmentation (SAM2 + Grounding DINO, optional VLM) and object X-Y pose from the RGB camera | In progress. **Not inspected**: its name and contents are deliberately not recorded here | TBD |
| `openarm-cartesian-sim-validated` | Simulation and feasibility lab: global IK with reach certificate, collision-aware `mink` servo, physics test plant, staged hardware test runner, calibration notes. **Stays unmerged** and is not on the real-robot path | Simulation-validated (34 tests in the commit), never run on the arm | TBD |
| `sim-cross-embodiment-demo` | Simulated FR3 and UR5e running the demo next to a follower OpenArm ([cross_embodiment.md](cross_embodiment.md)), the state-stream contract, the folder reorganization of `main` | Simulation-validated (24 tests); uncommitted. It copies `src/kinematics/arm_kinematics.py` from the sim lab as a plain file, so the two branches carry the same file | TBD |
| Weight-from-torque estimator | Mass estimation from joint torques | **Not pushed yet.** Only its contract is documented ([interfaces.md](interfaces.md)) | TBD |
| `demo-simulation`, `feat/scripted_pick_and_place` | Other branches exist on origin | Not inspected; not covered here | TBD |

## What the deploy branch has and lacks

Has: Cartesian control (`IKController.command_poses`, `move_l`, `command_delta`), forward kinematics, a joint safety
filter (limits, velocity limits, latched faults), CAN bring-up scripts, Docker for the robot PC.

Lacks: weight estimation, grasp/lift/place routines, torque-based contact detection, a vision interface, a hand-eye
calibration, any per-arm enable switch, and gripper force control or torque readout.

## Blockers before any run on the real robot

1. **The right arm is always enabled, and a silent right arm stops the driver.** `OpenArmBimanual.start()` enables
   both CAN buses and the config has no per-arm flag. It also requires every motor of **both** arms to reply; if any
   motor stays silent it raises `RuntimeError` and shuts the buses down. The right arm has a known hardware problem and
   must stay off, so as written the driver cannot run the left arm alone. A per-arm enable gate is needed in the
   deploy driver first. The sim-lab branch has an example of the rule (a `hardware.<side>.enabled` config flag and a
   helper that refuses a disabled side), written but not committed there. Until the deploy driver is gated, do not
   run it with both arms connected.
2. **Joint zero and signs.** Calibrate the left arm with
   `openarm-can-zero-position-calibration --arm-side left_arm --canport can1 --robot-version v1` (the script's
   defaults are the right arm on `can0` and version v2), then check the readings against the model.
3. **Gravity compensation** is off by default in the deploy config (`gravity_compensation_scale: 0.0`). Without it
   the arm sags; raise it gradually and watch the TCP error.
4. **Hand-eye calibration** between the camera and the robot base does not exist yet.

## Overlaps and duplicates

| Topic | Deploy branch | Sim-lab branch |
|---|---|---|
| IK | `src/planning/openarm_ik.py` (`BimanualIK`, `mink`) | `src/kinematics/openarm_ik.py` (global, structure-aware) and `src/control/openarm_servo.py` (`mink` servo) |
| Hardware driver | `src/hardware/openarm_bus.py`, `openarm_bimanual.py` (250 Hz, both arms) | `src/hardware/interface.py` (`OpenArmCanInterface`, one arm) |
| Config | `robot`, `hardware`, `controllers` sections in `configs/task_config.yaml` | A different `hardware` section in the same file |
| Docs | `docs/real_robot_deployment.md`, edited `README.md` and `agents.md` | `docs/control_stack.md`, `openarm_calibration.md`, `first_hardware_test.md`, `colleague_handoff.md` |

Expected conflicts when the lines meet: `README.md`, `agents.md`, `configs/task_config.yaml`, `pyproject.toml`,
`uv.lock`, `src/planning/ompl_planner.py`, `src/simulation/mujoco_env.py`, `tests/test_pipeline.py`.
The deploy branch also **deletes** files that `main` and the sim lab use, at their *old* paths: `docs/cross_embodiment.md`,
`docs/physical_workcell.md`, `scripts/run_pick_place_demo.py`, `scripts/setup_cross_embodiment_scene.py`,
`scripts/setup_openarm_v1_scene.py`, `scripts/view_cross_embodiment_scene.py`. On `main` these were since **moved** (see
[docs/README.md](../README.md) for the new layout), so a merge shows rename-versus-delete conflicts instead of a clean delete.
Merging as is would remove them. Decide that deliberately; do not let a merge decide it.

## Proposed integration order (for the team to confirm)

1. Add the per-arm enable gate to the deploy driver and config.
2. Merge the deploy controller as the base for real control, resolving the conflicts above in one reviewed pass and
   deciding the fate of the deleted scene scripts and workcell docs.
3. Add the vision branch as a module that produces `ObjectDetection` ([interfaces.md](interfaces.md)).
4. Add the mass estimator and a contact detector under a sensing module, then the task state machine
   ([demo_task.md](demo_task.md)).
5. Port selected sim-lab pieces by pull request, not by merging the branch.

## Sim-lab pieces that could move into the deploy stack later

| Piece | Where (sim-lab branch) | Why it helps |
|---|---|---|
| Per-arm enable gate | `src/hardware/arm_selection.py` plus the config flags (uncommitted) | Closes blocker 1 |
| Global IK with reach certificate | `src/kinematics/openarm_ik.py` | Finds all configurations and tells you a pose is unreachable, with a reason |
| Goal feasibility check with table/torso/other-arm clearance | `OpenArmServo.check_goal` in `src/control/openarm_servo.py` | Rejects bad grasp poses before moving |
| Physics test plant (gravity, friction, latency, noise) | `SimArmInterface` in `src/hardware/interface.py` | Test controllers and the mass estimator without the arm |
| Staged hardware test runner | `scripts/run_openarm_cartesian.py` | Mirror, ready, hold, move and precision stages with logging |
| Hold-stage gravity bias | the `hold` stage of that runner | A static-torque baseline, close to what mass estimation needs |
| Reachability measurements | [demo_task.md](demo_task.md) section "Constraints" | Table height and grasp angle planning |
