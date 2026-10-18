# Requirements

The demo is "vision cannot weigh":
1. An RGB camera finds two objects that look alike.
2. One OpenArm v1 arm lifts each object and estimates its mass from joint torques.
3. It compares the two masses, then sorts the objects into bins or hands one to a person.

A MuJoCo twin follows the real robot.

Each requirement has an ID. Tests name the IDs they cover (`tests/`). Hardware stages are in [hardware_bringup.md](hardware_bringup.md).

## Functional

| ID | Requirement |
|---|---|
| **FR1 Prompt** | A free-text instruction is parsed (rules, offline) into a `TaskIntent`: the object noun (GroundingDINO prompt), `mode` = `sort` or `handover`, the bin for the heavy and for the light object, and for handover which class (heavy/light) goes to the person. Defaults: object prompt from config; lighter object in the left bin. |
| **FR2 Detect** | Exactly 2 objects matching the prompt are located on the table in the robot base frame (x, y). Detections are fused over several frames and filtered by workspace. If the count is not 2, the task aborts before moving. `--objects manual` uses the positions in config instead. |
| **FR3 Pick & weigh** | The gripper is opened before any motion of a run. For each object in turn: approach, grasp (torque-mode gripper, closing at a regulated speed until contact), lift, move to the fixed weigh pose, estimate the mass (dithered joint-torque least squares), put it back where it was, and retreat through the approach pose. The result is stored as a belief `{name, x, y, mass_kg, se_kg}`. |
| **FR4 Compare** | `z = (m1 - m2) / hypot(se1, se2)`. With `|z| >= min_z` the heavier object is known. Otherwise the held (second) object is measured once more and the two estimates are combined by inverse variance. If it is still inconclusive, the object is put back and the arm goes home. |
| **FR5 Sort** | Each object is placed in the bin the intent assigns to its class (heavy/light). |
| **FR6 Handover** | The chosen object is carried to the handover pose. The arm records a torque baseline, then opens the gripper only when some joint's torque residual exceeds `max(k_sigma * std, min_delta_nm)` for `hold_s`. If there is no contact within `timeout_s`, it keeps gripping and places the object in its bin. The other object is sorted into its bin. |
| **FR7 Twin** | A MuJoCo viewer shows the measured arm and gripper state at `rate_hz` (default 30), plus the table, bins, the believed object positions (the held object moves with the hand), and the commanded vs measured TCP. It only ever reads the robot state. |
| **FR8 Calibration** | Two modes. *Intrinsics only*: `calib intrinsics` (ChArUco) writes `camera_intrinsics.yaml`, and the camera pose comes from the nominal mount in config. *Intrinsics + touch*: `calib touch` (robot fingertip touches + clicks, PnP) also writes `camera_extrinsics.yaml`, which overrides the nominal pose whenever it exists. The mode in use is logged. |
| **FR9 Bring-up** | `check`: read joints without enabling motors. `zero`: move safely to the rest pose. `grip`: close on and open an object. `weigh`: lift one object N times and report the mean and spread. `detect`: print and save detections. |

## Safety (hard; ported unchanged from the hardware-validated code on `main`)

| ID | Requirement |
|---|---|
| **SR1** | Inside the 250 Hz servo loop every target is clipped to the model joint limits minus a margin (widened to include the start pose) and rate-limited per joint; the gripper is rate-limited too. |
| **SR2** | Start is refused if any joint reads more than `startup_limit_tolerance_rad` outside its limits (bad zero calibration) or a motor does not answer. |
| **SR3** | A fault latches and freezes the arm at the measured pose on: a motor error code ≥ 0x8, no reply for `missing_reply_time_s`, an unhealthy CAN bus, or a tracking error > `max_tracking_error_rad` for `tracking_error_time_s`. Motions abort on a latched fault. |
| **SR4** | Before every Cartesian line the IK is checked at `check_points` along it (position/orientation tolerance). The arm bodies must stay `table_margin_m` above the table and `collision_clearance_m` from the torso and bins. Failing lines are refused without moving. |
| **SR5** | Gripper torque is capped at `max_torque_nm` and ramped at `torque_rate_nm_s`. |
| **SR6** | On Ctrl+C or any exception during the task, a held object is put back (if the arm is not faulted), then the arm parks at the rest pose and the motors are disabled. |

## Non-functional

| ID | Requirement |
|---|---|
| **NFR1** | One repo-local `.venv`, created by `./setup.sh` (uv). No Docker. `uv run robotokens <cmd>` runs everything. |
| **NFR2** | `uv run pytest` passes offline with no CAN, GPU, torch or weights. Vision-model tests are marked `vision`, hardware tests `hardware`. |
| **NFR3** | torch / GroundingDINO / SAM2 / `openarm_can` are imported lazily, only by code paths that need them. |
| **NFR4** | One config file (`configs/robotokens.yaml`), each value in one place, loaded into typed dataclasses. Unknown or missing keys raise errors. |
| **NFR5** | Every `run` writes `runs/<timestamp>/report.json`: intent, detections, beliefs, comparison, decision, phase log. |

## Out of scope (YAGNI)

Bimanual control, a physics simulation backend, motion planners (OMPL/RBS), hand-teleop, cross-embodiment, an LLM prompt parser, Docker.
