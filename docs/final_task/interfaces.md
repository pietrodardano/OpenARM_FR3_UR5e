# Interfaces Between Components

> **Status: final task.** Describes the demo the project is building toward.

Contracts between the vision, control, proprioception and task parts of the demo. Each item is marked:

- **existing**: found in code, with the path and symbol. The control items were read on
  `origin/deploy/openarm-real` at tip `814dcfc`; re-check them if that branch has moved.
- **proposed**: a contract suggested here for the owning team to confirm or change. Nothing in this file is
  implemented on `main`.

See [branches_and_integration.md](branches_and_integration.md) for who owns what, and
[demo_task.md](demo_task.md) for how the pieces are used.

## Frames and units

| Item | Value |
|---|---|
| Robot base frame | **existing** (deploy): $+X$ forward, $+Y$ left, $+Z$ up, origin on the floor under the torso |
| Units | metres, radians, newton-metres, seconds |
| Object Z | not measured by vision; taken from the table top height ([physical_workcell.md](../setup/physical_workcell.md): 0.750 m) plus the object grasp offset |
| Camera | fixed on the robot torso (nominally at $(0.070, 0, 1.600)$ m, pitched $20^\circ$ down; the values are provisional) |

## Vision to task

### `ObjectDetection` (proposed)

| Field | Type | Meaning |
|---|---|---|
| `object_id` | int | Stable id within one run, so "object 1" stays object 1 |
| `label` | str | Class from Grounding DINO / VLM (for example `can`, `box`) |
| `confidence` | float | 0 to 1 |
| `x_m`, `y_m` | float | Object centre in the robot base frame |
| `yaw_rad` | float, optional | Object orientation about $Z$ (boxes need it, cans do not) |
| `timestamp_s` | float | Capture time on a monotonic clock |
| `camera_to_base_version` | str | Which hand-eye transform produced these numbers |

Rules:
- Vision converts pixels to robot-base X-Y itself using the hand-eye calibration. The task layer never sees pixels.
- A detection older than a set age is stale and must not be used to move the arm.
- If the two objects are not both detected, vision reports that explicitly instead of repeating an old pose.

### Hand-eye calibration (required, missing)

Nothing on any inspected branch links the camera frame to the robot base frame. It is a required deliverable shared
by the vision and robot sides. The camera is fixed on the torso, so the transform is nominally known from the mount
geometry, but it must be calibrated (the ChArUco board assets in `configs/` and `scripts/setup/generate_charuco_board.py`
exist for this). Intrinsics also need calibrating; the current 70 degree field of view is a temporary estimate.

## Task to controller

### Controller API (existing, `origin/deploy/openarm-real`)

| Item | File and symbol | Notes |
|---|---|---|
| Read the TCP pose | `src/control/ik_controller.py`, `IKController.get_tcp_pose()` | Forward kinematics of the measured joints, per side |
| Absolute TCP command | `IKController.command_poses(poses, gripper_m=None, limit_step=True)` | Servo-style; clips to a per-arm workspace box and step limits |
| Straight-line move | `IKController.move_l(poses, duration=2.0, gripper_m=None, rate_hz=50)` | Blocking; minimum-jerk timing; raises `RuntimeError` if the backend faults |
| Relative command | `IKController.command_delta(dpos, drot=None, gripper_m=None)` | Applies to the previous commanded pose |
| Gripper | `gripper_m` argument, 0 to 0.044 m opening | Position only: no force command, no gripper torque readout |
| Robot state | `src/control/robot_interface.py`, `RobotBackend.get_state()` returning `RobotState` with `ArmState(q, dq, tau, gripper, q_setpoint, gripper_setpoint, motor_errors)` | `tau` is the measured joint torque, 7 values per arm |
| Faults | `RobotBackend.fault`, `clear_fault()` | Latched; the driver holds the last measured pose until cleared |
| Gravity torques | `src/planning/openarm_kinematics.py`, `OpenArmKinematics.gravity_torques(q_arm)` | Model gravity torques from the official MJCF; not fitted to real data |

Details checked in the code: `RobotState` holds `timestamp`, `arms` (a dict per side), `fault` and `stats`.
`ArmState.motor_errors` is the raw status per motor. `command_poses` and `command_delta` return the per-arm IK position
error in metres. Poses are per-side mappings (`Mapping[Side, Pose]`), so a single-arm command passes only `"left"`.

Gap: `OpenArmBimanual.start()` enables both arms and needs every motor of both arms to reply, otherwise it raises
and shuts the buses down. There is no per-arm flag, so the left arm cannot run alone yet (see
[branches_and_integration.md](branches_and_integration.md)).

### Pick and place commands (proposed)

The task layer should only need these from the controller side: move the TCP to a pose, set the gripper opening,
read the state, and report faults. Grasp, lift, place and hand-over routines belong in the task layer and are
**not implemented** on any branch inspected.

## Proprioception to task

The mass estimator is **not pushed yet**. The contract below is what the task layer needs; the code owner should
confirm or adjust it.

### `GraspMeasurement` (proposed)

| Field | Type | Meaning |
|---|---|---|
| `object_id` | int | Which object was weighed |
| `mass_kg` | float | Estimated mass |
| `mass_std_kg` | float | Uncertainty of the estimate (from window noise) |
| `pose_q` | 7 floats | Joint angles during the weighing window |
| `tau_loaded`, `tau_unloaded` | 7 floats each | Mean torque with and without the object at the same pose |
| `window_s` | float | Length of the averaging window |
| `steady` | bool | The arm was still during the window |

### `ContactEvent` (proposed)

| Field | Type | Meaning |
|---|---|---|
| `contact` | bool | A sustained torque change against the baseline was detected |
| `delta_tau` | 7 floats | Change from the baseline hold at the hand-over pose |
| `score` | float | Change relative to the baseline noise |
| `timestamp_s` | float | When the criterion was met |

## Real-arm state stream (proposed)

Used by the cross-embodiment demo ([cross_embodiment.md](cross_embodiment.md)) to move the rendered OpenArm and to
keep the simulated robots in step with the real task. Implemented on the simulation side in
`src/cross_embodiment/stream.py` (branch `sim-cross-embodiment-demo`); the real side still has to publish it.

One JSON object per line, over UDP, about 50 Hz:

```json
{"t": 12.34, "q": [0.1, -0.4, 0.2, 0.9, 0.0, -0.1, 0.3], "gripper_m": 0.044, "phase": "WEIGH_1"}
```

| Field | Type | Meaning |
|---|---|---|
| `t` | float | Seconds on the sender's clock |
| `q` | 7 floats | Measured left OpenArm joint angles (rad), joint 1 to 7 |
| `gripper_m` | float or null | Measured gripper opening (m), 0 to 0.044 |
| `phase` | string or null | Current real task phase, one of `INIT, PICK_1, LIFT_1, WEIGH_1, PLACE_BACK_1, PICK_2, LIFT_2, WEIGH_2, PLACE_BACK_2, COMPARE, ACT, DONE` |

Rules: phases only move forward; the receiver keeps the newest valid message and drops malformed ones; `q` is the
measured state, never the command. A reference publisher that sends the scripted stream is
`scripts/final_task/publish_mock_state.py`.

