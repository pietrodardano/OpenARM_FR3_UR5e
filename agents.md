# AGENTS.MD: Engineering Context & AI Agent Directives

## 1. Project Mission & Hackathon Context
- **Hackathon:** 30-Hour Robotics AI Sprint, Track 2 (OpenArms + RobotBlockSet + MuJoCo).
- **Final demo: "Vision cannot weigh".** Full spec: [docs/demo_task.md](docs/demo_task.md).
  1. An RGB camera with GroundingDINO + SAM2 (+ optional VLM) finds two look-alike objects (two cans or two boxes) on a table of known height.
  2. The OpenArm grasps and lifts each object, estimates its mass from **joint torques**, puts it back, and compares the two.
  3. It then **sorts** them into predefined bins by weight, or **hands the heavy one to the user**. In the handover it opens the gripper only after it *feels* the user's grasp through the torques.
- **Core claim:** vision alone cannot tell full from empty; interaction + proprioception can.

---

## 2. Branch Policy (read before editing)

| Line | Content | Rule |
|---|---|---|
| `main` | Task state machine ([src/task/](src/task/)), the proven real-robot controller (copied from `deploy/openarm-real`), simulation workcell, docs | Normal development |
| `deploy/openarm-real` | Source of the real-robot controller now on `main`: `RobotBackend`, Cartesian `IKController.move_l`, gripper torque mode, `payload.py` weighing, `OpenArmSim` | **Do not merge wholesale.** Its useful files were copied to `main` (see [docs/demo_task.md §9](docs/demo_task.md#9-integration-status)); change them on `main` |
| `demo-simulation` (team-mate) | Vision pipeline (GroundingDINO + SAM2, camera calibration) plus unrelated sandbox demos | Vision files of commit `6b5d670` copied to `main` (`src/vision/`, calibration scripts). **Do not merge the rest** |

- The task layer depends only on the Protocols in [src/task/interfaces.py](src/task/interfaces.py) (`ObjectDetector`, `ManipulationSkills`, `PayloadEstimator`). The robot side is implemented by [src/task/openarm_adapter.py](src/task/openarm_adapter.py). Vision is [src/vision/vision_detector.py](src/vision/vision_detector.py) (`VisionObjectDetector`); `demo.manual_objects` / `--objects` remain the manual fallback. torch / GroundingDINO / SAM2 are not installed on the integration Mac: import them lazily, and keep tests runnable without them.
- Do not re-implement vision on `main`. Treat the proven controller files (`src/hardware/`, `src/control/{robot_interface,safety,ik_controller,joint_position_controller}.py`, `src/estimation/payload.py`) as validated on hardware: change them only deliberately, and re-test on `--backend fake` / `sim`.

---

## 3. Technology Stack & Ecosystem References

| Component | Technology | Reference / Source |
|---|---|---|
| **Robotics Framework** | RobotBlockSet (`robotblockset_python` by JSI) | [abr-ijs/robotblockset_python](https://github.com/abr-ijs/robotblockset_python) |
| **OpenArm Control SDK** | `openarm_can` | [enactic/openarm_can](https://github.com/enactic/openarm_can) — official C++ SocketCAN control library; its Python bindings are experimental |
| **OpenArm Robot Model** | `openarm_description` | [enactic/openarm_description](https://github.com/enactic/openarm_description) — official URDF/xacro, joint metadata, and meshes |
| **OpenArm Simulator Model** | `openarm_mujoco` | [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) — official MuJoCo XML models and collision assets |
| **OpenArm CAD & BOM** | `openarm_hardware` | [enactic/openarm_hardware](https://github.com/enactic/openarm_hardware) — links to official STEP, BOM, and assembly sources |
| **Simulation Backend** | MuJoCo (v3.0+) | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) |
| **Motion Planning** | OMPL (optional; RBS only after model compatibility is verified) | [OMPL Library](https://ompl.kavrakilab.org/) |
| **Robot Hardware Setup** | OpenArm v1 bimanual, parallel-link grippers (left arm active) | [enactic/openarm](https://github.com/enactic/openarm) |
| **Camera** | HBV HD CAMERA, RGB only, torso-mounted | [docs/camera_specifications.md](docs/camera_specifications.md) |
| **Object Perception** | GroundingDINO (open-vocabulary boxes), SAM2 (masks), top-face PCA | `src/vision/`, deps in `requirements-vision.txt` |
| **Language & Tooling** | Python 3.10+, NumPy, SciPy, PyYAML, uv | Modern Python packaging |

---

## 4. System Architecture & Information Flow

```
[ RGB camera (torso mount) ]
    │  frames @ 30 Hz
    ▼
[ src.vision.vision_detector: GroundingDINO → SAM2 → top-face PCA ]
    │  mask centroid → ray ∩ table plane  ⇒  ObjectDetection(label, x, y, yaw, conf), base frame
    ▼
[ src.task.weigh_and_sort.WeighAndSortTask ]  ◄── demo: section of configs/task_config.yaml
    │  DETECT → (APPROACH → GRASP → LIFT → WEIGH → PLACE_BACK → RETRACT) × 2 → COMPARE
    │  → SORT (bins)  |  HANDOVER → WAIT_CONTACT (src.task.contact) → RELEASE
    ▼  ManipulationSkills / PayloadEstimator (Protocols)
[ src.task.openarm_adapter → IKController.move_l, set_gripper_torque, payload.measure_load ]
    ▼
[ RobotBackend: OpenArmBimanual (openarm_can @ 250 Hz)  |  OpenArmSim (official MuJoCo model) ]
```

---

## 5. Coordinate Frames & Transformations

1. **Camera frame ($C$):** OpenCV convention: $+X$ right, $+Y$ down, $+Z$ forward along the optical axis.
2. **Robot base frame ($R$):** the official OpenArm model frame used by the real-robot controller: $+X$ forward, $+Y$ left, $+Z$ up, shoulder axis at $z = 0.697$ m. Every `demo:` pose is expressed in it.
3. **TCP frames:** the `<side>_tcp` sites of the official MJCF.
4. **Table plane:** tabletop 0.725 m above the floor = $z = 0.092$ m in $R$ (derived in `src/task/workcell.py` from the measured `workcell` section).
5. **Pixel → table (`TableCalibration.pixel_to_plane`, at table + object top height):** $d = R_{C}^{R} K^{-1} [u, v, 1]^T$, $p = c + \lambda d$ with $\lambda = (z_{plane} - c_z)/d_z$. RGB only: the known table plane provides the depth.

---

## 6. Architectural Invariants & Constraints

1. **Official OpenArm SDK Rule:**
   - Use only `openarm_can`, `openarm_description`, and `openarm_mujoco` for OpenArm IK and real-arm control; RBS is optional planning only.
   - Use the v1 `parallel_link` model: `openarm_description/assets/end_effector/parallel_link/` and `openarm_mujoco/v1/openarm_bimanual.xml`.
   - Get dimensions, limits, actuator IDs, and collision meshes from the official sources, never this repository's placeholder model.
   - Prefer the supported C++ `openarm_can` API for hardware; the Python binding is experimental. Official local sources are under `third_party/openarm/` and Git-ignored.
   - Real-robot motion goes through the proven driver ([src/hardware/openarm_bimanual.py](src/hardware/openarm_bimanual.py), only `openarm_bus.py` imports `openarm_can`), never a new driver.
   - The demo's Cartesian targets are IK-checked along the whole straight line before motion (`OpenArmSkills.check_reachable`); never bypass it. The same goes for the collision guard (torso, table, bins, other arm) and the side route between zero and home: the hanging pose is below the tabletop ([docs/demo_task.md §7](docs/demo_task.md#7-reach-layout-and-motion-safety)).
   - OpenArm has 7 arm joints per side; simulation OMPL planning ([src/planning/ompl_planner.py](src/planning/ompl_planner.py)) plans over the combined 14-DOF arm space ($q_{left} \oplus q_{right}$) so arm–arm and arm–table collisions are checked; grippers are separate actuators.
2. **Demo Safety Rules:**
   - **Never open the gripper in the handover without detected external contact** from the joint torques. On timeout, keep gripping and place the object in the heavy bin.
   - **Never sort on an inconclusive comparison** ($z <$ `demo.weigh.min_z`). Re-weigh once, otherwise stop.
   - Both objects are weighed at the same fixed weigh pose; comparisons between different poses are invalid.
   - **Left arm only** until the right arm is repaired (`hardware.active_sides`). Grasp cans **upright** at ≈1 Nm; a full can held horizontally slips.
   - Any skill failure ends in `ManipulationSkills.abort()` (safe state), never an unhandled exception.
3. **Graceful Fallbacks (Hackathon Resilience Rule):**
   - If native OMPL or RBS bindings fail to compile on a developer's host machine, modules MUST provide a safe mock/simulation-only planning fallback without crashing.
   - A fallback MUST NOT replace the official OpenArm SDK for OpenArm IK, actuator control, or physical-hardware safety checks.
   - Every hardware/vision component needs a mock for offline testing (see [src/task/mocks.py](src/task/mocks.py)).
4. **Safety & Workspace Clipping:**
   - Never send raw perception output to the motors: targets go through Cartesian workspace clamping, velocity saturation, and joint-limit filtering (done inside the driver loop, `src/control/safety.py`).
5. **Decoupled Frequencies:**
   - Vision: ~30 Hz camera, detections on demand (once per demo, before motion).
   - Task state machine: event-driven, blocking skill calls.
   - Low-level control: 250 Hz on hardware; physics simulation 500 Hz.

---

## 7. Coding Standards & Conventions for Future AI Agents

- **Typing & Documentation:** All functions and methods must have full type annotations (`typing.Optional`, `typing.Tuple`, `numpy.typing.NDArray`) and descriptive docstrings.
- **File References:** When discussing or recommending files in comments or messages, use explicit workspace links: e.g. `[ompl_planner.py](file:///Users/Q682676/Documents/RoboTokens/src/planning/ompl_planner.py)`.
- **Modular Boundaries:**
  - `src/vision/` only reads and parses perceptual streams; never calls physics or motors.
  - `src/task/` sequences the demo through Protocols; only `openarm_adapter.py` touches the robot API.
  - `src/planning/` operates on kinematics, state spaces, collision models, and trajectories.
  - `src/control/` handles arbitration, rate limiting, and state transitions.
  - `src/simulation/` encapsulates the MuJoCo engine and RBS wrappers (`openarm_sim.py`: official-model backend with the real driver's API).
  - `src/hardware/` is the thin physical-device layer; only `openarm_bus.py` imports `openarm_can`. The real-robot control interface (`RobotBackend`, `JointPositionController`, `IKController`) is documented in `docs/real_robot_deployment.md`.
  - `src/simulation/real_mirror.py` is a display-only twin of a running robot: it may call only `RobotBackend.get_state()`, never a command method. The real robot is always the leader.
  - `src/estimation/` turns proprioceptive signals (joint torques) into object properties (payload mass).
- **Fail-Fast Configuration:** All tunable hyperparameters (camera extrinsics, PID gains, workspace limits) belong in `configs/task_config.yaml`, never hardcoded in scripts.
