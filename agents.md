# AGENTS.MD: Engineering Context & AI Agent Directives

## 1. Project Mission & Hackathon Context
- **Hackathon:** 30-Hour Robotics AI Sprint.
- **Track:** Track 2 — DaiDai Lab + OpenArms Robotics + RobotBlockSet (RBS by JSI) with MuJoCo Simulation and OMPL Motion Planning.
- **Core Objective (final demo, see [docs/final_task/demo_task.md](docs/final_task/demo_task.md)):** Prove that vision alone cannot tell a full object from an empty one. Vision (SAM2 + Grounding DINO, optional VLM, RGB camera only) finds two similar objects (two cans or two boxes) and their X-Y pose; the table height gives Z. The left OpenArm grasps and lifts each object, joint torques give its mass, and the result decides whether the objects are sorted into bins by weight or the heavy one is handed to the user. In the hand-over the gripper opens only after torque-sensed external contact.
- **Superseded:** the original egocentric hand-tracking teleoperation goal. That pipeline is deprecated ([docs/legacy/README.md](docs/legacy/README.md)).
- **Branches:** several lines of work run in parallel and are not merged yet. See rule 7 and [docs/final_task/branches_and_integration.md](docs/final_task/branches_and_integration.md).

---

## 2. Technology Stack & Ecosystem References

| Component | Technology | Reference / Source |
|---|---|---|
| **Robotics Framework** | RobotBlockSet (`robotblockset_python` by JSI) | [abr-ijs/robotblockset_python](https://github.com/abr-ijs/robotblockset_python) |
| **OpenArm Control SDK** | `openarm_can` | [enactic/openarm_can](https://github.com/enactic/openarm_can) — official C++ SocketCAN control library; its Python bindings are experimental |
| **OpenArm Robot Model** | `openarm_description` | [enactic/openarm_description](https://github.com/enactic/openarm_description) — official URDF/xacro, joint metadata, and meshes |
| **OpenArm Simulator Model** | `openarm_mujoco` | [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) — official MuJoCo XML models and collision assets |
| **OpenArm CAD & BOM** | `openarm_hardware` | [enactic/openarm_hardware](https://github.com/enactic/openarm_hardware) — links to official STEP, BOM, and assembly sources |
| **Simulation Backend** | MuJoCo (v3.0+) | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) |
| **Motion Planning** | OMPL (optional; RBS only after model compatibility is verified) | [OMPL Library](https://ompl.kavrakilab.org/) |
| **Robot Hardware Setup** | OpenArms Bimanual Robotics | OpenArms Hardware Platform |
| **Object Vision** | SAM2 + Grounding DINO (+ optional VLM) on an RGB camera; gives object X-Y in the robot base frame (vision branch) | [DaiDai Lab](https://wwnt.ai/) |
| **Egocentric Hand Tracking** | DaiDai Lab stream and hand tracker (deprecated, not part of the final demo) | [docs/legacy/README.md](docs/legacy/README.md) |
| **Language & Tooling** | Python 3.10+, NumPy, SciPy, PyYAML, uv | Modern Python packaging |

---

## 3. System Architecture & Information Flow

```
[ RGB camera on the torso ]
    │  (frames)
    ▼
[ Vision: SAM2 + Grounding DINO (+ VLM) ]            (vision branch)
    │  ObjectDetection: object id, X-Y in robot base frame (Z from table height)
    ▼
[ Task state machine ]  detect → pick → lift → weigh → put back → compare → sort | hand over   (planned)
    │  TCP poses, gripper opening                      ▲  mass, contact events
    ▼                                                  │
[ Controller: IKController + joint safety filter ]   [ Proprioception: mass estimator, contact detector ]
    │  (deploy branch)                                 │  (estimator not pushed; detector not started)
    ▼                                                  │
[ OpenArm driver + openarm_can ] ── joint q, dq, tau ──┘
    ▼
[ OpenArm v1, left arm only ]
```

Details and the contracts between the boxes: [docs/final_task/architecture.md](docs/final_task/architecture.md) and [docs/final_task/interfaces.md](docs/final_task/interfaces.md).
The older `src/simulation` + `src/planning` + `src/control` code on `main` is the previous simulation pipeline; parts of it are deprecated.

---

## 4. Coordinate Frames & Transformations

1. **Camera Frame ($C$):**
   - Standard OpenCV convention: $+X$ right, $+Y$ down, $+Z$ forward into the scene. The camera is fixed on the torso.
2. **Robot World / Base Frame ($R$):**
   - Robotics convention: $+X$ forward, $+Y$ left, $+Z$ up, origin on the floor under the torso. All commands and object poses use it.
3. **End-Effector Frame ($EE_{left}$):**
   - Attached to the wrist flange/tool center point of the left OpenArm.
4. **Camera to base (hand-eye):**
   - A fixed transform $T_{R \leftarrow C}$ that must be calibrated. It does not exist yet.
   - A pixel on the table plane maps to a base-frame point by intersecting its camera ray with $z = z_{table}$ (0.750 m); the known table height gives a 3D grasp point from RGB alone.

---

## 5. Architectural Invariants & Constraints

1. **Arms:**
   - The robot has two arms that share a workspace; OpenArm has 7 arm joints per side and the grippers are separate actuators.
   - **Current hardware state: only the left arm is operable.** The right arm has a known problem on the real robot, so it is OFF by default (see rule 6). The final demo uses the left arm only, so planning and control are single-arm (7 joints).
   - Joint-space planning over both arms ($q_{left} \oplus q_{right}$, 14 joints) is not implemented. It becomes relevant only if the right arm returns.
2. **Official OpenArm SDK Rule:**
   - Use only `openarm_can`, `openarm_description`, and `openarm_mujoco` for OpenArm IK and real-arm control; RBS is optional planning only.
   - Use the v1 `parallel_link` model: `openarm_description/assets/end_effector/parallel_link/` and `openarm_mujoco/v1/openarm_bimanual.xml`.
   - Get dimensions, limits, actuator IDs, and collision meshes from the official sources, never this repository's placeholder model.
   - Prefer the supported C++ `openarm_can` API for hardware; the Python binding is experimental. Official local sources are under `third_party/openarm/` and Git-ignored.
3. **Graceful Fallbacks (Hackathon Resilience Rule):**
   - If native OMPL or RBS bindings fail to compile on a developer's host machine, modules MUST provide a safe mock/simulation-only planning fallback without crashing.
   - A fallback MUST NOT replace the official OpenArm SDK for OpenArm IK, actuator control, or physical-hardware safety checks.
   - Every hardware or vision component needs a mock or simulated source so it can be tested offline without sensors or a physical arm.
4. **Safety & Workspace Clipping:**
   - Never send raw vision targets directly to joint motors without Cartesian workspace clamping, velocity saturation limits, a reachability check, and self-collision filtering.
   - Never open the gripper during the hand-over on a timeout; open only after torque-sensed contact.
5. **Decoupled Frequencies:**
   - Vision detection: as fast as the models run; the task layer needs a fresh, stable detection before each approach.
   - Motion planning & replanning: ~10–20 Hz (asynchronous or threshold-triggered).
   - Low-level control & physics simulation: 100–500 Hz.
6. **Arm Enablement (right arm OFF by default):**
   - `configs/task_config.yaml` sets `hardware.right.enabled: false` and `hardware.left.enabled: true`. Only enabled arms may be commanded.
   - Any script or module that can move an arm must check that the side is enabled in the config before touching hardware, and must not invoke right-arm code paths by default.
   - Scripts take a `--side` argument that defaults to `left`. `right` is accepted only when the config enables it.
   - The right-arm code is kept, not discarded: it is being worked on separately and will be reactivated once the hardware problem is fixed and re-verified (calibration on `can0`, then the staged checklist). Do not enable it as a side effect of another change.
   - Known gap: the real-robot driver on `origin/deploy/openarm-real` enables both arms, refuses to start if any motor of either arm is silent, and has no per-arm flag. It needs this gate before it is run ([docs/final_task/branches_and_integration.md](docs/final_task/branches_and_integration.md)).
7. **Branches and Merging:**
   - `origin/deploy/openarm-real` is the real-robot controller and is canonical for hardware. A vision branch produces object poses. `openarm-cartesian-sim-validated` is a simulation and feasibility lab. A weight-from-torque estimator exists but is not pushed.
   - **None of these is merged into another or into `main` yet.** Do not merge, rebase or cherry-pick code between them without an explicit decision by the team.
   - On `main`, documents, scripts and configs are sorted into `final_task/`, `setup/` and `legacy/` folders (see [docs/README.md](docs/README.md) and [scripts/README.md](scripts/README.md)). `src/` and `tests/` are **not** reorganized: do not move or delete files there that the other branches also edit. Deprecated code is recorded in [docs/legacy/README.md](docs/legacy/README.md) and moved after integration.
   - Do not inspect or depend on a branch you were not pointed to. Describe cross-branch interfaces as contracts in [docs/final_task/interfaces.md](docs/final_task/interfaces.md), marked existing or proposed.
8. **Interaction Decides Full vs Empty:**
   - Whether an object is full or empty is never inferred from images. It comes from interaction: lift the object and estimate its mass from joint torques.
   - Estimate mass from the torque difference between loaded and unloaded at the same pose, not from absolute gravity-model torques; the model masses are not exact.
   - Two objects can be in the same state. Report "indistinguishable" when the difference is below the measurement uncertainty instead of guessing.
   - Do not describe the estimator as tested until its code is pushed and its results are recorded.
9. **Simulation-only Robots (FR3, UR5e):**
   - They run the same task as the real OpenArm in the rendered scene, in step with the real task's phases, on their own objects ([docs/final_task/cross_embodiment.md](docs/final_task/cross_embodiment.md)).
   - They may cheat: true object positions instead of vision, weld grasps, injected user forces. Never present their results as evidence about the real robot or about vision.
   - The rendered OpenArm is a follower of the real arm's measured state. It never moves on its own and never acts on simulated objects.

---

## 6. Coding Standards & Conventions for Future AI Agents

- **Typing & Documentation:** All functions and methods must have full type annotations (`typing.Optional`, `typing.Tuple`, `numpy.typing.NDArray`) and descriptive docstrings.
- **File References:** When discussing or recommending files in comments or messages, use explicit workspace links: e.g. `[ompl_planner.py](file:///Users/Q682676/Documents/RoboTokens/src/planning/ompl_planner.py)`.
- **Modular Boundaries:**
  - `src/vision/` only reads and parses perceptual streams and produces object detections; never calls physics or motors.
  - `src/planning/` operates on kinematics, state spaces, collision models, and trajectories.
  - `src/control/` handles arbitration, rate limiting, and state transitions.
  - `src/simulation/` encapsulates the MuJoCo engine and RBS wrappers.
  - Planned, not created yet: `src/sensing/` (gravity model, mass estimator, contact detector) and `src/task/` (the demo state machine).
- **Deprecated code:** do not add features or new callers to anything in [docs/legacy/README.md](docs/legacy/README.md).
- **Fail-Fast Configuration:** All tunable hyperparameters (camera extrinsics, PID gains, workspace limits) belong in `configs/task_config.yaml`, never hardcoded in scripts.
