# AGENTS.MD: Engineering Context & AI Agent Directives

## 1. Project Mission & Hackathon Context
- **Hackathon:** 30-Hour Robotics AI Sprint.
- **Track:** Track 2 — DaiDai Lab (Egocentric Vision & Hand Tracking) + OpenArms Bimanual Robotics + RobotBlockSet (RBS by JSI) with MuJoCo Simulation and OMPL Motion Planning.
- **Core Objective:** Map human egocentric vision and hand tracking inputs into safe, robust, collision-free bimanual robot trajectories and control commands executed in MuJoCo and transferable to physical OpenArms hardware.

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
| **Egocentric Vision** | DaiDai Lab Stream & Hand Tracking | [DaiDai Lab](https://wwnt.ai/) |
| **Language & Tooling** | Python 3.10+, NumPy, SciPy, PyYAML, uv | Modern Python packaging |

---

## 3. System Architecture & Information Flow

```
[ DaiDai Lab Egocentric Stream ]
    │  (RGB-D + 3D Hand Landmarks @ 30Hz)
    ▼
[ src.vision.hand_tracker_listener ]
    │  (Confidence filtering, temporal smoothing, coordinate transformation)
    ▼
[ src.control.bimanual_controller ] ◄── Safety workspace limits & State machine
    │  (Cartesian End-Effector Setpoints: Left & Right)
    ▼
[ src.planning.ompl_planner ] ◄── Optional OMPL path planning / MuJoCo Collision World
   │  (Collision-free 14-DOF bimanual arm trajectory)
    ▼
[ src.simulation.mujoco_env ] ──► [ openarm_can Physical Hardware Bridge ]
       (Physics execution @ 500Hz)
```

---

## 4. Coordinate Frames & Transformations

1. **Egocentric Camera Frame ($C$):**
   - Standard OpenCV / XR convention: $+X$ right, $+Y$ down, $+Z$ forward into the scene.
2. **Robot World / Base Frame ($R$):**
   - Robotics convention: $+X$ forward, $+Y$ left, $+Z$ up.
3. **End-Effector Frames ($EE_{left}, EE_{right}$):**
   - Attached to the wrist flange/tool center point of OpenArms.
4. **Transform Pipeline:**
   $$p_{robot} = R_{cam}^{robot} \cdot (s \cdot (p_{cam} - p_{cam,0})) + p_{robot,0}$$
   where $s$ is the human-to-robot workspace scale factor, $p_{cam,0}$ is the operator calibration origin, and $p_{robot,0}$ is the robot nominal home pose.

---

## 5. Architectural Invariants & Constraints

1. **Dual-Arm Independence & Collision Coupling:**
   - The left and right arms share a common workspace in front of the robot.
   - OpenArm has 7 arm joints per side. OMPL state space must plan over the combined 14-DOF arm joint space ($q_{left} \oplus q_{right}$); grippers are separate actuators.
2. **Official OpenArm SDK Rule:**
   - Use only `openarm_can`, `openarm_description`, and `openarm_mujoco` for OpenArm IK and real-arm control; RBS is optional planning only.
   - Use the v1 `parallel_link` model: `openarm_description/assets/end_effector/parallel_link/` and `openarm_mujoco/v1/openarm_bimanual.xml`.
   - Get dimensions, limits, actuator IDs, and collision meshes from the official sources, never this repository's placeholder model.
   - Prefer the supported C++ `openarm_can` API for hardware; the Python binding is experimental. Official local sources are under `third_party/openarm/` and Git-ignored.
3. **Graceful Fallbacks (Hackathon Resilience Rule):**
   - If native OMPL or RBS bindings fail to compile on a developer's host machine, modules MUST provide a safe mock/simulation-only planning fallback without crashing.
   - A fallback MUST NOT replace the official OpenArm SDK for OpenArm IK, actuator control, or physical-hardware safety checks.
   - Mock streams (`MockHandTrackerStream`) must be provided for every hardware/vision component to allow offline testing without active sensors or physical robot arms.
4. **Safety & Workspace Clipping:**
   - Never send raw hand positions directly to joint motors without Cartesian workspace clamping, velocity saturation limits, and self-collision filtering.
5. **Decoupled Frequencies:**
   - Vision & hand-tracking input: ~30 Hz.
   - Motion planning & replanning: ~10–20 Hz (asynchronous or threshold-triggered).
   - Low-level control & physics simulation: 100–500 Hz.

---

## 6. Coding Standards & Conventions for Future AI Agents

- **Typing & Documentation:** All functions and methods must have full type annotations (`typing.Optional`, `typing.Tuple`, `numpy.typing.NDArray`) and descriptive docstrings.
- **File References:** When discussing or recommending files in comments or messages, use explicit workspace links: e.g. `[ompl_planner.py](file:///Users/Q682676/Documents/RoboTokens/src/planning/ompl_planner.py)`.
- **Modular Boundaries:**
  - `src/vision/` only reads and parses perceptual streams; never calls physics or motors.
  - `src/planning/` operates on kinematics, state spaces, collision models, and trajectories.
  - `src/control/` handles arbitration, rate limiting, and state transitions.
  - `src/simulation/` encapsulates the MuJoCo engine and RBS wrappers (`openarm_sim.py`: official-model backend with the real driver's API).
  - `src/hardware/` is the thin physical-device layer; only `openarm_bus.py` imports `openarm_can`. The real-robot control interface (`RobotBackend`, `JointPositionController`, `IKController`) is documented in `docs/real_robot_deployment.md`.
- **Fail-Fast Configuration:** All tunable hyperparameters (camera extrinsics, PID gains, workspace limits) belong in `configs/task_config.yaml`, never hardcoded in scripts.
