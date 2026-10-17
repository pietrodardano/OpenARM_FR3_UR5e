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
| **Simulation Backend** | MuJoCo (v3.0+) | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) |
| **Motion Planning** | OMPL (via RBS `JointPathPlanner`) | [OMPL Library](https://ompl.kavrakilab.org/) |
| **Robot Hardware Setup** | OpenArms (Bimanual LeRobot Ecosystem) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
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
[ src.planning.ompl_planner ] ◄── RBS JointPathPlanner / MuJoCo Collision World
    │  (Collision-free bimanual C-space trajectory)
    ▼
[ src.simulation.mujoco_env ] ──► [ OpenArms Physical Hardware Bridge ]
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
   - OMPL state space must plan either over the combined 12-DOF joint space ($q_{left} \oplus q_{right}$) or sequentially with swept-volume collision avoidance.
2. **Graceful Fallbacks (Hackathon Resilience Rule):**
   - If native C++ OMPL or RBS system bindings fail to compile on a developer's host machine, modules MUST provide robust numerical/Jacobian fallback planners and local obstacle avoidance without crashing.
   - Mock streams (`MockHandTrackerStream`) must be provided for every hardware/vision component to allow offline testing without active sensors or physical robot arms.
3. **Safety & Workspace Clipping:**
   - Never send raw hand positions directly to joint motors without Cartesian workspace clamping, velocity saturation limits, and self-collision filtering.
4. **Decoupled Frequencies:**
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
  - `src/simulation/` encapsulates the MuJoCo engine and RBS wrappers.
- **Fail-Fast Configuration:** All tunable hyperparameters (camera extrinsics, PID gains, workspace limits) belong in `configs/task_config.yaml`, never hardcoded in scripts.
