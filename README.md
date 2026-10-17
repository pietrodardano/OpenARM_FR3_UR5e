# RoboTokens: Egocentric Vision to Bimanual OpenArms Control via RobotBlockSet (RBS)

[![Track](https://img.shields.io/badge/Hackathon-Track%202%3A%20DaiDai%20Lab%20%2B%20OpenArms-blueviolet.svg)](#)
[![RBS](https://img.shields.io/badge/Robotics-RobotBlockSet%20(JSI)-blue.svg)](https://github.com/abr-ijs/robotblockset_python)
[![Simulation](https://img.shields.io/badge/Sim-MuJoCo%203.0+-red.svg)](https://github.com/google-deepmind/mujoco)
[![Planning](https://img.shields.io/badge/Planning-OMPL-green.svg)](https://ompl.kavrakilab.org/)
[![Hardware](https://img.shields.io/badge/Robot-OpenArms%20Bimanual-yellow.svg)](#)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](#)

RoboTokens maps human egocentric vision and real-time 3D hand tracking from the **DaiDai Lab** ecosystem into safe, collision-free bimanual robot control commands for dual **OpenArms**, powered by Jožef Stefan Institute's **RobotBlockSet (RBS)**, **MuJoCo** simulation backend, and **OMPL** motion planning.

---

## 1. Core Ecosystem References

| Ecosystem Component | Upstream Repository / Reference | Role in Stack |
|---|---|---|
| **RobotBlockSet (RBS by JSI)** | [abr-ijs/robotblockset_python](https://github.com/abr-ijs/robotblockset_python) | Unified robot kinematics, spatial transforms, and `JointPathPlanner` |
| **OpenArm Control** | [enactic/openarm](https://github.com/enactic/openarm) | `openarm_can` for low-level motor control, `openarm_description` for the robot model, and `openarm_mujoco` for simulation |
| **OpenArm Docs & CAD** | [enactic/openarm](https://github.com/enactic/openarm) | Reference geometry, mechanical assets, and control-oriented design details for MuJoCo and RBS integration |
| **OpenArms** | OpenArms Hardware Platform | Bimanual robot morphology, actuator interfaces, and physical hardware bridge |
| **DaiDai Lab Ecosystem** | [wwnt.ai](https://wwnt.ai/) | Wearable egocentric camera feed, 3D hand landmark tracking, and gesture streaming |
| **Egocentric Camera HW** | [HBV HD CAMERA Specs](docs/camera_specifications.md) | UVC HD webcam (0x0ac8:0x0346), 720p/1080p @ 30 FPS, plug-and-play |
| **MuJoCo** | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | High-fidelity contact physics, collision checking, and interactive visualization |
| **OMPL** | [ompl.kavrakilab.org](https://ompl.kavrakilab.org/) | Sampling-based collision-free motion planning in dual-arm configuration space |

---

## 2. Repository Structure

```
RoboTokens/
├── agents.md                   # AI agent directives & engineering context
├── README.md                   # Main documentation & setup guide
├── pyproject.toml              # Modern Python packaging configuration
├── requirements.txt            # Python dependencies
├── configs/
│   └── task_config.yaml        # Vision, planning, safety & simulation parameters
├── src/
│   ├── vision/                 # DaiDai Lab egocentric stream & hand tracking listener
│   │   ├── __init__.py
│   │   └── hand_tracker_listener.py
│   ├── planning/               # RBS wrappers, OMPL path planner, & trajectory generation
│   │   ├── __init__.py
│   │   └── ompl_planner.py
│   ├── control/                # Dual-arm coordination, safety bounding, & execution loop
│   │   ├── __init__.py
│   │   └── bimanual_controller.py
│   └── simulation/             # MuJoCo environment setup with RBS conventions
│       ├── __init__.py
│       └── mujoco_env.py
├── scripts/
│   ├── run_pipeline.py         # Main execution entrypoint (vision -> OMPL -> MuJoCo)
│   └── calibrate_camera.py     # Operator zero-calibration tool
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Comprehensive unit & integration tests
└── docs/
    ├── architecture.md         # Detailed mathematical derivations & system diagram
    └── hackathon_roadmap.md    # 30-hour sprint roadmap & pitch notes
```

---

## 3. Quickstart & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/pietrodardano/RoboTokens.git
cd RoboTokens
```

### Step 2: Environment Setup
Use `uv` to create the locked development environment:

```bash
uv venv --python 3.11
uv sync --extra rbs --extra dev
```

For the official OpenArm v1 sources, the parallel-link MuJoCo scene, validation, rendering, and hardware boundary, follow [docs/openarm_v1_environment.md](docs/openarm_v1_environment.md).

### Real OpenArm Deployment (Linux PC, container)
Joint position and IK control of the physical bimanual OpenArm, in plain Python (one streaming thread over the official `openarm_can` SDK, no ROS), with a MuJoCo backend exposing the same API for dry runs:

```bash
docker/dev.sh build && docker/dev.sh shell        # rootful Docker on the robot PC (SocketCAN)
scripts/setup_can.sh && python scripts/check_robot.py
python scripts/run_control_demo.py --backend sim --controller ik --render
python scripts/run_control_demo.py --backend real --controller joint --amplitude 0.3
```

See [docs/real_robot_deployment.md](docs/real_robot_deployment.md) for the interface, safety behaviour and bring-up checklist.

---

## 4. Running the Pipeline

### Run Unit Tests
Verify all coordinate transforms, OMPL planners, and simulation environments:
```bash
uv run python -m unittest discover -s tests -v
```

### Calibrate Egocentric Tracker
Hold your hands in the neutral resting pose in front of the camera to zero out reference offsets:
```bash
python3 scripts/calibrate_camera.py
```

### Launch the Full Pipeline (Vision -> RBS / OMPL -> MuJoCo)
Run the pipeline with the mock egocentric tracking generator:
```bash
python3 scripts/run_pipeline.py --steps 150
```

To run with an interactive MuJoCo graphical viewer:
```bash
python3 scripts/run_pipeline.py --render --steps 500
```

---

## 5. System Architecture Highlights

1. **Egocentric Coordinate Transformation:**
   Converts camera coordinates ($+X$ right, $+Y$ down, $+Z$ forward) to robot base coordinates ($+X$ forward, $+Y$ left, $+Z$ up) with automatic exponential moving average (EMA) smoothing and workspace bounding.
2. **RBS & OMPL Dual-Arm Planning:**
   Joint-space collision checking over the 14-DOF bimanual arm configuration space $q \in \mathbb{R}^{14}$, preventing arm-to-arm collision in the shared workspace. The active model is OpenArm v1 with seven arm joints and a parallel-link gripper per side.
3. **Safety Barriers:**
   - Cartesian bounding box $[0.15 \le x \le 0.55, -0.45 \le y \le 0.45, 0.38 \le z \le 0.75]$ m.
   - Dual-arm end-effector minimum clearance ($8\text{ cm}$).
   - Automatic contact E-Stop in simulation and hardware.

---

## 6. Hackathon Presentation & Documentation

For details on the system mathematics, control state machines, and pitch roadmap, see:
- [Architecture Overview](file:///Users/Q682676/Documents/RoboTokens/docs/architecture.md)
- [Hackathon Roadmap & Pitch Deck](file:///Users/Q682676/Documents/RoboTokens/docs/hackathon_roadmap.md)
- [AI Coding Directives](file:///Users/Q682676/Documents/RoboTokens/agents.md)
