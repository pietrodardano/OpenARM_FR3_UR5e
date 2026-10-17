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
│   ├── bimanual_openarms.xml   # Dual-arm OpenArms MuJoCo MJCF model & workspace
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
Using `uv` (recommended for ultra-fast setup) or standard `venv`:

```bash
# Using uv:
uv venv
source .venv/bin/activate

# Or using standard python:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .

# Install RobotBlockSet (RBS) from JSI:
pip install git+https://github.com/abr-ijs/robotblockset_python.git
```

> **Note on Fallback Mode:** If native C++ OMPL or MuJoCo libraries are not yet compiled on your target host, RoboTokens provides an automated fallback mock backend so you can develop, test, and run the complete architecture immediately!

---

## 4. Running the Pipeline

### Run Unit Tests
Verify all coordinate transforms, OMPL planners, and simulation environments:
```bash
python3 -m unittest discover -s tests -v
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
   Joint-space collision checking over the 10-DOF bimanual configuration space $q \in \mathbb{R}^{10}$, preventing arm-to-arm collision in the shared center table workspace.
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
