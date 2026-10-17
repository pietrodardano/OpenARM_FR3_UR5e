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

---

## 7. Dev Container & Tabletop Scene (OpenArm v1)

All Python dependencies live inside a Docker image; nothing is installed on the host.

```bash
git submodule update --init          # upstream OpenArm MJCF (third_party/openarm_mujoco)
docker/dev.sh build                  # build the image (re-run after changing dependencies)
docker/dev.sh run pytest -q          # one-off command in the container
docker/dev.sh shell                  # interactive shell
docker/dev.sh lock                   # re-lock after editing pyproject.toml
```

VS Code: "Dev Containers: Reopen in Container" uses the same `docker/compose.yaml`.
Rootless and rootful Docker are both handled (`docker/detect_user.sh`). macOS
teammates can skip Docker and run `uv sync` natively against the same `uv.lock`.

**Scene** — [assets/scenes/tabletop.xml](assets/scenes/tabletop.xml): OpenArm v1
bimanual arms (parallel-jaw grippers, upstream model unmodified), a table (top at
z = 0.30 m) and primitive objects: plate, cube, bottle, mug, apple, pen. Cameras:
`head` (the robot's only vision sensor; pose/fovy are estimates until calibrated)
plus debug views `overhead`, `front`, `three_quarter`.

[src/simulation/tabletop_scene.py](src/simulation/tabletop_scene.py) loads it and adds:
- `left_tcp` / `right_tcp` sites between the fingertips (the upstream
  `*_ee_control_point` is at the wrist flange, ~20 cm behind the tips),
- ideal gravity compensation on the robot links,
- a home pose with the upper arms swung back (grippers clear of the table),
- `reset()`, arm/gripper commands, TCP/object poses, offscreen rendering.

Top-down grasps at table height reach roughly x = 0.15-0.35 m in front of the
pedestal; objects are laid out inside that band.

**Scripted pick and place** — [src/skills/pick_place.py](src/skills/pick_place.py):
object poses are observed once (`perceive()`, ground truth in sim, standing in for a
head-camera estimate) and remembered. Skills `pick(side, obj)` and
`place(side, xy, surface_z)` are built from IK motion primitives
([src/control/ik.py](src/control/ik.py), mink): `move_joint` for transit and
`move_linear` for straight approach/lift lines (continuation IK, so a line stays on
one IK branch). Grasp yaw is chosen near the arm's natural wrist yaw within the
object's symmetry.

```bash
docker/dev.sh run python scripts/view_scene.py --render outputs/scene   # PNG per camera
docker/dev.sh run python scripts/run_pick_place.py                      # outputs/pick_place.mp4
docker/dev.sh shell  # then: python scripts/view_scene.py               # interactive viewer, needs $DISPLAY
```
