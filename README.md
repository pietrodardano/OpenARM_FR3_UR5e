# RoboTokens: Vision Cannot Weigh — Proprioceptive Object Discrimination on OpenArm

[![Track](https://img.shields.io/badge/Hackathon-Track%202%3A%20OpenArms-blueviolet.svg)](#)
[![Vision](https://img.shields.io/badge/Vision-GroundingDINO%20%2B%20SAM2-orange.svg)](#)
[![Simulation](https://img.shields.io/badge/Sim-MuJoCo%203.0+-red.svg)](https://github.com/google-deepmind/mujoco)
[![Planning](https://img.shields.io/badge/Planning-OMPL-green.svg)](https://ompl.kavrakilab.org/)
[![Hardware](https://img.shields.io/badge/Robot-OpenArm%20v1-yellow.svg)](https://github.com/enactic/openarm)

Two cans look identical: one is full, one is empty. No camera can tell which is which.
RoboTokens shows that a robot can, by **interacting**:

1. An RGB camera with **GroundingDINO + SAM2** (+ optional VLM) finds the two look-alike objects on a table of known height.
2. The **OpenArm** grasps and lifts each object, estimates its mass from its own **joint torques**, puts it back, and compares the two.
3. The robot acts on what it felt:
   - **Sort:** heavy and light objects go to predefined bins.
   - **Handover:** the robot offers the heavy object to a person and opens the gripper only when it **feels** them take it, again through the joint torques.

Full task spec: [docs/demo_task.md](docs/demo_task.md).

---

## 1. Components

| Component | Source | Role |
|---|---|---|
| **OpenArm v1** (parallel-link gripper) | [enactic/openarm](https://github.com/enactic/openarm): `openarm_can`, `openarm_description`, `openarm_mujoco` | Robot, official model, SocketCAN control. Left arm active |
| **Real-robot controller + weighing** | [src/control/](src/control/), [src/hardware/](src/hardware/), [src/estimation/](src/estimation/) (from `deploy/openarm-real`) | Proven `openarm_can` driver, Cartesian `move_l`, gripper torque mode, torque-based load comparison, `OpenArmSim` |
| **Object perception** | [src/vision/](src/vision/) (from `demo-simulation`) | GroundingDINO boxes, SAM2 masks, top-face PCA, calibrated pixel → base-frame XY (`VisionObjectDetector`) |
| **Task layer** | `main`: [src/task/](src/task/) | Demo state machine, component contracts, felt-handover contact detection, mocks |
| **HBV HD CAMERA** | [docs/camera_specifications.md](docs/camera_specifications.md) | RGB only, torso mount, 720p/1080p @ 30 FPS |
| **MuJoCo** | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Measured workcell, dry runs, collision checking |
| **RobotBlockSet + OMPL** (optional) | [abr-ijs/robotblockset_python](https://github.com/abr-ijs/robotblockset_python), [OMPL](https://ompl.kavrakilab.org/) | Collision-free planning in the simulated workcell |

---

## 2. Repository Structure

```
RoboTokens/
├── agents.md                    # Engineering directives, branch policy, invariants
├── README.md
├── pyproject.toml / uv.lock     # uv-managed environment
├── configs/
│   ├── task_config.yaml         # All tunables, incl. the `demo:` section
│   └── charuco_*.{pdf,png}      # Camera calibration boards
├── src/
│   ├── task/                    # Final demo
│   │   ├── interfaces.py        #   ObjectDetector / ManipulationSkills / PayloadEstimator contracts
│   │   ├── weigh_and_sort.py    #   WeighAndSortTask state machine (sort | handover)
│   │   ├── openarm_adapter.py   #   OpenArmSkills / OpenArmPayloadEstimator on the real controller
│   │   ├── weighing.py          #   Mass comparison (Δm, z) and re-weigh combination
│   │   ├── contact.py           #   External-contact detection from joint torques
│   │   └── mocks.py             #   Offline mocks of vision, robot and estimator
│   ├── control/                 # RobotBackend API, safety filter, joint + IK (Cartesian) controllers
│   ├── hardware/                # Real OpenArm driver; only openarm_bus.py imports openarm_can
│   ├── estimation/payload.py    # Load measurement from joint torques (dithered), comparison
│   ├── planning/                # Official-model kinematics + mink IK; OMPL / RBS for the workcell sim
│   ├── simulation/              # OpenArmSim (same API as the driver), real_mirror.py (live twin), workcell wrapper
│   └── vision/                  # GroundingDINO + SAM2 localizer, camera, calibration, VisionObjectDetector
├── docker/                      # Robot-PC container (rootful Docker, SocketCAN, openarm_can 1.4.0)
├── scripts/
│   ├── run_weigh_demo.py        # Final demo runner: --backend mock | sim | fake | real
│   ├── setup_can.sh, check_robot.py, read_ee_pose.py, go_zero.py, jog_joints.py   # Bring-up
│   ├── cartesian_points.py, ik_axis_test.py, run_control_demo.py                   # Motion tests
│   ├── grip_torque_test.py, compare_weights.py                                     # Grip + weighing
│   ├── safe_zero.py             # Shared zero-start/zero-stop lifecycle of every motion script
│   ├── setup_openarm_v1_scene.py, run_pick_place_demo.py   # Workcell scene + OMPL pick & place
│   ├── setup_cross_embodiment_scene.py, view_cross_embodiment_scene.py
│   └── generate_charuco_board.py
├── tests/                       # test_task.py (demo + adapter), test_real_mirror.py, test_openarm_control.py (driver), test_pipeline.py
└── docs/
    ├── demo_task.md             # Final demo spec, reach limits, integration status
    ├── real_robot_deployment.md # Driver, safety behaviour, bring-up checklist, grip limits
    ├── architecture.md          # Data flow and the math (projection, mass, decision, contact)
    ├── hackathon_roadmap.md     # Workstreams and pitch
    ├── physical_workcell.md, openarm_v1_environment.md, camera_specifications.md, cross_embodiment.md
```

The real-robot code was copied from `deploy/openarm-real`, and the vision pipeline from the vision commit of
`demo-simulation`; neither branch is merged wholesale.
See [docs/demo_task.md §9](docs/demo_task.md#9-integration-status).

---

## 3. Setup

```bash
git clone https://github.com/pietrodardano/RoboTokens.git
cd RoboTokens
uv venv --python 3.11
uv sync --extra rbs --extra dev --extra calib
```

For the official OpenArm v1 sources and the MuJoCo workcell scene, follow
[docs/openarm_v1_environment.md](docs/openarm_v1_environment.md).

---

## 4. Running

### Final demo
```bash
uv run python scripts/run_weigh_demo.py --mode sort                               # mocks: weigh both, sort
uv run python scripts/run_weigh_demo.py --mode handover                           # mocks: hand over the heavy one
uv run python scripts/run_weigh_demo.py --backend sim --render --rehearse --yes   # MuJoCo rehearsal
uv run python scripts/run_weigh_demo.py --backend real --mode sort                # real left arm
uv run python scripts/run_weigh_demo.py --backend real --mode sort --mirror       # + live MuJoCo twin
```
`--mirror` opens a read-only MuJoCo twin that follows the real arm's *measured* joints (the robot is the leader), with the commanded TCP, table, objects, bins and handover pose overlaid. On macOS run it under `mjpython`.
On the real robot, object positions come from the camera (GroundingDINO + SAM2): they are shown for confirmation
before anything moves. `--objects "x,y;x,y"` overrides them.

### Full run on the Linux workstation (robot + camera + GPU), in the container
```bash
ROBOTOKENS_VISION=1 docker/dev.sh build        # + torch (CUDA 12.8 wheels), GroundingDINO, SAM2
docker/dev.sh shell                            # everything below runs inside
scripts/download_vision_weights.sh             # weights/ (GroundingDINO SwinT, SAM2.1 small)
scripts/setup_can.sh && python scripts/check_robot.py
python scripts/calibrate_intrinsics.py ...     # lens, once (docs/camera_calibration.md)
python scripts/calibrate_camera_touch.py --side left --session outputs/touch_session   # camera pose
python scripts/detect_objects.py --prompt "soda can"                     # vision only: check with a ruler
python scripts/run_weigh_demo.py --backend sim --fast --yes              # dry run of the task
python scripts/run_weigh_demo.py --backend real --mirror --prompt "put the lighter can in the left bin"
python scripts/run_weigh_demo.py --backend real --mirror --prompt "give me the heavy can"
```
Other CUDA versions: `ROBOTOKENS_TORCH_INDEX=https://download.pytorch.org/whl/cu126 ROBOTOKENS_VISION=1 docker/dev.sh build`.
Without the container: torch for your GPU, then `pip install -r requirements-vision.txt`.
The real backend refuses to run until the table height is measured. See
[docs/demo_task.md §7–8](docs/demo_task.md#7-reach-layout-and-motion-safety).

### Real robot (Linux PC, container)
```bash
docker/dev.sh build && docker/dev.sh shell        # rootful Docker on the robot PC (SocketCAN)
scripts/setup_can.sh && python scripts/check_robot.py
python scripts/cartesian_points.py --backend sim --render --yes    # rehearse, then --backend real
python scripts/compare_weights.py --backend real --labels full empty --grip-torque 1.0
```
Bring-up checklist and safety behaviour: [docs/real_robot_deployment.md](docs/real_robot_deployment.md).

### Tests
```bash
uv run python -m unittest discover -s tests -v
```

### Simulation workcell
```bash
uv run python scripts/setup_openarm_v1_scene.py
uv run python scripts/run_pick_place_demo.py --show       # OMPL collision-checked pick & place
```
See [docs/physical_workcell.md](docs/physical_workcell.md) and [docs/cross_embodiment.md](docs/cross_embodiment.md).

---

## 5. Safety Rules of the Demo

- The gripper opens in the handover **only after external contact is felt** in the joint torques. On timeout the object goes to the heavy bin.
- The robot **never sorts on an inconclusive comparison** ($z < 3$ after one re-weigh).
- Both objects are weighed at the same fixed pose. Cans are grasped upright at ≈1 Nm. Only the left arm is used.
- Every Cartesian move is IK-checked along its whole line before the arm moves. Real-arm motion runs through the proven driver (joint-limit clamp, velocity limits, latched faults). Keep the hardware e-stop in reach.

---

## 6. Further Documentation

- [Demo task spec & integration status](docs/demo_task.md)
- [Real robot deployment](docs/real_robot_deployment.md)
- [Architecture & math](docs/architecture.md)
- [Roadmap & pitch](docs/hackathon_roadmap.md)
- [Engineering directives for agents](agents.md)
