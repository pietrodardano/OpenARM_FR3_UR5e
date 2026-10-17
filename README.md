# RoboTokens: Telling Full from Empty by Touch with OpenArm

[![Track](https://img.shields.io/badge/Hackathon-Track%202%3A%20DaiDai%20Lab%20%2B%20OpenArms-blueviolet.svg)](#)
[![Simulation](https://img.shields.io/badge/Sim-MuJoCo%203.0+-red.svg)](https://github.com/google-deepmind/mujoco)
[![Hardware](https://img.shields.io/badge/Robot-OpenArm%20v1%20(left%20arm)-yellow.svg)](#)

Two cans look identical. One is full, one is empty. **No camera can tell which.** RoboTokens shows that vision
alone is not enough: vision finds the objects, and the robot then picks each one up and feels its weight through
its own joint torques.

---

## 1. The Final Demo

1. **Vision** (SAM2 + Grounding DINO, optional VLM, RGB camera only) finds two similar objects (cans or boxes) and
   gives their X-Y pose. The table height is known, so Z is known.
2. The **left OpenArm** goes to object 1, grasps and lifts it, and estimates its mass from the joint torques. It puts the
   object back, then does the same with object 2.
3. It **compares** the two masses, then either:
   - **sorts** the objects into predefined bins by weight, or
   - **hands over** the heavy one (the full can) at an up-forward pose and opens the gripper only after it
     **feels** an external contact in the torques, not when the user merely looks close.

Full specification, state machine and success criteria: [docs/final_task/demo_task.md](docs/final_task/demo_task.md).

**Only the left arm is used.** The right arm has a known hardware problem and is off by default
(`hardware.right.enabled: false` in [configs/task_config.yaml](configs/task_config.yaml)).

---

## 2. Project Status

Several lines of work run in parallel on separate branches and **are not merged yet**.

| Line | What it is | Status |
|---|---|---|
| `main` (this branch) | Documentation, the older simulation pipeline, workcell scene generation | Docs aligned with the final demo |
| `origin/deploy/openarm-real` | Real-robot Cartesian controller | Reported working; needs a per-arm enable gate (right arm) before use |
| Vision branch | Object segmentation and X-Y pose | In progress |
| `openarm-cartesian-sim-validated` | Simulation and feasibility lab (not on the real-robot path) | Simulation-validated, never run on the arm |
| Weight-from-torque estimator | Mass from joint torques | Not pushed yet |

Branch details, overlaps, blockers and the proposed integration order: [docs/final_task/branches_and_integration.md](docs/final_task/branches_and_integration.md).

---

## 3. Ecosystem References

| Component | Upstream | Role |
|---|---|---|
| **OpenArm** | [enactic/openarm](https://github.com/enactic/openarm) | `openarm_can` motor control (C++ SocketCAN), `openarm_description` robot model, `openarm_mujoco` simulator model, `openarm_hardware` CAD |
| **MuJoCo** | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Physics, contact and kinematics |
| **SAM2 and Grounding DINO** | upstream model repositories | Open-vocabulary object segmentation in the vision branch |
| **OMPL** | [ompl.kavrakilab.org](https://ompl.kavrakilab.org/) | Sampling-based motion planning (used by the older simulation pipeline) |
| **RobotBlockSet (RBS by JSI)** | [abr-ijs/robotblockset_python](https://github.com/abr-ijs/robotblockset_python) | Optional; not used for OpenArm IK or control |
| **Camera** | [HBV HD CAMERA specs](docs/setup/camera_specifications.md) | RGB only, 720p/1080p at 30 FPS |

---

## 4. Repository Structure (current `main`)

Files are sorted by purpose: **final task** (what we are building now), **setup** (how the environment was brought up,
still valid) and **legacy** (what we used to get started; kept for reference). Each folder has a README.

```
RoboTokens/
├── agents.md                      # Engineering context and directives for AI agents
├── README.md
├── pyproject.toml, requirements.txt, uv.lock
├── configs/
│   ├── task_config.yaml           # Parameters; arm enablement under `hardware`
│   ├── charuco_*.png, *.pdf       # ChArUco boards for camera calibration
│   └── legacy/                    # bimanual_openarms.xml (5-DoF placeholder model)
├── docs/                          # Index: docs/README.md
│   ├── final_task/                # demo_task, architecture, interfaces, roadmap, branches_and_integration, cross_embodiment
│   ├── setup/                     # openarm_v1_environment, physical_workcell, camera_specifications
│   └── legacy/                    # register of deprecated items
├── scripts/                       # Index: scripts/README.md
│   ├── final_task/                # cross-embodiment demo: scene generator, runner, viewer, mock publisher
│   ├── setup/                     # setup_openarm_v1_scene, generate_charuco_board
│   └── legacy/                    # run_pipeline, calibrate_camera, run_pick_place_demo
├── src/
│   ├── cross_embodiment/          # Simulated FR3/UR5e running the demo, OpenArm follower, state stream (final task)
│   ├── kinematics/                # IK used by the simulated arms
│   ├── vision/                    # Egocentric hand tracker (legacy; older pipeline from here down)
│   ├── planning/                  # OMPL planner (partly legacy)
│   ├── control/                   # Hand-driven bimanual controller (legacy)
│   └── simulation/                # MuJoCo environment (partly legacy)
└── tests/                         # test_pipeline (older pipeline), test_cross_embodiment (demo)
```

`src/` and `tests/` were **not** reorganized: moving them would rewrite imports and touch files the other branches also
edit. Apart from `src/cross_embodiment/` and `src/kinematics/`, everything in `src/` belongs to the older simulation pipeline; the details and what replaces each part
are in [docs/legacy/README.md](docs/legacy/README.md).

Planned after integration (not created yet): `src/vision/` becomes the vision-branch code, `src/hardware/`,
`src/control/` and `src/planning/` come from the deploy branch, plus `src/sensing/` (gravity model, mass estimator,
contact detector) and `src/task/` (the real demo state machine). The legacy code in `src/` then moves
under `legacy/`.

---

## 5. Quickstart (simulation on `main`)

```bash
git clone https://github.com/pietrodardano/RoboTokens.git
cd RoboTokens
uv venv --python 3.11
uv sync --extra rbs --extra dev
uv run python -m unittest discover -s tests -v
```

For the official OpenArm v1 sources, the MuJoCo scene, rendering and the hardware boundary, follow
[docs/setup/openarm_v1_environment.md](docs/setup/openarm_v1_environment.md).

**Real robot:** the controller lives on `origin/deploy/openarm-real`, with its own setup and bring-up checklist in
that branch's `docs/real_robot_deployment.md`. Before running it, read the right-arm blocker in
[docs/final_task/branches_and_integration.md](docs/final_task/branches_and_integration.md). Always calibrate the left arm with
`openarm-can-zero-position-calibration --arm-side left_arm --canport can1 --robot-version v1`.

---

## 6. Documentation Index

The full index is [docs/README.md](docs/README.md). The essentials:

| Start here | Contents |
|---|---|
| [docs/final_task/demo_task.md](docs/final_task/demo_task.md) | The final demo: claim, sequence, state machine, mass and contact logic, constraints |
| [docs/final_task/architecture.md](docs/final_task/architecture.md) | Data flow, frames, timing, safety |
| [docs/final_task/interfaces.md](docs/final_task/interfaces.md) | Contracts between vision, control, proprioception and task (existing vs proposed) |
| [docs/final_task/branches_and_integration.md](docs/final_task/branches_and_integration.md) | Branch map, blockers, conflicts, integration order |
| [docs/final_task/hackathon_roadmap.md](docs/final_task/hackathon_roadmap.md) | Workstreams, dependencies, risks, pitch |
| [docs/final_task/cross_embodiment.md](docs/final_task/cross_embodiment.md) | Simulated FR3 and UR5e running the demo next to the real OpenArm |
| [docs/setup/](docs/setup/) | OpenArm sources and simulator, workcell geometry, camera hardware |
| [docs/legacy/README.md](docs/legacy/README.md) | What is deprecated and where the legacy files live |
| [scripts/README.md](scripts/README.md) | Scripts by purpose |
| [agents.md](agents.md) | Directives for AI coding agents |
