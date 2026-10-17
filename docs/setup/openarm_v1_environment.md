# OpenArm v1 Development Environment

> **Status: setup.** How the environment, workcell and hardware were brought up. Still valid reference, but not the final task itself.

This project uses a ROS-free Python simulation/planning stack and the official OpenArm v1 parallel-link model.

## Scope

- Python: 3.11 (validated on macOS)
- Dependency manager: `uv`
- Simulation: MuJoCo
- Planning utilities: RobotBlockSet (RBS) and OMPL
- Robot model: `openarm_description` v1 assets
- Simulator model: `openarm_mujoco/v1/openarm_bimanual.xml`
- Real hardware I/O: official `openarm_can` C++ SocketCAN SDK

Do not install or use CoppeliaSim, Genesis, ROS, or ROS2 for this project.

## Recreate the Python Environment

From the repository root:

```bash
uv venv --python 3.11
uv sync --extra rbs --extra dev
```

This creates `.venv/` and installs the locked, tested dependency set:

| Package | Verified version |
|---|---:|
| `robotblockset` | 1.3.1 |
| `mujoco` | 3.16.0 |
| `ompl` | 2.0.1 |

Use `uv run` to execute Python commands without manually activating the environment.

## Pull the Official OpenArm v1 Sources

The source checkouts are intentionally local and Git-ignored under `third_party/openarm/`.

```bash
mkdir -p third_party/openarm
git clone --depth 1 https://github.com/enactic/openarm_can.git third_party/openarm/openarm_can
git clone --depth 1 https://github.com/enactic/openarm_description.git third_party/openarm/openarm_description
git clone --depth 1 https://github.com/enactic/openarm_mujoco.git third_party/openarm/openarm_mujoco
```

Keep only the v1 assets required by this robot:

```bash
rm -rf third_party/openarm/openarm_mujoco/v2 \
       third_party/openarm/openarm_description/assets/robot/openarm_v2.0
```

The required model files are:

- `third_party/openarm/openarm_description/assets/end_effector/parallel_link/`
- `third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml`

## Validate the Installation

```bash
uv run python -c "import mujoco, ompl, robotblockset; print('RBS, MuJoCo, and OMPL are ready')"
uv run python -c "import mujoco; model = mujoco.MjModel.from_xml_path('third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml'); print(f'OpenArm v1 loaded: {model.nu} actuators')"
```

Expected result: the OpenArm v1 bimanual scene loads with 16 actuators: 7 arm actuators and 1 gripper actuator for each arm.

## Render the Official Robot

```bash
uv run python -m mujoco.viewer --mjcf=third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml
```

This is the authoritative simulator model for this project. Do not use `configs/legacy/bimanual_openarms.xml` for hardware geometry; it is a 5-DOF placeholder.

## Run the Current Project Tests

```bash
uv run python -m unittest discover -s tests -v
```

The existing tests exercise the current scaffold. They are not validation of the official 7-DOF OpenArm v1 model.

## Hardware Boundary

`openarm_can` is the only approved real-arm motor interface. Its supported API is C++ and requires Linux SocketCAN plus compatible CAN/CAN-FD hardware. The experimental Python binding must not be treated as the primary hardware interface.

macOS is suitable for development, planning, and MuJoCo rendering. Run physical-arm control on a Linux host configured for SocketCAN.
