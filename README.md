# RoboTokens: vision cannot weigh

Two cans look identical, but one is full and one is empty. A camera can't tell them apart. An OpenArm can, by lifting them:

1. An RGB camera with **GroundingDINO + SAM2** finds the two objects named in the prompt.
2. The arm lifts each object in turn and estimates its mass from its **joint torques**. It keeps that estimate as a belief, then puts the object back.
3. After the second object it compares the two masses, then either:
   - **sorts** the heavy and the light object into their bins, or
   - **hands one over** to a person, opening the gripper only when it *feels* them take it.

A MuJoCo twin follows the real robot. The twin only reads the robot's state; it never commands it.

The requirements are in [docs/requirements.md](docs/requirements.md). The staged real-robot procedure is in [docs/hardware_bringup.md](docs/hardware_bringup.md).

## Setup (one `.venv`, no Docker)

```bash
sudo apt install git cmake g++ libcli11-dev can-utils   # once; libcli11-dev builds openarm_can
curl -LsSf https://astral.sh/uv/install.sh | sh          # once, if uv is missing
./setup.sh                                               # .venv + official OpenArm model + check
```

`setup.sh` does three things:
- Clones `enactic/openarm_mujoco` at the commit validated on the robot.
- Installs everything with `uv sync`: `openarm_can` 1.4.0, torch cu128, GroundingDINO, SAM2.
- Checks that the model weights are present.

On a laptop without the robot, run `./setup.sh --no-hw` instead.

The model weights are not downloaded; they live in `weights/`, which git ignores:

| File | Source |
|---|---|
| `GroundingDINO_SwinT_OGC.py` | github.com/IDEA-Research/GroundingDINO `groundingdino/config/` |
| `groundingdino_swint_ogc.pth` | GroundingDINO release `v0.1.0-alpha` |
| `sam2.1_hiera_small.pt` (or `_large`) | dl.fbaipublicfiles.com/segment_anything_2/092824/ |

## Use

```bash
uv run pytest                                    # offline: no robot, GPU or weights needed
uv run robotokens run --fake --prompt "Sort the two cans: lighter in the left bin"   # rehearsal, no hardware

sudo scripts/can_up.sh                           # real robot (see docs/hardware_bringup.md)
uv run robotokens check                          # read-only
uv run robotokens detect                         # where are the objects? (base frame)
uv run robotokens run --prompt "put the lighter can in the left bin"
uv run robotokens run --prompt "give me the heavy can"
```

`run` shows the MuJoCo twin unless you pass `--no-twin`. Object positions come from vision on the real robot and from `task.manual_objects` with `--fake`. Use `--objects manual` or `--xy "x,y;x,y"` to set them yourself.

Each run writes `runs/<time>/report.json` with:
- the parsed intent
- the beliefs (position, mass ± SE, class)
- the comparison
- where each object went
- the phase log

## Layout

```
configs/robotokens.yaml   every setting, one place (strictly validated)
src/robotokens/
  config.py               typed config loader
  robot/                  bus.py (openarm_can | FakeBus), driver.py (250 Hz servo thread, safety, faults),
                          model.py (official MJCF + workcell), ik.py (mink), motion.py (checked moveL,
                          side route), gripper.py (torque-mode grasp), fake.py (payload for --fake)
  sensing/                payload.py (mass from torques, compare), contact.py (felt handover)
  task/                   prompt.py (instruction -> intent), belief.py, skills.py, demo.py (state machine)
  vision/                 camera, calibration (intrinsics + nominal | touch pose), localizer
                          (GroundingDINO + SAM2), top_face, detector, calib_cli
  twin/mirror.py          read-only MuJoCo twin
  cli.py                  `robotokens` command
tests/                    one file per module; IDs from docs/requirements.md
```

## Frames

Every pose is in the base frame of the official OpenArm v1 model:
- +X points forward, +Y left, +Z up.
- The origin is at `openarm_body_link0`, 0.633 m above the floor.
- The tabletop is at z = 0.092 m. The code derives this from `workcell.table.top_height_m` and `workcell.arm_mount_height_m`.
