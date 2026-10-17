# Real OpenArm v1 Deployment

Plain-Python control interface for the physical OpenArm v1 bimanual robot: no
ROS. One Python thread streams MIT (impedance) setpoints to the motors through
the official `openarm_can` SDK. On top of that sit a joint position controller
and an IK controller. The same API runs in MuJoCo on the official model, so
everything can be dry-run first.

```
your script / skill / policy
  │  TCP poses (abs / delta, moveL)          │  joint positions (abs / delta)
  ▼                                          │
IKController ── mink IK on official MJCF ────┤
                                             ▼
                              JointPositionController
                                             ▼  set_target(16)   (lock, non-blocking)
     OpenArmBimanual: Python thread @ 250 Hz ─ safety filter ─ openarm_can mit_control_all
     (or OpenArmSim: same API, official MuJoCo model)
                                             ▼
                 Damiao motors: can0 = right arm, can1 = left arm (7 joints + gripper each)
```

## Files

| File | Role |
|---|---|
| `src/control/robot_interface.py` | Backend API (`start/stop`, `get_state`, `set_target`, `move_joints`), state types, 16-dim command layout |
| `src/control/safety.py` | Joint-limit clamp + per-joint velocity limit, applied inside every backend loop |
| `src/control/joint_position_controller.py` | `command(q, gripper)`, `command_delta(dq)` |
| `src/control/ik_controller.py` | `command_poses(...)` (servoL-like), `command_delta(dpos, drot)`, `move_l(...)` (moveL-like), `get_tcp_pose()` |
| `src/planning/openarm_kinematics.py` | Official model: FK of the `<side>_tcp` sites, joint limits, gravity torques |
| `src/planning/openarm_ik.py` | mink differential IK (joint limits as hard constraints, posture pull to `home`) |
| `src/hardware/openarm_bus.py` | Only file importing `openarm_can` (one arm per CAN bus) + `FakeOpenArmBus` |
| `src/hardware/openarm_bimanual.py` | The real driver: streaming thread, safety, faults, startup/shutdown |
| `src/simulation/openarm_sim.py` | MuJoCo backend with the same API |
| `scripts/check_robot.py`, `scripts/jog_joints.py`, `scripts/run_control_demo.py` | Bring-up: read-only check, ±3° per-joint jog, controller demos (each takes `--backend real/sim/fake`) |
| `configs/task_config.yaml` | `robot`, `hardware`, `controllers` sections: every tunable |

## Conventions

- **Joints:** `openarm_<side>_joint1..7` in rad. Motor angle = URDF joint angle = MuJoCo joint angle, with the same limits and axis signs on both arms (verified against `openarm_description`).
- **Gripper:** finger opening in metres, `0.0` = closed, `0.044` = open. On the motor, 0 rad is closed and −1.0472 rad is open (the official `openarm_ros2` mapping). A full stroke takes 2 s (`max_gripper_velocity_m_s: 0.022`).
- **16-dim command vector:** `[left j1..j7, left gripper, right j1..j7, right gripper]`, the same order as the MJCF actuators.
- **Base frame:** the official model's world frame. +X forward, +Y left, +Z up, origin on the floor under the torso. At `home` (j4 = π/2) the TCPs are at about (0.41, ±0.15, 0.48) m, pointing forward.
- **Delta commands** apply to the previous *command*, not the measured pose, so gravity sag never accumulates.

## Usage

```python
import yaml
import numpy as np
from src.hardware import OpenArmBimanual           # or: from src.simulation.openarm_sim import OpenArmSim
from src.control.ik_controller import IKController

cfg = yaml.safe_load(open("configs/task_config.yaml"))
robot = OpenArmBimanual.from_config(cfg)           # OpenArmSim.from_config(cfg, realtime=True, render=True)
ik = IKController.from_config(cfg, robot)

robot.start()                                       # enable, hold the measured pose (no jump)
try:
    home = {s: np.array(cfg["robot"]["home_qpos"][s]) for s in ("left", "right")}
    robot.move_joints(home, gripper={"left": 0.044, "right": 0.044}, duration=4.0)
    ik.reset()

    p, R = ik.get_tcp_pose()["left"]
    ik.move_l({"left": (p + [0.05, 0, -0.05], R)}, duration=2.0)   # blocking, straight line

    for _ in range(100):                            # streaming at 20 Hz, e.g. a policy
        ik.command_delta({"right": np.array([0.0, 0.0, 0.002])}, gripper_m={"right": 0.0})
        robot.advance(0.05)                         # keeps a fixed loop rate
        state = robot.get_state()                   # q, dq, tau, gripper, setpoints, fault
finally:
    robot.stop()                                    # move to rest (arms hanging), disable motors
```

`JointPositionController(robot)` works the same way with `command({...})` and `command_delta({...})`.

## Safety behaviour (driver thread)

- **Start:** reads every motor and refuses to start if any joint reads more than 0.15 rad outside its limits (that means the motor zero calibration is off). Otherwise the setpoint starts at the measured pose.
- **Each tick:** the target is clipped to limits minus 0.02 rad, then rate-limited (`max_joint_velocity_rad_s`, `max_gripper_velocity_m_s`). A jump in a command becomes a bounded-velocity motion.
- **Faults (latched):** motor error codes (over-current, overheat, …), a motor that stops replying for 0.1 s, a CAN bus-off, or a tracking error above 0.5 rad for 0.3 s (collision or blocked joint). On a fault the robot holds its measured pose and ignores new targets until `robot.clear_fault()`.
- **Stop:** moves to `rest_qpos` (arms hanging), then disables the motors. Do not disable mid-air: the arms fall.
- These checks are software safety only. Keep the hardware e-stop within reach.

## Bring-up checklist (robot PC)

1. **Rootful Docker + NVIDIA runtime.** Rootless Docker's `network_mode: host` is not the real host network, so `can0`/`can1` would be invisible (`docker/dev.sh` warns about this).
2. **Official sources:** clone them into `third_party/openarm/` as described in [openarm_v1_environment.md](openarm_v1_environment.md). Validated against `openarm_can` 1.4.0 (`f340d4b`), `openarm_description` `206eb8a` and `openarm_mujoco` `36d4a44`.
3. **Build and enter the container:** `docker/dev.sh build`, then `docker/dev.sh shell`. The image builds the `openarm_can` Python binding pinned to 1.4.0 and installs `openarm-can-cli` from the Enactic PPA.
4. **CAN up:** `scripts/setup_can.sh` (CAN-FD 1M/5M on `can0` and `can1`). Check with `openarm-can-cli -i can0 discover`.
5. **Zero-position calibration (official tool).** Our interface assumes motor zero = model zero (both arms hanging straight down, gripper closed). The official tool finds it precisely: it drives each joint into its mechanical stop, moves back by the known stop-to-zero distance, and writes the zero to motor flash. Run it one arm at a time:
   ```bash
   openarm-can-zero-position-calibration --robot-version v1 --arm-side right_arm --canport can0
   openarm-can-zero-position-calibration --robot-version v1 --arm-side left_arm  --canport can1
   ```
   - **Always pass `--robot-version v1`.** The tool defaults to `v2`, which has different gripper limits and a different left-arm sequence.
   - The arm moves on its own and presses every joint against its hard stops. Before running it:
     - Mount the robot securely.
     - Clear the full range of motion.
     - Start with the arm hanging.
     - Keep a hand on the e-stop.
   - The zero survives power cycles, so calibrate once per assembly, and again after a motor swap or if `check_robot.py` shows offsets.
   - LeRobot's `lerobot-calibrate` writes the same kind of zero, but positioned by eye. Use the official tool when IK accuracy matters.
6. **Passive read, no motion:** `python scripts/check_robot.py`. With the arms hanging, every joint should read close to 0 rad and no motor should report a fault. Moving an arm by hand should change the readings in the expected direction.
7. **Rehearse the jog test:** `python scripts/jog_joints.py --backend sim --render --yes`. `--backend fake` runs the real driver code against a simulated bus.
8. **First motion, gentle: jog each joint ±3° at 2°/s.** Start with the arms hanging:
   ```bash
   python scripts/jog_joints.py --backend real                               # both arms, asks before every joint
   python scripts/jog_joints.py --backend real --side right --joints 5,6,7   # a subset
   ```
   - The script first holds the current pose for 3 s with no motion and reports the hold error and motor status.
   - It then plans every joint/direction from the current pose. A direction is skipped if it would leave the joint limits or bring the arm closer to the torso or the other arm (official collision meshes). From the hanging pose that skips joint 2 inward, one joint 6 direction, and joint 4 negative.
   - For each jog it reports the measured motion and tracking error, and flags any joint that moves the wrong way or too little (a sign or calibration problem).
   - It finishes with a small gripper move, then moves to rest and disables.
9. **Larger motions:** `python scripts/run_control_demo.py --backend real --controller joint --amplitude 0.3`, then `--controller ik`. Rehearse each one first with `--backend sim --render`.
10. **Gravity feedforward:** with `gravity_compensation_scale: 0.0` the arm sags about 0.06 rad at `home`, which puts the TCP a few cm low. Raise the scale gradually (0.3 → 0.7 → 1.0) and confirm that the measured-vs-commanded TCP error printed by the IK demo shrinks. If it grows, the torque sign is wrong; set it back to 0.
11. **Table:** measure the table height in the base frame and set `controllers.ik.workspace.*.low[2]` above it.

## Tests

```bash
docker/dev.sh run python -m pytest tests/ -q
```

The suite covers the safety filter and the real threaded driver on `FakeOpenArmBus`: no jump at start, velocity limiting, gripper unit mapping, a silent motor latching a fault, refusal to start outside limits, and rest-then-disable on stop. It also covers joint and IK control on the official MuJoCo model.
