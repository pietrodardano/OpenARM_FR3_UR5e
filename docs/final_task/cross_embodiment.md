# Cross-Embodiment Demo: Simulated FR3 and UR5e Next to the Real OpenArm

> **Status: final task.** Built and tested in simulation on branch `sim-cross-embodiment-demo`. Not merged.

The same weigh-by-touch task ([demo_task.md](demo_task.md)) runs on three robots at once in one MuJoCo scene:

| Robot | Role | How it moves | Where its objects come from |
|---|---|---|---|
| OpenArm (left arm) | The **real** robot | Follower: its joints are set from the real arm's measured state, every step. It never moves on its own and never touches the sim objects | Real cans, found by vision (not rendered) |
| FR3 (left arm) | Simulation only | Its own torque-controlled arm and task | True positions straight from the config: it may "cheat", there is no vision |
| UR5e (left arm) | Simulation only | Same as FR3 | Same layout as FR3 relative to its own left arm |

**In step, but independent.** Each sim robot starts a task phase only after the real task has entered the same phase,
so all three walk through the demo together. Within a phase each sim robot acts on its own objects at its own pace.

## What each simulated robot does

1. Records an unloaded reference torque at the weighing pose above the first can (gripper closed, empty).
2. Picks the can **laterally**: approaches along $+x$ with the tool horizontal and the fingers closing sideways.
3. Lifts it, holds still, and records the loaded torque at the same pose.
4. Estimates the mass from the loaded-minus-unloaded torque difference, puts the can back, and repeats for can 2.
5. Compares the two masses (or reports "indistinguishable"), then runs its ending:
   - **Sort** (FR3 by default): heavy can into the red bin, light can into the green bin.
   - **Hand-over** (UR5e by default): carries the heavy can to the hand-over pose, takes a baseline torque, and waits.
     A simulated user hand approaches and pulls on the can. The gripper opens only when the joint torque change is
     felt. If no contact comes, it never opens on a timeout: the can goes to the heavy bin instead.

## What is physical and what is a simulation cheat

| Physical | Cheat (allowed because these robots are simulation only) |
|---|---|
| Arms are torque-controlled (PD plus gravity feedforward through `qfrc_applied`); the reported torque is the applied torque plus noise, like a motor | Object positions are given directly, no vision |
| Gravity, joint friction (FR3), inertia, the can's weight loading the arm | A grasp is held by a weld created when the gripper closes; the can's collisions are off while it is held |
| Mass is estimated from torque, not read from the model | The user's pull is an injected external force on the can |
| Contact is detected from torque, not from the injected force | Positions were chosen inside each arm's reach |

## Measured in simulation

Twelve runs (six seeds, with the full can swapped between positions 1 and 2), both robots per run:

| Metric | Result |
|---|---|
| Runs finished without a motion error | 12 / 12 |
| Correct heavier-can decision | 12 / 12 |
| Mass error, mean / worst | 2 g / 4 g (cans of 60 g and 360 g) |
| Hand-over released on felt contact | 6 / 6; never before the simulated user pulled |
| No-contact case | Gripper stays closed, can goes to the heavy bin (tested) |

These numbers are optimistic: the controller's gravity model and the plant are the same MJCF, the can positions
were varied only between the two configured spots, and the sensor noise is 0.02 Nm. Treat them as a check that the
method and the code work, not as a prediction for the real arm.

Problems found and fixed while building it (each would have produced wrong masses silently):

- **Settling.** Measuring before the arm had fully stopped put the damping term into the torque (one estimate was 111 %
  off). The task now waits until the joints are still.
- **Friction.** FR3's joints have about 1.1 Nm of friction, which shifts the static torque depending on the direction
  the arm arrived from. The task now averages one arrival from below and one from above, which cancels symmetric friction.
- **The idle arm.** Each robot's unused right arm was unpowered, sagged, and pushed on the working arm, causing a
  5 to 10 Nm torque oscillation. Idle arms are now held at their pose and have no collisions.
- **Release.** Dropping the weld while the can still overlapped the finger pads threw the can across the room. The
  gripper now opens first, then the weld is dropped.
- **Masses from config.** The scene file bakes masses in; the world now applies them from the config with consistent
  inertia, so editing `full_can` cannot silently disagree with the scene.

## The link to the real arm

The real side publishes a small JSON stream (joint angles, gripper opening, task phase); the contract is in
[interfaces.md](interfaces.md#real-arm-state-stream-proposed). Three sources plug into the same interface:

| Source | Use |
|---|---|
| `mock` | A scripted stand-in for the real arm (lateral pick, lift, weigh, put back, present), so everything runs offline |
| `udp:PORT` | Live stream from the real side |
| `replay:FILE` | A recorded stream (JSON lines) |

## Run it

```text
uv run python scripts/setup/setup_openarm_v1_scene.py              # once: the OpenArm workcell scene
uv run python scripts/final_task/setup_cross_embodiment_scene.py   # after any change to the demo config
uv run python scripts/final_task/run_cross_embodiment_demo.py --show
```

Options: `--source udp:5600` for the live stream, `--sync free` to ignore the real phases, `--ur5e-ending sort` or
`--fr3-ending handover` to swap endings, `--no-touch` to test the no-contact fallback. To try the UDP path without the
robot, run `scripts/final_task/publish_mock_state.py` in a second terminal and use `--source udp:5600`.

## Layout

Everything comes from the `cross_embodiment_demo` block of [configs/task_config.yaml](../../configs/task_config.yaml).
The scene generator and the controllers read the same block.

- Each station has a 60 mm torso with two side-mounted arms (shoulders at 1.33 m), Robotiq 2F-85 grippers, and a table
  (top at 0.74 m) with two identical-looking cans (60 g empty, 360 g full), a red heavy bin and a green light bin.
- The UR5e station is at $y = +1.35$ m and the FR3 station at $y = -1.35$ m; the OpenArm is in the middle.
- Every robot works with its left arm, like the real OpenArm (`arm` per station in the config). Each station's table,
  cans, bins and hand-over pose sit in the same place relative to that arm.
- The idle right arm is parked as the mirror image of the working arm's home pose about the robot's own torso, computed
  when the scene is generated, so each robot looks symmetric. It is held at that pose and has no collisions.
- Home poses keep every link on the robot's own side of its torso, so the two arms of a robot never cross.

## Limitations

- The sim-side mass estimator in `src/cross_embodiment/mass_probe.py` is a stand-in. The real estimator is not pushed
  yet; compare the two when it lands.
- The OpenArm's own cans are not rendered. Rendering them from the vision poses would need the vision output in the
  stream.
- The mock stream is not the real controller and checks no collisions; it only walks through the same phases.
- Paths are checked against the table, bins and cans, but not against the other arm of the same robot (it is
  non-colliding by design) or against the OpenArm.
- The torque uncertainty used for the "indistinguishable" decision (0.25 Nm) is an assumption, not a measurement.
- If a sim robot is slower than the real task in some phase, it falls behind and catches up at the next phase it waits for.

## Files

| Path | Contents |
|---|---|
| `src/cross_embodiment/layout.py` | Reads the demo config |
| `src/cross_embodiment/world.py` | Scene, torque-controlled arms, parked arms, OpenArm follower, welds, simulated user |
| `src/cross_embodiment/motion.py` | Lateral-grasp orientation, straight Cartesian paths, contact checks, timing |
| `src/cross_embodiment/task.py` | The task state machine and the phase gate |
| `src/cross_embodiment/mass_probe.py`, `contact_detector.py` | Mass from torque, contact from torque change |
| `src/cross_embodiment/stream.py`, `mock_openarm.py`, `runner.py` | State stream, scripted real arm, tick loop |
| `src/kinematics/arm_kinematics.py` | IK; copied from `openarm-cartesian-sim-validated` (a duplicate to resolve at integration) |
| `scripts/final_task/` | Scene generator, demo runner, viewer, mock publisher |
| `tests/test_cross_embodiment.py` | 18 tests, including full runs of both robots |
