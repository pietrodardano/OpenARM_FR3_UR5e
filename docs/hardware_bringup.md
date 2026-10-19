# Hardware bring-up

These stages take you from power-on to the full demo. Do them in order. Each stage must pass before you start the next. Keep the e-stop within reach throughout.

After the first stage, every command moves the arm. Every command starts at the hanging rest pose and ends back there. The arm travels between rest and home through the side route beside the table. Ctrl+C stops safely: if the arm holds an object, it puts it back, then parks. While the arm returns, further Ctrl+C presses are ignored.

Only the left arm is driven (`robot.side: left`, `can1`).

## 0. Once per machine

```bash
sudo apt install git cmake g++ libcli11-dev can-utils   # libcli11-dev: openarm_can build
./setup.sh                                               # .venv, official model, report
```

## 0b. After every power-on

```bash
sudo scripts/can_up.sh            # CAN-FD 1 Mbit/5 Mbit on can0 / can1
```

If `check` reports joints outside their limits, the motor zero is off. Hang the arm straight down and run `openarm-can-zero-position-calibration` (from the `openarm-can-utils` package), then repeat `check`.

## Stages

| # | Command | Pass when |
|---|---|---|
| 1 | `uv run openarm-fr3-ur5e check` | All 7 joints and the gripper answer, joints are within limits, no status ≥ 0x8, and the TCP looks plausible. Nothing moves. |
| 2 | `uv run openarm-fr3-ur5e zero` | The arm enables at its current pose without a jump, then returns to rest and disables. |
| 3 | `uv run openarm-fr3-ur5e grip` | It closes on an empty can and on a full can, holds both, and does not crush the empty one. The jaw width it reports is about the can diameter. |
| 4 | `uv run openarm-fr3-ur5e run --objects manual --dry-path` | The arm goes home, then above and to each pre-grasp pose, above both bins, and to the handover pose, without grasping. Watch the twin: green (commanded) and yellow (measured) should overlap. |
| 5 | `uv run openarm-fr3-ur5e weigh --labels full empty --n 5` | The two masses separate clearly (z ≥ 3), and the spread over 5 runs is well below their difference. Record both numbers. |
| 6 | `uv run openarm-fr3-ur5e detect` | Measure the positions with a ruler or by touching each object with the TCP (`check` prints the TCP). The error must be within about ±1 cm. Do this with the nominal pose, then again after `calib touch`. |
| 7 | `uv run openarm-fr3-ur5e run --prompt "put the lighter can in the left bin"` | Both objects end up in the correct bins. `runs/<time>/report.json` has the masses. |
| 8 | `uv run openarm-fr3-ur5e run --prompt "give me the heavy can"` | It releases only after you take the can. If nobody takes it, after `task.contact.timeout_s` it puts the can in its bin. |

For stage 8, try the timeout path once: let the timeout run out without taking the can.

## Camera calibration

- **Intrinsics only.** `configs/camera_intrinsics.yaml` was measured (reprojection error 0.17 px). The camera pose comes from the nominal mount in `workcell.camera`. Expect errors of a few cm.
- **Intrinsics + touch.** Run `uv run openarm-fr3-ur5e calib touch`, which never commands motion:
  1. Close the gripper.
  2. Move the arm by hand to 12–20 points at 2–3 heights, pressing Enter at each.
  3. Click the fingertip in each frame.

  This writes `configs/camera_extrinsics.yaml`, which overrides the nominal pose. Delete that file to go back to the nominal pose.
- **Redo the intrinsics** (only after changing the camera mode or lens). Run `calib capture`, take 20–30 photos at varied angles, then run `calib intrinsics --square-mm <measured> --marker-mm <measured>`.

## Faults

A latched fault freezes the arm at its measured pose. The causes are a motor error, a silent motor, bus-off, or tracking error over 0.5 rad for 0.3 s. After a fault, the command does **not** move the arm on its own: clear the cause first.
