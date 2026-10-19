# OpenARM FR3 UR5e dashboard

A read-only view of what the robot saw, estimated and decided. It is standalone: it does not import or change any code in `src/`, and it never opens the camera, the CAN bus or the robot, so it is safe to run beside a live experiment.

```bash
.venv/bin/python dashboard/serve.py                 # watches ./runs, open http://127.0.0.1:8765
.venv/bin/python dashboard/serve.py --runs /path    # another runs directory (a synced or mounted copy works)
.venv/bin/python dashboard/serve.py --demo          # synthetic data, labelled DEMO, to preview the layout
```

It needs only `pyyaml` (already in the project environment). The page polls the server every 2 seconds.

## What it shows

| Panel | Source | Notes |
|---|---|---|
| Run status | `runs/<time>/report.json` | outcome, instruction, mode, heavier object, handover result, error |
| Camera | `runs/detect/detections.png`, `frame.png` | the last saved detection image, not live video |
| Digital twin (MuJoCo) | the report + `twin3d.py` | the project's own workcell model (`build_scene`, imported read-only) with each can or box placed at the position vision found; band color shows heavy/light once weighed; rendered on demand and cached in `dashboard/.cache/` |
| Workcell twin (top view) | the configured workcell + the report | table, bins, vision workspace, weigh pose, objects at their believed positions, where each went |
| Physical estimate | report beliefs | mass ± standard error per object, with the per-cycle estimates |
| Decision | report comparison | `z = Δm / SE` against the `min_z` threshold, conclusive or not |
| Phases | report phase log | what the state machine did |
| Run history | all reports | click a row to inspect that run; untick "follow latest run" to stay on it |
| Simulation | `docs/figures/` | the cross-embodiment GIF and results (MuJoCo, not live) |

## Live joint torques (passive CAN tap)

```bash
# on the robot PC, beside the running experiment
.venv/bin/python dashboard/serve.py --can            # interface, IDs and motor types come from configs/robotokens.yaml
# from your laptop: ssh -L 8765:127.0.0.1:8765 <robot-pc>, then open http://127.0.0.1:8765
.venv/bin/python dashboard/serve.py --live-demo      # synthetic feed, labelled SYNTHETIC, to preview the panel
.venv/bin/python -m pytest dashboard/tests           # offline tests: decoder, config channels, no-transmit guarantee
```

**Why it is safe**
- The tap is a second, **receive-only** SocketCAN socket. Linux hands every listener a copy of each frame, so the robot's own connection is not involved.
- The socket object is wrapped so only `recv` is reachable: there is no send path in the code, and a test checks that.
- It never configures the interface (no `ip link`, no `can_up.sh`) and never opens an openarm_can connection.
- A kernel-side filter passes only the 8 motor-reply IDs, so the extra work is small. The server also lowers its own priority (`nice`) so it cannot compete with the servo thread.
- The server binds to `127.0.0.1` by default. Use an SSH tunnel to see it from another machine.
- It reads `configs/robotokens.yaml` but never writes anything outside `dashboard/`.

**What it shows:** the motor-reported torque per joint and for the gripper, in N·m, the same quantity the robot code reads as `ArmState.tau`. Gripper torque is in motor sign. Motor status nibbles `>= 0x8` raise a red fault banner, and a yellow banner appears if no frames arrive for 1.5 s.

**Not verified on hardware.** The frame decoding follows `parse_motor_state_data` in `openarm_can` and is unit-tested against that formula, but it has not been run against the real bus. Before trusting it, check with the arm idle: torques should be small at the rest pose and the shoulder and elbow torques should rise when it lifts, matching what you see in the robot's own reports. If you see no frames, check the interface name and whether the bus is CAN-FD (`hardware.can_fd`).

## Limits

- **Reports appear when a run finishes.** `save_report` writes `report.json` at the end, so the dashboard cannot show a run in progress. Live joint states and the live camera would need a change on the robot side, which this dashboard deliberately does not make.
- **The MuJoCo panel is a rebuild, not a live mirror.** It shows the scene as detected at the start of the run, with the arm at the home pose. It needs `mujoco` and the OpenArm model (`./setup.sh`); on a machine with no display, set `MUJOCO_GL=egl` or `osmesa`. It renders in a subprocess, so a GL failure only blanks that panel (the reason is written to `dashboard/.cache/twin/last_error.txt`).
- **The top-view map** is drawn from the config and the report, not from MuJoCo.
- **Camera panel:** `openarm-fr3-ur5e detect` writes `runs/detect/detections.png`. `run` does not currently save its frames, so the panel shows the last `detect` call.
