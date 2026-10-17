# Scripts

Scripts are sorted by what they are for. Run them from the repository root, for example
`uv run python scripts/setup/setup_openarm_v1_scene.py`.

## Final task (`final_task/`)

The cross-embodiment demo ([../docs/final_task/cross_embodiment.md](../docs/final_task/cross_embodiment.md)). The real-robot
controller and the vision pipeline live on other branches and are not merged
([../docs/final_task/branches_and_integration.md](../docs/final_task/branches_and_integration.md)).

| Script | Purpose |
|---|---|
| [final_task/setup_cross_embodiment_scene.py](final_task/setup_cross_embodiment_scene.py) | Generates the scene: OpenArm, side-mounted FR3 and UR5e, tables, cans, bins, simulated user hands. Re-run after changing the demo config |
| [final_task/run_cross_embodiment_demo.py](final_task/run_cross_embodiment_demo.py) | Runs the demo (`--show` for the viewer, `--source mock / udp:PORT / replay:FILE`) |
| [final_task/publish_mock_state.py](final_task/publish_mock_state.py) | Publishes the scripted real-arm stream over UDP; a reference for the real side |
| [final_task/view_cross_embodiment_scene.py](final_task/view_cross_embodiment_scene.py) | Opens the scene at its home pose, nothing moving |

## Setup (`setup/`)

Used to bring up the environment. Still valid.

| Script | Purpose |
|---|---|
| [setup/setup_openarm_v1_scene.py](setup/setup_openarm_v1_scene.py) | Generates the physical workcell MuJoCo scene from the official OpenArm v1 model (needs `third_party/openarm`, see [../docs/setup/openarm_v1_environment.md](../docs/setup/openarm_v1_environment.md)) |
| [setup/generate_charuco_board.py](setup/generate_charuco_board.py) | Generates the printable ChArUco boards in `configs/` for camera calibration |

## Legacy (`legacy/`)

Kept for reference. Do not add features or callers. Register: [../docs/legacy/README.md](../docs/legacy/README.md).

| Script | What it was |
|---|---|
| [legacy/run_pipeline.py](legacy/run_pipeline.py) | Egocentric hand-tracking teleoperation entry point |
| [legacy/calibrate_camera.py](legacy/calibrate_camera.py) | Zero-calibration of the hand tracker (not camera intrinsics or hand-eye) |
| [legacy/run_pick_place_demo.py](legacy/run_pick_place_demo.py) | Older simulation pick-and-place (superseded) |
