#!/usr/bin/env python3
"""Open the visual-only OpenArm, UR5e, and FR3 MuJoCo comparison scene."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_MJPYTHON = PROJECT_ROOT / ".venv" / "bin" / "mjpython"
SCENE_PATH = PROJECT_ROOT / "third_party/mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml"


def _ensure_macos_gui_interpreter() -> None:
    """Re-execute with ``mjpython`` on macOS so the passive viewer can start."""
    if sys.platform != "darwin" or os.environ.get("ROBOTTOKENS_MJPYTHON") == "1":
        return
    if not VENV_MJPYTHON.is_file():
        raise FileNotFoundError(
            f"Project MuJoCo interpreter not found: {VENV_MJPYTHON}. Run `uv sync --extra rbs --extra dev`."
        )
    environment = os.environ.copy()
    environment["ROBOTTOKENS_MJPYTHON"] = "1"
    os.execve(str(VENV_MJPYTHON), [str(VENV_MJPYTHON), str(Path(__file__).resolve())], environment)


def main() -> None:
    """Load the generated CAD-backed comparison scene and keep its viewer open."""
    _ensure_macos_gui_interpreter()
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(
            f"Comparison scene not found: {SCENE_PATH}. Run scripts/final_task/setup_cross_embodiment_scene.py first."
        )

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("cross_embodiment_overview").id
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()