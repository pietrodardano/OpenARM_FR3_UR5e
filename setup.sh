#!/usr/bin/env bash
# One-shot, idempotent setup of the repo-local .venv (uv). Run from anywhere:
#
#   ./setup.sh             # everything: hardware (openarm_can) + vision (GroundingDINO, SAM2) + tests
#   ./setup.sh --no-hw     # skip the openarm_can build (laptop without the robot)
#
# System prerequisites (Ubuntu 24.04), once:
#   sudo apt install git cmake g++ libcli11-dev can-utils     # libcli11-dev: openarm_can C++ build
#   curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
cd "$(dirname "$0")"

OPENARM_MUJOCO_REV=36d4a44   # enactic/openarm_mujoco commit validated on the real robot

sync_args=()
if [[ "${1:-}" == "--no-hw" ]]; then
  sync_args+=(--no-group hardware)
  export UV_NO_SYNC=1  # later `uv run` calls must not try to build openarm_can either
  echo "NOTE: without openarm_can, use the venv directly: source .venv/bin/activate"
elif ! dpkg -s libcli11-dev >/dev/null 2>&1; then
  echo "ERROR: openarm_can needs libcli11-dev: sudo apt install libcli11-dev  (or ./setup.sh --no-hw)" >&2
  exit 1
fi

# 1) Official OpenArm MuJoCo model (MJCF + meshes): kinematics, IK, gravity, twin.
if [[ ! -e third_party/openarm_mujoco/.git ]]; then
  git clone https://github.com/enactic/openarm_mujoco.git third_party/openarm_mujoco
fi
git -C third_party/openarm_mujoco checkout --quiet "$OPENARM_MUJOCO_REV"

# 2) Python environment.
uv sync "${sync_args[@]}"

# 3) Model weights (not downloaded here; see README).
for f in GroundingDINO_SwinT_OGC.py groundingdino_swint_ogc.pth sam2.1_hiera_small.pt; do
  [[ -s weights/$f ]] && echo "ok    weights/$f" || echo "MISSING weights/$f (vision will not run)"
done

# 4) Report.
uv run python - <<'EOF'
import importlib
for name, check in [("cv2", lambda m: hasattr(m.aruco, "CharucoDetector")),
                    ("mujoco", None), ("mink", None),
                    ("torch", lambda m: m.cuda.is_available()),
                    ("groundingdino", None), ("sam2", None), ("openarm_can", None)]:
    try:
        m = importlib.import_module(name)
        extra = "" if check is None else f"  ({'ok' if check(m) else 'CHECK FAILED'})"
        print(f"ok    {name}{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"--    {name}: {e.__class__.__name__}")
EOF
echo "Done. Try: uv run pytest   |   uv run openarm-fr3-ur5e --help"
