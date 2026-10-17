"""View or render the OpenArm sandbox scene (build it first with setup_sandbox_scene.py).

    python scripts/view_sandbox.py                 # interactive viewer (needs a display)
    python scripts/view_sandbox.py --png out.png   # offscreen render of the overview camera
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from setup_sandbox_scene import SANDBOX_MODEL  # noqa: E402


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer (the container ships no PIL / OpenCV)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=SANDBOX_MODEL)
    parser.add_argument("--png", type=Path, help="render the overview camera to this file and exit")
    parser.add_argument("--camera", default="sandbox_overview")
    parser.add_argument("--seconds", type=float, default=0.0, help="simulate this long before rendering")
    parser.add_argument("--lever", type=float, default=0.0, help="torque pressing the dispenser lever down (Nm, e.g. 0.3)")
    parser.add_argument("--size", type=int, nargs=2, default=(640, 480))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    lever = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dispenser_lever")
    if args.png:
        for _ in range(int(args.seconds / model.opt.timestep)):
            if args.lever and lever >= 0:
                data.qfrc_applied[model.jnt_dofadr[lever]] = args.lever
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=args.size[1], width=args.size[0])
        renderer.update_scene(data, camera=args.camera)
        write_png(args.png, renderer.render())
        renderer.close()
        print(args.png)
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
