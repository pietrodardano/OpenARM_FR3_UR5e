"""View or render the tabletop scene.

Interactive viewer (needs a display):
    python scripts/view_scene.py

Headless: let the scene settle, then save one PNG per camera:
    python scripts/view_scene.py --render outputs/scene
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v3 as iio  # noqa: E402
import mujoco.viewer  # noqa: E402

from src.simulation.tabletop_scene import SCENE_CAMERAS, TabletopScene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", type=Path, metavar="DIR", help="save PNGs here instead of opening the viewer")
    parser.add_argument("--cameras", nargs="+", default=list(SCENE_CAMERAS), choices=SCENE_CAMERAS)
    parser.add_argument("--settle", type=float, default=1.0, help="seconds to simulate before rendering")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    scene = TabletopScene()

    if args.render is None:
        mujoco.viewer.launch(scene.model, scene.data)
        return

    scene.step(int(args.settle / scene.model.opt.timestep))
    args.render.mkdir(parents=True, exist_ok=True)
    for camera in args.cameras:
        path = args.render / f"{camera}.png"
        iio.imwrite(path, scene.render(camera, args.width, args.height))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
