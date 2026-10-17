"""Scripted bimanual pick-and-place demo, recorded to MP4.

Observe once, then: right arm puts the cube on the plate, left arm puts the
apple on the plate, both arms return home. Head camera and a debug view are
recorded side by side; robot contacts are logged and placements checked.

    python scripts/run_pick_place.py --out outputs/pick_place.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.simulation.tabletop_scene import HEAD_CAMERA, OBJECT_NAMES, Side, TabletopScene  # noqa: E402
from src.skills.pick_place import SkillRunner  # noqa: E402

PLATE_RADIUS = 0.09
PLATE_THICKNESS = 0.008

# object -> (arm, placement offset from the plate centre)
TASK: Dict[str, Tuple[Side, Tuple[float, float]]] = {
    "cube": ("right", (0.0, -0.04)),
    "apple": ("left", (0.0, 0.045)),
}


def run_task(runner: SkillRunner) -> None:
    runner.hold(0.5)
    objects = runner.perceive()
    plate = objects["plate"]
    surface = plate[2] + PLATE_THICKNESS
    for name, (side, offset) in TASK.items():
        print(f"{side} arm: pick {name}")
        if not runner.pick(side, name):
            print(f"  grasp of {name} failed")
            continue
        print(f"{side} arm: place {name} on plate")
        runner.place(side, plate[:2] + np.array(offset), surface)
        runner.go_home((side,), duration=1.5)  # clear the shared workspace for the other arm
    runner.hold(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/pick_place.mp4"))
    parser.add_argument("--side-camera", default="three_quarter")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scene = TabletopScene()
    m, d = scene.model, scene.data

    # Label bodies; log every contact between an arm and anything with a different label.
    label = {m.body(name).id: name for name in (*OBJECT_NAMES, "table")}
    for i in range(m.nbody):
        name = m.body(i).name
        if name == "openarm_body_link0":
            label[i] = "pedestal"
        elif name.startswith("openarm_left_"):
            label[i] = "left arm"
        elif name.startswith("openarm_right_"):
            label[i] = "right arm"
    contacts: Counter = Counter()

    renderers = [mujoco.Renderer(m, args.height, args.width) for _ in range(2)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    next_frame = [0.0]

    def on_tick(_: TabletopScene) -> None:
        touching = set()
        for c in d.contact[: d.ncon]:
            l1, l2 = label.get(m.geom_bodyid[c.geom1]), label.get(m.geom_bodyid[c.geom2])
            if l1 and l2 and l1 != l2 and ("arm" in l1 or "arm" in l2):
                touching.add(tuple(sorted((l1, l2))))
        contacts.update(touching)
        if d.time >= next_frame[0]:
            next_frame[0] += 1.0 / args.fps
            frames = []
            for renderer, camera in zip(renderers, (HEAD_CAMERA, args.side_camera)):
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())
            writer.append_data(np.hstack(frames))

    try:
        run_task(SkillRunner(scene, on_tick=on_tick))
    finally:
        writer.close()
        for r in renderers:
            r.close()

    print(f"\nwrote {args.out} ({d.time:.1f} s simulated)")
    print("robot contacts: control ticks in contact")
    for key, n in sorted(contacts.items()):
        print(f"  {key}: {n}")
    plate = scene.object_pose("plate")[0]
    for name in TASK:
        pos = scene.object_pose(name)[0]
        r = np.linalg.norm(pos[:2] - plate[:2])
        on_plate = r < PLATE_RADIUS and pos[2] > plate[2] + PLATE_THICKNESS
        print(f"{name}: {np.round(pos, 3)}, {r * 100:.1f} cm from plate centre -> {'ON PLATE' if on_plate else 'NOT on plate'}")


if __name__ == "__main__":
    main()
