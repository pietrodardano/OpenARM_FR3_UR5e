"""End-to-end check of the scripted pick-and-place skills (no rendering)."""

import numpy as np

from src.simulation.tabletop_scene import TabletopScene
from src.skills.pick_place import SkillRunner

PLATE_RADIUS = 0.09
PLATE_THICKNESS = 0.008


def test_cube_and_apple_onto_plate() -> None:
    scene = TabletopScene()
    m, d = scene.model, scene.data
    table = m.body("table").id
    robot = {i for i in range(m.nbody) if m.body(i).name.startswith("openarm_")}
    table_hits = []

    def on_tick(_: TabletopScene) -> None:
        for c in d.contact[: d.ncon]:
            bodies = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
            if table in bodies and bodies & robot:
                table_hits.append(d.time)

    runner = SkillRunner(scene, on_tick=on_tick)
    objects = runner.perceive()
    plate = objects["plate"]
    surface = plate[2] + PLATE_THICKNESS
    for name, side, offset in (("cube", "right", (0.0, -0.04)), ("apple", "left", (0.0, 0.045))):
        assert runner.pick(side, name), f"{side} arm failed to grasp {name}"
        runner.place(side, plate[:2] + np.array(offset), surface)
        runner.go_home((side,), duration=1.5)
    runner.hold(0.5)

    assert not table_hits, "robot touched the table"
    for name in ("cube", "apple"):
        pos = scene.object_pose(name)[0]
        assert np.linalg.norm(pos[:2] - plate[:2]) < PLATE_RADIUS, f"{name} missed the plate"
        assert pos[2] > surface, f"{name} is not resting on the plate"
