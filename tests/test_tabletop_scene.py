"""Sanity checks for the OpenArm v1 tabletop scene."""

import os

import numpy as np
import pytest

from src.simulation.tabletop_scene import (
    HEAD_CAMERA,
    OBJECT_NAMES,
    SCENE_CAMERAS,
    SIDES,
    TABLE_HEIGHT,
    TabletopScene,
)


@pytest.fixture(scope="module")
def scene() -> TabletopScene:
    return TabletopScene()


def test_model_structure(scene: TabletopScene) -> None:
    m = scene.model
    assert m.nu == 16  # 2 x (7 arm + 1 gripper)
    assert m.nq == 18 + 7 * len(OBJECT_NAMES)  # 2 x (7 arm + 2 finger) + free joints
    assert sorted(m.camera(i).name for i in range(m.ncam)) == sorted(SCENE_CAMERAS)


def test_reset_has_no_penetration(scene: TabletopScene) -> None:
    scene.reset()
    d = scene.data
    worst = min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
    assert worst > -2e-3, f"initial penetration of {-worst * 1000:.1f} mm"


def test_scene_settles(scene: TabletopScene) -> None:
    scene.reset()
    start = {name: scene.object_pose(name)[0] for name in OBJECT_NAMES}
    tcp_start = {side: scene.tcp_pose(side)[0] for side in SIDES}
    scene.step(2000)  # 2 s
    for name in OBJECT_NAMES:
        pos = scene.object_pose(name)[0]
        assert np.linalg.norm(pos - start[name]) < 5e-3, f"{name} drifted"
        assert pos[2] >= TABLE_HEIGHT - 1e-3, f"{name} fell through the table"
    for side in SIDES:
        assert np.linalg.norm(scene.tcp_pose(side)[0] - tcp_start[side]) < 1e-2, f"{side} arm sagged"


def test_tcp_is_between_fingertips(scene: TabletopScene) -> None:
    scene.reset()
    m, d = scene.model, scene.data
    for side in SIDES:
        tcp = scene.tcp_pose(side)[0]
        fingers = [m.body(f"openarm_{side}_{f}_finger").id for f in ("left", "right")]
        mid = d.xpos[fingers].mean(axis=0)
        # Same lateral position as the finger midpoint, ahead of the finger bases.
        approach = scene.tcp_pose(side)[1][:, 2]
        lateral = (tcp - mid) - approach * np.dot(tcp - mid, approach)
        assert np.linalg.norm(lateral) < 5e-3
        assert 0.0 < np.dot(tcp - mid, approach) < 0.1


def test_gripper_open_close(scene: TabletopScene) -> None:
    scene.reset()
    assert scene.gripper_opening("left") == pytest.approx(0.044, abs=1e-3)
    for side in SIDES:
        scene.set_gripper(side, 0.0)
    scene.step(500)
    for side in SIDES:
        assert scene.gripper_opening(side) < 2e-3
    for side in SIDES:
        scene.set_gripper(side, 1.0)
    scene.step(500)
    for side in SIDES:
        assert scene.gripper_opening(side) > 0.04


def test_object_noise_is_seeded(scene: TabletopScene) -> None:
    scene.reset(seed=3, object_xy_noise=0.02)
    first = scene.object_pose("cube")[0]
    scene.reset(seed=3, object_xy_noise=0.02)
    np.testing.assert_allclose(scene.object_pose("cube")[0], first)
    scene.reset()
    assert not np.allclose(scene.object_pose("cube")[0], first)


@pytest.mark.skipif(not os.environ.get("MUJOCO_GL"), reason="needs an offscreen GL backend (MUJOCO_GL)")
def test_head_camera_renders(scene: TabletopScene) -> None:
    scene.reset()
    img = scene.render(HEAD_CAMERA, width=160, height=120)
    assert img.shape == (120, 160, 3)
    assert img.std() > 10  # not a blank frame
