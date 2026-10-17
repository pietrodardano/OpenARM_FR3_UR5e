"""Sandbox scene: builds, stays calm, nothing overlaps the arms at home, the dispenser lever works."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if not (ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml").is_file():
    pytest.skip("official OpenArm sources not cloned", allow_module_level=True)

from setup_sandbox_scene import DEFAULT_TABLE_TOP_M, ELBOW_HEIGHT_M, PROPS, build_sandbox  # noqa: E402

TABLE_TOP = DEFAULT_TABLE_TOP_M


@pytest.fixture(scope="module")
def scene():
    out = ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_sandbox_test.xml"
    build_sandbox(list(PROPS), TABLE_TOP, out)
    model = mujoco.MjModel.from_xml_path(str(out))
    yield model
    out.unlink(missing_ok=True)


def fresh(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return data


def test_table_is_just_below_the_elbows():
    assert ELBOW_HEIGHT_M - 0.15 < TABLE_TOP < ELBOW_HEIGHT_M


def test_home_pose_has_no_robot_contact_with_props(scene):
    data = fresh(scene)
    mujoco.mj_forward(scene, data)
    robot = {b for b in range(scene.nbody) if scene.body(b).name.startswith("openarm_")}
    bad = [(scene.geom(c.geom1).name, scene.geom(c.geom2).name) for c in data.contact[: data.ncon]
           if (scene.geom_bodyid[c.geom1] in robot) != (scene.geom_bodyid[c.geom2] in robot) and c.dist < 0]
    assert not bad, bad


def test_props_rest_on_the_table(scene):
    data = fresh(scene)
    start = data.qpos.copy()
    for _ in range(2000):
        mujoco.mj_step(scene, data)
    assert np.all(np.isfinite(data.qpos))
    for j in range(scene.njnt):
        if scene.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = scene.jnt_qposadr[j]
            assert data.qpos[a + 2] > TABLE_TOP - 0.01, scene.joint(j).name
            assert np.linalg.norm(data.qpos[a:a + 2] - start[a:a + 2]) < 0.01, scene.joint(j).name  # did not slide


def test_cup_is_under_the_nozzle(scene):
    data = fresh(scene)
    mujoco.mj_forward(scene, data)
    nozzle = data.site("dispenser_nozzle").xpos
    cup = data.body("cup_tumbler").xpos
    assert np.linalg.norm(nozzle[:2] - cup[:2]) < 0.005
    assert nozzle[2] - cup[2] > 0.10 + 0.015  # tumbler (10 cm) fits under with a gap


def test_pushing_the_lever_engages_and_the_spring_returns_it(scene):
    data = fresh(scene)
    jid = scene.joint("dispenser_lever").id
    dof, adr = scene.jnt_dofadr[jid], scene.jnt_qposadr[jid]
    for _ in range(1000):
        data.qfrc_applied[dof] = 0.3  # ~ a fingertip pressing the tip down with ~6 N
        mujoco.mj_step(scene, data)
    assert data.qpos[adr] > 0.15  # "dispensing"
    data.qfrc_applied[dof] = 0.0
    for _ in range(1500):
        mujoco.mj_step(scene, data)
    assert data.qpos[adr] < 0.03  # back to rest (15 deg up)
