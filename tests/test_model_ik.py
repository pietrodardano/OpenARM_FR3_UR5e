"""Official-model kinematics, gravity, workcell checks and IK (SR4 building blocks)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest


def test_limits_from_official_model(model):
    assert model.lower[3] == pytest.approx(0.0) and model.upper[3] == pytest.approx(2.443461)
    np.testing.assert_allclose(model.home, [0, 0, 0, 1.570796, 0, 0, 0])


def test_fk_matches_mujoco_site(model):
    q = np.array([0.1, -0.3, 0.2, 1.2, 0.1, -0.2, 0.3])
    p, R = model.fk(q)
    d = mujoco.MjData(model.model)
    d.qpos[model.qadr] = q
    mujoco.mj_kinematics(model.model, d)
    sid = model.model.site(f"{model.side}_tcp").id
    np.testing.assert_allclose(p, d.site_xpos[sid], atol=1e-12)
    np.testing.assert_allclose(R, d.site_xmat[sid].reshape(3, 3), atol=1e-12)


def test_home_tcp_is_forward_over_the_table(model, cfg):
    p, _ = model.fk(model.home)
    assert p[0] > 0.3 and p[1] > 0 and p[2] > cfg.table_z  # left arm, in front, above the table


def test_gravity_matches_bias_forces(model):
    q = model.home
    g = model.gravity(q)
    assert g.shape == (7,) and np.abs(g).max() > 0.5  # elbow at 90 deg carries the forearm


def test_jz_is_vertical_jacobian_row(model):
    q = model.home
    eps = 1e-6
    p0, _ = model.fk(q)
    jz_num = np.array([(model.fk(q + eps * np.eye(7)[j])[0][2] - p0[2]) / eps for j in range(7)])
    np.testing.assert_allclose(model.jz(q), jz_num, atol=1e-5)


def test_blocked_detects_table_and_allows_home(model, cfg):
    assert model.blocked(model.home) is None
    q_low = np.array([-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # straight arm, hand on the tabletop
    assert model.fk(q_low)[0][2] < cfg.table_z + cfg.motion.table_margin_m
    assert "table guard" in model.blocked(q_low)


def test_blocked_detects_bin_collision(model):
    hit = model.blocked(np.array([-0.5, 0, 0, 0, 0, 0, 0]))  # straight arm pitched onto the left bin
    assert hit is not None and "bin_left" in hit


def test_ik_reaches_reachable_target(model):
    from robotokens.robot.ik import IK

    ik = IK(model)
    q_goal = np.array([0.0, -0.2, 0.1, 1.3, 0.0, 0.3, 0.0])
    p, R = model.fk(q_goal)
    res = ik.solve(p, R, model.home, max_iters=200)
    assert res.pos_error < 0.005 and res.rot_error < 0.035
    np.testing.assert_allclose(model.fk(res.q)[0], p, atol=0.005)


def test_ik_reports_unreachable_target(model):
    from robotokens.robot.ik import IK

    _, R = model.fk(model.home)
    res = IK(model).solve(np.array([2.0, 0.2, 0.5]), R, model.home, max_iters=50)
    assert res.pos_error > 0.5
