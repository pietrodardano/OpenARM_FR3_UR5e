"""FR3 / FR4: torque-based mass estimate and the comparison rule."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from robotokens.sensing.payload import G, MassEstimate, combine, compare, equivalent_mass, measure_load


def test_equivalent_mass_inverts_jz_m_g():
    jz = np.array([-0.4, 0.0, 0.0, 0.4, 0.0, 0.0, -0.2])
    tau = jz * 0.3 * G
    assert equivalent_mass(tau, jz, [0, 3, 6]) == pytest.approx(0.3)


@pytest.mark.parametrize("mass", [0.06, 0.36])
def test_measure_load_recovers_mass_with_friction_and_noise(make_fake_arm, model, cfg, mass):
    from robotokens.robot.model import ArmModel

    rng = np.random.default_rng(0)
    plant = ArmModel(cfg)  # own instance: tau_fn runs in the servo thread

    def tau_fn(q, dq, _gripper_q):
        friction = 0.15 * np.sign(dq)  # Coulomb friction: cancelled by the dither
        return plant.gravity(q) + plant.jz(q) * mass * G + friction + rng.normal(0, 0.02, 7)

    arm = make_fake_arm(q0=model.home, tau_fn=tau_fn)
    weigh = dataclasses.replace(cfg.task.weigh, settle_s=0.1, dither_period_s=0.4)
    est = measure_load(arm, model, weigh, "obj")
    assert est.mass_kg == pytest.approx(mass, abs=0.01)
    assert 0 < est.se_kg < 0.01
    np.testing.assert_allclose(arm.targets()[0], model.home, atol=1e-9)  # back on the start targets


def _est(label, m, se):
    return MassEstimate(label=label, mass_kg=m, se_kg=se)


def test_compare_conclusive_and_inconclusive():
    c = compare(_est("a", 0.36, 0.01), _est("b", 0.06, 0.01), min_z=3.0)
    assert (c.heavier, c.lighter, c.conclusive) == ("a", "b", True)
    assert c.z == pytest.approx(0.30 / math.hypot(0.01, 0.01))
    c = compare(_est("a", 0.10, 0.05), _est("b", 0.12, 0.05), min_z=3.0)
    assert c.heavier == "b" and not c.conclusive


def test_compare_nan_se_is_inconclusive():
    assert not compare(_est("a", 0.3, float("nan")), _est("b", 0.0, 0.01), 3.0).conclusive


def test_combine_is_inverse_variance_mean():
    c = combine(_est("a", 0.10, 0.01), _est("a", 0.20, 0.02))
    w1, w2 = 1 / 0.01**2, 1 / 0.02**2
    assert c.mass_kg == pytest.approx((w1 * 0.10 + w2 * 0.20) / (w1 + w2))
    assert c.se_kg == pytest.approx(1 / math.sqrt(w1 + w2))
