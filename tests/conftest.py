"""Shared fixtures. Nothing here needs CAN, a GPU, torch or the model weights (NFR2)."""

from __future__ import annotations

import pytest

from robotokens.config import REPO_ROOT, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def model(cfg):
    if not (REPO_ROOT / cfg.robot.model_path).exists():
        pytest.skip("official OpenArm model missing: run ./setup.sh")
    from robotokens.robot.model import ArmModel

    return ArmModel(cfg)


@pytest.fixture
def make_fake_arm(cfg, model):
    """Builds a started arm on a FakeBus; stops (rest move + disable) after the test."""
    from robotokens.robot import make_arm

    arms = []

    def _make(q0=None, tau_fn=None, **bus_kwargs):
        arm = make_arm(cfg, fake=True, q0=q0, tau_fn=tau_fn, **bus_kwargs)
        arm.start()
        arms.append(arm)
        return arm

    yield _make
    for arm in arms:
        arm.stop(rest=False)
