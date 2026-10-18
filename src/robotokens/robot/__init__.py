"""One OpenArm v1 arm: CAN bus, 250 Hz driver, official model, IK, checked motion, gripper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from .bus import CanBus, FakeBus, TauFn
from .driver import Arm, ArmState

__all__ = ["Arm", "ArmState", "make_arm"]


def make_arm(cfg: Config, fake: bool = False, q0: Optional[NDArray] = None, tau_fn: Optional[TauFn] = None,
             silent: bool = False) -> Arm:
    """The configured arm on CAN, or on a :class:`FakeBus` (``fake``; ``q0`` defaults to rest).

    Limits and the gravity feedforward come from a dedicated :class:`ArmModel` (the servo thread
    calls it).
    """
    from .model import ArmModel

    model = ArmModel(cfg)
    if fake:
        start = np.asarray(cfg.robot.rest_qpos if q0 is None else q0, dtype=np.float64)
        bus = FakeBus(q0=start, dt=1.0 / cfg.hardware.control_rate_hz, tau_fn=tau_fn,
                      motor_open_rad=cfg.gripper.motor_open_rad, silent=silent)
    else:
        bus = CanBus(cfg.hardware, cfg.gripper)
    return Arm(cfg, bus, model.lower, model.upper, gravity_fn=model.gravity)
