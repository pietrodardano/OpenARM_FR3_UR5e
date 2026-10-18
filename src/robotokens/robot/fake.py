"""Joint torques for ``--fake`` runs: model gravity + the payload of a held object + noise.

The FakeBus stops the closing fingers at ``fake.object_width_m``. While they are stopped there,
the plant adds ``J_z m g`` of the object nearest to the TCP when the grasp started (masses from
``fake.masses_kg`` in the order of ``task.manual_objects``). Nobody takes an offered object, so
a fake handover exercises the timeout path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..sensing.payload import G
from .model import ArmModel


class FakePlant:
    def __init__(self, cfg: Config, block_at_motor_rad: float, seed: int = 0) -> None:
        self.model = ArmModel(cfg)  # own instance: called from the servo thread
        self.objects = np.array([[o.x, o.y] for o in cfg.task.manual_objects])
        self.masses = list(cfg.fake.masses_kg)
        self.noise = cfg.fake.torque_noise_nm
        self.block_at = block_at_motor_rad
        self.rng = np.random.default_rng(seed)
        self.held: Optional[int] = None

    def __call__(self, q: NDArray, dq: NDArray, gripper_q: float) -> NDArray:
        holding = abs(gripper_q - self.block_at) < 0.01
        if holding and self.held is None:
            p, _ = self.model.fk(q)
            self.held = int(np.argmin(np.linalg.norm(self.objects - p[:2], axis=1)))
        elif not holding:
            self.held = None
        tau = self.model.gravity(q) + self.rng.normal(0.0, self.noise, 7)
        if self.held is not None:
            tau += self.model.jz(q) * self.masses[self.held] * G
        return tau
