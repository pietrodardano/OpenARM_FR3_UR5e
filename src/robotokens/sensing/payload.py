"""Payload mass from the arm's joint torques (no force sensor), and the comparison rule (FR3, FR4).

Holding a mass ``m`` at the TCP adds ``J_z(q) * m * g`` to the joint torques (``J_z``: vertical
row of the TCP position Jacobian). At the weigh pose that is ~1 Nm per 300 g on the shoulder
and elbow. :func:`measure_load`

- dithers the load-bearing joints (|J_z| > 5 cm) with a small sine and averages the torques over
  whole cycles: Coulomb friction flips sign with the motion and cancels;
- subtracts the model's arm gravity at the measured mean pose;
- fits ``m`` by least squares per cycle; the cycle-to-cycle spread gives the standard error.

The absolute value is rough (model mass errors, cable drag, current offsets). Both objects are
measured the same way at the same pose, so the *difference* is reliable; :func:`compare` acts on
``z = diff / combined SE``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import WeighConfig
from ..robot.driver import Arm
from ..robot.model import ArmModel
from ..robot.safety import minimum_jerk

logger = logging.getLogger(__name__)

G = 9.81
MIN_LEVER_M = 0.05  # joints with |J_z| below this see too little of the payload
RATE_HZ = 100.0


@dataclass
class MassEstimate:
    label: str
    mass_kg: float
    se_kg: float  # standard error; NaN = unknown
    cycle_mass: List[float] = field(default_factory=list)
    joints: List[int] = field(default_factory=list)  # 0-based joints used


@dataclass(frozen=True)
class Comparison:
    heavier: str
    lighter: str
    diff_kg: float  # >= 0
    z: float  # diff / combined SE
    conclusive: bool


def equivalent_mass(tau_residual: NDArray, jz: NDArray, joints: Sequence[int]) -> float:
    """Least-squares mass (kg) for a torque residual: tau_res ~ J_z m g."""
    j = list(joints)
    return float(np.dot(jz[j], tau_residual[j]) / (G * np.dot(jz[j], jz[j])))


def measure_load(arm: Arm, model: ArmModel, w: WeighConfig, label: str) -> MassEstimate:
    """Settles, then measures the TCP load at the current targets (blocking, ~(cycles + 2) periods).

    One ramp-in cycle, ``dither_cycles`` measured cycles, one ramp-out cycle; the arm ends on
    the exact start targets.
    """
    arm.advance(w.settle_s)
    base = arm.targets()[0]
    jz0 = model.jz(arm.state().q)
    used = [j for j in range(7) if abs(jz0[j]) > MIN_LEVER_M]
    if not used:
        raise ValueError(f"no joint has a lever arm > {MIN_LEVER_M} m on the TCP load at this pose")
    amp = np.zeros(7)
    amp[used] = np.radians(w.dither_deg)
    n_per = int(round(w.dither_period_s * RATE_HZ))
    cycles = w.dither_cycles
    total = (cycles + 2) * n_per
    taus, qs = [], []
    try:
        for k in range(1, total + 1):
            cyc = k // n_per
            env = minimum_jerk(k / n_per) if cyc == 0 else (minimum_jerk((total - k) / n_per) if cyc >= cycles + 1 else 1.0)
            arm.set_targets(q=base + env * amp * np.sin(2 * np.pi * k / n_per))
            arm.advance(1.0 / RATE_HZ)
            st = arm.state()
            if st.fault:
                raise RuntimeError(f"load measurement aborted: {st.fault}")
            if n_per < k <= (cycles + 1) * n_per:
                taus.append(st.tau.copy())
                qs.append(st.q.copy())
    finally:
        arm.set_targets(q=base)
    tau = np.asarray(taus).reshape(cycles, n_per, 7)
    q_mean = np.asarray(qs).mean(axis=0)
    jz, tau_model = model.jz(q_mean), model.gravity(q_mean)
    cycle_mass = [equivalent_mass(tau[c].mean(axis=0) - tau_model, jz, used) for c in range(cycles)]
    se = float(np.std(cycle_mass, ddof=1) / np.sqrt(cycles)) if cycles > 1 else float("nan")
    est = MassEstimate(label=label, mass_kg=float(np.mean(cycle_mass)), se_kg=se, cycle_mass=cycle_mass,
                       joints=used)
    logger.info("%s: %.0f +- %.0f g (joints %s)", label, est.mass_kg * 1e3, est.se_kg * 1e3, [j + 1 for j in used])
    return est


def compare(a: MassEstimate, b: MassEstimate, min_z: float) -> Comparison:
    """Heavier of two estimates; conclusive when ``z >= min_z`` (NaN SE: never conclusive)."""
    heavier, lighter = (a, b) if a.mass_kg >= b.mass_kg else (b, a)
    diff = heavier.mass_kg - lighter.mass_kg
    se = math.hypot(a.se_kg, b.se_kg)
    z = float("nan") if math.isnan(se) else (diff / se if se > 0 else (math.inf if diff > 0 else 0.0))
    return Comparison(heavier=heavier.label, lighter=lighter.label, diff_kg=diff, z=z, conclusive=bool(z >= min_z))


def combine(a: MassEstimate, b: MassEstimate) -> MassEstimate:
    """Inverse-variance mean of two measurements of the same object."""
    if a.se_kg > 0 and b.se_kg > 0:
        wa, wb = 1 / a.se_kg**2, 1 / b.se_kg**2
        mass, se = (wa * a.mass_kg + wb * b.mass_kg) / (wa + wb), 1 / math.sqrt(wa + wb)
    else:
        mass, se = 0.5 * (a.mass_kg + b.mass_kg), 0.5 * math.hypot(a.se_kg, b.se_kg)
    return MassEstimate(label=a.label, mass_kg=mass, se_kg=se, cycle_mass=a.cycle_mass + b.cycle_mass, joints=a.joints)
