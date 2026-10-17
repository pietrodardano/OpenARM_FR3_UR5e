"""
Comparative payload sensing from the arm's joint torques (no force sensor).

The Damiao motors report output torque from their phase current. Holding an object
at the TCP adds ``J_z(q) * m * g`` to the joint torques, where ``J_z`` is the vertical
row of the TCP position Jacobian. At the elbow-90 pose (forearm forward) ``J_z`` is
about [-0.41, 0, 0, 0.41, 0, 0, -0.19] m, so 300 g adds ~1.2 Nm on joints 1 and 4 and
~0.56 Nm on joint 7.

Absolute mass from the motor torques is unreliable (model mass errors, cable drag,
current offsets, gearbox friction are a few tenths of Nm). Comparing two objects
measured the same way at the same pose cancels most of that, which is all a
"which box is heavier" decision needs:

- :func:`measure_load` dithers the load-bearing joints with a small sine
  (+-``amplitude_deg``) and averages the torques over whole cycles: Coulomb friction
  flips sign with the motion and averages out, so the stick-slip ambiguity of a
  static hold does not decide the result.
- The model's arm gravity at the measured pose is subtracted first, so small pose
  differences between the two measurements do not show up as a weight difference.
- :func:`compare` returns the heavier sample, the equivalent mass difference and a
  confidence ``z`` (difference / combined standard error of the per-cycle values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import SIDES, RobotBackend, Side, arm_slice, minimum_jerk
from ..planning.openarm_kinematics import OpenArmKinematics

G = 9.81
MIN_LEVER_M = 0.05  # joints whose |J_z| is below this see too little of the payload


@dataclass
class LoadSample:
    """One load measurement of one arm."""

    label: str
    side: Side
    q: NDArray[np.float64]  # (7,) mean measured joints, rad
    tau: NDArray[np.float64]  # (7,) mean measured torques, Nm
    tau_model: NDArray[np.float64]  # (7,) model arm gravity (no payload) at q, Nm
    jz: NDArray[np.float64]  # (7,) vertical TCP Jacobian row at q, m
    joints: List[int]  # 0-based joints used for the estimate
    cycle_mass: NDArray[np.float64]  # (n_cycles,) equivalent mass of each dither cycle, kg
    gripper_m: float = 0.0  # mean finger opening during the measurement
    extra: dict = field(default_factory=dict)

    @property
    def mass(self) -> float:
        """Equivalent payload mass (kg) relative to the model: rough as an absolute value."""
        return float(np.mean(self.cycle_mass))

    @property
    def mass_se(self) -> float:
        """Standard error of :attr:`mass` from the cycle-to-cycle spread (kg)."""
        n = len(self.cycle_mass)
        return float(np.std(self.cycle_mass, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    def per_joint_mass(self) -> dict:
        """Equivalent mass seen by each used joint alone (kg): a consistency check."""
        res = self.tau - self.tau_model
        return {j + 1: float(res[j] / (G * self.jz[j])) for j in self.joints}


def equivalent_mass(tau_residual: NDArray[np.float64], jz: NDArray[np.float64], joints: Sequence[int]) -> float:
    """Least-squares payload mass (kg) for a torque residual: tau_res ~ J_z * m * g."""
    j = list(joints)
    return float(np.dot(jz[j], tau_residual[j]) / (G * np.dot(jz[j], jz[j])))


def tcp_jz(kin: OpenArmKinematics, side: Side, q_arms: dict) -> NDArray[np.float64]:
    """Vertical row of the TCP position Jacobian of ``side`` (7,), in m/rad."""
    kin.fk(q_arms)
    mujoco.mj_forward(kin.model, kin.data)
    jp = np.zeros((3, kin.model.nv))
    mujoco.mj_jacSite(kin.model, kin.data, jp, None, kin.site_id[side])
    return jp[2, kin.dof_adr[side]].copy()


def measure_load(
    robot: RobotBackend,
    kin: OpenArmKinematics,
    side: Side,
    label: str = "",
    joints: Optional[Sequence[int]] = None,
    amplitude_deg: float = 1.0,
    period_s: float = 2.0,
    cycles: int = 3,
    rate_hz: float = 100.0,
) -> LoadSample:
    """Measures the TCP load of ``side`` at its current pose (blocking, ``(cycles + 2) * period_s``).

    The arm dithers the used joints around the current targets: one ramp-in cycle,
    ``cycles`` measured cycles, one ramp-out cycle, then it is back on the exact start
    targets. ``joints`` (0-based) defaults to every joint with |J_z| > 5 cm.
    """
    base = robot.get_joint_targets()[arm_slice(side)].copy()
    st = robot.get_state()
    jz0 = tcp_jz(kin, side, {s: st.arms[s].q for s in SIDES})
    used = list(joints) if joints is not None else [j for j in range(7) if abs(jz0[j]) > MIN_LEVER_M]
    if not used:
        raise ValueError(f"no joint has a lever arm > {MIN_LEVER_M} m on the TCP load at this pose")

    amp = np.zeros(7)
    amp[used] = np.radians(amplitude_deg)
    n_per = int(round(period_s * rate_hz))
    total = (cycles + 2) * n_per
    taus, qs, grips = [], [], []
    try:
        for k in range(1, total + 1):
            cyc = k // n_per
            env = minimum_jerk(k / n_per) if cyc == 0 else (minimum_jerk((total - k) / n_per) if cyc >= cycles + 1 else 1.0)
            robot.set_joint_targets({side: base + env * amp * np.sin(2 * np.pi * k / n_per)})
            robot.advance(1.0 / rate_hz)
            st = robot.get_state()
            if st.fault:
                raise RuntimeError(f"load measurement aborted: {st.fault}")
            if n_per < k <= (cycles + 1) * n_per:  # the measured cycles
                taus.append(st.arms[side].tau.copy())
                qs.append(st.arms[side].q.copy())
                grips.append(st.arms[side].gripper)
    finally:
        robot.set_joint_targets({side: base})

    tau = np.asarray(taus).reshape(cycles, n_per, 7)
    q = np.asarray(qs).reshape(cycles, n_per, 7)
    q_mean = q.mean(axis=(0, 1))
    q_arms = {s: st.arms[s].q for s in SIDES}
    q_arms[side] = q_mean
    jz = tcp_jz(kin, side, q_arms)
    tau_model = kin.gravity_torques(q_arms)[side]
    cycle_mass = np.array([equivalent_mass(tau[c].mean(axis=0) - tau_model, jz, used) for c in range(cycles)])
    return LoadSample(label=label, side=side, q=q_mean, tau=tau.mean(axis=(0, 1)), tau_model=tau_model, jz=jz,
                      joints=used, cycle_mass=cycle_mass, gripper_m=float(np.mean(grips)))


@dataclass
class Comparison:
    heavier: LoadSample
    lighter: LoadSample
    diff_kg: float  # equivalent mass difference, >= 0
    z: float  # diff / combined standard error: > 3 is a clear call
    pose_diff_deg: float  # largest joint difference between the two measurement poses


def compare(a: LoadSample, b: LoadSample) -> Comparison:
    """Which of two load samples is heavier, by how much, and how confidently."""
    heavier, lighter = (a, b) if a.mass >= b.mass else (b, a)
    diff = heavier.mass - lighter.mass
    se = float(np.hypot(a.mass_se, b.mass_se))
    z = diff / se if se > 0 else (float("inf") if diff > 0 else 0.0)
    pose = float(np.degrees(np.max(np.abs(a.q - b.q))))
    return Comparison(heavier=heavier, lighter=lighter, diff_kg=diff, z=z, pose_diff_deg=pose)
