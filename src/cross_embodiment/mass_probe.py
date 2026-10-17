"""Mass from joint torque for the simulated arms (sim-side stand-in for the real estimator).

The real weight-from-torque estimator is not pushed yet (see docs/final_task/branches_and_integration.md).
This module implements the same idea against the simulated plant so the sim robots can run the full
task, and so the method can be compared with the real estimator once that lands.

Idea: holding a mass ``m`` at the TCP adds a static joint torque ``dtau = J_v(q)^T (m g z_hat)``.
Estimate ``m`` by least squares over the joints, using the **difference** between the loaded and the
unloaded torque at the same pose so that model error in the arm's own gravity cancels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class MassEstimate:
    mass_kg: float
    std_kg: float
    residual_nm: float  # norm of the torque change left unexplained by the best-fit mass


def estimate_mass(delta_tau: np.ndarray, jacobian_pos: np.ndarray, torque_std_nm: float) -> MassEstimate:
    """Least-squares mass from a loaded-minus-unloaded torque change.

    Args:
        delta_tau: (n,) torque change between loaded and unloaded at the same pose, Nm.
        jacobian_pos: (3, n) position Jacobian of the grasp point at that pose.
        torque_std_nm: standard deviation of ``delta_tau`` per joint (sensor noise, friction floor).

    Returns:
        ``MassEstimate``; ``std_kg`` is ``torque_std_nm / ||a||`` for the sensitivity vector ``a``.
    """
    sensitivity = GRAVITY * jacobian_pos.T @ np.array([0.0, 0.0, 1.0])
    norm_sq = float(sensitivity @ sensitivity)
    if norm_sq < 1e-9:
        raise ValueError("Grasp point has no vertical sensitivity at this pose; cannot estimate mass.")
    mass = float(sensitivity @ delta_tau / norm_sq)
    residual = float(np.linalg.norm(delta_tau - mass * sensitivity))
    return MassEstimate(mass, torque_std_nm / float(np.sqrt(norm_sq)), residual)


def heavier(first: MassEstimate, second: MassEstimate, sigmas: float = 2.0) -> int:
    """Which of two estimates is heavier: 1, 2, or 0 when they cannot be told apart.

    They are indistinguishable when the difference is below ``sigmas`` combined standard deviations.
    """
    spread = float(np.hypot(first.std_kg, second.std_kg))
    diff = first.mass_kg - second.mass_kg
    if abs(diff) <= sigmas * spread:
        return 0
    return 1 if diff > 0 else 2
