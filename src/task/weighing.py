"""
Comparison of two torque-based mass estimates.

Same decision rule as ``compare()`` in ``src/estimation/payload.py``:
the heavier object is decided by the mass difference relative to the combined
standard error, ``z = diff / sqrt(se_a^2 + se_b^2)``. ``z >= 3`` is a clear call.
"""

from __future__ import annotations

import math

from .interfaces import ComparisonResult, MassEstimate


def compare_estimates(a: MassEstimate, b: MassEstimate, min_z: float) -> ComparisonResult:
    """Returns which estimate is heavier and whether the call is conclusive.

    Args:
        a: First estimate.
        b: Second estimate.
        min_z: Minimum ``z`` for a conclusive decision.

    Returns:
        The comparison. A NaN standard error makes it inconclusive.
    """
    heavier, lighter = (a, b) if a.mass_kg >= b.mass_kg else (b, a)
    diff = heavier.mass_kg - lighter.mass_kg
    se = math.hypot(a.mass_se_kg, b.mass_se_kg)
    if math.isnan(se):
        z = float("nan")
    elif se > 0.0:
        z = diff / se
    else:
        z = float("inf") if diff > 0.0 else 0.0
    return ComparisonResult(heavier=heavier, lighter=lighter, diff_kg=diff, z=z, conclusive=bool(z >= min_z))


def combine_estimates(a: MassEstimate, b: MassEstimate) -> MassEstimate:
    """Inverse-variance weighted mean of two measurements of the same object.

    Args:
        a: First measurement.
        b: Second measurement (same object, same pose).

    Returns:
        The combined estimate, labelled like ``a``, with both raw samples in ``raw``.
    """
    if a.mass_se_kg > 0.0 and b.mass_se_kg > 0.0:
        wa, wb = 1.0 / a.mass_se_kg**2, 1.0 / b.mass_se_kg**2
        mass = (wa * a.mass_kg + wb * b.mass_kg) / (wa + wb)
        se = 1.0 / math.sqrt(wa + wb)
    else:
        mass = 0.5 * (a.mass_kg + b.mass_kg)
        se = 0.5 * math.hypot(a.mass_se_kg, b.mass_se_kg)
    return MassEstimate(label=a.label, mass_kg=mass, mass_se_kg=se, raw=[a.raw, b.raw])
