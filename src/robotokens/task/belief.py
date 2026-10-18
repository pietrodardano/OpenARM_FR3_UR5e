"""The robot's belief about each object: where it stands, how heavy it felt, its weight class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..sensing.payload import Comparison, MassEstimate


@dataclass
class ObjectBelief:
    name: str
    x: float  # base frame, m
    y: float
    confidence: float = 1.0  # detector confidence
    mass: Optional[MassEstimate] = None  # set after weighing
    weight_class: Optional[str] = None  # "heavy" / "light", after a conclusive comparison


def assign_classes(beliefs: Sequence[ObjectBelief], comparison: Comparison) -> None:
    """Marks the heavier object "heavy" and the other "light" (conclusive comparisons only)."""
    if not comparison.conclusive:
        raise ValueError("cannot classify on an inconclusive comparison")
    for b in beliefs:
        b.weight_class = "heavy" if b.name == comparison.heavier else "light"
