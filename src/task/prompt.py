"""
Free-text instruction -> what the demo should do with the two objects.

Rule-based on purpose (deterministic, offline, testable); a VLM can replace
:func:`parse_prompt` later behind the same :class:`TaskIntent`. Understood examples:

- "Sort the two cans: put the lighter one in the left bin."
- "Heavy box goes right." / "Put the full can in the red bin."
- "Give me the heavy can." / "Hand me the empty one, the other goes in the right bin."

Bins are named from the **robot's** point of view: ``left`` = +Y, ``right`` = -Y, also by
color (``red`` = left, ``blue`` = right). With two bins, naming one bin for one weight class
puts the other class in the other bin. No bin named: ``default_light_bin``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class DemoMode(str, Enum):
    """Ending of the demo after both objects are weighed."""

    SORT = "sort"
    HANDOVER = "handover"

LIGHT_WORDS = {"light", "lighter", "lightest", "empty", "emptier", "emptiest"}
HEAVY_WORDS = {"heavy", "heavier", "heaviest", "full", "fuller", "fullest", "filled"}
HANDOVER_PATTERNS = ("hand me", "hand over", "handover", "hand it", "give me", "give it", "pass me", "pass it",
                     "bring me", "bring it", "to me", "offer")
BIN_ALIASES = {"left": "left", "red": "left", "right": "right", "blue": "right"}
OBJECT_WORDS = {"can": "soda can", "cans": "soda can", "box": "cardboard box", "boxes": "cardboard box",
                "bottle": "bottle", "bottles": "bottle", "cup": "cup", "cups": "cup"}


@dataclass(frozen=True)
class TaskIntent:
    """What to do once both objects are weighed.

    Attributes:
        mode: ``SORT`` (both objects into bins) or ``HANDOVER`` (one object to the user,
            the other into its bin).
        bin_for: Bin of each weight class, e.g. ``{"light": "left", "heavy": "right"}``.
        handover_class: Weight class handed to the user (``"heavy"`` unless asked otherwise).
        object_prompt: Text prompt for the open-vocabulary detector.
        text: The original instruction.
    """

    mode: DemoMode
    bin_for: Dict[str, str]
    handover_class: str
    object_prompt: str
    text: str


def _words(text: str) -> List[Tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in re.finditer(r"[a-z]+", text.lower())]


def parse_prompt(text: str, bins: Sequence[str] = ("left", "right"), default_light_bin: str = "left",
                 default_object: str = "soda can") -> TaskIntent:
    """Parses a free-text instruction.

    Args:
        text: The user's instruction.
        bins: The two available bin names.
        default_light_bin: Bin of the lighter object when the text names no bin.
        default_object: Detector prompt when the text names no known object.

    Returns:
        The intent.

    Raises:
        ValueError: If the text sends both weight classes to the same bin or names an unknown bin.
    """
    if len(bins) != 2:
        raise ValueError("the demo sorts into exactly two bins")
    low = text.lower()
    words = _words(low)
    mode = DemoMode.HANDOVER if any(p in low for p in HANDOVER_PATTERNS) else DemoMode.SORT

    classes = [(pos, "light" if w in LIGHT_WORDS else "heavy") for pos, w in words if w in LIGHT_WORDS | HEAVY_WORDS]
    bin_mentions = [(pos, BIN_ALIASES[w]) for pos, w in words if w in BIN_ALIASES]
    for _, b in bin_mentions:
        if b not in bins:
            raise ValueError(f"unknown bin '{b}' (available: {list(bins)})")

    # Pair every bin mention with the closest weight word before it (else the closest after it).
    pairs: Dict[str, str] = {}
    for pos, b in bin_mentions:
        before = [c for c in classes if c[0] < pos]
        cls: Optional[str] = before[-1][1] if before else (min(classes, key=lambda c: abs(c[0] - pos))[1] if classes else None)
        if cls is None:
            continue
        if cls in pairs and pairs[cls] != b:
            raise ValueError(f"the {cls} object is sent to two bins: '{pairs[cls]}' and '{b}'")
        pairs[cls] = b
    if len(set(pairs.values())) < len(pairs):
        raise ValueError("the lighter and the heavier object cannot go to the same bin")
    other = {bins[0]: bins[1], bins[1]: bins[0]}
    if "light" in pairs:
        bin_for = {"light": pairs["light"], "heavy": pairs.get("heavy", other[pairs["light"]])}
    elif "heavy" in pairs:
        bin_for = {"heavy": pairs["heavy"], "light": other[pairs["heavy"]]}
    else:
        bin_for = {"light": default_light_bin, "heavy": other[default_light_bin]}

    handover_class = "heavy"
    if mode is DemoMode.HANDOVER:
        # The weight word closest to the handover phrase decides which object the user gets.
        hpos = min(low.find(p) for p in HANDOVER_PATTERNS if p in low)
        if classes:
            handover_class = min(classes, key=lambda c: abs(c[0] - hpos))[1]
            # "give me the heavy one, the light one goes left": the bin-paired class is not handed over.
            if handover_class in pairs and len(classes) > 1:
                unpaired = [c for c in classes if c[1] not in pairs]
                if unpaired:
                    handover_class = unpaired[0][1]

    obj = next((OBJECT_WORDS[w] for _, w in words if w in OBJECT_WORDS), default_object)
    return TaskIntent(mode=mode, bin_for=bin_for, handover_class=handover_class, object_prompt=obj, text=text)
