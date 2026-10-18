"""Free-text instruction -> what to do with the two weighed objects (FR1).

Rule-based on purpose: deterministic, offline, testable. Understood examples:

- "Sort the two cans: put the lighter one in the left bin."
- "Heavy box goes right." / "Put the full can in the red bin."
- "Give me the heavy can." / "Hand me the empty one, the other goes in the right bin."

Bins are named from the robot's point of view (left = +Y), also by colour (red = left,
blue = right). Naming one bin for one weight class puts the other class in the other bin.
"other" / "remaining" refers to the class opposite of the weight word before it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Mode(str, Enum):
    SORT = "sort"
    HANDOVER = "handover"


LIGHT_WORDS = {"light", "lighter", "lightest", "empty", "emptier", "emptiest"}
HEAVY_WORDS = {"heavy", "heavier", "heaviest", "full", "fuller", "fullest", "filled"}
OTHER_WORDS = {"other", "remaining"}
HANDOVER_PHRASES = ("hand me", "hand over", "handover", "hand it", "give me", "give it", "pass me", "pass it",
                    "bring me", "bring it", "to me", "offer")
BIN_ALIASES = {"left": "left", "red": "left", "right": "right", "blue": "right"}
# Prompts tested on the real camera: "soda can" scores ~0.4, "aluminum can" ~0.5, bare "can" finds nothing.
OBJECT_WORDS = {"can": "aluminum can", "cans": "aluminum can", "box": "box", "boxes": "box",
                "bottle": "bottle", "bottles": "bottle", "cup": "cup", "cups": "cup"}
OPPOSITE = {"light": "heavy", "heavy": "light"}
BINS = ("left", "right")


@dataclass(frozen=True)
class TaskIntent:
    mode: Mode
    bin_for: Dict[str, str]  # weight class -> bin, e.g. {"light": "left", "heavy": "right"}
    handover_class: str  # weight class handed to the person (handover mode)
    object_prompt: str  # GroundingDINO text prompt
    text: str


def parse_prompt(text: str, default_object: str = "aluminum can . box", default_light_bin: str = "left") -> TaskIntent:
    """Parses an instruction. Raises ValueError if both classes are sent to the same bin."""
    low = text.lower()
    words = [(m.start(), m.group(0)) for m in re.finditer(r"[a-z]+", low)]
    mode = Mode.HANDOVER if any(p in low for p in HANDOVER_PHRASES) else Mode.SORT

    classes: List[Tuple[int, str]] = []  # (position, weight class)
    for pos, w in words:
        if w in LIGHT_WORDS or w in HEAVY_WORDS:
            classes.append((pos, "light" if w in LIGHT_WORDS else "heavy"))
        elif w in OTHER_WORDS and classes:
            classes.append((pos, OPPOSITE[classes[-1][1]]))
    bins = [(pos, BIN_ALIASES[w]) for pos, w in words if w in BIN_ALIASES]

    # Pair each bin with the closest weight word before it (else the closest one after it).
    pairs: Dict[str, str] = {}
    for pos, b in bins:
        before = [c for c in classes if c[0] < pos]
        cls: Optional[str] = before[-1][1] if before else (
            min(classes, key=lambda c: abs(c[0] - pos))[1] if classes else None)
        if cls is None:
            continue
        if pairs.get(cls, b) != b:
            raise ValueError(f"the {cls} object is sent to two bins: '{pairs[cls]}' and '{b}'")
        pairs[cls] = b
    if len(set(pairs.values())) < len(pairs):
        raise ValueError("the lighter and the heavier object cannot go to the same bin")
    other_bin = {BINS[0]: BINS[1], BINS[1]: BINS[0]}
    if "light" in pairs:
        bin_for = {"light": pairs["light"], "heavy": pairs.get("heavy", other_bin[pairs["light"]])}
    elif "heavy" in pairs:
        bin_for = {"heavy": pairs["heavy"], "light": other_bin[pairs["heavy"]]}
    else:
        bin_for = {"light": default_light_bin, "heavy": other_bin[default_light_bin]}

    handover_class = "heavy"
    if mode is Mode.HANDOVER and classes:
        # The weight word closest to the handover phrase decides; a class paired with a bin is not handed over.
        hpos = min(low.find(p) for p in HANDOVER_PHRASES if p in low)
        handover_class = min(classes, key=lambda c: abs(c[0] - hpos))[1]
        if handover_class in pairs and OPPOSITE[handover_class] not in pairs:
            handover_class = OPPOSITE[handover_class]

    obj = next((OBJECT_WORDS[w] for _, w in words if w in OBJECT_WORDS), default_object)
    return TaskIntent(mode=mode, bin_for=bin_for, handover_class=handover_class, object_prompt=obj, text=text)
