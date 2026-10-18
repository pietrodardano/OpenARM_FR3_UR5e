"""FR1: free-text instruction -> TaskIntent (rule-based)."""

from __future__ import annotations

import pytest

from robotokens.task.prompt import Mode, parse_prompt


@pytest.mark.parametrize(
    "text, mode, bin_for, handover_class, obj",
    [
        ("Sort the two cans: put the lighter one in the left bin.", Mode.SORT,
         {"light": "left", "heavy": "right"}, "heavy", "aluminum can"),
        ("Heavy box goes right.", Mode.SORT, {"heavy": "right", "light": "left"}, "heavy", "box"),
        ("Put the full can in the red bin.", Mode.SORT, {"heavy": "left", "light": "right"}, "heavy", "aluminum can"),
        ("Give me the heavy can.", Mode.HANDOVER, {"light": "left", "heavy": "right"}, "heavy", "aluminum can"),
        ("Hand me the empty one, the other goes in the right bin.", Mode.HANDOVER,
         {"light": "left", "heavy": "right"}, "light", "aluminum can . box"),
        ("Sort them.", Mode.SORT, {"light": "left", "heavy": "right"}, "heavy", "aluminum can . box"),
    ],
)
def test_parse(text, mode, bin_for, handover_class, obj):
    intent = parse_prompt(text)
    assert intent.mode is mode
    assert intent.bin_for == bin_for
    assert intent.handover_class == handover_class
    assert intent.object_prompt == obj


def test_default_object_used_when_none_named():
    assert parse_prompt("Sort them.", default_object="bottle").object_prompt == "bottle"


def test_same_bin_for_both_rejected():
    with pytest.raises(ValueError):
        parse_prompt("light one left, heavy one left")


def test_no_object_named_searches_cans_and_boxes(cfg):
    assert parse_prompt("Sort them.", default_object=cfg.task.object_prompt).object_prompt == "aluminum can . box"
    assert parse_prompt("Sort the boxes.", default_object=cfg.task.object_prompt).object_prompt == "box"
