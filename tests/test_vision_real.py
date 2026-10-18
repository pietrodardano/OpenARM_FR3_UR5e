"""GroundingDINO + SAM2 on a saved camera frame (run with: uv run pytest -m vision)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotokens.config import REPO_ROOT

pytestmark = pytest.mark.vision
FRAMES = sorted((REPO_ROOT / "tests" / "data").glob("cans_*.png"))


@pytest.mark.skipif(not FRAMES, reason="no saved frames in tests/data/cans_*.png (save one with `robotokens detect --save`)")
@pytest.mark.parametrize("path", FRAMES, ids=lambda p: Path(p).name)
def test_localizer_finds_two_cans(cfg, path):
    import cv2

    from robotokens.vision.localizer import Localizer

    loc = Localizer(cfg.vision)
    hits = loc.locate(cv2.imread(str(path)), "soda can")
    assert len([h for h in hits if h.score >= cfg.task.min_confidence]) == 2
    assert all(h.centroid is not None for h in hits)
