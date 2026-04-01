from __future__ import annotations

import numpy as np

from ftd.differ import compute_diff_regions, compute_solution_overlay


def test_compute_diff_regions_finds_single_change() -> None:
    original = np.full((100, 100, 3), 255, dtype=np.uint8)
    modified = original.copy()
    modified[30:50, 40:60] = [255, 0, 0]

    regions = compute_diff_regions(original, modified)
    assert len(regions) == 1
    x, y, w, h = regions[0].bbox
    assert x <= 40 <= x + w
    assert y <= 30 <= y + h


def test_compute_solution_overlay_draws_border() -> None:
    original = np.full((100, 100, 3), 255, dtype=np.uint8)
    modified = original.copy()
    modified[30:50, 40:60] = [255, 0, 0]

    overlay = compute_solution_overlay(original, modified)
    assert np.any(np.all(overlay == [255, 0, 0], axis=2))
