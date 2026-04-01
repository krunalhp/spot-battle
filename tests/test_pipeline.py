from __future__ import annotations

import numpy as np

from ftd.detector import DetectedObject
from ftd.difficulty import DifficultyLevel
from ftd.modifier import ChangeType
from ftd.pipeline import PuzzleGenerator


class StubDetector:
    def __init__(self, objects: list[DetectedObject]) -> None:
        self.objects = objects

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        return self.objects


def build_objects() -> list[DetectedObject]:
    masks = []
    for start_x in [10, 40, 70]:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, start_x : start_x + 15] = True
        masks.append(mask)

    return [
        DetectedObject(
            id=index,
            mask=mask,
            bbox=(10 + index * 30, 20, 15, 20),
            area=int(mask.sum()),
            stability_score=0.95 - index * 0.1,
            label=f"object_{index}",
        )
        for index, mask in enumerate(masks)
    ]


def test_manual_generate_applies_selected_changes() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:40, 10:25] = [255, 0, 0]
    image[20:40, 40:55] = [0, 255, 0]

    objects = build_objects()
    generator = PuzzleGenerator(StubDetector(objects))
    result = generator.manual_generate(
        image,
        [
            (objects[0], ChangeType.RECOLOR, {"hue_shift": 60, "saturation_scale": 1.0}),
            (objects[1], ChangeType.REMOVE, {}),
        ],
    )

    assert len(result.changes) == 2
    assert result.puzzle_image.ndim == 3
    assert result.solution_image.ndim == 3


def test_auto_generate_raises_when_no_objects() -> None:
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    generator = PuzzleGenerator(StubDetector([]))

    try:
        generator.auto_generate(image, DifficultyLevel.EASY)
    except ValueError as exc:
        assert "No objects detected" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no objects are detected")
