from __future__ import annotations

import numpy as np

from ftd.detector import DetectedObject, ObjectDetector


def test_get_thumbnail_returns_rgb_image() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:50, 30:60] = [0, 128, 255]
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 30:60] = True
    obj = DetectedObject(
        id=1,
        mask=mask,
        bbox=(30, 20, 30, 30),
        area=int(mask.sum()),
        stability_score=0.99,
        label="object_1",
    )

    detector = ObjectDetector.__new__(ObjectDetector)
    thumb = detector.get_thumbnail(image, obj)
    assert thumb.ndim == 3
    assert thumb.shape[2] == 3
