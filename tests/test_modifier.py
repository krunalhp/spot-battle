from __future__ import annotations

import numpy as np

from ftd.modifier import flip_object, recolor_object, remove_object, resize_object, shift_object


def sample_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[20:40, 20:40] = [255, 0, 0]
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:40, 20:40] = True
    return image, mask


def test_recolor_object_changes_masked_pixels() -> None:
    image, mask = sample_image_and_mask()
    result = recolor_object(image, mask, hue_shift=60, saturation_scale=1.0)
    assert not np.array_equal(result[mask], image[mask])
    assert np.array_equal(result[~mask], image[~mask])


def test_remove_object_changes_region() -> None:
    image, mask = sample_image_and_mask()
    result = remove_object(image, mask)
    assert not np.array_equal(result[mask], image[mask])


def test_shift_object_moves_content() -> None:
    image, mask = sample_image_and_mask()
    result = shift_object(image, mask, dx=10, dy=0)
    assert np.any(np.all(result[20:40, 30:50] == [255, 0, 0], axis=2))


def test_resize_object_keeps_image_shape() -> None:
    image, mask = sample_image_and_mask()
    result = resize_object(image, mask, scale=0.5)
    assert result.shape == image.shape


def test_flip_object_keeps_image_shape() -> None:
    image, mask = sample_image_and_mask()
    image[20:40, 20:30] = [0, 255, 0]
    result = flip_object(image, mask, axis=1)
    assert result.shape == image.shape
    assert np.any(np.all(result[20:40, 30:40] == [0, 255, 0], axis=2))
