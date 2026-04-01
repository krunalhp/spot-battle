from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ftd.detector import DetectedObject
from ftd.utils import clamp_bbox, dilate_mask, mask_to_bbox


class ChangeType(Enum):
    RECOLOR = "recolor"
    REMOVE = "remove"
    SHIFT = "shift"
    RESIZE = "resize"
    FLIP = "flip"


@dataclass
class AppliedChange:
    object_id: int
    change_type: ChangeType
    bbox: tuple[int, int, int, int]
    params: dict = field(default_factory=dict)


def recolor_object(
    image: np.ndarray,
    mask: np.ndarray,
    hue_shift: int = 30,
    saturation_scale: float = 1.0,
) -> np.ndarray:
    original = image.copy()
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0][mask] = (hsv[..., 0][mask] + hue_shift) % 180
    hsv[..., 1][mask] = np.clip(hsv[..., 1][mask] * saturation_scale, 0, 255).astype(np.int16)
    recolored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigmaX=1.5)[..., None]
    blended = recolored.astype(np.float32) * blurred_mask + original.astype(np.float32) * (1.0 - blurred_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def remove_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    dilated = dilate_mask(mask, iterations=3).astype(np.uint8) * 255
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, dilated, inpaintRadius=7, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def shift_object(image: np.ndarray, mask: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    obj_pixels = np.zeros_like(image)
    obj_pixels[mask] = image[mask]

    result = remove_object(image, mask)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    shifted_pixels = cv2.warpAffine(obj_pixels, matrix, (width, height))
    shifted_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, matrix, (width, height)) > 127
    result[shifted_mask] = shifted_pixels[shifted_mask]
    return result


def resize_object(image: np.ndarray, mask: np.ndarray, scale: float = 0.8) -> np.ndarray:
    result = remove_object(image, mask)
    height, width = image.shape[:2]
    bx, by, bw, bh = mask_to_bbox(mask)
    if bw == 0 or bh == 0:
        return image.copy()

    crop = image[by : by + bh, bx : bx + bw]
    mask_crop = mask[by : by + bh, bx : bx + bw]
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_crop.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 127

    cx = bx + bw // 2
    cy = by + bh // 2
    nx = cx - new_w // 2
    ny = cy - new_h // 2

    src_x1 = max(0, -nx)
    src_y1 = max(0, -ny)
    dst_x1 = max(0, nx)
    dst_y1 = max(0, ny)
    paste_w = min(new_w - src_x1, width - dst_x1)
    paste_h = min(new_h - src_y1, height - dst_y1)

    if paste_w > 0 and paste_h > 0:
        src_slice = np.s_[src_y1 : src_y1 + paste_h, src_x1 : src_x1 + paste_w]
        dst_slice = np.s_[dst_y1 : dst_y1 + paste_h, dst_x1 : dst_x1 + paste_w]
        dst_mask = resized_mask[src_slice]
        result_patch = result[dst_slice]
        crop_patch = resized_crop[src_slice]
        result_patch[dst_mask] = crop_patch[dst_mask]
        result[dst_slice] = result_patch

    return result


def flip_object(image: np.ndarray, mask: np.ndarray, axis: int = 1) -> np.ndarray:
    result = image.copy()
    bx, by, bw, bh = mask_to_bbox(mask)
    if bw == 0 or bh == 0:
        return result

    crop = image[by : by + bh, bx : bx + bw]
    mask_crop = mask[by : by + bh, bx : bx + bw]
    flipped_crop = cv2.flip(crop, axis)
    flipped_mask = cv2.flip(mask_crop.astype(np.uint8) * 255, axis) > 127

    patch = result[by : by + bh, bx : bx + bw]
    patch[flipped_mask] = flipped_crop[flipped_mask]
    result[by : by + bh, bx : bx + bw] = patch
    return result


def is_symmetric(image: np.ndarray, mask: np.ndarray, threshold: float = 5.0) -> bool:
    bx, by, bw, bh = mask_to_bbox(mask)
    if bw == 0 or bh == 0:
        return True

    crop = image[by : by + bh, bx : bx + bw]
    mask_crop = mask[by : by + bh, bx : bx + bw]
    flipped = cv2.flip(crop, 1)
    diff = np.abs(crop.astype(np.float32) - flipped.astype(np.float32)).mean(axis=2)
    masked_diff = diff[mask_crop]
    if masked_diff.size == 0:
        return True
    return float(masked_diff.mean()) < threshold


CHANGE_FUNCTIONS = {
    ChangeType.RECOLOR: recolor_object,
    ChangeType.REMOVE: remove_object,
    ChangeType.SHIFT: shift_object,
    ChangeType.RESIZE: resize_object,
    ChangeType.FLIP: flip_object,
}


def apply_change(
    image: np.ndarray,
    obj: DetectedObject,
    change_type: ChangeType,
    params: dict | None = None,
) -> tuple[np.ndarray, AppliedChange]:
    params = params or {}
    function = CHANGE_FUNCTIONS[change_type]
    modified = function(image=image, mask=obj.mask, **params)
    record = AppliedChange(
        object_id=obj.id,
        change_type=change_type,
        bbox=obj.bbox,
        params=params,
    )
    return modified, record
