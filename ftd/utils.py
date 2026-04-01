from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return _ensure_rgb(image)


def decode_uploaded_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode image data")
    return _ensure_rgb(image)


def save_image(image: np.ndarray, path: str | Path) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 4:
        alpha = (img[:, :, 3:4].astype(np.float32) / 255.0)
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        composited = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)

    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def resize_to_max(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def estimate_background_color(
    image: np.ndarray,
    mask: np.ndarray,
    border_width: int = 10,
) -> tuple[int, int, int]:
    iterations = max(1, border_width // 2 + 1)
    dilated = dilate_mask(mask, iterations=iterations)
    ring = dilated & ~mask.astype(bool)

    if not np.any(ring):
        pixels = image.reshape(-1, 3)
    else:
        pixels = image[ring]

    median = np.median(pixels, axis=0)
    return tuple(int(value) for value in median)


def dilate_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=iterations)
    return dilated > 127


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    height, width = img_shape[:2]

    x1 = min(max(0, x), width)
    y1 = min(max(0, y), height)
    x2 = min(max(x1, x + w), width)
    y2 = min(max(y1, y + h), height)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, 0, 0

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def encode_image_to_png(image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return encoded.tobytes()
