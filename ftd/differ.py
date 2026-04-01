from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DiffRegion:
    bbox: tuple[int, int, int, int]
    area: int


def compute_diff_regions(
    original: np.ndarray,
    modified: np.ndarray,
    threshold: int = 25,
    min_contour_area: int = 100,
    merge_distance: int = 20,
) -> list[DiffRegion]:
    diff = np.abs(original.astype(np.int16) - modified.astype(np.int16)).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            boxes.append(cv2.boundingRect(contour))

    merged = _merge_nearby_boxes(boxes, distance=merge_distance)
    return [DiffRegion(bbox=box, area=box[2] * box[3]) for box in merged]


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    distance: int,
) -> list[tuple[int, int, int, int]]:
    merged = boxes[:]
    changed = True

    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)

        for i, box_a in enumerate(merged):
            if used[i]:
                continue

            current = box_a
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _boxes_overlap_or_close(current, merged[j], distance):
                    current = _union_box(current, merged[j])
                    used[j] = True
                    changed = True
            next_boxes.append(current)

        merged = next_boxes

    return merged


def _boxes_overlap_or_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    distance: int,
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + distance < bx
        or bx + bw + distance < ax
        or ay + ah + distance < by
        or by + bh + distance < ay
    )


def _union_box(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def draw_solution_borders(
    image: np.ndarray,
    regions: list[DiffRegion],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
    padding: int = 8,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]

    for region in regions:
        x, y, w, h = region.bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width - 1, x + w + padding)
        y2 = min(height - 1, y + h + padding)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    return result


def compute_solution_overlay(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    regions = compute_diff_regions(original, modified)
    return draw_solution_borders(modified, regions)
