from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ftd.differ import compute_solution_overlay


def stitch_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    separator_width: int = 4,
    separator_color: tuple[int, int, int] = (200, 200, 200),
    labels: tuple[str, str] = ("Original", "Modified"),
    label_height: int = 40,
    show_labels: bool = True,
) -> np.ndarray:
    target_h = max(left.shape[0], right.shape[0])
    left_resized = _resize_to_height(left, target_h)
    right_resized = _resize_to_height(right, target_h)

    if (left_resized.shape[1] + separator_width + right_resized.shape[1]) / target_h > 3.5:
        return _stitch_vertical(
            left_resized,
            right_resized,
            separator_width=separator_width,
            separator_color=separator_color,
            labels=labels,
            label_height=label_height,
            show_labels=show_labels,
        )

    separator = np.full((target_h, separator_width, 3), separator_color, dtype=np.uint8)
    stitched = np.concatenate([left_resized, separator, right_resized], axis=1)

    if not show_labels:
        return stitched

    label_bar = np.full((label_height, stitched.shape[1], 3), 40, dtype=np.uint8)
    _draw_centered_text(label_bar, labels[0], left_resized.shape[1] // 2, label_height)
    right_center_x = left_resized.shape[1] + separator_width + right_resized.shape[1] // 2
    _draw_centered_text(label_bar, labels[1], right_center_x, label_height)
    return np.concatenate([label_bar, stitched], axis=0)


def _stitch_vertical(
    top: np.ndarray,
    bottom: np.ndarray,
    separator_width: int = 4,
    separator_color: tuple[int, int, int] = (200, 200, 200),
    labels: tuple[str, str] = ("Original", "Modified"),
    label_height: int = 40,
    show_labels: bool = True,
) -> np.ndarray:
    target_w = max(top.shape[1], bottom.shape[1])
    top_resized = _resize_to_width(top, target_w)
    bottom_resized = _resize_to_width(bottom, target_w)

    if not show_labels:
        separator = np.full((separator_width, target_w, 3), separator_color, dtype=np.uint8)
        return np.concatenate([top_resized, separator, bottom_resized], axis=0)

    top_bar = np.full((label_height, target_w, 3), 40, dtype=np.uint8)
    bottom_bar = np.full((label_height, target_w, 3), 40, dtype=np.uint8)
    _draw_centered_text(top_bar, labels[0], target_w // 2, label_height)
    _draw_centered_text(bottom_bar, labels[1], target_w // 2, label_height)

    separator = np.full((separator_width, target_w, 3), separator_color, dtype=np.uint8)
    return np.concatenate([top_bar, top_resized, separator, bottom_bar, bottom_resized], axis=0)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _draw_centered_text(canvas: np.ndarray, text: str, center_x: int, height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, center_x - text_w // 2)
    y = max(text_h, (height + text_h) // 2 - baseline)
    cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def build_puzzle_image(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    return stitch_side_by_side(original, modified)


def build_solution_image(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    annotated = compute_solution_overlay(original, modified)
    return stitch_side_by_side(original, annotated, labels=("Original", "Solution"))


def build_video_puzzle_image(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    return stitch_side_by_side(original, modified, show_labels=False)


def build_video_solution_image(original: np.ndarray, modified: np.ndarray) -> np.ndarray:
    annotated = compute_solution_overlay(original, modified)
    return stitch_side_by_side(original, annotated, labels=("Original", "Solution"), show_labels=False)


def save_outputs(
    original: np.ndarray,
    modified: np.ndarray,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    from ftd.utils import save_image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    puzzle_path = output_path / "puzzle.png"
    solution_path = output_path / "solution.png"
    save_image(build_puzzle_image(original, modified), puzzle_path)
    save_image(build_solution_image(original, modified), solution_path)
    return puzzle_path, solution_path
