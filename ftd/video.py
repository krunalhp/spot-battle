from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from ftd.output import build_video_puzzle_image, build_video_solution_image

FULL_HD = (1920, 1080)
DEFAULT_FPS = 30
DEFAULT_PUZZLE_DURATION = 90.0
DEFAULT_FLASH_INTERVAL = 0.5
DEFAULT_REVEAL_CYCLES = 3


def build_video_segments(
    puzzle_duration: float = DEFAULT_PUZZLE_DURATION,
    flash_interval: float = DEFAULT_FLASH_INTERVAL,
    reveal_cycles: int = DEFAULT_REVEAL_CYCLES,
) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = [("puzzle", float(puzzle_duration))]
    for _ in range(reveal_cycles):
        segments.append(("puzzle", float(flash_interval)))
        segments.append(("solution", float(flash_interval)))
    return segments


def render_video_frame(
    image: np.ndarray,
    progress: float,
    resolution: tuple[int, int] = FULL_HD,
) -> np.ndarray:
    width, height = resolution
    frame = np.full((height, width, 3), 18, dtype=np.uint8)

    bar_margin = 48
    bar_height = 22
    image_area_height = height - (bar_margin * 2 + bar_height)
    image_area_width = width - 80

    scaled = _fit_image(image, image_area_width, image_area_height)
    y_offset = max(20, (image_area_height - scaled.shape[0]) // 2 + 20)
    x_offset = (width - scaled.shape[1]) // 2
    frame[y_offset : y_offset + scaled.shape[0], x_offset : x_offset + scaled.shape[1]] = scaled

    _draw_progress_bar(frame, progress, bar_margin=bar_margin, bar_height=bar_height)
    return frame


def save_puzzle_video(
    original: np.ndarray,
    modified: np.ndarray,
    output_path: str | Path,
    puzzle_duration: float = DEFAULT_PUZZLE_DURATION,
    flash_interval: float = DEFAULT_FLASH_INTERVAL,
    reveal_cycles: int = DEFAULT_REVEAL_CYCLES,
    fps: int = DEFAULT_FPS,
    resolution: tuple[int, int] = FULL_HD,
) -> Path:
    puzzle_image = build_video_puzzle_image(original, modified)
    solution_image = build_video_solution_image(original, modified)
    return save_video_from_images(
        puzzle_image=puzzle_image,
        solution_image=solution_image,
        output_path=output_path,
        puzzle_duration=puzzle_duration,
        flash_interval=flash_interval,
        reveal_cycles=reveal_cycles,
        fps=fps,
        resolution=resolution,
    )


def export_puzzle_video_bytes(
    original: np.ndarray,
    modified: np.ndarray,
    puzzle_duration: float = DEFAULT_PUZZLE_DURATION,
    flash_interval: float = DEFAULT_FLASH_INTERVAL,
    reveal_cycles: int = DEFAULT_REVEAL_CYCLES,
    fps: int = DEFAULT_FPS,
    resolution: tuple[int, int] = FULL_HD,
) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "puzzle_video.mp4"
        save_puzzle_video(
            original=original,
            modified=modified,
            output_path=path,
            puzzle_duration=puzzle_duration,
            flash_interval=flash_interval,
            reveal_cycles=reveal_cycles,
            fps=fps,
            resolution=resolution,
        )
        return path.read_bytes()


def save_video_from_images(
    puzzle_image: np.ndarray,
    solution_image: np.ndarray,
    output_path: str | Path,
    puzzle_duration: float = DEFAULT_PUZZLE_DURATION,
    flash_interval: float = DEFAULT_FLASH_INTERVAL,
    reveal_cycles: int = DEFAULT_REVEAL_CYCLES,
    fps: int = DEFAULT_FPS,
    resolution: tuple[int, int] = FULL_HD,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for {output_path}")

    segments = build_video_segments(
        puzzle_duration=puzzle_duration,
        flash_interval=flash_interval,
        reveal_cycles=reveal_cycles,
    )
    total_duration = sum(duration for _, duration in segments)
    elapsed = 0.0

    try:
        for segment_name, duration in segments:
            frame_count = max(1, int(round(duration * fps)))
            for frame_index in range(frame_count):
                segment_progress = frame_index / max(1, frame_count - 1)
                current_time = elapsed + segment_progress * duration
                progress = min(1.0, current_time / total_duration)
                source = puzzle_image if segment_name == "puzzle" else solution_image
                frame = render_video_frame(source, progress=progress, resolution=resolution)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            elapsed += duration
    finally:
        writer.release()

    return output_path


def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _draw_progress_bar(
    frame: np.ndarray,
    progress: float,
    bar_margin: int,
    bar_height: int,
) -> None:
    height, width = frame.shape[:2]
    x1 = bar_margin
    x2 = width - bar_margin
    y1 = height - bar_margin
    y2 = y1 + bar_height

    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), thickness=-1)
    filled_width = int(round((x2 - x1) * np.clip(progress, 0.0, 1.0)))
    if filled_width > 0:
        color = _progress_color(progress)
        cv2.rectangle(frame, (x1, y1), (x1 + filled_width, y2), color, thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=2)


def _progress_color(progress: float) -> tuple[int, int, int]:
    if progress < 0.75:
        return 70, 200, 90
    if progress < 0.9:
        mix = (progress - 0.75) / 0.15
        start = np.array([70, 200, 90], dtype=np.float32)
        end = np.array([240, 180, 60], dtype=np.float32)
        color = start * (1.0 - mix) + end * mix
        return tuple(int(value) for value in color)

    mix = (progress - 0.9) / 0.1
    start = np.array([240, 180, 60], dtype=np.float32)
    end = np.array([220, 70, 60], dtype=np.float32)
    color = start * (1.0 - mix) + end * mix
    return tuple(int(value) for value in color)
