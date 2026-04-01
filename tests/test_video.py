from __future__ import annotations

from pathlib import Path

import numpy as np

from ftd.video import build_video_segments, render_video_frame, save_video_from_images


def test_build_video_segments_matches_expected_timing() -> None:
    segments = build_video_segments(puzzle_duration=90.0, flash_interval=0.5, reveal_cycles=3)
    assert segments[0] == ("puzzle", 90.0)
    assert segments[1:] == [
        ("puzzle", 0.5),
        ("solution", 0.5),
        ("puzzle", 0.5),
        ("solution", 0.5),
        ("puzzle", 0.5),
        ("solution", 0.5),
    ]
    assert sum(duration for _, duration in segments) == 93.0


def test_render_video_frame_outputs_full_hd() -> None:
    image = np.full((500, 900, 3), 255, dtype=np.uint8)
    frame = render_video_frame(image, progress=0.5)
    assert frame.shape == (1080, 1920, 3)


def test_save_video_from_images_creates_mp4(tmp_path: Path) -> None:
    puzzle = np.full((400, 800, 3), 240, dtype=np.uint8)
    solution = puzzle.copy()
    solution[100:200, 100:200] = [255, 0, 0]

    video_path = tmp_path / "sample.mp4"
    save_video_from_images(
        puzzle_image=puzzle,
        solution_image=solution,
        output_path=video_path,
        puzzle_duration=1.0,
        flash_interval=0.1,
        reveal_cycles=1,
        fps=10,
    )

    assert video_path.exists()
    assert video_path.stat().st_size > 0
