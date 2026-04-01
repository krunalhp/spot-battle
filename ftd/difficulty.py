from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from ftd.modifier import ChangeType


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyConfig:
    num_differences: tuple[int, int]
    allowed_changes: list[ChangeType]
    hue_shift_range: tuple[int, int]
    shift_pixel_range: tuple[int, int]
    resize_scale_range: tuple[float, float]


PRESETS = {
    DifficultyLevel.EASY: DifficultyConfig(
        num_differences=(3, 5),
        allowed_changes=[ChangeType.RECOLOR, ChangeType.REMOVE],
        hue_shift_range=(40, 80),
        shift_pixel_range=(15, 30),
        resize_scale_range=(0.6, 0.75),
    ),
    DifficultyLevel.MEDIUM: DifficultyConfig(
        num_differences=(5, 7),
        allowed_changes=[ChangeType.RECOLOR, ChangeType.REMOVE, ChangeType.SHIFT, ChangeType.RESIZE],
        hue_shift_range=(20, 50),
        shift_pixel_range=(8, 20),
        resize_scale_range=(0.75, 0.9),
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        num_differences=(7, 10),
        allowed_changes=list(ChangeType),
        hue_shift_range=(10, 30),
        shift_pixel_range=(4, 12),
        resize_scale_range=(0.85, 0.95),
    ),
}


def resolve_config(level: DifficultyLevel) -> DifficultyConfig:
    return PRESETS[level]


def sample_num_differences(config: DifficultyConfig) -> int:
    return random.randint(config.num_differences[0], config.num_differences[1])


def sample_params(config: DifficultyConfig, change_type: ChangeType) -> dict:
    if change_type == ChangeType.RECOLOR:
        return {
            "hue_shift": random.randint(*config.hue_shift_range),
            "saturation_scale": random.uniform(0.8, 1.2),
        }
    if change_type == ChangeType.REMOVE:
        return {}
    if change_type == ChangeType.SHIFT:
        return {
            "dx": random.choice([-1, 1]) * random.randint(*config.shift_pixel_range),
            "dy": random.choice([-1, 1]) * random.randint(*config.shift_pixel_range),
        }
    if change_type == ChangeType.RESIZE:
        return {"scale": random.uniform(*config.resize_scale_range)}
    if change_type == ChangeType.FLIP:
        return {"axis": 1}
    raise ValueError(f"Unsupported change type: {change_type}")
