from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ftd.detector import DetectedObject, ObjectDetector
from ftd.difficulty import DifficultyLevel, resolve_config, sample_num_differences, sample_params
from ftd.modifier import AppliedChange, ChangeType, apply_change, is_symmetric
from ftd.output import build_puzzle_image, build_solution_image

CHANGE_PRIORITY = {
    ChangeType.SHIFT: 0,
    ChangeType.RESIZE: 1,
    ChangeType.FLIP: 2,
    ChangeType.RECOLOR: 3,
    ChangeType.REMOVE: 4,
}


@dataclass
class PuzzleResult:
    original: np.ndarray
    modified: np.ndarray
    changes: list[AppliedChange]
    puzzle_image: np.ndarray
    solution_image: np.ndarray


class PuzzleGenerator:
    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector

    def detect_objects(self, image: np.ndarray) -> list[DetectedObject]:
        return self.detector.detect(image)

    def auto_generate(self, image: np.ndarray, difficulty: DifficultyLevel) -> PuzzleResult:
        objects = self.detect_objects(image)
        if not objects:
            raise ValueError("No objects detected in the image. Try a clearer cartoon scene.")

        config = resolve_config(difficulty)
        target_count = min(sample_num_differences(config), len(objects))
        selected = self._pick_non_overlapping(objects, target_count)
        assignments: list[tuple[DetectedObject, ChangeType, dict]] = []

        for obj in selected:
            change_type = random.choice(config.allowed_changes)
            if change_type == ChangeType.FLIP and is_symmetric(image, obj.mask):
                alternatives = [candidate for candidate in config.allowed_changes if candidate != ChangeType.FLIP]
                change_type = random.choice(alternatives) if alternatives else ChangeType.RECOLOR
            params = sample_params(config, change_type)
            assignments.append((obj, change_type, params))

        assignments.sort(key=lambda item: CHANGE_PRIORITY[item[1]])
        return self._apply_and_build(image, assignments)

    def manual_generate(
        self,
        image: np.ndarray,
        selections: list[tuple[DetectedObject, ChangeType, dict]],
    ) -> PuzzleResult:
        if not selections:
            raise ValueError("No changes selected.")

        assignments = sorted(selections, key=lambda item: CHANGE_PRIORITY[item[1]])
        return self._apply_and_build(image, assignments)

    def _apply_and_build(
        self,
        image: np.ndarray,
        assignments: list[tuple[DetectedObject, ChangeType, dict]],
    ) -> PuzzleResult:
        modified = image.copy()
        changes: list[AppliedChange] = []

        for obj, change_type, params in assignments:
            modified, record = apply_change(modified, obj, change_type, params)
            changes.append(record)

        puzzle_image = build_puzzle_image(image, modified)
        solution_image = build_solution_image(image, modified)
        return PuzzleResult(
            original=image,
            modified=modified,
            changes=changes,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
        )

    @staticmethod
    def _pick_non_overlapping(
        objects: list[DetectedObject],
        n: int,
        max_overlap: float = 0.5,
    ) -> list[DetectedObject]:
        candidates = objects[:]
        random.shuffle(candidates)
        selected: list[DetectedObject] = []

        for obj in candidates:
            overlaps = False
            for existing in selected:
                intersection = int((obj.mask & existing.mask).sum())
                overlap_ratio = intersection / max(1, min(obj.area, existing.area))
                if overlap_ratio > max_overlap:
                    overlaps = True
                    break
            if overlaps:
                continue

            selected.append(obj)
            if len(selected) >= n:
                break

        return selected
