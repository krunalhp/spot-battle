from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ftd.detector import DetectedObject
from ftd.modifier import AppliedChange, ChangeType, apply_change
from ftd.output import build_puzzle_image, build_solution_image


@dataclass
class EditCommand:
    object_id: int
    change_type: ChangeType
    params: dict = field(default_factory=dict)


@dataclass
class EditorSnapshot:
    original: np.ndarray
    modified: np.ndarray
    changes: list[AppliedChange]


class EditorSession:
    def __init__(self, image: np.ndarray, objects: list[DetectedObject]) -> None:
        self.original = image.copy()
        self.objects = {obj.id: obj for obj in objects}
        self.history: list[EditCommand] = []
        self._snapshot = EditorSnapshot(
            original=self.original,
            modified=self.original.copy(),
            changes=[],
        )

    @property
    def modified_image(self) -> np.ndarray:
        return self._snapshot.modified

    @property
    def applied_changes(self) -> list[AppliedChange]:
        return self._snapshot.changes

    def apply_edit(self, object_id: int, change_type: ChangeType, params: dict | None = None) -> EditorSnapshot:
        if object_id not in self.objects:
            raise ValueError(f"Unknown object id: {object_id}")

        self.history.append(EditCommand(object_id=object_id, change_type=change_type, params=params or {}))
        self._snapshot = self._rebuild()
        return self._snapshot

    def undo(self) -> EditorSnapshot:
        if self.history:
            self.history.pop()
            self._snapshot = self._rebuild()
        return self._snapshot

    def clear(self) -> EditorSnapshot:
        self.history.clear()
        self._snapshot = self._rebuild()
        return self._snapshot

    def can_undo(self) -> bool:
        return bool(self.history)

    def build_puzzle_image(self) -> np.ndarray:
        return build_puzzle_image(self.original, self.modified_image)

    def build_solution_image(self) -> np.ndarray:
        return build_solution_image(self.original, self.modified_image)

    def _rebuild(self) -> EditorSnapshot:
        modified = self.original.copy()
        changes: list[AppliedChange] = []

        for command in self.history:
            obj = self.objects[command.object_id]
            modified, record = apply_change(modified, obj, command.change_type, command.params)
            changes.append(record)

        return EditorSnapshot(
            original=self.original,
            modified=modified,
            changes=changes,
        )
