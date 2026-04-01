from __future__ import annotations

import numpy as np

from ftd.detector import DetectedObject
from ftd.editor_session import EditorSession
from ftd.modifier import ChangeType


def build_editor_session() -> tuple[EditorSession, np.ndarray]:
    image = np.full((90, 90, 3), 255, dtype=np.uint8)
    image[20:45, 20:45] = [255, 0, 0]
    image[50:75, 50:75] = [0, 255, 0]

    mask_a = np.zeros((90, 90), dtype=bool)
    mask_a[20:45, 20:45] = True
    mask_b = np.zeros((90, 90), dtype=bool)
    mask_b[50:75, 50:75] = True

    objects = [
        DetectedObject(id=1, mask=mask_a, bbox=(20, 20, 25, 25), area=int(mask_a.sum()), stability_score=0.9, label="object_1"),
        DetectedObject(id=2, mask=mask_b, bbox=(50, 50, 25, 25), area=int(mask_b.sum()), stability_score=0.8, label="object_2"),
    ]
    return EditorSession(image, objects), image


def test_editor_session_apply_and_undo() -> None:
    session, original = build_editor_session()

    session.apply_edit(1, ChangeType.RECOLOR, {"hue_shift": 60, "saturation_scale": 1.0})
    assert len(session.applied_changes) == 1
    assert not np.array_equal(session.modified_image, original)

    session.undo()
    assert len(session.applied_changes) == 0
    assert np.array_equal(session.modified_image, original)


def test_editor_session_clear_resets_multiple_edits() -> None:
    session, original = build_editor_session()

    session.apply_edit(1, ChangeType.RECOLOR, {"hue_shift": 60, "saturation_scale": 1.0})
    session.apply_edit(2, ChangeType.REMOVE, {})
    assert len(session.applied_changes) == 2

    session.clear()
    assert len(session.applied_changes) == 0
    assert np.array_equal(session.modified_image, original)
