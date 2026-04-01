from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ftd.utils import clamp_bbox

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


@dataclass
class DetectedObject:
    id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    stability_score: float
    label: str


class ObjectDetector:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu") -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {self.checkpoint_path}. "
                "Download sam_vit_b_01ec64.pth into the models directory."
            )

        sam = sam_model_registry[MODEL_TYPE](checkpoint=str(self.checkpoint_path))
        sam.to(device=device)
        self._generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            min_mask_region_area=200,
        )

    def detect(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.25,
        max_objects: int = 40,
    ) -> list[DetectedObject]:
        masks = self._generator.generate(image)
        height, width = image.shape[:2]
        total_pixels = height * width
        min_area = total_pixels * min_area_ratio
        max_area = total_pixels * max_area_ratio

        detected: list[DetectedObject] = []
        for idx, item in enumerate(masks):
            area = int(item["area"])
            if not (min_area <= area <= max_area):
                continue

            x, y, w, h = item["bbox"]
            detected.append(
                DetectedObject(
                    id=idx,
                    mask=item["segmentation"].astype(bool),
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    stability_score=float(item["stability_score"]),
                    label=f"object_{idx}",
                )
            )

        detected.sort(key=lambda obj: obj.stability_score, reverse=True)
        return detected[:max_objects]

    def get_thumbnail(
        self,
        image: np.ndarray,
        obj: DetectedObject,
        padding: int = 10,
        size: int = 120,
    ) -> np.ndarray:
        x, y, w, h = obj.bbox
        bbox = clamp_bbox((x - padding, y - padding, w + 2 * padding, h + 2 * padding), image.shape)
        x, y, w, h = bbox
        crop = image[y : y + h, x : x + w].copy()
        mask_crop = obj.mask[y : y + h, x : x + w]

        overlay = crop.copy().astype(np.float32)
        blue = np.array([100, 150, 255], dtype=np.float32)
        overlay[mask_crop] = (overlay[mask_crop] * 0.5 + blue * 0.5).astype(np.float32)
        thumb = overlay.astype(np.uint8)

        if thumb.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)

        scale = size / thumb.shape[0]
        new_width = max(1, int(round(thumb.shape[1] * scale)))
        return cv2.resize(thumb, (new_width, size), interpolation=cv2.INTER_AREA)
