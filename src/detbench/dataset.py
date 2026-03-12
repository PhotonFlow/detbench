"""COCO crop dataset for classification-level benchmarks.

Parses a COCO-format annotation JSON, crops ground-truth bounding
boxes from source images, and returns ``(crop_tensor, label_index)``
pairs suitable for standard classification training loops.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CocoCropDataset(Dataset):  # type: ignore[type-arg]
    """Dataset that yields crops from COCO bounding-box annotations.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing the source images.
    json_path : str or Path
        Path to the COCO-format annotation JSON.
    transform : callable or None
        Torchvision-style transform applied to each PIL crop.
    min_crop_size : int
        Crops smaller than this in either dimension are skipped.
    """

    def __init__(
        self,
        root_dir: str | Path,
        json_path: str | Path,
        transform: Any = None,
        min_crop_size: int = 2,
    ) -> None:
        self.root = str(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int, int, int, int, int]] = []

        with open(json_path, encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)

        images = {img["id"]: img for img in data["images"]}
        self.classes = sorted(c["id"] for c in data["categories"])
        self.num_classes = len(self.classes)
        self.cat_to_idx = {cat_id: i for i, cat_id in enumerate(self.classes)}

        for ann in data["annotations"]:
            img_info = images.get(ann["image_id"])
            if img_info is None:
                continue
            x, y, w, h = map(int, ann["bbox"])
            if w < min_crop_size or h < min_crop_size:
                continue
            label_idx = self.cat_to_idx.get(ann["category_id"])
            if label_idx is None:
                continue
            full_path = os.path.join(self.root, img_info["file_name"])
            self.samples.append((full_path, x, y, w, h, label_idx))

        logger.info("Loaded %d crops from %s", len(self.samples), json_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, x, y, w, h, label = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            crop = cv2.resize(img, (224, 224))

        crop_pil = Image.fromarray(crop)
        if self.transform:
            crop_pil = self.transform(crop_pil)

        return crop_pil, label
