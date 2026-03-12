"""COCO-format to YOLO-format annotation conversion.

Converts bounding boxes from COCO format ``[x, y, w, h]`` (absolute
pixels, top-left origin) to YOLO format ``[class x_center y_center
width height]`` (normalised 0–1, centre origin), writing one ``.txt``
file per image.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def convert_coco_to_yolo(
    json_path: str | Path,
    output_labels_dir: str | Path,
    *,
    category_offset: int = 1,
) -> list[str]:
    """Convert a COCO annotation JSON to YOLO label text files.

    Parameters
    ----------
    json_path : str or Path
        Path to COCO-format annotation JSON.
    output_labels_dir : str or Path
        Directory for the output ``.txt`` label files.
    category_offset : int
        Value subtracted from ``category_id`` to produce a zero-based
        class index (default 1, since COCO IDs are typically 1-based).

    Returns
    -------
    list of str
        Ordered list of class names (index matches label integer).
    """
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    img_map = {img["id"]: img for img in data["images"]}

    for ann in data["annotations"]:
        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue

        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = ann["bbox"]

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h
        cls_id = ann["category_id"] - category_offset

        fname = os.path.splitext(img_info["file_name"])[0]
        txt_path = output_labels_dir / f"{fname}.txt"

        with open(txt_path, "a", encoding="utf-8") as fout:
            fout.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

    cats = {cat["id"]: cat["name"] for cat in data["categories"]}
    class_names = [cats[i] for i in sorted(cats.keys())]
    logger.info(
        "Converted %d annotations → %s (%d classes)",
        len(data["annotations"]),
        output_labels_dir,
        len(class_names),
    )
    return class_names
