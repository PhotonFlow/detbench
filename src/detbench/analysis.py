"""Error analysis: false positive / false negative visualisation.

Performs inference on validation images and visually annotates
detections as TP (green), FP (red), or FN (blue), producing
diagnostic images that pinpoint where a model struggles.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import cv2
import yaml

logger = logging.getLogger(__name__)


def compute_iou_xyxy(
    box1: tuple[float, ...],
    box2: tuple[float, ...],
) -> float:
    """IoU for ``[x1, y1, x2, y2]`` format boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def generate_error_analysis(
    model: Any,
    data_yaml_path: str | Path,
    output_dir: str | Path,
    *,
    conf_thres: float = 0.4,
    iou_thres: float = 0.5,
    max_examples: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """Generate FP / FN visualisation images.

    Parameters
    ----------
    model : YOLO
        A loaded model with a ``.predict()`` method.
    data_yaml_path : str or Path
        Path to the YOLO ``dataset.yaml``.
    output_dir : str or Path
        Directory for output images.
    conf_thres : float
        Confidence threshold for predictions.
    iou_thres : float
        IoU threshold for TP / FP / FN classification.
    max_examples : int
        Maximum number of FP and FN images to save.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        ``{"fp_saved": int, "fn_saved": int}``
    """
    errors_dir = Path(output_dir) / "error_examples"
    errors_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml_path, encoding="utf-8") as fh:
        data_cfg = yaml.safe_load(fh)

    val_img_dir = data_cfg["val"]
    if not os.path.isabs(val_img_dir):
        val_img_dir = os.path.join(os.path.dirname(str(data_yaml_path)), val_img_dir)

    label_dir = val_img_dir.replace("images", "labels")
    image_files = [
        f for f in os.listdir(val_img_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    rng = random.Random(seed)
    rng.shuffle(image_files)

    fp_count = 0
    fn_count = 0

    for img_file in image_files:
        if fp_count >= max_examples and fn_count >= max_examples:
            break

        img_path = os.path.join(val_img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load ground truth
        gt_boxes: list[list[int]] = []
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    _, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    gt_boxes.append([x1, y1, x2, y2])

        # Predict
        results = model.predict(img_path, conf=conf_thres, verbose=False)[0]
        pred_boxes = results.boxes.xyxy.cpu().numpy()

        # Classify FP / FN
        fp_idx = []
        for i, p_box in enumerate(pred_boxes):
            best = max(
                (compute_iou_xyxy(tuple(p_box), tuple(g)) for g in gt_boxes),
                default=0.0,
            )
            if best < iou_thres:
                fp_idx.append(i)

        fn_idx = []
        for i, g_box in enumerate(gt_boxes):
            best = max(
                (compute_iou_xyxy(tuple(g_box), tuple(p)) for p in pred_boxes),
                default=0.0,
            )
            if best < iou_thres:
                fn_idx.append(i)

        has_fp = len(fp_idx) > 0
        has_fn = len(fn_idx) > 0

        if not (
            (has_fp and fp_count < max_examples) or (has_fn and fn_count < max_examples)
        ):
            continue

        # Draw
        for i, p_box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, p_box)
            is_fp = i in fp_idx
            color = (0, 0, 255) if is_fp else (0, 255, 0)
            label = "FP" if is_fp else "TP"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        for i in fn_idx:
            x1, y1, x2, y2 = map(int, gt_boxes[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img,
                "FN",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        if has_fp:
            fp_count += 1
        if has_fn:
            fn_count += 1

        out_name = f"Error_{fp_count + fn_count}_{img_file}"
        cv2.imwrite(str(errors_dir / out_name), img)

    stats = {"fp_saved": fp_count, "fn_saved": fn_count}
    logger.info("Error analysis: %s", stats)
    return stats
