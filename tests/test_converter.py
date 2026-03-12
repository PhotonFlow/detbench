"""Unit tests for detbench.converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detbench.converter import convert_coco_to_yolo


class TestCocoToYolo:
    def _make_coco(self, tmp_path: Path) -> Path:
        coco = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 100, "height": 200},
            ],
            "categories": [
                {"id": 1, "name": "car"},
                {"id": 2, "name": "truck"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 60, 20, 30]},
            ],
        }
        path = tmp_path / "train.json"
        path.write_text(json.dumps(coco))
        return path

    def test_creates_label_file(self, tmp_path: Path) -> None:
        json_path = self._make_coco(tmp_path)
        labels_dir = tmp_path / "labels"
        names = convert_coco_to_yolo(json_path, labels_dir)

        assert names == ["car", "truck"]
        label_file = labels_dir / "img.txt"
        assert label_file.exists()

        lines = label_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_normalised_coordinates(self, tmp_path: Path) -> None:
        json_path = self._make_coco(tmp_path)
        labels_dir = tmp_path / "labels"
        convert_coco_to_yolo(json_path, labels_dir)

        lines = (labels_dir / "img.txt").read_text().strip().split("\n")
        parts = lines[0].split()
        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])

        assert cls_id == 0  # category_id 1 - offset 1
        # x_center = (10 + 30/2) / 100 = 0.25
        assert x_center == pytest.approx(0.25, abs=0.01)
        # y_center = (20 + 40/2) / 200 = 0.20
        assert y_center == pytest.approx(0.20, abs=0.01)

    def test_empty_annotations(self, tmp_path: Path) -> None:
        coco = {
            "images": [],
            "categories": [{"id": 1, "name": "obj"}],
            "annotations": [],
        }
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(coco))
        labels_dir = tmp_path / "labels"
        names = convert_coco_to_yolo(path, labels_dir)
        assert names == ["obj"]
