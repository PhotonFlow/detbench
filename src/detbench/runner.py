"""Benchmark orchestration for detection model training and evaluation.

Coordinates end-to-end benchmarking:
1. COCO → YOLO dataset conversion
2. Train across multiple datasets × models × regimes
3. Per-class metric logging
4. FP / FN error analysis
5. Summary CSV generation
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

from detbench.converter import convert_coco_to_yolo

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark experiment.

    Parameters
    ----------
    train_sets : dict
        Mapping of dataset name → ``{"root": str, "json": str}``.
    val_root : str
        Path to validation images.
    val_json : str
        Path to validation annotation JSON.
    output_dir : str
        Root output directory.
    models : dict
        Mapping of model alias → weights path.
    regimes : list of str
        Training regimes (e.g. ``Full_FT``, ``Classifier_Only``, ``LP_FT``).
    epochs : int
        Total training epochs.
    batch_size : int
        Batch size.
    img_size : int
        Training image size.
    workers : int
        DataLoader workers.
    lp_epochs : int
        Linear-probe epochs for the ``LP_FT`` regime.
    """

    train_sets: dict[str, dict[str, str]] = field(default_factory=dict)
    val_root: str = ""
    val_json: str = ""
    output_dir: str = "benchmark_results"
    models: dict[str, str] = field(default_factory=lambda: {"YOLOv8n": "yolov8n.pt"})
    regimes: list[str] = field(default_factory=lambda: ["Classifier_Only"])
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    workers: int = 8
    lp_epochs: int = 15


class BenchmarkRunner:
    """Run multi-dataset, multi-model, multi-regime benchmarks.

    Parameters
    ----------
    config : BenchmarkConfig
        Experiment configuration.

    Example
    -------
    >>> runner = BenchmarkRunner(BenchmarkConfig(
    ...     train_sets={"clean": {"root": "images/", "json": "train.json"}},
    ...     val_root="val_images/",
    ...     val_json="val.json",
    ... ))
    >>> runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> pd.DataFrame:
        """Execute the full benchmark sweep.

        Returns
        -------
        pd.DataFrame
            Summary table with mAP, precision, recall, and fitness
            for each (dataset, model, regime) combination.
        """
        from ultralytics import YOLO  # Lazy import — heavy dependency

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Clean previous logs
        for fname in ["detailed_training_history.csv", "per_class_metrics.csv"]:
            path = os.path.join(cfg.output_dir, fname)
            if os.path.exists(path):
                os.remove(path)

        all_results: list[dict[str, Any]] = []

        for dset_name, dset_paths in cfg.train_sets.items():
            logger.info("Dataset: %s", dset_name)
            project_dir = os.path.join(cfg.output_dir, dset_name)

            data_yaml, class_names = self._setup_dataset(
                dset_paths["root"],
                dset_paths["json"],
                project_dir,
            )

            for model_alias, model_weights in cfg.models.items():
                for regime in cfg.regimes:
                    run_id = f"{model_alias}_{regime}"
                    final_path = self._train(
                        YOLO,
                        model_weights,
                        data_yaml,
                        project_dir,
                        run_id,
                        regime,
                    )

                    if final_path and os.path.exists(final_path):
                        model = YOLO(final_path)
                        metrics = model.val(split="val")

                        self._log_per_class(
                            metrics, dset_name, model_alias, regime, class_names
                        )

                        # Error analysis (import lazily)
                        from detbench.analysis import generate_error_analysis

                        generate_error_analysis(
                            model,
                            data_yaml,
                            os.path.join(project_dir, run_id),
                        )

                        fitness = 0.1 * metrics.box.map50 + 0.9 * metrics.box.map
                        all_results.append(
                            {
                                "Dataset": dset_name,
                                "Model": model_alias,
                                "Regime": regime,
                                "mAP@50": round(metrics.box.map50, 4),
                                "mAP@50-95": round(metrics.box.map, 4),
                                "Precision": round(metrics.box.mp, 4),
                                "Recall": round(metrics.box.mr, 4),
                                "Fitness": round(fitness, 4),
                            }
                        )

        df = pd.DataFrame(all_results)
        summary_path = os.path.join(cfg.output_dir, "overall_benchmark_summary.csv")
        df.to_csv(summary_path, index=False)
        logger.info("Benchmark complete. Summary → %s", summary_path)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_dataset(
        self,
        train_root: str,
        train_json: str,
        project_dir: str,
    ) -> tuple[str, list[str]]:
        """Convert COCO datasets → YOLO format and generate dataset.yaml."""
        cfg = self.config

        train_labels = os.path.join(project_dir, "labels/train")
        val_labels = os.path.join(project_dir, "labels/val")

        class_names = convert_coco_to_yolo(train_json, train_labels)
        convert_coco_to_yolo(cfg.val_json, val_labels)

        # Copy images
        train_img = os.path.join(project_dir, "images/train")
        val_img = os.path.join(project_dir, "images/val")
        self._link_images(train_root, train_img)
        self._link_images(cfg.val_root, val_img)

        # Write YAML
        yaml_path = os.path.join(project_dir, "dataset.yaml")
        yaml_data = {
            "path": os.path.abspath(project_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(class_names)},
        }
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh)

        return yaml_path, class_names

    def _train(
        self,
        yolo_cls: type,
        weights: str,
        data_yaml: str,
        project_dir: str,
        run_id: str,
        regime: str,
    ) -> str | None:
        """Execute a single training run and return path to best weights."""
        cfg = self.config

        if regime == "Full_FT":
            return self._train_single(
                yolo_cls,
                weights,
                data_yaml,
                project_dir,
                run_id,
                cfg.epochs,
                freeze_n=0,
            )
        elif regime == "Classifier_Only":
            return self._train_single(
                yolo_cls,
                weights,
                data_yaml,
                project_dir,
                run_id,
                cfg.epochs,
                freeze_n=10,
            )
        elif regime == "LP_FT":
            if cfg.epochs <= cfg.lp_epochs:
                lp = max(1, cfg.epochs // 2)
                ft = max(1, cfg.epochs - lp)
            else:
                lp = cfg.lp_epochs
                ft = cfg.epochs - cfg.lp_epochs

            s1_name = f"{run_id}_stage1"
            self._train_single(
                yolo_cls, weights, data_yaml, project_dir, s1_name, lp, 10
            )

            s1_weights = os.path.join(project_dir, s1_name, "weights/best.pt")
            if os.path.exists(s1_weights):
                s2_name = f"{run_id}_stage2"
                return self._train_single(
                    yolo_cls, s1_weights, data_yaml, project_dir, s2_name, ft, 0
                )
        return None

    def _train_single(
        self,
        yolo_cls: type,
        weights: str,
        data_yaml: str,
        project_dir: str,
        run_name: str,
        epochs: int,
        freeze_n: int,
    ) -> str | None:
        """Single YOLO training call."""
        if epochs <= 0:
            return None

        model = yolo_cls(weights)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            project=project_dir,
            name=run_name,
            workers=self.config.workers,
            freeze=freeze_n,
            exist_ok=True,
            pretrained=True,
            verbose=False,
        )
        best = os.path.join(project_dir, run_name, "weights/best.pt")
        return best if os.path.exists(best) else None

    def _log_per_class(
        self,
        metrics: Any,
        dataset: str,
        model_name: str,
        regime: str,
        class_names: list[str],
    ) -> None:
        """Append per-class P/R/mAP to a CSV."""
        rows = []
        for i, name in enumerate(class_names):
            if i < len(metrics.box.p):
                rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model_name,
                        "Regime": regime,
                        "Class": name,
                        "Precision": round(float(metrics.box.p[i]), 4),
                        "Recall": round(float(metrics.box.r[i]), 4),
                        "mAP50-95": round(float(metrics.box.maps[i]), 4),
                    }
                )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.config.output_dir, "per_class_metrics.csv")
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=write_header, index=False)

    @staticmethod
    def _link_images(src_dir: str, dst_dir: str) -> None:
        """Copy images from src to dst if not already present."""
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith((".jpg", ".png", ".jpeg")):
                tgt = os.path.join(dst_dir, f)
                if not os.path.exists(tgt):
                    shutil.copy(os.path.join(src_dir, f), tgt)
