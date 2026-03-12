"""detbench: Benchmarking toolkit for object detection experiments.

Provides reproducible training, evaluation, and analysis workflows for
comparing detection models across datasets, training regimes, and loss
functions.  Generates publication-ready metrics, error analysis, and
data-scaling curves.

Typical usage::

    from detbench import BenchmarkRunner, BenchmarkConfig

    runner = BenchmarkRunner(BenchmarkConfig(
        train_sets={"clean": ("images/", "train.json")},
        val_images="val_images/",
        val_json="val.json",
    ))
    runner.run()
"""

from __future__ import annotations

__version__ = "0.1.0"
