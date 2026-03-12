<div align="center">

# 📊 detbench

**Benchmarking Toolkit for Object Detection Experiments**

*Reproducible multi-dataset, multi-model, multi-regime training sweeps.*

[![CI](https://github.com/PhotonFlow/detbench/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonFlow/detbench/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The Problem

Detection research involves comparing many variables: datasets, architectures, training regimes, loss functions.  Running these comparisons manually is error-prone and hard to reproduce.

## The Solution

**detbench** automates end-to-end benchmark sweeps: train → evaluate → analyse → report.  One config, one command, publication-ready results.

```
              ┌────────────────────────────────┐
              │      Benchmark Configuration   │
              │  datasets × models × regimes   │
              └────────────┬───────────────────┘
                           │
              ┌────────────▼───────────────────┐
              │  1. COCO → YOLO Converter      │
              │     Annotations + dataset.yaml  │
              └────────────┬───────────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         │                 │                     │
  ┌──────▼─────┐   ┌──────▼──────┐   ┌──────────▼──────┐
  │ Full Fine- │   │ Classifier  │   │ LP → Fine-Tune  │
  │ Tune       │   │ Only (Head) │   │ (Two-Stage)     │
  │ freeze=0   │   │ freeze=10   │   │ freeze=10→0     │
  └──────┬─────┘   └──────┬──────┘   └──────────┬──────┘
         │                │                     │
         └─────────────────┼─────────────────────┘
                           │
              ┌────────────▼───────────────────┐
              │  2. Evaluation & Analysis      │
              │  • Per-class P/R/mAP           │
              │  • FP / FN error images         │
              │  • Training curves              │
              └────────────┬───────────────────┘
                           │
              ┌────────────▼───────────────────┐
              │  3. Summary CSV + Reports      │
              │  overall_benchmark_summary.csv  │
              │  per_class_metrics.csv          │
              └────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **Multi-Sweep** | Sweep datasets × models × regimes in one run |
| **3 Training Regimes** | Full fine-tune, classifier-only, LP→FT (two-stage) |
| **9 Noise-Robust Losses** | CE, FL, GCE, SCE, MAE, NCE+MAE, NCE+RCE, NCE+AGCE, ANL-FL |
| **Error Analysis** | Visual FP/FN diagnostics with IoU-based classification |
| **COCO → YOLO** | Built-in format converter |
| **Per-Class Metrics** | Precision, Recall, mAP50-95 logged per class |

## Installation

```bash
pip install detbench
```

**With YOLO training support:**
```bash
pip install "detbench[yolo]"
```

**From source:**
```bash
git clone https://github.com/PhotonFlow/detbench.git
cd detbench
pip install -e ".[dev,yolo]"
```

## Quickstart

### CLI

```bash
# Convert COCO annotations to YOLO format
detbench convert --json train.json --output-labels labels/train

# List available noise-robust loss functions
detbench losses
```

### Python API — Benchmark Sweep

```python
from detbench.runner import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    train_sets={
        "raw": {"root": "images/train", "json": "train.json"},
        "cleaned": {"root": "images/clean", "json": "train_clean.json"},
    },
    val_root="images/val",
    val_json="val.json",
    models={"YOLOv8n": "yolov8n.pt", "YOLOv8s": "yolov8s.pt"},
    regimes=["Full_FT", "Classifier_Only"],
    epochs=50,
)
summary_df = BenchmarkRunner(config).run()
print(summary_df)
```

### Loss Functions

```python
from detbench.losses import get_loss, LOSS_REGISTRY

# List available losses
for key, desc in LOSS_REGISTRY.items():
    print(f"{key}: {desc}")

# Use in a training loop
criterion = get_loss("FL", num_classes=5)  # Focal Loss
loss = criterion(logits, targets)
```

## Noise-Robust Loss Functions

| Loss | Reference | Key Idea |
|---|---|---|
| **CE** | — | Standard baseline |
| **FL** | Lin et al., ICCV 2017 | Down-weight easy examples |
| **GCE** | Zhang & Sabuncu, NeurIPS 2018 | Noise-robust CE variant (q-parameter) |
| **SCE** | Wang et al., ICCV 2019 | CE + Reverse CE for symmetry |
| **MAE** | Ghosh et al., AAAI 2017 | Inherently noise-tolerant |
| **NCE+MAE** | Ma et al., ICML 2020 | Active-Passive combination |
| **NCE+RCE** | Ma et al., ICML 2020 | Normalised CE + Reverse CE |
| **ANL-FL** | Ma et al., ICML 2020 | Focal + Reverse CE |

## Project Structure

```
detbench/
├── src/detbench/
│   ├── __init__.py      # Package init
│   ├── cli.py           # CLI with subcommands
│   ├── runner.py        # Benchmark orchestrator
│   ├── converter.py     # COCO → YOLO converter
│   ├── losses.py        # 9 noise-robust loss functions
│   ├── dataset.py       # COCO crop Dataset
│   └── analysis.py      # FP / FN error visualiser
├── tests/               # Unit tests (pytest)
├── pyproject.toml       # Package configuration
└── .github/workflows/   # CI pipeline
```

## References

- **Lin et al.** "Focal Loss for Dense Object Detection." *ICCV 2017*.
- **Ghosh et al.** "Making Risk Minimization Tolerant to Label Noise." *AAAI 2017*.
- **Zhang & Sabuncu.** "Generalised Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels." *NeurIPS 2018*.
- **Wang et al.** "Symmetric Cross Entropy for Robust Learning with Noisy Labels." *ICCV 2019*.
- **Ma et al.** "Normalised Loss Functions: Unifying PPL with Active Passive Losses." *ICML 2020*.

## License

[MIT](LICENSE)
