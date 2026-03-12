"""Command-line interface for detbench.

Usage
-----
.. code-block:: bash

    # Convert COCO annotations to YOLO format
    detbench convert --json train.json --output-labels labels/train

    # Run a benchmark sweep (requires a YAML config file)
    detbench run --config benchmark.yaml
"""

from __future__ import annotations

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detbench",
        description=(
            "Benchmarking toolkit for object detection — train, evaluate, "
            "and compare models across datasets and training regimes."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- convert ---
    p_conv = sub.add_parser(
        "convert",
        help="Convert COCO annotations to YOLO label format.",
    )
    p_conv.add_argument(
        "--json",
        required=True,
        help="Input COCO annotation JSON.",
    )
    p_conv.add_argument(
        "--output-labels",
        required=True,
        help="Output labels directory.",
    )
    p_conv.add_argument(
        "--category-offset",
        type=int,
        default=1,
        help="Offset subtracted from category IDs (default: 1).",
    )

    # --- losses ---
    p_loss = sub.add_parser(
        "losses",
        help="List all available loss functions.",
    )

    # --- Shared ---
    for p in [p_conv, p_loss]:
        p.add_argument(
            "--verbose",
            action="store_true",
            help="Debug logging.",
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "convert":
        from detbench.converter import convert_coco_to_yolo

        class_names = convert_coco_to_yolo(
            args.json,
            args.output_labels,
            category_offset=args.category_offset,
        )
        print(f"Converted {len(class_names)} classes: {class_names}")

    elif args.command == "losses":
        from detbench.losses import LOSS_REGISTRY

        print("Available loss functions:")
        for key, desc in LOSS_REGISTRY.items():
            print(f"  {key:12s}  {desc}")


if __name__ == "__main__":
    main()
