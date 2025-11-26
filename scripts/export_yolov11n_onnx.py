"""Utility script to export YOLOv11 weights to ONNX.

Uses ultralytics.YOLO to convert the provided .pt weights into an ONNX model
that can be consumed by the backend service.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is required. Install it via `pip install ultralytics`."
    ) from exc


def export_to_onnx(weights: Path, img_size: int, output: Path | None, dynamic: bool) -> Path:
    """Export the given weights to ONNX and return the final path."""
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    model = YOLO(str(weights))
    print(f"Loaded weights: {weights}")

    onnx_temp = Path(
        model.export(format="onnx", imgsz=img_size, dynamic=dynamic, simplify=True)
    )
    print(f"Temporary ONNX path: {onnx_temp}")

    if output is None:
        output = Path("models/weights") / f"{weights.stem}.onnx"

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(onnx_temp), output)
    print(f"Saved ONNX model to: {output.resolve()}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    if size_mb > 80:
        print("Warning: model size exceeds 80 MB target.")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv11 to ONNX")
    parser.add_argument("--weights", default="yolo11n.pt", help="Path to .pt weights")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--output", help="Optional output path for the .onnx file")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic axes for height/width/batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    output = Path(args.output) if args.output else None
    export_to_onnx(weights, args.img_size, output, args.dynamic)


if __name__ == "__main__":
    main()
