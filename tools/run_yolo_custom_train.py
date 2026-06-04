"""Fine-tune YOLOv8m-seg on the custom masks dataset."""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path so `launcher` package imports work when
# the script is executed directly (not as a module).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def main() -> None:
    data_yaml = REPO_ROOT / "datasets" / "custom" / "masks_for_training_yolo" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Prefer the latest TACO fine-tuned weights if available; otherwise fallback to the base model.
    candidate_weights = [
        REPO_ROOT / "datasets" / "TACO" / "yolo_seg" / "runs" / "yolov8m_seg_taco" / "weights" / "best.pt",
        REPO_ROOT / "datasets" / "TACO" / "yolo_seg" / "weights" / "best.pt",
        REPO_ROOT / "weights" / "yolov8_seg_best.pt",
        Path("yolov8m-seg.pt"),
    ]
    model_weights = next((p for p in candidate_weights if p.exists()), Path("yolov8m-seg.pt"))
    model = YOLO(str(model_weights))
    model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=640,
        batch=1,
        device=0,
        amp=True,
        project=str((REPO_ROOT / "datasets" / "custom" / "masks_for_training_yolo" / "runs")),
        name="yolov8m_seg_custom",
        patience=50,
    )


if __name__ == "__main__":
    main()
