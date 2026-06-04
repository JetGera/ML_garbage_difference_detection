"""Run YOLOv8m-seg training on TACO with memory-conservative settings.
Creates a training run under datasets/TACO/yolo_seg/runs and writes logs/weights there.
"""
import traceback
from pathlib import Path
import sys
import os

# Ensure repository root is on sys.path so `launcher` package imports work when
# the script is executed directly (not as a module).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from launcher.method_scripts.yolov8_seg import YoloV8SegRunner
except Exception:
    # allow running when package layout is different
    from method_scripts.yolov8_seg import YoloV8SegRunner


def main():
    # Create runner for the configured method
    runner = YoloV8SegRunner(method_id="yolov8_seg", device="auto", force_cpu=False, model_name="yolov8m-seg.pt")

    # Training settings tuned for 8GB-ish GPUs: small imgsz, batch=1, use AMP and accumulate
    epochs = 500  # large number per request
    batch = 1
    imgsz = 640
    accumulate = 4  # simulate larger batch

    print(f"Starting YOLOv8m-seg training on TACO: epochs={epochs}, imgsz={imgsz}, batch={batch}, accumulate={accumulate}")
    try:
        result = runner.train_on_taco(
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=None,  # let runner decide (auto)
            patience=100,
            workers=2,
            cache="disk",
            amp=True,
            optimizer=None,
            lr0=None,
            weight_decay=None,
            allow_oom_fallback=True,
        )
        print("Training finished. Result summary:")
        print(result)
    except Exception as exc:
        print("Training failed with exception:\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()
