#!/usr/bin/env python
"""Smoke test for separate hybrid methods: Siamese+DINOv2 and ChangeFormer+DINOv2."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from launcher.runners import create_runner


TEST_PAIR_DIR = Path(r"C:\Coding\AI\Dataset\1")


def _resolve_pair() -> tuple[Path, Path]:
    before = TEST_PAIR_DIR / "before.png"
    for after_name in ("after.png", "AI_after.png"):
        after = TEST_PAIR_DIR / after_name
        if before.exists() and after.exists():
            return before, after
    return before, TEST_PAIR_DIR / "after.png"


def main() -> int:
    before, after = _resolve_pair()

    if not before.exists() or not after.exists():
        print("Test images not found")
        print(f"  before: {before} exists={before.exists()}")
        print(f"  after: {after} exists={after.exists()}")
        return 1

    for method_id in ("siamese_dinov2", "changeformer_dinov2"):
        print(f"Running method: {method_id}")
        runner = create_runner(method_id, device="auto")
        result = runner.analyze(before, after)
        print(f"  method_name: {result.method_name}")
        print(f"  preview: {result.preview_image_path}")
        print(f"  analysis_mode: {result.metrics.get('analysis_mode')}")
        if method_id == "siamese_dinov2":
            if result.metrics.get("analysis_mode") != "siamese_dinov2_pair":
                raise RuntimeError("Unexpected analysis mode for siamese_dinov2")
        if method_id == "changeformer_dinov2":
            if result.metrics.get("analysis_mode") != "changeformer_dinov2_pair":
                raise RuntimeError("Unexpected analysis mode for changeformer_dinov2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
