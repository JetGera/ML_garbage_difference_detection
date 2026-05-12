#!/usr/bin/env python
"""Smoke test for Siamese U-Net inference with DINOv2 semantic hint map."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.dinov2_cd import DinoV2CdRunner
from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner


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
        print("Test images not found:")
        print(f"  Before: {before} exists={before.exists()}")
        print(f"  After: {after} exists={after.exists()}")
        return 1

    print("Building DINOv2 semantic hint:")
    dino_runner = DinoV2CdRunner("dinov2_cd", device="auto")
    dinov2_map, dino_meta = dino_runner.predict_probability_map(before, after)

    print(f"  model_source: {dino_meta.get('model_source')}")
    print(f"  fallback_reason: {dino_meta.get('fallback_reason')}")
    print(f"  torch_import_ok: {dino_meta.get('torch_import_ok')}")
    print(f"  timm_import_ok: {dino_meta.get('timm_import_ok')}")

    if dino_meta.get("fallback_reason") is not None:
        raise RuntimeError(f"DINOv2 hint generation returned fallback_reason={dino_meta.get('fallback_reason')!r}")
    if dino_meta.get("torch_import_ok") is False or dino_meta.get("timm_import_ok") is False:
        raise RuntimeError("DINOv2 hint generation reports missing torch/timm")

    print("Running Siamese U-Net with DINOv2 hint map")
    siamese_runner = SiameseUnetCdRunner("siamese_unet_cd", device="auto")
    result = siamese_runner.analyze(str(before), str(after), dinov2_map=dinov2_map)

    print("Result:")
    print(f"  method: {result.method_name}")
    print(f"  model_source: {result.metrics.get('model_source')}")
    print(f"  dinov2_hint_used: {result.metrics.get('dinov2_hint_used')}")
    print(f"  change_ratio: {result.metrics.get('change_ratio')}")
    print(f"  preview: {result.preview_image_path}")

    if result.metrics.get("dinov2_hint_used") is not True:
        raise RuntimeError("Siamese result did not mark dinov2_hint_used=True")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
