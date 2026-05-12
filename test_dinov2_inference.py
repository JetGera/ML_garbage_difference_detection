#!/usr/bin/env python
"""Quick smoke test for DINOv2 cleanup-degree inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.dinov2_cd import DinoV2CdRunner


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

    print("Running DINOv2 inference on:")
    print(f"  Before: {before}")
    print(f"  After: {after}")

    runner = DinoV2CdRunner("dinov2_cd", device="auto")
    result = runner.analyze(str(before), str(after))

    print("\nResult:")
    print(f"  Method: {result.method_name}")
    print(f"  model_source: {result.metrics.get('model_source', 'N/A')}")
    print(f"  fallback_reason: {result.metrics.get('fallback_reason', 'N/A')}")
    print(f"  torch_import_ok: {result.metrics.get('torch_import_ok', 'N/A')}")
    print(f"  timm_import_ok: {result.metrics.get('timm_import_ok', 'N/A')}")
    print(f"  cleanup_percent: {result.metrics.get('cleanup_percent', 'N/A')}")
    print(f"  confidence: {result.metrics.get('cleanup_confidence', 'N/A')}")
    print(f"  semantic_change_ratio: {result.metrics.get('semantic_change_ratio', 'N/A')}")
    print(f"  Preview saved to: {result.preview_image_path}")

    if result.metrics.get('model_source') == 'cv_fallback':
        raise RuntimeError('DINOv2 smoke test unexpectedly used cv_fallback')
    if result.metrics.get('fallback_reason') is not None:
        raise RuntimeError(f"DINOv2 smoke test returned fallback_reason={result.metrics.get('fallback_reason')!r}")
    if result.metrics.get('torch_import_ok') is False or result.metrics.get('timm_import_ok') is False:
        raise RuntimeError('DINOv2 smoke test reports missing torch/timm in a successful run')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
