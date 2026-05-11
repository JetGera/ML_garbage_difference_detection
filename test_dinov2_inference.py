#!/usr/bin/env python
"""Quick smoke test for DINOv2 cleanup-degree inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.dinov2_cd import DinoV2CdRunner


def main() -> int:
    test_dir = Path("datasets/TACO/cd_pairs/test")
    before = test_dir / "before" / "test__img000013__pos__before.jpg"
    after = test_dir / "after" / "test__img000013__pos__after.jpg"

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
    print(f"  cleanup_percent: {result.metrics.get('cleanup_percent', 'N/A')}")
    print(f"  confidence: {result.metrics.get('cleanup_confidence', 'N/A')}")
    print(f"  semantic_change_ratio: {result.metrics.get('semantic_change_ratio', 'N/A')}")
    print(f"  Preview saved to: {result.preview_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
