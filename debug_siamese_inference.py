#!/usr/bin/env python
"""Debug Siamese U-Net CD inference to identify alignment issues."""
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner


TEST_PAIR_DIR = Path(r"C:\Coding\AI\Dataset\1")


def _resolve_pair() -> tuple[Path, Path]:
    before = TEST_PAIR_DIR / "before.png"
    for after_name in ("after.png", "AI_after.png"):
        after = TEST_PAIR_DIR / after_name
        if before.exists() and after.exists():
            return before, after
    return before, TEST_PAIR_DIR / "after.png"

def main():
    before_path, after_path = _resolve_pair()
    
    before_img = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after_img = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    
    print(f"Before shape: {before_img.shape}")
    print(f"After shape: {after_img.shape}")
    print(f"Images are identical: {np.array_equal(before_img, after_img)}")
    
    # Check basic image difference
    if before_img.shape == after_img.shape:
        diff = cv2.absdiff(before_img, after_img)
        diff_ratio = np.count_nonzero(diff) / diff.size
        print(f"Pixel difference ratio: {diff_ratio:.4f}")
    
    runner = SiameseUnetCdRunner("siamese_unet_cd", device="auto")
    result = runner.analyze(str(before_path), str(after_path))
    
    print(f"\nResult:")
    print(f"  change_ratio: {result.metrics.get('change_ratio', 'N/A')}")
    print(f"  change_pixels: {result.metrics.get('change_pixels', 'N/A')}")
    print(f"  overlap_pixels: {result.metrics.get('overlap_pixels', 'N/A')}")
    print(f"  alignment_mode: {result.metrics.get('alignment_mode', 'N/A')}")
    print(f"  threshold: {result.metrics.get('threshold', 'N/A')}")
    print(f"\nPreview: {result.preview_image_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
