#!/usr/bin/env python
"""Quick test of Siamese U-Net CD inference."""
from pathlib import Path
import sys

# Add launcher to path
sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner

def main():
    test_dir = Path("datasets/TACO/cd_pairs/test")
    before = test_dir / "before" / "test__img000013__pos__before.jpg"
    after = test_dir / "after" / "test__img000013__pos__after.jpg"
    
    if not before.exists() or not after.exists():
        print(f"Test images not found:")
        print(f"  Before: {before} exists={before.exists()}")
        print(f"  After: {after} exists={after.exists()}")
        return 1
    
    print(f"Running inference on:")
    print(f"  Before: {before}")
    print(f"  After: {after}")
    
    runner = SiameseUnetCdRunner("siamese_unet_cd", device="auto")
    result = runner.analyze(str(before), str(after))
    
    print(f"\nResult:")
    print(f"  Method: {result.method_name}")
    print(f"  change_ratio: {result.metrics.get('change_ratio', 'N/A')}")
    print(f"  Preview saved to: {result.preview_image_path}")
    print(f"  Summary: {result.summary}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
