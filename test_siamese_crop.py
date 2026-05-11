#!/usr/bin/env python3
"""Quick test of Siamese U-Net CD with cropping fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner

test_dir = Path(__file__).parent / "datasets" / "TACO" / "cd_pairs" / "test"
before_files = sorted((test_dir / "before").glob("*.jpg"))[:1]

if before_files:
    before_file = before_files[0]
    after_file = test_dir / "after" / before_file.name.replace("__before", "__after")
    
    if before_file.exists() and after_file.exists():
        print(f"Testing: {before_file.name}")
        runner = SiameseUnetCdRunner("siamese_unet_cd")
        result = runner.analyze(str(before_file), str(after_file))
        print(f"✓ Analysis complete")
        print(f"  Change pixels: {result.metrics.get('change_pixels', 'N/A')}")
        print(f"  Overlap pixels: {result.metrics.get('overlap_pixels', 'N/A')}")
        print(f"  Preview: {result.preview_image_path}")
    else:
        print(f"✗ Test pair files missing")
        print(f"  Before: {before_file.exists()}: {before_file}")
        print(f"  After: {after_file.exists()}: {after_file}")
else:
    print("✗ No test pairs found")
