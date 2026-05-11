"""Quick test of the tuned SIFT+RANSAC settings on bottle cap pair."""
import sys
from pathlib import Path

# Add launcher to path for imports
sys.path.insert(0, str(Path(__file__).parent / "launcher"))

from method_scripts.sift_ransac import SiftRansacRunner

# Test with TACO dataset images
before_path = r"C:\Coding\AI\projekt\datasets\TACO\cd_pairs\test\before\test__img000013__pos__before.jpg"
after_path = r"C:\Coding\AI\projekt\datasets\TACO\cd_pairs\test\after\test__img000013__pos__after.jpg"

# Check if test images exist
if not Path(before_path).exists() or not Path(after_path).exists():
    print("Test images not found. Expected:")
    print(f"  {before_path}")
    print(f"  {after_path}")
    print("\nUsing TACO dataset pair for testing.")
    sys.exit(1)

runner = SiftRansacRunner("01")
result = runner.analyze(before_path, after_path)

print("\n" + "="*60)
print(f"Method: {result.method_name}")
print("="*60)
print(f"Alignment mode: {result.metrics['alignment_mode']}")
print(f"Good matches: {result.metrics['good_matches']}")
print(f"Inliers: {result.metrics['inliers']}")
print(f"Change ratio: {result.metrics['change_ratio']:.6f}")
print(f"Overlap ratio: {result.metrics['overlap_ratio']:.4f}")
print(f"\nPreview saved to: {result.preview_image_path}")
print(f"All artifacts saved to: {result.artifacts}")
