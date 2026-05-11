#!/usr/bin/env python
"""Diagnose raw model predictions without thresholding."""
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner

def main():
    test_dir = Path("datasets/TACO/cd_pairs/test")
    before_path = test_dir / "before" / "test__img000013__pos__before.jpg"
    after_path = test_dir / "after" / "test__img000013__pos__after.jpg"
    
    runner = SiameseUnetCdRunner("siamese_unet_cd", device="auto")
    
    # Hack: directly get predictions without thresholding
    before_img = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after_img = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    
    # Internal method call to get raw prediction
    aligned_before, aligned_after, overlap_mask, alignment_mode = runner._align_after_to_before(before_img, after_img)
    prediction = runner._predict_change_map(aligned_before, aligned_after)
    probability_map = prediction.probability_map
    
    print("=== Raw Predictions Diagnosis ===")
    print(f"Probability map shape: {probability_map.shape}")
    print(f"Min: {probability_map.min():.4f}")
    print(f"Max: {probability_map.max():.4f}")
    print(f"Mean: {probability_map.mean():.4f}")
    print(f"Std: {probability_map.std():.4f}")
    print(f"Median: {np.median(probability_map):.4f}")
    print(f"Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}%: {np.percentile(probability_map, p):.4f}")
    
    print(f"\nThreshold used: {prediction.threshold_value:.3f}")
    print(f"Pixels above 0.3: {(probability_map >= 0.3).sum()}")
    print(f"Pixels above 0.4: {(probability_map >= 0.4).sum()}")
    print(f"Pixels above 0.45: {(probability_map >= 0.45).sum()}")
    print(f"Pixels above 0.5: {(probability_map >= 0.5).sum()}")
    print(f"Pixels above 0.6: {(probability_map >= 0.6).sum()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
