#!/usr/bin/env python
"""Test ECC alignment fix for Siamese U-Net CD."""
from pathlib import Path
import sys
import csv
import argparse
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner


TEST_DATASET_ROOT = Path(r"C:\Coding\AI\Dataset")


def _discover_pairs(test_dir: Path) -> list[tuple[Path, Path]]:
    pair_dirs = [path for path in sorted(test_dir.iterdir()) if path.is_dir()]
    pairs: list[tuple[Path, Path]] = []

    for pair_dir in pair_dirs:
        if (pair_dir / "before.png").exists() and (pair_dir / "after.png").exists():
            pairs.append((pair_dir / "before.png", pair_dir / "after.png"))
            continue
        if (pair_dir / "AI_before.png").exists() and (pair_dir / "AI_after.png").exists():
            pairs.append((pair_dir / "AI_before.png", pair_dir / "AI_after.png"))
            continue

        before_dir = pair_dir / "before"
        after_dir = pair_dir / "after"
        if before_dir.exists() and after_dir.exists():
            before_files = sorted([p for p in before_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
            for before_path in before_files:
                if "__before" in before_path.name:
                    after_path = after_dir / before_path.name.replace("__before", "__after")
                else:
                    after_path = after_dir / before_path.name
                if after_path.exists():
                    pairs.append((before_path, after_path))

    return pairs


def run_dataset(test_dir: Path, out_csv: Path, min_overlap: float = 0.5):
    runner = SiameseUnetCdRunner("siamese_unet_cd", device="auto")
    pair_paths = _discover_pairs(test_dir)

    out_rows = []
    for before_path, after_path in pair_paths:

        try:
            result = runner.analyze(str(before_path), str(after_path))
            metrics = result.metrics
            change_ratio = float(metrics.get("change_ratio", 0.0))
            overlap_ratio = float(metrics.get("overlap_ratio", 0.0))
            alignment_mode = metrics.get("alignment_mode", "unknown")
            row = {
                "pair": before_path.stem,
                "before": str(before_path),
                "after": str(after_path),
                "alignment_mode": alignment_mode,
                "overlap_ratio": overlap_ratio,
                "change_ratio": change_ratio,
                "change_pixels": int(metrics.get("change_pixels", 0)),
                "overlap_pixels": int(metrics.get("overlap_pixels", 0)),
                "preview": str(result.preview_image_path),
            }
            out_rows.append(row)
            print(f"Processed: {before_path.name} | align={alignment_mode} | overlap={overlap_ratio:.4f} | change={change_ratio:.6f}")

        except Exception as e:
            print(f"Error processing {before_path.name}: {e}")

    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "before", "after", "alignment_mode", "overlap_ratio", "change_ratio", "change_pixels", "overlap_pixels", "preview"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    # summary stats
    total = len(out_rows)
    high_changes = [r for r in out_rows if r["change_ratio"] > 0.01]
    print("\nDataset run complete")
    print(f"Total pairs processed: {total}")
    print(f"Pairs with change_ratio > 0.01: {len(high_changes)}")
    return out_rows


def main():
    p = argparse.ArgumentParser(description="Run ECC alignment + Siamese U-Net tests over test dataset")
    p.add_argument("--test-dir", default=str(TEST_DATASET_ROOT), help="Path to test dataset folder")
    p.add_argument("--out-csv", default="results/analysis/siamese_unet/test_summary_$(date).csv", help="Output CSV path")
    args = p.parse_args()

    test_dir = Path(args.test_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(f"results/analysis/siamese_unet/test_summary_{timestamp}.csv")

    run_dataset(test_dir, out_csv)


if __name__ == "__main__":
    main()
