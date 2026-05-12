"""Quick test of the tuned SIFT+RANSAC settings on bottle cap pair."""
import argparse
import sys
from pathlib import Path

# Add launcher to path for imports
sys.path.insert(0, str(Path(__file__).parent / "launcher"))

from core import list_image_files, select_before_after
from method_scripts.sift_ransac import SiftRansacRunner

DEFAULT_PAIRS_ROOT = Path(__file__).parent / "datasets" / "custom" / "cd_pairs_17_yolo" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SIFT+RANSAC smoke test on your custom before/after dataset.")
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=DEFAULT_PAIRS_ROOT,
        help="Dataset split folder containing before/ and after/ directories.",
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        default="",
        help="Optional pair id (for example 'test__pair_1') to select a specific pair from before/after.",
    )
    parser.add_argument(
        "--pair-dir",
        type=Path,
        default=None,
        help="Optional folder with mixed images; before/after will be auto-selected from filenames.",
    )
    parser.add_argument("--before", type=Path, default=None, help="Optional explicit path to before image.")
    parser.add_argument("--after", type=Path, default=None, help="Optional explicit path to after image.")
    return parser.parse_args()


def normalize_pair_key(stem: str) -> str:
    key = stem.casefold()
    for suffix in ("__before", "__after", "_before", "_after", "before", "after"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.strip("_- ")


def resolve_pair_from_split(pairs_root: Path, pair_id: str) -> tuple[Path, Path]:
    before_dir = pairs_root / "before"
    after_dir = pairs_root / "after"
    if not before_dir.exists() or not after_dir.exists():
        raise ValueError(f"Expected before/ and after/ directories under: {pairs_root}")

    before_files = sorted(path for path in before_dir.iterdir() if path.is_file())
    after_files = sorted(path for path in after_dir.iterdir() if path.is_file())
    if not before_files or not after_files:
        raise ValueError(f"No files found under {before_dir} or {after_dir}")

    before_map = {normalize_pair_key(path.stem): path for path in before_files}
    after_map = {normalize_pair_key(path.stem): path for path in after_files}
    common_ids = sorted(set(before_map) & set(after_map))
    if not common_ids:
        raise ValueError(f"No matching pair ids between {before_dir} and {after_dir}")

    selected_id = pair_id.casefold().strip()
    if selected_id:
        if selected_id not in before_map or selected_id not in after_map:
            raise ValueError(f"Pair id '{pair_id}' not found. Available: {', '.join(common_ids)}")
        return before_map[selected_id], after_map[selected_id]

    first_id = common_ids[0]
    return before_map[first_id], after_map[first_id]


def resolve_input_pair(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.before and args.after:
        return args.before, args.after

    if args.pair_dir is not None:
        files = list_image_files(args.pair_dir)
        before_path, after_path = select_before_after(files)
        return before_path, after_path

    return resolve_pair_from_split(args.pairs_root, args.pair_id)


args = parse_args()
before_path, after_path = resolve_input_pair(args)

if not before_path.exists() or not after_path.exists():
    print("Resolved test images do not exist:")
    print(f"  before: {before_path}")
    print(f"  after:  {after_path}")
    sys.exit(1)

runner = SiftRansacRunner("sift_ransac")
result = runner.analyze(str(before_path), str(after_path))

print("\n" + "="*60)
print(f"Method: {result.method_name}")
print("="*60)
print(f"Before: {before_path}")
print(f"After:  {after_path}")
print(f"Alignment mode: {result.metrics['alignment_mode']}")
print(f"Good matches: {result.metrics['good_matches']}")
print(f"Inliers: {result.metrics['inliers']}")
print(f"Change ratio: {result.metrics['change_ratio']:.6f}")
print(f"Overlap ratio: {result.metrics['overlap_ratio']:.4f}")
print(f"\nPreview saved to: {result.preview_image_path}")
print(f"All artifacts saved to: {result.artifacts}")
