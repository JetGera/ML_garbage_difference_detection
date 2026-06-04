"""Build Siamese change detection pairs from custom before/after/mask folders.

Expected source structure:
  <src_root>\<folder>\before (n).png|jpg
  <src_root>\<folder>\after (n).png|jpg
  <src_root>\<folder>\mask (n).png|jpg

Pairing rule:
- For each before(n) with mask(n), pair with every after(m) in the same folder.
- If a mask is missing for before(n), that before is skipped.

Output structure:
  <out_root>\index.csv
  <out_root>\train|val|test\before|after|masks\<pair_id>__before.jpg
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    split: str
    pair_id: str
    before_path: Path
    after_path: Path
    mask_path: Path
    source_folder: str
    source_before: str
    source_after: str
    mask_area_px: int


def _parse_numbered_name(path: Path) -> tuple[str, str | None]:
    stem = path.stem.casefold()
    if "(" in stem and ")" in stem:
        prefix = stem.split("(")[0].strip()
        num = stem.split("(")[1].split(")")[0].strip()
        return prefix, num if num.isdigit() else None
    return stem, None


def _list_images(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS]


def _deterministic_split(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _write_mask(path: Path, mask: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), binary)
    return int(np.count_nonzero(binary))


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src), str(dst))


def _build_pairs(
    src_root: Path,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    limit_folders: int | None,
    limit_pairs: int | None,
    dry_run: bool,
) -> tuple[list[PairRecord], dict[str, object]]:
    rng = random.Random(seed)
    folders = [p for p in sorted(src_root.iterdir()) if p.is_dir()]
    if limit_folders:
        folders = folders[: int(limit_folders)]

    all_pairs: list[PairRecord] = []
    stats = {
        "folders": len(folders),
        "pairs_total": 0,
        "pairs_by_split": {"train": 0, "val": 0, "test": 0},
        "mask_pixels": 0,
        "mask_area_max": 0,
    }

    for folder in folders:
        images = _list_images(folder)
        if not images:
            continue

        befores: dict[str | None, Path] = {}
        afters: dict[str | None, Path] = {}
        masks: dict[str | None, Path] = {}

        for img in images:
            prefix, num = _parse_numbered_name(img)
            if prefix.startswith("before"):
                befores[num] = img
            elif prefix.startswith("after"):
                afters[num] = img
            elif prefix.startswith("mask"):
                masks[num] = img

        if not befores or not afters:
            continue

        default_after: Path | None = None
        if len(afters) == 1:
            default_after = next(iter(afters.values()))

        for before_num, before_path in befores.items():
            mask_path = masks.get(before_num)
            if mask_path is None:
                continue

            after_path = afters.get(before_num) or default_after
            if after_path is None:
                continue

            key = f"{folder.name}|{before_path.name}|{after_path.name}"
            split = _deterministic_split(key, train_ratio, val_ratio)

            before_tag = _safe_stem(before_path.stem)
            after_tag = _safe_stem(after_path.stem)
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
            pair_id = f"{split}__{folder.name}__{before_tag}__{after_tag}__{digest}"

            before_out = out_root / split / "before" / f"{pair_id}__before{before_path.suffix.casefold()}"
            after_out = out_root / split / "after" / f"{pair_id}__after{after_path.suffix.casefold()}"
            mask_out = out_root / split / "masks" / f"{pair_id}__mask.png"

            if not dry_run:
                _copy_image(before_path, before_out)
                _copy_image(after_path, after_out)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask_area_px = _write_mask(mask_out, mask)
            else:
                mask_area_px = 0

            record = PairRecord(
                split=split,
                pair_id=pair_id,
                before_path=before_out,
                after_path=after_out,
                mask_path=mask_out,
                source_folder=folder.name,
                source_before=before_path.name,
                source_after=after_path.name,
                mask_area_px=mask_area_px,
            )
            all_pairs.append(record)

            stats["pairs_total"] += 1
            stats["pairs_by_split"][split] += 1
            stats["mask_pixels"] += mask_area_px
            stats["mask_area_max"] = max(stats["mask_area_max"], mask_area_px)

            if limit_pairs and len(all_pairs) >= int(limit_pairs):
                return all_pairs, stats

    rng.shuffle(all_pairs)
    return all_pairs, stats


def _write_index_csv(index_path: Path, pairs: Iterable[PairRecord]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "pair_id",
        "label_type",
        "before_path",
        "after_path",
        "mask_path",
        "source_image_id",
        "source_file_name",
        "mask_area_px",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "split": pair.split,
                    "pair_id": pair.pair_id,
                    "label_type": "positive",
                    "before_path": str(pair.before_path),
                    "after_path": str(pair.after_path),
                    "mask_path": str(pair.mask_path),
                    "source_image_id": pair.source_folder,
                    "source_file_name": f"{pair.source_before}|{pair.source_after}",
                    "mask_area_px": str(pair.mask_area_px),
                }
            )


def _signature(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Siamese CD pairs from custom masks dataset.")
    parser.add_argument("--src-root", type=Path, required=True, help="Source root with before/after/mask folders.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for cd_pairs.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-folders", type=int, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.train + args.val >= 1.0:
        raise ValueError("Train + val must be < 1.0")

    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source root not found: {src_root}")

    if not args.dry_run:
        for split in ("train", "val", "test"):
            (out_root / split / "before").mkdir(parents=True, exist_ok=True)
            (out_root / split / "after").mkdir(parents=True, exist_ok=True)
            (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    pairs, stats = _build_pairs(
        src_root=src_root,
        out_root=out_root,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
        limit_folders=args.limit_folders,
        limit_pairs=args.limit_pairs,
        dry_run=args.dry_run,
    )

    if not pairs:
        raise RuntimeError("No valid pairs were generated. Check the dataset layout.")

    if not args.dry_run:
        index_path = out_root / "index.csv"
        _write_index_csv(index_path, pairs)

        stats["index_csv"] = str(index_path)
        stats["source_root"] = str(src_root)
        stats["signature"] = _signature(stats)
        metadata_path = out_root / "metadata.json"
        metadata_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Wrote {len(pairs)} pairs to {out_root}")
        print(f"Index: {index_path}")
    else:
        print(f"Dry run: would write {len(pairs)} pairs to {out_root}")


if __name__ == "__main__":
    main()
