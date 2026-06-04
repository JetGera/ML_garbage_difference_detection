"""Build a YOLOv8-seg dataset from custom before/mask pairs.

Expected source structure:
  <src_root>\<folder>\before (n).png|jpg
  <src_root>\<folder>\mask (n).png|jpg

The script matches mask(n) with before(n) and converts masks into YOLO
segmentation labels (single class "trash").
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import yaml


def _parse_numbered_name(path: Path) -> Tuple[str, str | None]:
    stem = path.stem.lower()
    # matches "before (12)" or "mask (3)" with optional spaces
    if "(" in stem and ")" in stem:
        prefix = stem.split("(")[0].strip()
        num = stem.split("(")[1].split(")")[0].strip()
        return prefix, num if num.isdigit() else None
    return stem, None


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def _deterministic_split(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _mask_to_polygons(mask: np.ndarray, min_area: int) -> List[List[Tuple[float, float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[Tuple[float, float]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        points = contour.squeeze(axis=1)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        polygons.append([(float(x), float(y)) for x, y in points])
    return polygons


def _write_label(label_path: Path, polygons: List[List[Tuple[float, float]]], width: int, height: int) -> None:
    lines: List[str] = []
    for poly in polygons:
        coords: List[str] = []
        for x, y in poly:
            coords.append(f"{x / width:.6f}")
            coords.append(f"{y / height:.6f}")
        if coords:
            lines.append("0 " + " ".join(coords))
    label_path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(
    src_root: Path,
    out_root: Path,
    image_key: str,
    train_ratio: float,
    val_ratio: float,
    min_area: int,
    dry_run: bool,
    limit: int | None,
) -> None:
    folders = [p for p in src_root.iterdir() if p.is_dir()]
    if not folders:
        raise RuntimeError(f"No folders found in {src_root}")

    images_written = 0
    skipped = 0

    for folder in sorted(folders):
        images = _list_images(folder)
        if not images:
            continue

        befores: dict[str | None, Path] = {}
        masks: dict[str | None, Path] = {}

        for img in images:
            prefix, num = _parse_numbered_name(img)
            if prefix.startswith("before"):
                befores[num] = img
            elif prefix.startswith("mask"):
                masks[num] = img

        for num, before_path in befores.items():
            mask_path = masks.get(num)
            if mask_path is None:
                # fallback: if only one mask and no numbers, use it
                if num is None and len(masks) == 1:
                    mask_path = next(iter(masks.values()))
                else:
                    skipped += 1
                    continue

            split = _deterministic_split(f"{folder.name}_{before_path.name}", train_ratio, val_ratio)
            image_id = f"{folder.name}__{before_path.stem}"

            if dry_run:
                images_written += 1
                if limit and images_written >= limit:
                    return
                continue

            img_out = out_root / "images" / split / f"{image_id}{before_path.suffix.lower()}"
            label_out = out_root / "labels" / split / f"{image_id}.txt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                skipped += 1
                continue

            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            binary = (mask > 0).astype(np.uint8) * 255
            polygons = _mask_to_polygons(binary, min_area=min_area)
            if not polygons:
                skipped += 1
                continue

            cv2.imwrite(str(img_out), image)
            _write_label(label_out, polygons, image.shape[1], image.shape[0])

            images_written += 1
            if limit and images_written >= limit:
                return

    print(f"Images written: {images_written}")
    print(f"Skipped samples: {skipped}")


def _write_data_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "trash"},
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLOv8-seg dataset from custom masks.")
    parser.add_argument("--src-root", type=Path, required=True, help="Source root with numbered before/mask files.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for YOLO dataset.")
    parser.add_argument("--image-key", type=str, default="before", choices=["before"], help="Which image to use.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum mask area per instance.")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not write files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples.")

    args = parser.parse_args()
    src_root = args.src_root
    out_root = args.out_root

    if args.train + args.val >= 1.0:
        raise ValueError("Train + val must be < 1.0")

    if not args.dry_run:
        (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (out_root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / "test").mkdir(parents=True, exist_ok=True)

    build_dataset(
        src_root=src_root,
        out_root=out_root,
        image_key=args.image_key,
        train_ratio=args.train,
        val_ratio=args.val,
        min_area=args.min_area,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    if not args.dry_run:
        _write_data_yaml(out_root)
        print(f"Wrote data.yaml to {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
