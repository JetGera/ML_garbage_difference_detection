from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception as exc:  # pragma: no cover - environment specific
    raise RuntimeError("torch/torchvision are required to train EfficientNet") from exc

try:
    import timm
except Exception as exc:  # pragma: no cover - environment specific
    raise RuntimeError("timm is required to train EfficientNet") from exc

try:
    from .core import select_before_after
except ImportError:
    from core import select_before_after


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CANONICAL_WEIGHTS = Path("results") / "models" / "efficientnet" / "best.pt"


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


@dataclass(frozen=True)
class PairSample:
    pair_dir: Path
    before_path: Path
    after_path: Path


class BinaryImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image), sample.label, str(sample.path)


def discover_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def make_samples(folder: Path, label: int) -> list[Sample]:
    return [Sample(path=path, label=label) for path in discover_images(folder)]


def discover_pair_dirs(pairs_root: Path) -> list[Path]:
    if not pairs_root.exists():
        return []
    return [path for path in sorted(pairs_root.iterdir(), key=lambda path: path.name.casefold()) if path.is_dir()]


def discover_pair_samples(pairs_root: Path) -> list[PairSample]:
    samples: list[PairSample] = []
    for pair_dir in discover_pair_dirs(pairs_root):
        files = [path for path in sorted(pair_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if len(files) < 2:
            continue
        before_path, after_path = select_before_after(files)
        samples.append(PairSample(pair_dir=pair_dir, before_path=before_path, after_path=after_path))

    if not samples:
        raise RuntimeError(f"No usable before/after pairs were found under {pairs_root}")
    return samples


def resolve_taco_annotations_path(taco_root: Path) -> Path | None:
    for annotation_name in ("annotations.json", "annotations_unofficial.json"):
        candidate = taco_root / "data" / annotation_name
        if candidate.exists():
            return candidate
    fallback = taco_root / "data" / "annotations.json"
    return fallback if fallback.exists() else None


def prepare_taco_binary_dataset(
    taco_root: Path,
    output_root: Path,
    crop_size: int,
    dirty_crops_per_image: int,
    clean_crops_per_dirty: int,
    clean_overlap_threshold: float,
    val_fraction: float,
    seed: int,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    annotations_path = resolve_taco_annotations_path(taco_root)
    if annotations_path is None:
        raise RuntimeError(
            f"Could not find TACO annotations in {taco_root / 'data'}; expected annotations.json or annotations_unofficial.json"
        )

    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TACO annotations: {annotations_path}") from exc

    images = payload.get("images", []) or []
    annotations = payload.get("annotations", []) or []

    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        annotations_by_image.setdefault(int(image_id), []).append(annotation)

    signature_payload = {
        "annotations_path": str(annotations_path.resolve()),
        "annotations_mtime": round(annotations_path.stat().st_mtime, 6),
        "crop_size": int(crop_size),
        "dirty_crops_per_image": int(dirty_crops_per_image),
        "clean_crops_per_dirty": int(clean_crops_per_dirty),
        "clean_overlap_threshold": float(clean_overlap_threshold),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "split_strategy": "image_level",
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    prepared_root = output_root / f"taco_binary_dataset__{signature}"
    train_clean_dir = prepared_root / "train" / "clean"
    train_dirty_dir = prepared_root / "train" / "dirty"
    val_clean_dir = prepared_root / "val" / "clean"
    val_dirty_dir = prepared_root / "val" / "dirty"
    metadata_path = prepared_root / "metadata.json"

    if (
        metadata_path.exists()
        and train_clean_dir.exists()
        and train_dirty_dir.exists()
        and val_clean_dir.exists()
        and val_dirty_dir.exists()
    ):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                return train_clean_dir, train_dirty_dir, val_clean_dir, val_dirty_dir, metadata
        except Exception:
            pass

    train_clean_dir.mkdir(parents=True, exist_ok=True)
    train_dirty_dir.mkdir(parents=True, exist_ok=True)
    val_clean_dir.mkdir(parents=True, exist_ok=True)
    val_dirty_dir.mkdir(parents=True, exist_ok=True)

    train_images, val_images = _split_taco_images(images, val_fraction, seed)

    train_stats = _export_taco_subset(
        images_subset=train_images,
        source_root=taco_root / "data",
        annotations_by_image=annotations_by_image,
        dirty_dir=train_dirty_dir,
        clean_dir=train_clean_dir,
        crop_size=crop_size,
        dirty_crops_per_image=dirty_crops_per_image,
        clean_crops_per_dirty=clean_crops_per_dirty,
        clean_overlap_threshold=clean_overlap_threshold,
        rng=random.Random(seed),
    )
    val_stats = _export_taco_subset(
        images_subset=val_images,
        source_root=taco_root / "data",
        annotations_by_image=annotations_by_image,
        dirty_dir=val_dirty_dir,
        clean_dir=val_clean_dir,
        crop_size=crop_size,
        dirty_crops_per_image=dirty_crops_per_image,
        clean_crops_per_dirty=clean_crops_per_dirty,
        clean_overlap_threshold=clean_overlap_threshold,
        rng=random.Random(seed + 1),
    )

    metadata = {
        "signature": signature,
        "annotations_path": str(annotations_path),
        "split_strategy": "image_level",
        "val_fraction": float(val_fraction),
        "source_images": len(images),
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "train": train_stats,
        "val": val_stats,
        "crop_size": int(crop_size),
        "dirty_crops_per_image": int(dirty_crops_per_image),
        "clean_crops_per_dirty": int(clean_crops_per_dirty),
        "clean_overlap_threshold": float(clean_overlap_threshold),
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_clean_dir, train_dirty_dir, val_clean_dir, val_dirty_dir, metadata


def prepare_pair_binary_dataset(
    pairs_root: Path,
    output_root: Path,
    val_fraction: float,
    seed: int,
    export_max_side: int,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    samples = discover_pair_samples(pairs_root)

    signature_payload = {
        "pairs_root": str(pairs_root.resolve()),
        "pairs_root_mtime": round(pairs_root.stat().st_mtime, 6),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "split_strategy": "pair_level",
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    prepared_root = output_root / f"pairs_binary_dataset__{signature}"
    train_clean_dir = prepared_root / "train" / "clean"
    train_dirty_dir = prepared_root / "train" / "dirty"
    val_clean_dir = prepared_root / "val" / "clean"
    val_dirty_dir = prepared_root / "val" / "dirty"
    metadata_path = prepared_root / "metadata.json"

    if (
        metadata_path.exists()
        and train_clean_dir.exists()
        and train_dirty_dir.exists()
        and val_clean_dir.exists()
        and val_dirty_dir.exists()
    ):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                return train_clean_dir, train_dirty_dir, val_clean_dir, val_dirty_dir, metadata
        except Exception:
            pass

    train_clean_dir.mkdir(parents=True, exist_ok=True)
    train_dirty_dir.mkdir(parents=True, exist_ok=True)
    val_clean_dir.mkdir(parents=True, exist_ok=True)
    val_dirty_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs = _split_pair_samples(samples, val_fraction, seed)

    train_stats = _export_pair_subset(train_pairs, train_dirty_dir, train_clean_dir, export_max_side)
    val_stats = _export_pair_subset(val_pairs, val_dirty_dir, val_clean_dir, export_max_side)

    metadata = {
        "signature": signature,
        "pairs_root": str(pairs_root),
        "split_strategy": "pair_level",
        "val_fraction": float(val_fraction),
        "export_max_side": int(export_max_side),
        "source_pairs": len(samples),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "train": train_stats,
        "val": val_stats,
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_clean_dir, train_dirty_dir, val_clean_dir, val_dirty_dir, metadata


def _split_taco_images(images: list[dict[str, object]], val_fraction: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not images:
        return [], []

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_images = shuffled[:val_count]
    train_images = shuffled[val_count:]
    if not train_images:
        train_images = val_images[:1]
        val_images = val_images[1:]
    return train_images, val_images


def _split_pair_samples(samples: list[PairSample], val_fraction: float, seed: int) -> tuple[list[PairSample], list[PairSample]]:
    if not samples:
        return [], []

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples:
        train_samples = val_samples[:1]
        val_samples = val_samples[1:]
    return train_samples, val_samples


def _resize_export_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    if max(image.size) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _export_pair_subset(pairs: list[PairSample], dirty_dir: Path, clean_dir: Path, max_side: int) -> dict[str, object]:
    stats: dict[str, object] = {
        "source_pairs": 0,
        "dirty_samples": 0,
        "clean_samples": 0,
        "missing_images": 0,
    }

    for pair in pairs:
        if not pair.before_path.exists() or not pair.after_path.exists():
            stats["missing_images"] = int(stats["missing_images"]) + 1
            continue

        dirty_name = f"{pair.pair_dir.name}__before__dirty{pair.before_path.suffix.lower()}"
        clean_name = f"{pair.pair_dir.name}__after__clean{pair.after_path.suffix.lower()}"
        with Image.open(pair.before_path) as before_image:
            before_export = _resize_export_image(ImageOps.exif_transpose(before_image).convert("RGB"), max_side)
            before_export.save(dirty_dir / dirty_name)
        with Image.open(pair.after_path) as after_image:
            after_export = _resize_export_image(ImageOps.exif_transpose(after_image).convert("RGB"), max_side)
            after_export.save(clean_dir / clean_name)
        stats["source_pairs"] = int(stats["source_pairs"]) + 1
        stats["dirty_samples"] = int(stats["dirty_samples"]) + 1
        stats["clean_samples"] = int(stats["clean_samples"]) + 1

    return stats


def _export_taco_subset(
    images_subset: list[dict[str, object]],
    source_root: Path,
    annotations_by_image: dict[int, list[dict[str, object]]],
    dirty_dir: Path,
    clean_dir: Path,
    crop_size: int,
    dirty_crops_per_image: int,
    clean_crops_per_dirty: int,
    clean_overlap_threshold: float,
    rng: random.Random,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "source_images": 0,
        "dirty_samples": 0,
        "clean_samples": 0,
        "missing_images": 0,
        "skipped_annotations": 0,
        "images_with_annotations": 0,
    }

    for image_info in images_subset:
        file_name = str(image_info.get("file_name", "")).strip()
        if not file_name:
            continue

        image_id = int(image_info.get("id", -1))
        image_path = source_root / file_name
        if not image_path.exists():
            stats["missing_images"] = int(stats["missing_images"]) + 1
            continue

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        annotations_for_image = annotations_by_image.get(image_id, [])
        if not annotations_for_image:
            continue

        image_mask = build_taco_mask(image.size, annotations_for_image)
        dirty_candidates = sorted(
            [annotation for annotation in annotations_for_image if _taco_annotation_area(annotation) > 0],
            key=_taco_annotation_area,
            reverse=True,
        )[: max(1, int(dirty_crops_per_image))]

        if not dirty_candidates:
            stats["skipped_annotations"] = int(stats["skipped_annotations"]) + 1
            continue

        stats["source_images"] = int(stats["source_images"]) + 1
        stats["images_with_annotations"] = int(stats["images_with_annotations"]) + 1
        source_stem = Path(file_name).stem

        for annotation_index, annotation in enumerate(dirty_candidates):
            bbox = annotation.get("bbox")
            crop_box = _taco_crop_box_from_annotation(bbox, image.size, crop_size)
            if crop_box is None:
                stats["skipped_annotations"] = int(stats["skipped_annotations"]) + 1
                continue

            dirty_crop = crop_with_padding(image, crop_box)
            dirty_name = f"{source_stem}__img{image_id:06d}__ann{int(annotation.get('id', annotation_index)):06d}__dirty.png"
            dirty_crop.save(dirty_dir / dirty_name)
            stats["dirty_samples"] = int(stats["dirty_samples"]) + 1

            clean_needed = max(0, int(clean_crops_per_dirty))
            for clean_index in range(clean_needed):
                clean_box = _sample_clean_crop_box(
                    image.size,
                    image_mask,
                    crop_size,
                    rng,
                    clean_overlap_threshold,
                )
                if clean_box is None:
                    stats["skipped_annotations"] = int(stats["skipped_annotations"]) + 1
                    continue

                clean_crop = crop_with_padding(image, clean_box)
                clean_name = (
                    f"{source_stem}__img{image_id:06d}__ann{int(annotation.get('id', annotation_index)):06d}"
                    f"__clean{clean_index:02d}.png"
                )
                clean_crop.save(clean_dir / clean_name)
                stats["clean_samples"] = int(stats["clean_samples"]) + 1

    return stats


def build_taco_mask(image_size: tuple[int, int], annotations: list[dict[str, object]]) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list) and segmentation:
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                    continue
                points = [(float(polygon[index]), float(polygon[index + 1])) for index in range(0, len(polygon), 2)]
                draw.polygon(points, fill=1)
            continue

        bbox = annotation.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x, y, box_width, box_height = bbox[:4]
            draw.rectangle([float(x), float(y), float(x) + float(box_width), float(y) + float(box_height)], fill=1)

    return np.asarray(mask_image, dtype=np.uint8) > 0


def crop_with_padding(image: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = crop_box
    crop_width = max(1, int(right - left))
    crop_height = max(1, int(bottom - top))
    canvas = Image.new("RGB", (crop_width, crop_height), (0, 0, 0))

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image.width, right)
    source_bottom = min(image.height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        return canvas

    pasted = image.crop((source_left, source_top, source_right, source_bottom))
    canvas.paste(pasted, (source_left - left, source_top - top))
    return canvas


def _taco_annotation_area(annotation: dict[str, object]) -> float:
    area = annotation.get("area")
    if isinstance(area, (int, float)) and float(area) > 0:
        return float(area)

    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])))

    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        polygon_points = 0
        for polygon in segmentation:
            if isinstance(polygon, list):
                polygon_points += len(polygon) // 2
        return float(polygon_points)

    return 0.0


def _taco_crop_box_from_annotation(
    bbox: list[object] | tuple[object, ...] | None,
    image_size: tuple[int, int],
    crop_size: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    image_width, image_height = image_size
    box_x = float(bbox[0])
    box_y = float(bbox[1])
    box_width = max(1.0, float(bbox[2]))
    box_height = max(1.0, float(bbox[3]))
    side = max(int(crop_size), int(round(max(box_width, box_height) * 1.8)))
    side = max(64, side)

    center_x = box_x + box_width / 2.0
    center_y = box_y + box_height / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = left + side
    bottom = top + side

    if image_width <= 0 or image_height <= 0:
        return None
    return left, top, right, bottom


def _sample_clean_crop_box(
    image_size: tuple[int, int],
    mask: np.ndarray,
    crop_size: int,
    rng: random.Random,
    overlap_threshold: float,
) -> tuple[int, int, int, int] | None:
    image_width, image_height = image_size
    side = max(64, int(crop_size))
    if image_width <= 0 or image_height <= 0:
        return None

    max_x = max(0, image_width - side)
    max_y = max(0, image_height - side)

    candidate_boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    for _ in range(20):
        left = rng.randint(0, max_x) if max_x > 0 else 0
        top = rng.randint(0, max_y) if max_y > 0 else 0
        right = min(image_width, left + side)
        bottom = min(image_height, top + side)
        if right <= left or bottom <= top:
            continue
        overlap = float(np.mean(mask[top:bottom, left:right])) if mask.size else 0.0
        if overlap <= overlap_threshold:
            return left, top, right, bottom
        candidate_boxes.append((overlap, (left, top, right, bottom)))

    if candidate_boxes:
        candidate_boxes.sort(key=lambda item: item[0])
        return candidate_boxes[0][1]
    return None


def split_samples(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not samples:
        return [], []

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples:
        train_samples = val_samples[:1]
        val_samples = val_samples[1:]
    return train_samples, val_samples


def prepare_splits(
    clean_dir: Path,
    dirty_dir: Path,
    val_clean_dir: Path | None,
    val_dirty_dir: Path | None,
    val_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if val_clean_dir is not None and val_dirty_dir is not None:
        train_samples = make_samples(clean_dir, 0) + make_samples(dirty_dir, 1)
        val_samples = make_samples(val_clean_dir, 0) + make_samples(val_dirty_dir, 1)
        return train_samples, val_samples

    clean_samples = make_samples(clean_dir, 0)
    dirty_samples = make_samples(dirty_dir, 1)
    clean_train, clean_val = split_samples(clean_samples, val_fraction, seed)
    dirty_train, dirty_val = split_samples(dirty_samples, val_fraction, seed + 1)
    return clean_train + dirty_train, clean_val + dirty_val


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.04),
            transforms.RandomAutocontrast(p=0.20),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.12, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=0),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.18)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, val_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision_dirty": float(precision),
        "recall_dirty": float(recall),
        "f1_dirty": float(f1),
        "roc_auc": float(auc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def make_output_dir(base_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    safe_run_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_name)
    output_dir = base_dir / f"{timestamp}__{safe_run_name}__{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def sync_canonical_checkpoint(source: Path, target: Path, metadata: dict[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    manifest_path = target.with_suffix(".json")
    manifest = {
        "source_checkpoint": str(source),
        "canonical_checkpoint": str(target),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": _to_json_safe(metadata),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Unexpected checkpoint payload in {path}")
    return payload


def apply_checkpoint_args(args: argparse.Namespace, checkpoint: dict[str, object]) -> None:
    train_args = checkpoint.get("train_args") if isinstance(checkpoint, dict) else None
    if not isinstance(train_args, dict):
        train_args = {}
    args.model_name = str(train_args.get("model_name", checkpoint.get("model_name", args.model_name)))
    args.image_size = int(train_args.get("image_size", checkpoint.get("image_size", args.image_size)))


def build_model(model_name: str, device: torch.device) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    return model.to(device)


def freeze_backbone(model: nn.Module, head_only: bool) -> None:
    for name, parameter in model.named_parameters():
        if head_only:
            parameter.requires_grad = any(token in name for token in ("classifier", "head", "fc"))
        else:
            parameter.requires_grad = True


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0.0
    all_targets: list[int] = []
    all_probs: list[float] = []

    for images, targets, _paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item()) * int(images.size(0))
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        all_targets.extend(targets.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_targets), np.asarray(all_probs)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_probs: list[float] = []

    for images, targets, _paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.detach().item()) * int(images.size(0))
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_targets.extend(targets.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_targets), np.asarray(all_probs)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, best_score: float, args: argparse.Namespace, extra: dict[str, object]) -> None:
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "state_dict": model.state_dict(),
        "train_args": vars(args),
        "extra": extra,
    }
    torch.save(payload, path)


def write_metrics_csv(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet for clean/dirty classification.")
    parser.add_argument("--clean-dir", type=Path, default=None, help="Directory with clean images")
    parser.add_argument("--dirty-dir", type=Path, default=None, help="Directory with dirty images (for example TACO images)")
    parser.add_argument("--val-clean-dir", type=Path, default=None, help="Optional validation clean directory")
    parser.add_argument("--val-dirty-dir", type=Path, default=None, help="Optional validation dirty directory")
    parser.add_argument("--taco-root", type=Path, default=None, help="Optional TACO dataset root with data/annotations.json and data/images")
    parser.add_argument("--taco-crop-size", type=int, default=512, help="Square crop size used when building the TACO binary dataset")
    parser.add_argument(
        "--taco-dirty-crops-per-image",
        type=int,
        default=3,
        help="How many dirty crops to export from each TACO image before splitting",
    )
    parser.add_argument(
        "--taco-clean-crops-per-dirty",
        type=int,
        default=1,
        help="How many background clean crops to export for each dirty crop",
    )
    parser.add_argument(
        "--taco-clean-overlap-threshold",
        type=float,
        default=0.01,
        help="Maximum allowed trash-mask overlap for a crop to count as clean",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results") / "training", help="Root directory for training runs")
    parser.add_argument("--run-name", type=str, default="efficientnet_cls", help="Human-readable run name")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0", help="timm model name")
    parser.add_argument("--pairs-root", type=Path, default=None, help="Optional root folder containing your own before/after pair subfolders")
    parser.add_argument("--pairs-export-max-side", type=int, default=768, help="Max side length used when exporting own pairs to training folders")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation on the validation split and skip training")
    parser.add_argument(
        "--weights-in",
        type=Path,
        default=DEFAULT_CANONICAL_WEIGHTS,
        help="Checkpoint to load for eval-only (defaults to canonical best.pt)",
    )
    parser.add_argument("--weights-out", type=Path, default=None, help="Explicit checkpoint path for best.pt")
    parser.add_argument("--canonical-weights-out", type=Path, default=DEFAULT_CANONICAL_WEIGHTS, help="Stable checkpoint path used by the GUI after training")
    args = parser.parse_args()

    if args.taco_root is None and args.pairs_root is None and (args.clean_dir is None or args.dirty_dir is None):
        parser.error("provide --pairs-root, or --taco-root, or both --clean-dir and --dirty-dir")

    if args.taco_root is not None and (args.clean_dir is not None or args.dirty_dir is not None or args.pairs_root is not None):
        print("TACO mode is enabled; manual clean/dirty directories will be ignored.")

    if args.pairs_root is not None and (args.clean_dir is not None or args.dirty_dir is not None):
        print("Pair mode is enabled; manual clean/dirty directories will be ignored.")

    return args


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.force_cpu:
        return torch.device("cpu")
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint: dict[str, object] | None = None
    if args.eval_only:
        if not args.weights_in.exists():
            raise RuntimeError(f"Checkpoint not found: {args.weights_in}")
        checkpoint = load_checkpoint(args.weights_in)
        apply_checkpoint_args(args, checkpoint)

    taco_metadata: dict[str, object] | None = None
    pair_metadata: dict[str, object] | None = None
    if args.taco_root is not None:
        prepared_dataset_root = args.output_root / "prepared_datasets"
        clean_dir, dirty_dir, val_clean_dir, val_dirty_dir, taco_metadata = prepare_taco_binary_dataset(
            taco_root=args.taco_root,
            output_root=prepared_dataset_root,
            crop_size=args.taco_crop_size,
            dirty_crops_per_image=args.taco_dirty_crops_per_image,
            clean_crops_per_dirty=args.taco_clean_crops_per_dirty,
            clean_overlap_threshold=args.taco_clean_overlap_threshold,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    elif args.pairs_root is not None:
        prepared_dataset_root = args.output_root / "prepared_datasets"
        clean_dir, dirty_dir, val_clean_dir, val_dirty_dir, pair_metadata = prepare_pair_binary_dataset(
            pairs_root=args.pairs_root,
            output_root=prepared_dataset_root,
            val_fraction=args.val_fraction,
            seed=args.seed,
            export_max_side=args.pairs_export_max_side,
        )
    else:
        clean_dir = args.clean_dir
        dirty_dir = args.dirty_dir
        val_clean_dir = args.val_clean_dir
        val_dirty_dir = args.val_dirty_dir

    train_samples, val_samples = prepare_splits(
        clean_dir=clean_dir,
        dirty_dir=dirty_dir,
        val_clean_dir=val_clean_dir,
        val_dirty_dir=val_dirty_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError("Training split is empty")
    if not val_samples:
        raise RuntimeError("Validation split is empty")

    output_dir = make_output_dir(args.output_root, args.run_name)
    best_path = args.weights_out or (output_dir / "best.pt")
    last_path = output_dir / "last.pt"

    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = BinaryImageDataset(train_samples, train_transform)
    val_dataset = BinaryImageDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = resolve_device(args)
    model = build_model(args.model_name, device)

    class_counts = np.bincount(np.asarray([sample.label for sample in train_samples], dtype=np.int64), minlength=2)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1) / 2.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    freeze_backbone(model, head_only=args.freeze_backbone_epochs > 0)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda") if device.type == "cuda" else None

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    metadata = {
        "dataset_mode": "taco_binary" if args.taco_root is not None else ("pair_binary" if args.pairs_root is not None else "manual_folders"),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "class_counts_train": class_counts.tolist(),
        "clean_dir": str(clean_dir),
        "dirty_dir": str(dirty_dir),
        "val_clean_dir": str(val_clean_dir) if val_clean_dir else None,
        "val_dirty_dir": str(val_dirty_dir) if val_dirty_dir else None,
        "taco_root": str(args.taco_root) if args.taco_root else None,
        "pairs_root": str(args.pairs_root) if args.pairs_root else None,
        "taco_metadata": taco_metadata,
        "pair_metadata": pair_metadata,
    }

    if args.eval_only:
        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        val_loss, val_targets, val_probs = evaluate(model, val_loader, criterion, device)
        val_metrics = binary_metrics(val_targets, val_probs)
        summary = {
            "mode": "eval_only",
            "weights_in": str(args.weights_in),
            "val_loss": round(val_loss, 6),
            "val_metrics": _to_json_safe(val_metrics),
            "metadata": metadata,
            "output_dir": str(output_dir),
        }
        metrics_path = output_dir / "metrics.json"
        summary_path = output_dir / "summary.json"
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Eval only | val_loss={val_loss:.4f} val_f1_dirty={val_metrics['f1_dirty']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )
        print(f"Output dir: {output_dir}")
        return

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            freeze_backbone(model, head_only=False)
            optimizer = make_optimizer(model, args.lr, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - epoch + 1, 1))

        train_loss, train_targets, train_probs = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_targets, val_probs = evaluate(model, val_loader, criterion, device)
        train_metrics = binary_metrics(train_targets, train_probs)
        val_metrics = binary_metrics(val_targets, val_probs)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_f1_dirty": round(train_metrics["f1_dirty"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_precision_dirty": round(val_metrics["precision_dirty"], 6),
            "val_recall_dirty": round(val_metrics["recall_dirty"], 6),
            "val_f1_dirty": round(val_metrics["f1_dirty"], 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6) if math.isfinite(val_metrics["roc_auc"]) else None,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(row)

        improved = val_metrics["f1_dirty"] > best_f1
        if improved:
            best_f1 = val_metrics["f1_dirty"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, epoch, best_f1, args, {**metadata, "val_metrics": val_metrics})
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, epoch, best_f1, args, {**metadata, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1_dirty={val_metrics['f1_dirty']:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"best_f1={best_f1:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement.")
            break

    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(history_path, history)
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_dirty": best_f1,
        "best_checkpoint": str(best_path),
        "canonical_checkpoint": str(args.canonical_weights_out),
        "last_checkpoint": str(last_path),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "training_rows": history,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sync_canonical_checkpoint(best_path, args.canonical_weights_out, {**metadata, "best_epoch": best_epoch, "best_f1_dirty": best_f1})

    print()
    print(f"Best checkpoint: {best_path}")
    print(f"Canonical checkpoint: {args.canonical_weights_out}")
    print(f"Last checkpoint: {last_path}")
    print(f"Output dir: {output_dir}")
    print(f"Set EFFICIENTNET_CLS_WEIGHTS={best_path}")


if __name__ == "__main__":
    main()