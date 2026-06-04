from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
except Exception as exc:  # pragma: no cover - environment specific
    raise RuntimeError("torch/torchvision are required to train Siamese U-Net") from exc

try:
    from .method_scripts.siamese_unet_cd import SiameseUNet, SIAMESE_UNET_CD_CONFIG
except ImportError:
    from method_scripts.siamese_unet_cd import SiameseUNet, SIAMESE_UNET_CD_CONFIG


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_CSV = BASE_DIR / "datasets" / "custom" / "cd_pairs_masks_for_training" / "index.csv"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "training" / "siamese_unet_cd"
DEFAULT_CANONICAL_WEIGHTS = BASE_DIR / "weights" / "siamese_unet_cd_best.pt"


@dataclass(frozen=True)
class PairRecord:
    split: str
    pair_id: str
    before_path: Path
    after_path: Path
    mask_path: Path
    mask_area_px: int


def _read_index(index_path: Path) -> list[PairRecord]:
    if not index_path.exists():
        raise RuntimeError(f"index.csv not found: {index_path}")

    records: list[PairRecord] = []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            pair_id = str(row.get("pair_id", "")).strip()
            before_path = Path(str(row.get("before_path", "")).strip())
            after_path = Path(str(row.get("after_path", "")).strip())
            mask_path = Path(str(row.get("mask_path", "")).strip())
            mask_area_px = int(float(row.get("mask_area_px", 0) or 0))
            if not pair_id or not before_path.exists() or not after_path.exists() or not mask_path.exists():
                continue
            if split not in {"train", "val", "test"}:
                continue
            records.append(
                PairRecord(
                    split=split,
                    pair_id=pair_id,
                    before_path=before_path,
                    after_path=after_path,
                    mask_path=mask_path,
                    mask_area_px=mask_area_px,
                )
            )

    if not records:
        raise RuntimeError(f"No usable rows found in {index_path}")
    return records


def _split_records(records: list[PairRecord]) -> tuple[list[PairRecord], list[PairRecord]]:
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    if not train_records or not val_records:
        raise RuntimeError("index.csv must contain both train and val splits")
    return train_records, val_records


def _load_mask_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask_img = Image.open(path).convert("L")
    if size is not None and mask_img.size != size:
        mask_img = mask_img.resize(size, Image.NEAREST)
    return (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.uint8)


class SiamesePairDataset(Dataset):
    def __init__(self, records: list[PairRecord], image_size: int, augment: bool):
        self.records = records
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        before_img = Image.open(record.before_path).convert("RGB")
        after_img = Image.open(record.after_path).convert("RGB")
        mask_img = Image.open(record.mask_path).convert("L")

        before_img, after_img, mask_img = self._resize_pair(before_img, after_img, mask_img)
        if self.augment:
            before_img, after_img, mask_img = self._augment_pair(before_img, after_img, mask_img)

        before_tensor = TF.to_tensor(before_img)
        after_tensor = TF.to_tensor(after_img)
        before_tensor = TF.normalize(before_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        after_tensor = TF.normalize(after_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask_tensor = TF.to_tensor(mask_img).float()
        mask_tensor = (mask_tensor > 0.5).float()
        return before_tensor, after_tensor, mask_tensor, record.pair_id

    def _resize_pair(self, before_img: Image.Image, after_img: Image.Image, mask_img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        size = (self.image_size, self.image_size)
        before_img = TF.resize(before_img, size, interpolation=InterpolationMode.BILINEAR)
        after_img = TF.resize(after_img, size, interpolation=InterpolationMode.BILINEAR)
        mask_img = TF.resize(mask_img, size, interpolation=InterpolationMode.NEAREST)
        return before_img, after_img, mask_img

    def _augment_pair(self, before_img: Image.Image, after_img: Image.Image, mask_img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        if random.random() < 0.55:
            scale = random.uniform(0.80, 1.00)
            ratio = random.uniform(0.92, 1.08)
            crop_top, crop_left, crop_height, crop_width = T.RandomResizedCrop.get_params(
                before_img,
                scale=(scale, scale),
                ratio=(ratio, ratio),
            )
            size = (self.image_size, self.image_size)
            before_img = TF.resized_crop(before_img, crop_top, crop_left, crop_height, crop_width, size, interpolation=InterpolationMode.BILINEAR)
            after_img = TF.resized_crop(after_img, crop_top, crop_left, crop_height, crop_width, size, interpolation=InterpolationMode.BILINEAR)
            mask_img = TF.resized_crop(mask_img, crop_top, crop_left, crop_height, crop_width, size, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            before_img = TF.hflip(before_img)
            after_img = TF.hflip(after_img)
            mask_img = TF.hflip(mask_img)

        if random.random() < 0.2:
            before_img = TF.vflip(before_img)
            after_img = TF.vflip(after_img)
            mask_img = TF.vflip(mask_img)

        angle = random.uniform(-8.0, 8.0)
        if abs(angle) >= 0.25:
            before_img = TF.rotate(before_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            after_img = TF.rotate(after_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask_img = TF.rotate(mask_img, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < 0.75:
            brightness = 1.0 + random.uniform(-0.18, 0.18)
            contrast = 1.0 + random.uniform(-0.20, 0.20)
            saturation = 1.0 + random.uniform(-0.12, 0.12)
            hue = random.uniform(-0.04, 0.04)
            before_img = TF.adjust_brightness(before_img, brightness)
            after_img = TF.adjust_brightness(after_img, brightness)
            before_img = TF.adjust_contrast(before_img, contrast)
            after_img = TF.adjust_contrast(after_img, contrast)
            before_img = TF.adjust_saturation(before_img, saturation)
            after_img = TF.adjust_saturation(after_img, saturation)
            before_img = TF.adjust_hue(before_img, hue)
            after_img = TF.adjust_hue(after_img, hue)

        if random.random() < 0.25:
            blur_radius = random.uniform(0.2, 1.0)
            before_img = before_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            after_img = after_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return before_img, after_img, mask_img


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_output_dir(base_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    safe_run_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_name)
    output_dir = base_dir / f"{timestamp}__{safe_run_name}__{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def binary_segmentation_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
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
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    dice = (2.0 * tp) / max((2 * tp) + fp + fn, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _flatten_probabilities(targets: list[torch.Tensor], probs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    if not targets:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    y_true = torch.cat([tensor.detach().cpu().flatten() for tensor in targets], dim=0).numpy()
    y_prob = torch.cat([tensor.detach().cpu().flatten() for tensor in probs], dim=0).numpy()
    return y_true, y_prob


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0.0
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for before_tensor, after_tensor, masks, _pair_ids in loader:
        before_tensor = before_tensor.to(device, non_blocking=True)
        after_tensor = after_tensor.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(before_tensor, after_tensor)
            bce_loss = criterion(logits, masks)
            loss = (0.6 * bce_loss) + (0.4 * dice_loss_from_logits(logits, masks))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item()) * int(before_tensor.size(0))
        probabilities = torch.sigmoid(logits.detach())
        all_targets.append(masks.detach())
        all_probs.append(probabilities)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    y_true, y_prob = _flatten_probabilities(all_targets, all_probs)
    return avg_loss, y_true, y_prob


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for before_tensor, after_tensor, masks, _pair_ids in loader:
        before_tensor = before_tensor.to(device, non_blocking=True)
        after_tensor = after_tensor.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(before_tensor, after_tensor)
        bce_loss = criterion(logits, masks)
        loss = (0.6 * bce_loss) + (0.4 * dice_loss_from_logits(logits, masks))
        total_loss += float(loss.detach().item()) * int(before_tensor.size(0))
        all_targets.append(masks.detach())
        all_probs.append(torch.sigmoid(logits.detach()))

    avg_loss = total_loss / max(len(loader.dataset), 1)
    y_true, y_prob = _flatten_probabilities(all_targets, all_probs)
    return avg_loss, y_true, y_prob


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


def save_checkpoint(path: Path, model: nn.Module, epoch: int, best_score: float, args: argparse.Namespace, extra: dict[str, object]) -> None:
    payload = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "image_size": int(args.image_size),
        "state_dict": model.state_dict(),
        "train_args": _to_json_safe(vars(args)),
        "extra": _to_json_safe(extra),
    }
    torch.save(payload, path)


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


def write_metrics_csv(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Siamese U-Net on index.csv pairs.")
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV, help="Path to index.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for training runs")
    parser.add_argument("--run-name", type=str, default="siamese_unet_cd", help="Human-readable run name")
    parser.add_argument("--image-size", type=int, default=int(SIAMESE_UNET_CD_CONFIG["input_size"]))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--weights-init",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize weights before training",
    )
    parser.add_argument(
        "--weights-in",
        type=Path,
        default=DEFAULT_CANONICAL_WEIGHTS,
        help="Checkpoint to load for eval-only",
    )
    parser.add_argument("--weights-out", type=Path, default=None, help="Explicit checkpoint path for best.pt")
    parser.add_argument("--canonical-weights-out", type=Path, default=DEFAULT_CANONICAL_WEIGHTS)
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.force_cpu:
        return torch.device("cpu")
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_limit(records: list[PairRecord], limit: int | None) -> list[PairRecord]:
    if limit is None or limit <= 0:
        return records
    return records[: min(len(records), int(limit))]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = _read_index(args.index_csv)
    train_records, val_records = _split_records(records)
    train_records = maybe_limit(train_records, args.limit_train_samples)
    val_records = maybe_limit(val_records, args.limit_val_samples)
    if not train_records or not val_records:
        raise RuntimeError("Train/val splits are empty after applying limits")

    output_dir = make_output_dir(args.output_root, args.run_name)
    best_path = args.weights_out or (output_dir / "best.pt")
    last_path = output_dir / "last.pt"

    device = resolve_device(args)
    model = SiameseUNet(pretrained=True, input_channels=3)
    model = model.to(device)

    if args.weights_init is not None:
        if not args.weights_init.exists():
            raise RuntimeError(f"weights-init not found: {args.weights_init}")
        payload = torch.load(args.weights_init, map_location="cpu")
        state_dict = None
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict") or payload.get("model_state_dict")
            if state_dict is None and all(hasattr(v, "shape") for v in payload.values()):
                state_dict = payload
        elif hasattr(payload, "items"):
            state_dict = payload

        if not isinstance(state_dict, dict):
            raise RuntimeError(f"weights-init payload missing state_dict: {args.weights_init}")
        model.load_state_dict(state_dict, strict=False)

    positive_pixels = 0
    total_pixels = 0
    for record in train_records:
        mask_array = _load_mask_array(record.mask_path)
        positive_pixels += int(mask_array.sum())
        total_pixels += int(mask_array.size)
    negative_pixels = max(total_pixels - positive_pixels, 1)
    pos_weight_value = float(min(25.0, max(1.0, negative_pixels / max(positive_pixels, 1))))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda") if device.type == "cuda" else None

    train_dataset = SiamesePairDataset(train_records, args.image_size, augment=True)
    val_dataset = SiamesePairDataset(val_records, args.image_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_iou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    metadata = {
        "index_csv": str(args.index_csv),
        "output_dir": str(output_dir),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "positive_pixels_train": int(positive_pixels),
        "negative_pixels_train": int(negative_pixels),
        "pos_weight": round(pos_weight_value, 6),
        "image_size": int(args.image_size),
        "device": str(device),
    }

    if args.eval_only:
        if not args.weights_in.exists():
            raise RuntimeError(f"Checkpoint not found: {args.weights_in}")
        payload = torch.load(args.weights_in, map_location="cpu")
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=True)
        val_loss, val_targets, val_probs = evaluate(model, val_loader, criterion, device)
        val_metrics = binary_segmentation_metrics(val_targets, val_probs)
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
            f"Eval only | val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} val_iou={val_metrics['iou']:.4f}"
        )
        print(f"Output dir: {output_dir}")
        return

    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_probs = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_targets, val_probs = evaluate(model, val_loader, criterion, device)
        train_metrics = binary_segmentation_metrics(train_targets, train_probs)
        val_metrics = binary_segmentation_metrics(val_targets, val_probs)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_iou": round(train_metrics["iou"], 6),
            "train_dice": round(train_metrics["dice"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "val_iou": round(val_metrics["iou"], 6),
            "val_dice": round(val_metrics["dice"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(row)

        improved = (val_metrics["f1"] > best_f1) or (math.isclose(val_metrics["f1"], best_f1) and val_metrics["iou"] > best_iou)
        if improved:
            best_f1 = val_metrics["f1"]
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, epoch, best_f1, args, {**metadata, "val_metrics": _to_json_safe(val_metrics)})
            sync_canonical_checkpoint(
                best_path,
                args.canonical_weights_out,
                {**metadata, "val_metrics": _to_json_safe(val_metrics), "best_epoch": epoch, "best_f1": best_f1, "best_iou": best_iou},
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, epoch, best_f1, args, {**metadata, "val_metrics": _to_json_safe(val_metrics)})

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_iou={val_metrics['iou']:.4f} best_f1={best_f1:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement.")
            break

    history_path = output_dir / "history.csv"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(history_path, history)
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_val_iou": best_iou,
        "best_checkpoint": str(best_path),
        "canonical_checkpoint": str(args.canonical_weights_out),
        "last_checkpoint": str(last_path),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "history": history,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Best checkpoint: {best_path}")
    print(f"Canonical checkpoint: {args.canonical_weights_out}")
    print(f"Last checkpoint: {last_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
