from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

try:
    from torchvision.models import ResNet34_Weights, resnet34
except Exception:
    ResNet34_Weights = None
    resnet34 = None

try:
    from ..core import AnalysisResult
    from ..methods import get_method_spec
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec


SIAMESE_UNET_CD_CONFIG = {
    "encoder_name": "resnet34",
    "encoder_pretrained": True,
    "input_size": 384,
    "threshold": 0.65,
    "morph_kernel": (3, 3),
    "morph_open_iterations": 2,
    "morph_close_iterations": 1,
    "min_component_area_px": 128,
    "preview_max_width": 3200,
    "preview_max_height": 2400,
    "panel_title_height": 44,
    "overlay_alpha": 0.60,
    "alignment_downscale_max_side": 1200,
    "ecc_max_iterations": 80,
    "ecc_eps": 1e-5,
    "colormap": cv2.COLORMAP_TURBO,
}


if nn is not None:

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.block(tensor)


    class _UpBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.block = _ConvBlock(out_channels + skip_channels, out_channels)

        def forward(self, tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            tensor = self.up(tensor)
            if tensor.shape[-2:] != skip.shape[-2:]:
                tensor = torch.nn.functional.interpolate(tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            tensor = torch.cat([tensor, skip], dim=1)
            return self.block(tensor)


    class SiameseUNet(nn.Module):
        def __init__(self, pretrained: bool = True, input_channels: int = 3, base_channels: int = 64):
            super().__init__()
            if input_channels != 3:
                raise ValueError("This implementation currently expects RGB inputs per branch")

            self.encoder = None
            if resnet34 is not None:
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained and ResNet34_Weights is not None else None
                self.encoder = resnet34(weights=weights)
                self.encoder.fc = nn.Identity()

            self.enc0 = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.MaxPool2d(2)
            self.enc1 = _ConvBlock(base_channels, base_channels * 2)
            self.enc2 = _ConvBlock(base_channels * 2, base_channels * 4)
            self.enc3 = _ConvBlock(base_channels * 4, base_channels * 8)
            self.bottleneck = _ConvBlock(base_channels * 8, base_channels * 16)

            self.up3 = _UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
            self.up2 = _UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
            self.up1 = _UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
            self.up0 = _UpBlock(base_channels * 2, base_channels, base_channels)
            self.head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            )

        def _branch(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x0 = self.enc0(tensor)
            x1 = self.enc1(self.pool(x0))
            x2 = self.enc2(self.pool(x1))
            x3 = self.enc3(self.pool(x2))
            x4 = self.bottleneck(self.pool(x3))
            return x0, x1, x2, x3, x4

        def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
            before_features = self._branch(before)
            after_features = self._branch(after)

            x0 = torch.abs(before_features[0] - after_features[0])
            x1 = torch.abs(before_features[1] - after_features[1])
            x2 = torch.abs(before_features[2] - after_features[2])
            x3 = torch.abs(before_features[3] - after_features[3])
            x4 = torch.abs(before_features[4] - after_features[4])

            x = self.up3(x4, x3)
            x = self.up2(x, x2)
            x = self.up1(x, x1)
            x = self.up0(x, x0)
            return self.head(x)

else:

    class SiameseUNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch/torchvision are required for SiameseUNet")


@dataclass(frozen=True)
class _PairPrediction:
    probability_map: np.ndarray
    threshold_value: float
    inference_ms: float
    model_source: str
    device_used: str


class SiameseUnetCdRunner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        input_size: int | None = None,
        threshold: float | None = None,
        weights_path: str | Path | None = None,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = str(device)
        self.force_cpu = bool(force_cpu)
        self.input_size = int(input_size or SIAMESE_UNET_CD_CONFIG["input_size"])
        self.threshold = float(threshold or SIAMESE_UNET_CD_CONFIG["threshold"])
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self._model = None
        self._model_source = None

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        before_img = self._read_color_image(before)
        after_img = self._read_color_image(after)
        aligned_before, aligned_after, overlap_mask, alignment_mode = self._align_after_to_before(before_img, after_img)
        prediction = self._predict_change_map(aligned_before, aligned_after)
        probability_map = prediction.probability_map
        threshold_value = prediction.threshold_value

        change_mask = (probability_map >= threshold_value).astype(np.uint8) * 255
        change_mask = self._postprocess_mask(change_mask, overlap_mask)
        change_pixels = int(np.count_nonzero(change_mask))
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        canvas_pixels = int(overlap_mask.size)
        change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)

        probability_u8 = (np.clip(probability_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        probability_heatmap = cv2.applyColorMap(probability_u8, int(SIAMESE_UNET_CD_CONFIG["colormap"]))
        probability_overlay = self._blend_overlay(aligned_after, probability_heatmap, float(SIAMESE_UNET_CD_CONFIG["overlay_alpha"]))
        mask_overlay = self._blend_mask_overlay(aligned_after, change_mask)
        preview = self._compose_preview(aligned_before, aligned_after, probability_overlay, mask_overlay, alignment_mode)

        output_dir = self._prepare_output_dir(before, after)
        artifacts = self._save_artifacts(
            output_dir=output_dir,
            aligned_before=aligned_before,
            aligned_after=aligned_after,
            overlap_mask=overlap_mask,
            probability_map=probability_u8,
            probability_overlay=probability_overlay,
            change_mask=change_mask,
            preview=preview,
        )

        metrics = {
            "analysis_mode": "siamese_unet_cd_pair",
            "model_source": prediction.model_source,
            "device_requested": self.device,
            "device_used": prediction.device_used,
            "force_cpu": bool(self.force_cpu),
            "input_size": int(self.input_size),
            "threshold": float(self.threshold),
            "alignment_mode": alignment_mode,
            "change_pixels": change_pixels,
            "overlap_pixels": overlap_pixels,
            "canvas_pixels": canvas_pixels,
            "overlap_ratio": round(overlap_pixels / max(canvas_pixels, 1), 6),
            "change_ratio": change_ratio,
            "inference_ms": round(float(prediction.inference_ms), 3),
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }

        summary = "Siamese U-Net change detection on a before/after pair with alignment, postprocessing, and checkpoint-aware inference."
        preview_text = (
            f"Mode: siamese_unet_cd\n"
            f"Device: {prediction.device_used} (requested: {self.device})\n"
            f"Alignment: {alignment_mode}\n"
            f"Threshold: {threshold_value:.3f}\n"
            f"Change ratio: {change_ratio:.6f}"
        )

        return AnalysisResult(
            method_id=self.method_id,
            method_name=self.label,
            summary=summary,
            metrics=metrics,
            before_path=before,
            after_path=after,
            preview_text=preview_text,
            preview_image_path=artifacts["preview"],
            artifacts=artifacts,
        )

    def _read_color_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return image

    def _align_after_to_before(self, before_img: np.ndarray, after_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        before_h, before_w = before_img.shape[:2]
        if (before_h, before_w) != after_img.shape[:2]:
            after_img = cv2.resize(after_img, (before_w, before_h), interpolation=cv2.INTER_LINEAR)

        try:
            gray_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(SIAMESE_UNET_CD_CONFIG["ecc_max_iterations"]),
                float(SIAMESE_UNET_CD_CONFIG["ecc_eps"]),
            )
            cv2.findTransformECC(gray_before, gray_after, warp_matrix, cv2.MOTION_AFFINE, criteria)
            aligned_after = cv2.warpAffine(
                after_img,
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            # Compute valid support after warping to avoid border artifacts being treated as change.
            valid_before = np.full((before_h, before_w), 255, dtype=np.uint8)
            valid_after = np.full((before_h, before_w), 255, dtype=np.uint8)
            valid_after = cv2.warpAffine(
                valid_after,
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            overlap_mask = cv2.bitwise_and(valid_before, valid_after)
            overlap_mask = cv2.erode(overlap_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
            
            # Crop to valid region (remove black borders) like SIFT RANSAC does
            union_mask = cv2.bitwise_or(valid_before, valid_after)
            non_zero_points = cv2.findNonZero(union_mask)
            if non_zero_points is not None:
                x, y, w, h = cv2.boundingRect(non_zero_points)
                before_img = before_img[y : y + h, x : x + w]
                aligned_after = aligned_after[y : y + h, x : x + w]
                overlap_mask = overlap_mask[y : y + h, x : x + w]
            
            return before_img, aligned_after, overlap_mask, "ecc_affine"
        except Exception:
            overlap_mask = np.full((before_h, before_w), 255, dtype=np.uint8)
            return before_img, after_img, overlap_mask, "resize_fallback"

    def _predict_change_map(self, before_img: np.ndarray, after_img: np.ndarray) -> _PairPrediction:
        if torch is None or nn is None:
            return self._heuristic_prediction(before_img, after_img, device_used="cpu", model_source="cv_heuristic")

        model = self._load_model()
        device_used = self._resolve_device()
        model = model.to(device_used)
        before_tensor = self._image_to_tensor(before_img, device_used)
        after_tensor = self._image_to_tensor(after_img, device_used)

        start = perf_counter()
        with torch.no_grad():
            logits = model(before_tensor, after_tensor)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        inference_ms = (perf_counter() - start) * 1000.0
        probability = cv2.resize(probability.astype(np.float32), (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return _PairPrediction(probability, float(self.threshold), inference_ms, str(self._model_source), device_used)

    def _heuristic_prediction(self, before_img: np.ndarray, after_img: np.ndarray, device_used: str, model_source: str) -> _PairPrediction:
        start = perf_counter()
        diff = cv2.absdiff(before_img, after_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        diff_norm = cv2.normalize(diff_blur.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        inference_ms = (perf_counter() - start) * 1000.0
        return _PairPrediction(diff_norm, float(self.threshold), inference_ms, model_source, device_used)

    def _load_model(self):
        if self._model is not None:
            return self._model

        model = SiameseUNet(pretrained=bool(SIAMESE_UNET_CD_CONFIG["encoder_pretrained"]))
        model_source = "random_init"
        checkpoint_path = self.weights_path or self._discover_latest_training_checkpoint()
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = torch.load(str(checkpoint_path), map_location="cpu")
            # Handle both trainer format (model_state_dict) and direct state_dict
            if isinstance(payload, dict):
                state_dict = payload.get("model_state_dict") or payload.get("state_dict")
            else:
                state_dict = payload
            if isinstance(state_dict, dict):
                normalized_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
                model.load_state_dict(normalized_state, strict=False)
                model_source = str(checkpoint_path)
        self._model = model.eval()
        self._model_source = model_source
        return self._model

    def _discover_latest_training_checkpoint(self) -> Path | None:
        training_root = Path(__file__).resolve().parent.parent.parent / "results" / "training"
        if not training_root.exists():
            return None

        # best trained data is real17
        real17_preferred = training_root / "siamese_unet_cd_real17_finetuned" / "best.pt"
        if real17_preferred.exists():
            return real17_preferred

        preferred = training_root / "latest" / "best.pt"
        if preferred.exists():
            return preferred

        original = training_root / "siamese_unet_cd" / "best.pt"
        if original.exists():
            return original

        candidates = [path for path in training_root.rglob("best.pt") if path.is_file()]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]

        candidates = [path for path in training_root.rglob("last.pt") if path.is_file()]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]

        return None

    def _resolve_device(self) -> str:
        if self.force_cpu:
            return "cpu"
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        if self.device == "cpu":
            return "cpu"
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def _image_to_tensor(self, image: np.ndarray, device_used: str) -> torch.Tensor:
        resized = cv2.resize(image, (int(self.input_size), int(self.input_size)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
        if device_used == "cuda":
            tensor = tensor.cuda(non_blocking=True)
        return tensor

    def _postprocess_mask(self, mask: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(SIAMESE_UNET_CD_CONFIG["morph_kernel"]))
        mask = cv2.bitwise_and(mask, overlap_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(SIAMESE_UNET_CD_CONFIG["morph_open_iterations"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(SIAMESE_UNET_CD_CONFIG["morph_close_iterations"]))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(mask)
        min_area = int(SIAMESE_UNET_CD_CONFIG["min_component_area_px"])
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    def _compose_preview(self, before_img: np.ndarray, after_img: np.ndarray, probability_overlay: np.ndarray, mask_overlay: np.ndarray, alignment_mode: str) -> np.ndarray:
        panels = [
            self._add_title(before_img, f"Before | {alignment_mode}"),
            self._add_title(after_img, "After"),
            self._add_title(probability_overlay, "Probability map"),
            self._add_title(mask_overlay, "Change mask"),
        ]
        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])
        preview = np.vstack([top, bottom])
        return self._resize_for_preview(preview)

    def _add_title(self, image: np.ndarray, title: str) -> np.ndarray:
        title_h = int(SIAMESE_UNET_CD_CONFIG["panel_title_height"])
        canvas = np.zeros((image.shape[0] + title_h, image.shape[1], 3), dtype=np.uint8)
        canvas[title_h:, :] = image
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], title_h), (18, 18, 18), thickness=-1)
        cv2.putText(canvas, title, (12, int(title_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def _resize_for_preview(self, preview: np.ndarray) -> np.ndarray:
        max_width = int(SIAMESE_UNET_CD_CONFIG["preview_max_width"])
        max_height = int(SIAMESE_UNET_CD_CONFIG["preview_max_height"])
        height, width = preview.shape[:2]
        scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
        if scale < 1.0:
            preview = cv2.resize(preview, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return preview

    def _blend_overlay(self, base_image: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(base_rgb, 1.0 - alpha, overlay_rgb, alpha, 0.0)

    def _blend_mask_overlay(self, after_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.7
        return (overlay.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).astype(np.uint8)

    def _prepare_output_dir(self, before: Path, after: Path) -> Path:
        safe_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        token = uuid4().hex[:8]
        output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "analysis" / self.method_id / f"{safe_stamp}__{before.stem}__{after.stem}__{token}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_artifacts(self, output_dir: Path, aligned_before: np.ndarray, aligned_after: np.ndarray, overlap_mask: np.ndarray, probability_map: np.ndarray, probability_overlay: np.ndarray, change_mask: np.ndarray, preview: np.ndarray) -> dict[str, Path]:
        artifacts = {
            "aligned_before": output_dir / "aligned_before.png",
            "aligned_after": output_dir / "aligned_after.png",
            "overlap_mask": output_dir / "overlap_mask.png",
            "probability_map": output_dir / "probability_map.png",
            "probability_overlay": output_dir / "probability_overlay.png",
            "change_mask": output_dir / "change_mask.png",
            "preview": output_dir / "preview.png",
        }
        cv2.imwrite(str(artifacts["aligned_before"]), aligned_before)
        cv2.imwrite(str(artifacts["aligned_after"]), aligned_after)
        cv2.imwrite(str(artifacts["overlap_mask"]), overlap_mask)
        cv2.imwrite(str(artifacts["probability_map"]), probability_map)
        cv2.imwrite(str(artifacts["probability_overlay"]), cv2.cvtColor(probability_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(artifacts["change_mask"]), change_mask)
        cv2.imwrite(str(artifacts["preview"]), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        return artifacts
