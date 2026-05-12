from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
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
    from ..utils.alignment_utils import build_validity_mask, compute_overlap_mask, try_ecc_alignment
    from ..utils.io_utils import write_image
    from ..utils.viz_utils import annotate_panel, blend_with_alpha, resize_if_too_large
    try:
        from ..method_scripts.sift_ransac import SiftRansacRunner
    except Exception:
        SiftRansacRunner = None
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec
    from utils.alignment_utils import build_validity_mask, compute_overlap_mask, try_ecc_alignment
    from utils.io_utils import write_image
    from utils.viz_utils import annotate_panel, blend_with_alpha, resize_if_too_large
    try:
        from method_scripts.sift_ransac import SiftRansacRunner
    except Exception:
        SiftRansacRunner = None


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
            input_channels = int(input_channels)
            if input_channels <= 0:
                raise ValueError("input_channels must be positive")

            self.encoder = None
            if resnet34 is not None:
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained and ResNet34_Weights is not None else None
                self.encoder = resnet34(weights=weights)
                self.encoder.fc = nn.Identity()

            self.enc0 = nn.Sequential(
                nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False),
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
        self._models: dict[int, Any] = {}
        self._model_sources: dict[int, str] = {}

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path, dinov2_map: np.ndarray | None = None) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        before_img = self._read_color_image(before)
        after_img = self._read_color_image(after)
        aligned_before, aligned_after, overlap_mask, alignment_mode = self._align_after_to_before(before_img, after_img)
        prediction = self._predict_change_map(aligned_before, aligned_after, dinov2_map=dinov2_map)
        probability_map = prediction.probability_map
        threshold_value = prediction.threshold_value

        change_mask = (probability_map >= threshold_value).astype(np.uint8) * 255
        change_mask = self._postprocess_mask(change_mask, overlap_mask)
        change_pixels = int(np.count_nonzero(change_mask))
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        canvas_pixels = int(overlap_mask.size)
        change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)

        probability_u8 = (np.clip(probability_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        probability_u8 = cv2.bitwise_and(probability_u8, overlap_mask)
        probability_heatmap = cv2.applyColorMap(probability_u8, int(SIAMESE_UNET_CD_CONFIG["colormap"]))
        probability_overlay = blend_with_alpha(
            cv2.cvtColor(aligned_after, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(probability_heatmap, cv2.COLOR_BGR2RGB),
            float(SIAMESE_UNET_CD_CONFIG["overlay_alpha"]),
        )
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
            "dinov2_hint_used": bool(dinov2_map is not None),
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
        after_h, after_w = after_img.shape[:2]
        
        ecc_result = try_ecc_alignment(
            before_img,
            after_img,
            motion_type=cv2.MOTION_AFFINE,
            max_iterations=int(SIAMESE_UNET_CD_CONFIG["ecc_max_iterations"]),
            eps=float(SIAMESE_UNET_CD_CONFIG["ecc_eps"]),
            erode_kernel=(5, 5),
            allow_crop=False,
        )

        # If ECC succeeded and residual is low, use it.
        if ecc_result is not None and float(ecc_result.get("residual", 1.0)) < 0.02:
            warp_matrix = ecc_result["warp_matrix"]  # 2x3 affine matrix
            aligned_before, aligned_after, overlap_mask = self._warp_to_shared_canvas_ecc(
                before_img, after_img, warp_matrix
            )
            return aligned_before, aligned_after, overlap_mask, "ecc_affine"

        # If ECC poor quality or too high residual, try SIFT+RANSAC alignment (if available)
        if SiftRansacRunner is not None:
            try:
                sift_runner = SiftRansacRunner("sift_ransac")
                res = sift_runner._align_after_to_before(before_img, after_img)
                if res is not None:
                    aligned_before, aligned_after, overlap_mask, info, match_vis = res
                    if info.get("alignment_mode", "").startswith("homography") or info.get("alignment_mode", "").startswith("affine"):
                        return aligned_before, aligned_after, overlap_mask, info.get("alignment_mode", "sift_ransac")
            except Exception:
                pass

        # Fallback: resize after to before dimensions
        if (before_h, before_w) != (after_h, after_w):
            after_img = cv2.resize(after_img, (before_w, before_h), interpolation=cv2.INTER_LINEAR)
        overlap_mask = build_validity_mask((before_h, before_w))
        return before_img, after_img, overlap_mask, "resize_fallback"

    def _predict_change_map(self, before_img: np.ndarray, after_img: np.ndarray, dinov2_map: np.ndarray | None = None) -> _PairPrediction:
        if torch is None or nn is None:
            return self._heuristic_prediction(before_img, after_img, device_used="cpu", model_source="cv_heuristic")

        input_channels = 4 if dinov2_map is not None else 3
        model = self._load_model(input_channels=input_channels)
        device_used = self._resolve_device()
        model = model.to(device_used)
        before_tensor = self._image_to_tensor(before_img, device_used, dinov2_map=dinov2_map)
        after_tensor = self._image_to_tensor(after_img, device_used, dinov2_map=dinov2_map)

        start = perf_counter()
        with torch.no_grad():
            logits = model(before_tensor, after_tensor)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        inference_ms = (perf_counter() - start) * 1000.0
        probability = cv2.resize(probability.astype(np.float32), (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        model_source = str(self._model_sources.get(input_channels, "unknown"))
        if dinov2_map is not None:
            model_source = f"{model_source}|dinov2_hint"
        return _PairPrediction(probability, float(self.threshold), inference_ms, model_source, device_used)

    def _heuristic_prediction(self, before_img: np.ndarray, after_img: np.ndarray, device_used: str, model_source: str) -> _PairPrediction:
        start = perf_counter()
        diff = cv2.absdiff(before_img, after_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        diff_norm = cv2.normalize(diff_blur.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        inference_ms = (perf_counter() - start) * 1000.0
        return _PairPrediction(diff_norm, float(self.threshold), inference_ms, model_source, device_used)

    def _load_model(self, input_channels: int = 3):
        input_channels = int(input_channels)
        if input_channels in self._models:
            return self._models[input_channels]

        model = SiameseUNet(
            pretrained=bool(SIAMESE_UNET_CD_CONFIG["encoder_pretrained"]),
            input_channels=input_channels,
        )
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
                normalized_state = self._adapt_state_dict_input_channels(normalized_state, input_channels)
                model.load_state_dict(normalized_state, strict=False)
                model_source = str(checkpoint_path)
        self._models[input_channels] = model.eval()
        self._model_sources[input_channels] = model_source
        return self._models[input_channels]

    def _adapt_state_dict_input_channels(self, state_dict: dict[str, Any], input_channels: int) -> dict[str, Any]:
        if torch is None:
            return state_dict

        conv_key = "enc0.0.weight"
        weight = state_dict.get(conv_key)
        if weight is None or not hasattr(weight, "shape"):
            return state_dict

        # Adapt first convolution weights so RGB checkpoints can initialize RGB+hint runs.
        in_ch = int(weight.shape[1])
        if in_ch == input_channels:
            return state_dict

        if in_ch == 3 and input_channels > 3:
            mean_channel = weight.mean(dim=1, keepdim=True)
            extra = mean_channel.repeat(1, input_channels - 3, 1, 1)
            state_dict[conv_key] = torch.cat([weight, extra], dim=1)
            return state_dict

        if in_ch > input_channels:
            state_dict[conv_key] = weight[:, :input_channels, :, :]
            return state_dict

        pad = torch.zeros((int(weight.shape[0]), input_channels - in_ch, int(weight.shape[2]), int(weight.shape[3])), dtype=weight.dtype)
        state_dict[conv_key] = torch.cat([weight, pad], dim=1)
        return state_dict

    def _discover_latest_training_checkpoint(self) -> Path | None:
        weights_root = Path(__file__).resolve().parent.parent.parent / "weights"
        
        # Priority 1: best trained data is real17 finetuned weights in central weights location
        real17_finetuned = weights_root / "siamese_unet_cd_real17_finetuned_best.pt"
        if real17_finetuned.exists():
            return real17_finetuned
        
        # Priority 2: generic central weights location
        central_weights = weights_root / "siamese_unet_cd_best.pt"
        if central_weights.exists():
            return central_weights

        return None

    def _resolve_device(self) -> str:
        if self.force_cpu:
            return "cpu"
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        if self.device == "cpu":
            return "cpu"
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def _image_to_tensor(self, image: np.ndarray, device_used: str, dinov2_map: np.ndarray | None = None) -> torch.Tensor:
        resized = cv2.resize(image, (int(self.input_size), int(self.input_size)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if dinov2_map is not None:
            hint = np.asarray(dinov2_map, dtype=np.float32)
            if hint.ndim != 2:
                raise ValueError("dinov2_map must be a 2D float map")
            hint = cv2.resize(hint, (int(self.input_size), int(self.input_size)), interpolation=cv2.INTER_LINEAR)
            hint = np.clip(hint, 0.0, 1.0)
            rgb = np.concatenate([rgb, hint[:, :, None]], axis=2)
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
        if device_used == "cuda":
            tensor = tensor.cuda(non_blocking=True)
        return tensor

    def _postprocess_mask(self, mask: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
        # Ensure overlap_mask matches mask size
        if mask.shape != overlap_mask.shape:
            overlap_mask = cv2.resize(overlap_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
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

    def _warp_to_shared_canvas_ecc(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a shared canvas for both images using ECC affine transformation (like SIFT RANSAC does)."""
        before_h, before_w = before_img.shape[:2]
        after_h, after_w = after_img.shape[:2]

        # Convert 2x3 affine matrix to 3x3 for perspective transform
        affine_3x3 = np.array(
            [
                [warp_matrix[0, 0], warp_matrix[0, 1], warp_matrix[0, 2]],
                [warp_matrix[1, 0], warp_matrix[1, 1], warp_matrix[1, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Get corners of after image
        after_corners = np.float32([[0, 0], [after_w, 0], [after_w, after_h], [0, after_h]]).reshape(-1, 1, 2)
        warped_after_corners = cv2.perspectiveTransform(after_corners, affine_3x3)

        # Get before corners and compute bounding box
        before_corners = np.float32([[0, 0], [before_w, 0], [before_w, before_h], [0, before_h]]).reshape(-1, 1, 2)
        all_corners = np.vstack([before_corners, warped_after_corners]).reshape(-1, 2)
        min_x = int(np.floor(all_corners[:, 0].min()))
        min_y = int(np.floor(all_corners[:, 1].min()))
        max_x = int(np.ceil(all_corners[:, 0].max()))
        max_y = int(np.ceil(all_corners[:, 1].max()))

        canvas_w = max(1, max_x - min_x)
        canvas_h = max(1, max_y - min_y)

        # Create translation matrix to center on (0, 0)
        translation = np.array(
            [
                [1.0, 0.0, -float(min_x)],
                [0.0, 1.0, -float(min_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        before_homography = translation
        after_homography = translation @ affine_3x3

        # Warp both images to the shared canvas
        aligned_before = cv2.warpPerspective(
            before_img,
            before_homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        aligned_after = cv2.warpPerspective(
            after_img,
            after_homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Create validity masks for both warped images
        before_valid = cv2.warpPerspective(
            build_validity_mask((before_h, before_w)),
            before_homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        after_valid = cv2.warpPerspective(
            build_validity_mask((after_h, after_w)),
            after_homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Compute overlap mask as intersection of validity masks
        overlap_mask = compute_overlap_mask(before_valid, after_valid, erode_kernel=(5, 5))

        # Do NOT crop images - keep them at canvas size for proper aspect ratio preservation
        # when they get resized by the model. Only return overlap_mask as-is for masking.
        return aligned_before, aligned_after, overlap_mask

    def _compose_preview(self, before_img: np.ndarray, after_img: np.ndarray, probability_overlay: np.ndarray, mask_overlay: np.ndarray, alignment_mode: str) -> np.ndarray:
        panels = [
            annotate_panel(before_img, f"Before | {alignment_mode}"),
            annotate_panel(after_img, "After"),
            annotate_panel(probability_overlay, "Probability map"),
            annotate_panel(mask_overlay, "Change mask"),
        ]
        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])
        preview = np.vstack([top, bottom])
        return resize_if_too_large(preview, int(SIAMESE_UNET_CD_CONFIG["preview_max_width"]), int(SIAMESE_UNET_CD_CONFIG["preview_max_height"]))

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
        write_image(artifacts["aligned_before"], aligned_before)
        write_image(artifacts["aligned_after"], aligned_after)
        write_image(artifacts["overlap_mask"], overlap_mask)
        write_image(artifacts["probability_map"], probability_map)
        write_image(artifacts["probability_overlay"], cv2.cvtColor(probability_overlay, cv2.COLOR_RGB2BGR))
        write_image(artifacts["change_mask"], change_mask)
        write_image(artifacts["preview"], cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        return artifacts
