from __future__ import annotations

import os
import re
import sys
import unicodedata
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
    TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import timm
    TIMM_IMPORT_ERROR: str | None = None
except Exception as exc:
    timm = None
    TIMM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from ..core import AnalysisResult
    from ..methods import get_method_spec
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec


CHANGEFORMER_CONFIG = {
    # Pyramid Vision Transformer backbone used for feature extraction.
    "backbone_name": "pvt_v2_b0",
    # Fallback transformer backbone when the primary one cannot be loaded.
    "fallback_backbone_name": "mobilevitv2_100",
    # Optional env var that points to a supervised ChangeFormer checkpoint.
    "weights_env_var": "CHANGEFORMER_WEIGHTS",
    # Canonical checkpoint path used by the GUI when a trained model exists.
    "canonical_weights_path": "results/models/changeformer/best.pt",
    # Decoder width used by the supervised segmentation model.
    "decoder_channels": 128,
    # Side length for transformer inference.
    "input_size": 512,
    # Percentile used to convert the probability map into a binary change mask.
    "threshold_percentile": 93.0,
    # Additional threshold offset relative to Otsu on probability values.
    "otsu_threshold_offset": 0.05,
    # Morphology and component filtering settings for the final mask.
    "morph_kernel": (7, 7),
    "morph_open_iterations": 2,
    "morph_close_iterations": 1,
    "overlap_erode_kernel": 9,
    "min_component_area_px": 320,
    "min_component_area_ratio": 0.0022,
    "min_component_width_px": 10,
    "min_component_height_px": 8,
    "max_component_aspect_ratio": 8.0,
    "drop_border_components": True,
    "border_margin_px": 4,
    # Rendering settings.
    "colormap": cv2.COLORMAP_TURBO,
    "overlay_alpha": 0.58,
    "panel_title_height": 44,
    "preview_max_width": 3600,
    "preview_max_height": 2600,
    # ECC alignment settings.
    "ecc_max_iterations": 120,
    "ecc_eps": 1e-6,
    "ecc_rotation_candidates": (0, 90, 180, 270),
    "ecc_motion_models": (
        cv2.MOTION_HOMOGRAPHY,
        cv2.MOTION_AFFINE,
        cv2.MOTION_EUCLIDEAN,
    ),
    "alignment_downscale_max_side": 1400,
    # If aligned pair consistency is too low, suppress detections for precision.
    "min_scene_consistency_for_detection": 0.62,
    # Fusion weights for transformer and photometric maps.
    "transformer_map_weight": 0.84,
    "photometric_map_weight": 0.16,
    # Test-time augmentation for transformer inference.
    "enable_tta": True,
    "tta_scales": (0.85, 1.0, 1.15),
    "tta_horizontal_flip": True,
    # Suppress edge-dominated noise (tree branches, texture flicker).
    "edge_suppression_weight": 0.22,
    # Confidence filters for connected components.
    "min_component_mean_prob": 0.48,
    "min_component_peak_prob": 0.68,
    "max_component_area_ratio": 0.10,
    "min_component_center_y_ratio": 0.30,
    "top_region_peak_override": 0.88,
    # Relaxed rescue rule for compact high-confidence detections in the upper image region.
    "top_region_relaxed_peak_prob": 0.74,
    "top_region_relaxed_min_mean_prob": 0.50,
    "top_region_relaxed_max_area_ratio": 0.015,
}


if nn is not None and timm is not None:

    def _make_group_norm(num_channels: int):
        groups = min(32, max(1, int(num_channels)))
        while groups > 1 and int(num_channels) % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, int(num_channels))


    class ChangeFormerSegmentationModel(nn.Module):
        def __init__(
            self,
            backbone_name: str = str(CHANGEFORMER_CONFIG["backbone_name"]),
            decoder_channels: int = int(CHANGEFORMER_CONFIG["decoder_channels"]),
            dropout: float = 0.10,
        ):
            super().__init__()
            self.backbone_name = str(backbone_name)
            self.decoder_channels = int(decoder_channels)
            self.dropout = float(dropout)

            self.backbone, self.model_source = self._create_backbone(self.backbone_name)
            feature_channels = self._feature_channels()
            if not feature_channels:
                raise RuntimeError(f"Backbone {self.model_source} did not expose feature channels")

            self.level_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(int(channels) * 3, self.decoder_channels, kernel_size=1, bias=False),
                        _make_group_norm(self.decoder_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.decoder_channels, self.decoder_channels, kernel_size=3, padding=1, bias=False),
                        _make_group_norm(self.decoder_channels),
                        nn.ReLU(inplace=True),
                    )
                    for channels in feature_channels
                ]
            )
            self.fusion_head = nn.Sequential(
                nn.Conv2d(self.decoder_channels, self.decoder_channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(self.decoder_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout),
                nn.Conv2d(self.decoder_channels, 1, kernel_size=1),
            )

        def _create_backbone(self, backbone_name: str) -> tuple[Any, str]:
            requested = str(backbone_name)
            fallback = str(CHANGEFORMER_CONFIG["fallback_backbone_name"])
            for candidate_name in (requested, fallback):
                for pretrained in (True, False):
                    try:
                        backbone = timm.create_model(candidate_name, pretrained=pretrained, features_only=True)
                        source = candidate_name if pretrained else f"{candidate_name}_scratch"
                        return backbone, source
                    except Exception:
                        continue
            raise RuntimeError(f"Unable to create a ChangeFormer backbone from {requested}")

        def _feature_channels(self) -> list[int]:
            feature_info = getattr(self.backbone, "feature_info", None)
            if feature_info is None:
                return []
            channels = feature_info.channels() if hasattr(feature_info, "channels") else []
            return [int(channel) for channel in channels]

        def _extract_feature_list(self, outputs) -> list[Any]:
            features: list[Any] = []
            if isinstance(outputs, (list, tuple)):
                for item in outputs:
                    if torch.is_tensor(item):
                        features.append(item)
                return features

            if isinstance(outputs, dict):
                for item in outputs.values():
                    if torch.is_tensor(item):
                        features.append(item)
                return features

            if torch.is_tensor(outputs):
                return [outputs]

            return []

        def forward(self, before_tensor, after_tensor):
            before_features = self._extract_feature_list(self.backbone(before_tensor))
            after_features = self._extract_feature_list(self.backbone(after_tensor))
            level_count = min(len(before_features), len(after_features), len(self.level_blocks))
            if level_count == 0:
                raise RuntimeError("ChangeFormer backbone did not return any aligned feature levels")

            reference_size = None
            for feature in before_features[:level_count]:
                feature_size = (int(feature.shape[-2]), int(feature.shape[-1]))
                if reference_size is None:
                    reference_size = feature_size
                    continue
                if feature_size[0] * feature_size[1] > reference_size[0] * reference_size[1]:
                    reference_size = feature_size

            fused = None
            for index in range(level_count):
                before_level = before_features[index]
                after_level = after_features[index]
                if before_level.shape[-2:] != after_level.shape[-2:]:
                    target_h = min(int(before_level.shape[-2]), int(after_level.shape[-2]))
                    target_w = min(int(before_level.shape[-1]), int(after_level.shape[-1]))
                    before_level = torch.nn.functional.interpolate(
                        before_level,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    after_level = torch.nn.functional.interpolate(
                        after_level,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                diff = torch.cat(
                    [
                        before_level,
                        after_level,
                        torch.abs(before_level - after_level),
                    ],
                    dim=1,
                )
                level_map = self.level_blocks[index](diff)
                if reference_size is not None and level_map.shape[-2:] != reference_size:
                    level_map = torch.nn.functional.interpolate(
                        level_map,
                        size=reference_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                fused = level_map if fused is None else fused + level_map

            logits = self.fusion_head(fused)
            logits = torch.nn.functional.interpolate(
                logits,
                size=(int(before_tensor.shape[-2]), int(before_tensor.shape[-1])),
                mode="bilinear",
                align_corners=False,
            )
            return logits


else:

    class ChangeFormerSegmentationModel:  # pragma: no cover - only used when torch/timm are unavailable
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch and timm are required for ChangeFormer training or supervised inference")


def _strip_state_dict_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        sanitized[new_key] = value
    return sanitized


def _torch_load_checkpoint(path: Path, map_location: Any):
    if torch is None:
        raise RuntimeError("torch is not installed")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_latest_changeformer_checkpoint() -> Path | None:
    project_root = Path(__file__).resolve().parent.parent.parent

    canonical_checkpoint = project_root / str(CHANGEFORMER_CONFIG["canonical_weights_path"])
    if canonical_checkpoint.exists():
        return canonical_checkpoint

    training_root = project_root / "results" / "training"
    if not training_root.exists():
        return None

    best_candidates = [path for path in training_root.rglob("best.pt") if path.is_file()]
    if not best_candidates:
        return None

    best_candidates.sort(
        key=lambda path: (
            0 if "changeformer" in path.as_posix().lower() else 1,
            -path.stat().st_mtime,
        )
    )
    return best_candidates[0]


class ChangeformerRunner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        backbone_name: str | None = None,
        input_size: int | None = None,
        threshold_percentile: float | None = None,
        weights_path: str | Path | None = None,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = str(device)
        self.force_cpu = bool(force_cpu)
        self.backbone_name = str(backbone_name or CHANGEFORMER_CONFIG["backbone_name"])
        self.input_size = int(input_size or CHANGEFORMER_CONFIG["input_size"])
        self.threshold_percentile = float(threshold_percentile or CHANGEFORMER_CONFIG["threshold_percentile"])
        # Kept for API compatibility with other runners.
        self.weights_path = self._resolve_weights_path(weights_path)

        self._feature_model = None
        self._supervised_model = None
        self._supervised_model_source = None
        self._model_source = None
        self._torch_available = torch is not None and timm is not None
        self._inference_mode = "transformer_features" if self._torch_available else "cv_fallback"

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        before_img = self._read_color_image(before)
        after_img = self._read_color_image(after)
        aligned_before, aligned_after, overlap_mask, align_info = self._align_after_to_before(before_img, after_img)
        fallback_reason: str | None = None

        if self._torch_available:
            try:
                probability_map, inference_ms, device_used, backbone_used = self._predict_change_map_transformer(
                    aligned_before,
                    aligned_after,
                )
                self._inference_mode = "trained_checkpoint" if self.weights_path is not None else "transformer_features"
            except Exception as exc:
                probability_map, inference_ms = self._predict_change_map_cv_fallback(aligned_before, aligned_after)
                device_used = "cpu"
                backbone_used = "cv_fallback"
                self._inference_mode = "cv_fallback"
                fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            probability_map, inference_ms = self._predict_change_map_cv_fallback(aligned_before, aligned_after)
            device_used = "cpu"
            backbone_used = "cv_fallback"
            self._inference_mode = "cv_fallback"
            fallback_reason = (
                f"torch_or_timm_not_available; torch_error={TORCH_IMPORT_ERROR}; timm_error={TIMM_IMPORT_ERROR}"
            )

        scene_consistency = 1.0 - self._alignment_residual(aligned_before, aligned_after, overlap_mask)
        scene_mismatch_suppressed = scene_consistency < float(CHANGEFORMER_CONFIG["min_scene_consistency_for_detection"])
        if scene_mismatch_suppressed:
            probability_map = np.zeros_like(probability_map, dtype=np.float32)

        change_mask, threshold_value = self._probability_to_mask(probability_map, overlap_mask)
        change_pixels = int(np.count_nonzero(change_mask))
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        canvas_pixels = int(overlap_mask.size)
        change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)

        probability_u8 = (np.clip(probability_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        probability_heatmap = cv2.applyColorMap(probability_u8, int(CHANGEFORMER_CONFIG["colormap"]))
        probability_overlay = self._build_probability_overlay(aligned_after, probability_heatmap)
        mask_overlay = self._build_mask_overlay(aligned_before, aligned_after, change_mask)
        preview = self._compose_preview(
            aligned_before=aligned_before,
            aligned_after=aligned_after,
            probability_overlay=probability_overlay,
            mask_overlay=mask_overlay,
            alignment_mode=str(align_info["alignment_mode"]),
            inference_mode=self._inference_mode,
        )

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
            "analysis_mode": "changeformer_pair",
            "inference_mode": self._inference_mode,
            "model_name": self.backbone_name,
            "model_source": str(backbone_used),
            "python_executable": sys.executable,
            "torch_import_ok": TORCH_IMPORT_ERROR is None,
            "timm_import_ok": TIMM_IMPORT_ERROR is None,
            "torch_version": getattr(torch, "__version__", None),
            "timm_version": getattr(timm, "__version__", None),
            "torch_import_error": TORCH_IMPORT_ERROR,
            "timm_import_error": TIMM_IMPORT_ERROR,
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "device_requested": self.device,
            "device_used": device_used,
            "force_cpu": bool(self.force_cpu),
            "cuda_available": bool(torch is not None and torch.cuda.is_available()),
            "input_size": int(self.input_size),
            "threshold_percentile": float(self.threshold_percentile),
            "threshold_value": round(float(threshold_value), 6),
            "alignment_mode": str(align_info["alignment_mode"]),
            "alignment_ecc_score": None
            if align_info["ecc_score"] is None
            else round(float(align_info["ecc_score"]), 6),
            "alignment_quality": round(float(align_info.get("alignment_quality", 0.0)), 6),
            "alignment_rotation_deg": int(align_info.get("rotation_deg", 0)),
            "scene_consistency": round(float(scene_consistency), 6),
            "scene_mismatch_suppressed": bool(scene_mismatch_suppressed),
            "fallback_reason": fallback_reason,
            "change_pixels": change_pixels,
            "overlap_pixels": overlap_pixels,
            "canvas_pixels": canvas_pixels,
            "overlap_ratio": round(overlap_pixels / max(canvas_pixels, 1), 6),
            "change_ratio": change_ratio,
            "inference_ms": round(float(inference_ms), 3),
        }

        summary = (
            "ChangeFormer-style inference on a before/after pair using multi-scale transformer feature differencing "
            "with automatic fallback to a classical CV map."
        )
        preview_text = (
            f"Mode: {self._inference_mode}\n"
            f"Backbone: {backbone_used}\n"
            f"Device: {device_used} (requested: {self.device})\n"
            f"Alignment: {metrics['alignment_mode']}\n"
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

    def _align_after_to_before(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        before_h, before_w = before_img.shape[:2]
        if (before_h, before_w) != after_img.shape[:2]:
            resized_after = cv2.resize(after_img, (before_w, before_h), interpolation=cv2.INTER_LINEAR)
            default_mode = "resize_then_alignment"
        else:
            resized_after = after_img.copy()
            default_mode = "identity_then_alignment"

        before_ref = before_img.copy()
        best_candidate: dict[str, Any] | None = None

        for rotation_deg in CHANGEFORMER_CONFIG["ecc_rotation_candidates"]:
            rotated_after = self._rotate_to_reference_shape(resized_after, int(rotation_deg), before_w, before_h)
            for motion_type in CHANGEFORMER_CONFIG["ecc_motion_models"]:
                candidate = self._try_ecc_alignment(before_ref, rotated_after, int(rotation_deg), int(motion_type))
                if candidate is None:
                    continue
                if best_candidate is None or candidate["quality"] > best_candidate["quality"]:
                    best_candidate = candidate

        if best_candidate is not None:
            info = {
                "alignment_mode": best_candidate["mode"],
                "ecc_score": best_candidate["ecc_score"],
                "alignment_quality": best_candidate["quality"],
                "rotation_deg": best_candidate["rotation_deg"],
            }
            return before_ref, best_candidate["aligned_after"], best_candidate["overlap_mask"], info

        orb_candidate = self._align_with_orb_homography(before_ref, resized_after)
        if orb_candidate is not None:
            info = {
                "alignment_mode": "orb_homography",
                "ecc_score": None,
                "alignment_quality": orb_candidate["quality"],
                "rotation_deg": int(orb_candidate["rotation_deg"]),
            }
            return before_ref, orb_candidate["aligned_after"], orb_candidate["overlap_mask"], info

        fallback_overlap = np.full((before_h, before_w), 255, dtype=np.uint8)
        info = {
            "alignment_mode": f"{default_mode}_fallback",
            "ecc_score": None,
            "alignment_quality": 0.0,
            "rotation_deg": 0,
        }
        return before_ref, resized_after, fallback_overlap, info

    def _rotate_to_reference_shape(self, image: np.ndarray, rotation_deg: int, target_w: int, target_h: int) -> np.ndarray:
        if rotation_deg == 0:
            rotated = image
        elif rotation_deg == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_deg == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_deg == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = image
        if rotated.shape[1] != target_w or rotated.shape[0] != target_h:
            rotated = cv2.resize(rotated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return rotated

    def _prepare_alignment_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        return enhanced.astype(np.float32) / 255.0

    def _try_ecc_alignment(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        rotation_deg: int,
        motion_type: int,
    ) -> dict[str, Any] | None:
        before_h, before_w = before_img.shape[:2]
        max_side = int(CHANGEFORMER_CONFIG["alignment_downscale_max_side"])
        scale = min(max_side / max(before_h, before_w, 1), 1.0)
        work_size = (before_w, before_h)
        if scale < 1.0:
            work_size = (max(64, int(before_w * scale)), max(64, int(before_h * scale)))

        before_work = cv2.resize(before_img, work_size, interpolation=cv2.INTER_AREA)
        after_work = cv2.resize(after_img, work_size, interpolation=cv2.INTER_AREA)
        before_gray = self._prepare_alignment_gray(before_work)
        after_gray = self._prepare_alignment_gray(after_work)

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(CHANGEFORMER_CONFIG["ecc_max_iterations"]),
            float(CHANGEFORMER_CONFIG["ecc_eps"]),
        )

        try:
            ecc_score, warp_matrix = cv2.findTransformECC(
                templateImage=before_gray,
                inputImage=after_gray,
                warpMatrix=warp_matrix,
                motionType=motion_type,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error:
            return None

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            aligned_after = cv2.warpPerspective(
                after_img,
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            warped_valid = cv2.warpPerspective(
                np.full((before_h, before_w), 255, dtype=np.uint8),
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            motion_name = "homography"
        else:
            aligned_after = cv2.warpAffine(
                after_img,
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            warped_valid = cv2.warpAffine(
                np.full((before_h, before_w), 255, dtype=np.uint8),
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            motion_name = "affine" if motion_type == cv2.MOTION_AFFINE else "euclidean"

        overlap_mask = cv2.bitwise_and(
            np.full((before_h, before_w), 255, dtype=np.uint8),
            warped_valid,
        )
        overlap_ratio = float(np.count_nonzero(overlap_mask) / max(overlap_mask.size, 1))
        if overlap_ratio <= 0.05:
            return None

        residual = self._alignment_residual(before_img, aligned_after, overlap_mask)
        quality = float((float(ecc_score) + 1.0) * overlap_ratio * (1.0 - residual))
        return {
            "aligned_after": aligned_after,
            "overlap_mask": overlap_mask,
            "ecc_score": float(ecc_score),
            "quality": quality,
            "rotation_deg": int(rotation_deg),
            "mode": f"ecc_{motion_name}_rot{int(rotation_deg)}",
        }

    def _align_with_orb_homography(self, before_img: np.ndarray, after_img: np.ndarray) -> dict[str, Any] | None:
        before_h, before_w = before_img.shape[:2]
        orb = cv2.ORB_create(nfeatures=6000)

        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        kp_before, desc_before = orb.detectAndCompute(before_gray, None)
        if desc_before is None or len(kp_before) < 12:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        best_candidate: dict[str, Any] | None = None

        for rotation_deg in CHANGEFORMER_CONFIG["ecc_rotation_candidates"]:
            candidate_after = self._rotate_to_reference_shape(after_img, int(rotation_deg), before_w, before_h)
            after_gray = cv2.cvtColor(candidate_after, cv2.COLOR_BGR2GRAY)
            kp_after, desc_after = orb.detectAndCompute(after_gray, None)
            if desc_after is None or len(kp_after) < 12:
                continue

            raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)
            good_matches = []
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                first, second = pair
                if first.distance < 0.78 * second.distance:
                    good_matches.append(first)

            if len(good_matches) < 10:
                continue

            src_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if homography is None:
                continue

            aligned_after = cv2.warpPerspective(
                candidate_after,
                homography,
                (before_w, before_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            warped_valid = cv2.warpPerspective(
                np.full((before_h, before_w), 255, dtype=np.uint8),
                homography,
                (before_w, before_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            overlap_mask = cv2.bitwise_and(np.full((before_h, before_w), 255, dtype=np.uint8), warped_valid)
            overlap_ratio = float(np.count_nonzero(overlap_mask) / max(overlap_mask.size, 1))
            if overlap_ratio <= 0.05:
                continue

            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            residual = self._alignment_residual(before_img, aligned_after, overlap_mask)
            quality = float((inliers / max(len(good_matches), 1)) * overlap_ratio * (1.0 - residual))
            candidate = {
                "aligned_after": aligned_after,
                "overlap_mask": overlap_mask,
                "quality": quality,
                "rotation_deg": int(rotation_deg),
            }
            if best_candidate is None or candidate["quality"] > best_candidate["quality"]:
                best_candidate = candidate

        return best_candidate

    def _alignment_residual(self, before_img: np.ndarray, aligned_after: np.ndarray, overlap_mask: np.ndarray) -> float:
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        after_gray = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(before_gray - after_gray)
        valid = overlap_mask > 0
        if not np.any(valid):
            return 1.0
        residual = float(np.mean(diff[valid]) / 255.0)
        return float(np.clip(residual, 0.0, 1.0))

    def _predict_change_map_transformer(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
    ) -> tuple[np.ndarray, float, str, str]:
        if torch is None or timm is None:
            raise RuntimeError("torch/timm are not installed in the active environment")

        if self.weights_path is not None:
            device_obj, device_used = self._resolve_device()
            model, backbone_used = self._load_supervised_model(device_obj)
            start = perf_counter()
            tta_maps: list[np.ndarray] = []
            tta_scales = CHANGEFORMER_CONFIG["tta_scales"] if CHANGEFORMER_CONFIG.get("enable_tta", False) else (1.0,)

            for scale in tta_scales:
                side = max(128, int(round(self.input_size * float(scale))))
                tta_maps.append(self._infer_supervised_map_single(model, device_obj, before_img, after_img, side=side, hflip=False))
                if bool(CHANGEFORMER_CONFIG.get("tta_horizontal_flip", False)):
                    tta_maps.append(self._infer_supervised_map_single(model, device_obj, before_img, after_img, side=side, hflip=True))

            if not tta_maps:
                raise RuntimeError("No supervised ChangeFormer TTA maps were generated")

            map_np = np.mean(np.stack(tta_maps, axis=0), axis=0).astype(np.float32)
            map_np = self._apply_edge_suppression(map_np, before_img, after_img)
            inference_ms = (perf_counter() - start) * 1000.0
            return map_np, inference_ms, device_used, backbone_used

        device_obj, device_used = self._resolve_device()
        model, backbone_used = self._load_feature_model(device_obj)

        start = perf_counter()
        tta_maps: list[np.ndarray] = []
        tta_scales = CHANGEFORMER_CONFIG["tta_scales"] if CHANGEFORMER_CONFIG.get("enable_tta", False) else (1.0,)

        for scale in tta_scales:
            side = max(128, int(round(self.input_size * float(scale))))
            tta_maps.append(self._infer_transformer_map_single(model, device_obj, before_img, after_img, side=side, hflip=False))
            if bool(CHANGEFORMER_CONFIG.get("tta_horizontal_flip", False)):
                tta_maps.append(self._infer_transformer_map_single(model, device_obj, before_img, after_img, side=side, hflip=True))

        if not tta_maps:
            raise RuntimeError("No transformer TTA maps were generated")

        map_np = np.mean(np.stack(tta_maps, axis=0), axis=0).astype(np.float32)
        map_np = self._normalize_to_unit_interval(map_np)
        photometric_map = self._build_photometric_change_map(before_img, after_img)
        map_np = (
            float(CHANGEFORMER_CONFIG["transformer_map_weight"]) * map_np
            + float(CHANGEFORMER_CONFIG["photometric_map_weight"]) * photometric_map
        )
        map_np = self._normalize_to_unit_interval(map_np)
        map_np = self._apply_edge_suppression(map_np, before_img, after_img)
        inference_ms = (perf_counter() - start) * 1000.0
        return map_np, inference_ms, device_used, backbone_used

    def _resolve_weights_path(self, weights_path: str | Path | None) -> Path | None:
        if weights_path is not None:
            return Path(weights_path)

        env_value = os.environ.get(str(CHANGEFORMER_CONFIG["weights_env_var"]))
        if env_value:
            return Path(env_value)

        return resolve_latest_changeformer_checkpoint()

    def _load_supervised_model(self, device_obj) -> tuple[Any, str]:
        if self._supervised_model is not None and self._supervised_model_source is not None:
            self._supervised_model = self._supervised_model.to(device_obj)
            self._supervised_model.eval()
            return self._supervised_model, str(self._supervised_model_source)

        if self.weights_path is None:
            raise RuntimeError("No ChangeFormer checkpoint is available")

        checkpoint = _torch_load_checkpoint(self.weights_path, map_location=device_obj)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
            backbone_name = str(checkpoint.get("backbone_name") or self.backbone_name)
            decoder_channels = int(checkpoint.get("decoder_channels") or CHANGEFORMER_CONFIG["decoder_channels"])
            dropout = float(checkpoint.get("dropout") or 0.10)
        else:
            state_dict = checkpoint
            backbone_name = self.backbone_name
            decoder_channels = int(CHANGEFORMER_CONFIG["decoder_channels"])
            dropout = 0.10

        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported checkpoint payload in {self.weights_path}")

        model = ChangeFormerSegmentationModel(
            backbone_name=backbone_name,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        model.load_state_dict(_strip_state_dict_prefix(state_dict), strict=True)
        model = model.to(device_obj).eval()
        self._supervised_model = model
        self._supervised_model_source = self.weights_path
        return model, str(self.weights_path)

    def _infer_supervised_map_single(
        self,
        model,
        device_obj,
        before_img: np.ndarray,
        after_img: np.ndarray,
        side: int,
        hflip: bool,
    ) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch is not installed")

        if hflip:
            before_work = cv2.flip(before_img, 1)
            after_work = cv2.flip(after_img, 1)
        else:
            before_work = before_img
            after_work = after_img

        before_tensor = self._preprocess_for_model(before_work, side=side).to(device_obj)
        after_tensor = self._preprocess_for_model(after_work, side=side).to(device_obj)

        with torch.no_grad():
            logits = model(before_tensor, after_tensor)
            probability = torch.sigmoid(logits)

        probability = torch.nn.functional.interpolate(
            probability,
            size=(before_work.shape[0], before_work.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        map_np = probability[0, 0].detach().float().cpu().numpy().astype(np.float32)
        if hflip:
            map_np = cv2.flip(map_np, 1)
        return np.clip(map_np, 0.0, 1.0)

    def _infer_transformer_map_single(
        self,
        model,
        device_obj,
        before_img: np.ndarray,
        after_img: np.ndarray,
        side: int,
        hflip: bool,
    ) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch is not installed")

        if hflip:
            before_work = cv2.flip(before_img, 1)
            after_work = cv2.flip(after_img, 1)
        else:
            before_work = before_img
            after_work = after_img

        before_tensor = self._preprocess_for_model(before_work, side=side).to(device_obj)
        after_tensor = self._preprocess_for_model(after_work, side=side).to(device_obj)

        with torch.no_grad():
            before_features = self._model_features(model, before_tensor)
            after_features = self._model_features(model, after_tensor)
            map_tensor = self._build_feature_change_map(before_features, after_features)

        if map_tensor.ndim != 4:
            raise RuntimeError("Unexpected feature map shape from transformer backbone")

        map_tensor = torch.nn.functional.interpolate(
            map_tensor,
            size=(before_work.shape[0], before_work.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        map_np = map_tensor[0, 0].detach().float().cpu().numpy()
        map_np = self._normalize_to_unit_interval(map_np)
        if hflip:
            map_np = cv2.flip(map_np, 1)
        return map_np.astype(np.float32)

    def _predict_change_map_cv_fallback(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        start = perf_counter()
        photo_map = self._build_photometric_change_map(before_img, after_img)
        structure_map = self._build_structure_change_map(before_img, after_img)
        merged = (0.68 * photo_map) + (0.32 * structure_map)
        merged = cv2.GaussianBlur(merged.astype(np.float32), (5, 5), 0)
        merged = self._normalize_to_unit_interval(merged)
        merged = self._apply_edge_suppression(merged, before_img, after_img)
        inference_ms = (perf_counter() - start) * 1000.0
        return merged.astype(np.float32), inference_ms

    def _apply_edge_suppression(self, probability_map: np.ndarray, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
        weight = float(CHANGEFORMER_CONFIG.get("edge_suppression_weight", 0.0))
        if weight <= 0.0:
            return probability_map

        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        edges_before = cv2.Canny(before_gray, 60, 140)
        edges_after = cv2.Canny(after_gray, 60, 140)
        edge_union = cv2.bitwise_or(edges_before, edges_after)
        edge_union = cv2.dilate(edge_union, np.ones((3, 3), np.uint8), iterations=1)
        edge_weight = edge_union.astype(np.float32) / 255.0

        suppressed = probability_map * (1.0 - (weight * edge_weight))
        return self._normalize_to_unit_interval(np.clip(suppressed, 0.0, 1.0))

    def _build_photometric_change_map(self, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
        lab_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        ab_delta = np.linalg.norm(lab_before[:, :, 1:3] - lab_after[:, :, 1:3], axis=2)
        ab_delta = self._normalize_to_unit_interval(ab_delta)

        hsv_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_diff = np.abs(hsv_before[:, :, 0] - hsv_after[:, :, 0])
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0
        sat_diff = np.abs(hsv_before[:, :, 1] - hsv_after[:, :, 1]) / 255.0

        photometric = (0.56 * ab_delta) + (0.28 * hue_diff) + (0.16 * sat_diff)
        photometric = cv2.GaussianBlur(photometric.astype(np.float32), (5, 5), 0)
        return self._normalize_to_unit_interval(photometric)

    def _build_structure_change_map(self, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        before_grad_x = cv2.Sobel(before_gray, cv2.CV_32F, 1, 0, ksize=3)
        before_grad_y = cv2.Sobel(before_gray, cv2.CV_32F, 0, 1, ksize=3)
        after_grad_x = cv2.Sobel(after_gray, cv2.CV_32F, 1, 0, ksize=3)
        after_grad_y = cv2.Sobel(after_gray, cv2.CV_32F, 0, 1, ksize=3)

        before_mag = cv2.magnitude(before_grad_x, before_grad_y)
        after_mag = cv2.magnitude(after_grad_x, after_grad_y)
        grad_delta = np.abs(before_mag - after_mag)
        return self._normalize_to_unit_interval(grad_delta)

    def _resolve_device(self) -> tuple[Any, str]:
        if torch is None:
            raise RuntimeError("torch is not installed")

        if self.force_cpu:
            return torch.device("cpu"), "cpu"

        requested = self.device.strip().lower()
        cuda_available = torch.cuda.is_available()

        if requested == "cpu":
            return torch.device("cpu"), "cpu"
        if requested == "cuda":
            if cuda_available:
                return torch.device("cuda"), "cuda"
            return torch.device("cpu"), "cpu"
        if requested == "auto" and cuda_available:
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    def _load_feature_model(self, device_obj) -> tuple[Any, str]:
        requested_backbone = self.backbone_name
        if self._feature_model is not None and self._model_source == requested_backbone:
            self._feature_model = self._feature_model.to(device_obj)
            self._feature_model.eval()
            return self._feature_model, requested_backbone

        model = None
        used_backbone = requested_backbone
        try:
            model = timm.create_model(requested_backbone, pretrained=True, features_only=True)
        except Exception:
            fallback_backbone = str(CHANGEFORMER_CONFIG["fallback_backbone_name"])
            model = timm.create_model(fallback_backbone, pretrained=True, features_only=True)
            used_backbone = fallback_backbone

        model = model.to(device_obj).eval()
        self._feature_model = model
        self._model_source = used_backbone
        return model, used_backbone

    def _preprocess_for_model(self, image_bgr: np.ndarray, side: int | None = None):
        if torch is None:
            raise RuntimeError("torch is not installed")

        work_side = max(128, int(side or self.input_size))
        resized = cv2.resize(image_bgr, (work_side, work_side), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0

        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std

        chw = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor

    def _model_features(self, model, image_tensor) -> list[Any]:
        outputs = model(image_tensor)
        features = self._extract_feature_list(outputs)
        if not features:
            raise RuntimeError("Feature model returned no usable features")
        return features

    def _extract_feature_list(self, outputs) -> list[Any]:
        if torch is None:
            return []

        features: list[Any] = []
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if torch.is_tensor(item):
                    features.append(item)
            return features

        if isinstance(outputs, dict):
            for item in outputs.values():
                if torch.is_tensor(item):
                    features.append(item)
            return features

        if torch.is_tensor(outputs):
            return [outputs]

        return []

    def _build_feature_change_map(self, before_features: list[Any], after_features: list[Any]):
        if torch is None:
            raise RuntimeError("torch is not installed")

        level_count = min(len(before_features), len(after_features))
        if level_count == 0:
            raise RuntimeError("No paired feature levels available")

        weighted_sum = None
        total_weight = 0.0
        reference_size: tuple[int, int] | None = None

        for index in range(level_count):
            before_level = self._to_spatial_feature(before_features[index])
            after_level = self._to_spatial_feature(after_features[index])

            if before_level.shape != after_level.shape:
                target_h = min(before_level.shape[-2], after_level.shape[-2])
                target_w = min(before_level.shape[-1], after_level.shape[-1])
                before_level = torch.nn.functional.interpolate(
                    before_level,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                after_level = torch.nn.functional.interpolate(
                    after_level,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

            l1_map = torch.mean(torch.abs(before_level - after_level), dim=1, keepdim=True)
            before_norm = torch.nn.functional.normalize(before_level, p=2, dim=1, eps=1e-6)
            after_norm = torch.nn.functional.normalize(after_level, p=2, dim=1, eps=1e-6)
            cosine_map = 1.0 - torch.sum(before_norm * after_norm, dim=1, keepdim=True)
            diff_map = (0.68 * l1_map) + (0.32 * cosine_map)

            if reference_size is None:
                reference_size = (int(diff_map.shape[-2]), int(diff_map.shape[-1]))
            elif diff_map.shape[-2:] != reference_size:
                diff_map = torch.nn.functional.interpolate(
                    diff_map,
                    size=reference_size,
                    mode="bilinear",
                    align_corners=False,
                )

            weight = float(index + 1)
            total_weight += weight
            weighted_sum = diff_map * weight if weighted_sum is None else weighted_sum + (diff_map * weight)

        combined = weighted_sum / max(total_weight, 1e-6)
        min_val = torch.amin(combined, dim=(2, 3), keepdim=True)
        max_val = torch.amax(combined, dim=(2, 3), keepdim=True)
        combined = (combined - min_val) / (max_val - min_val + 1e-8)
        return combined

    def _to_spatial_feature(self, tensor):
        if torch is None:
            raise RuntimeError("torch is not installed")

        if tensor.ndim == 4:
            return tensor

        if tensor.ndim == 3:
            batch, token_count, channels = tensor.shape
            if token_count <= 1:
                raise RuntimeError("Token feature map is too small")

            candidate_token_count = token_count - 1
            side = int(round(candidate_token_count ** 0.5))
            if side * side == candidate_token_count:
                tensor = tensor[:, 1:, :]
                token_count = candidate_token_count
            else:
                side = int(round(token_count ** 0.5))
                if side * side != token_count:
                    raise RuntimeError("Cannot reshape token sequence into a 2D map")

            spatial = tensor.transpose(1, 2).reshape(batch, channels, side, side)
            return spatial

        raise RuntimeError(f"Unsupported tensor rank for feature map: {tensor.ndim}")

    def _normalize_to_unit_interval(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if max_value - min_value < 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        return (image - min_value) / (max_value - min_value)

    def _probability_to_mask(self, probability_map: np.ndarray, overlap_mask: np.ndarray) -> tuple[np.ndarray, float]:
        stable_overlap = overlap_mask.copy()
        erode_size = int(CHANGEFORMER_CONFIG.get("overlap_erode_kernel", 0))
        if erode_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            stable_overlap = cv2.erode(stable_overlap, kernel, iterations=1)

        valid_values = probability_map[stable_overlap > 0]
        if valid_values.size == 0:
            return np.zeros_like(overlap_mask, dtype=np.uint8), 1.0

        percentile_threshold = float(np.percentile(valid_values, self.threshold_percentile))
        otsu_threshold = self._otsu_threshold_from_values(valid_values)
        threshold_value = max(
            percentile_threshold,
            otsu_threshold + float(CHANGEFORMER_CONFIG["otsu_threshold_offset"]),
        )
        threshold_value = float(np.clip(threshold_value, 0.0, 0.995))
        raw_mask = np.where(probability_map >= threshold_value, 255, 0).astype(np.uint8)
        raw_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=stable_overlap)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(CHANGEFORMER_CONFIG["morph_kernel"]))
        opened = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=int(CHANGEFORMER_CONFIG["morph_open_iterations"]),
        )
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(CHANGEFORMER_CONFIG["morph_close_iterations"]),
        )

        filtered = self._remove_small_components(closed, stable_overlap, probability_map)
        filtered = cv2.bitwise_and(filtered, filtered, mask=overlap_mask)
        return filtered, threshold_value

    def _otsu_threshold_from_values(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 1.0
        clipped = np.clip(values.astype(np.float32), 0.0, 1.0)
        bins = 256
        hist, _ = np.histogram(clipped, bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return 1.0

        probability = hist / total
        cumulative_prob = np.cumsum(probability)
        cumulative_mean = np.cumsum(probability * np.arange(bins))
        global_mean = cumulative_mean[-1]

        denominator = cumulative_prob * (1.0 - cumulative_prob)
        denominator = np.where(denominator <= 0.0, np.nan, denominator)
        between = ((global_mean * cumulative_prob - cumulative_mean) ** 2) / denominator
        best_idx = int(np.nanargmax(between)) if np.any(np.isfinite(between)) else int(self.threshold_percentile / 100.0 * (bins - 1))
        return float(best_idx / (bins - 1))

    def _remove_small_components(self, mask: np.ndarray, overlap_mask: np.ndarray, probability_map: np.ndarray) -> np.ndarray:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        is_transformer_mode = self._inference_mode == "transformer_features"

        min_area = max(
            int(CHANGEFORMER_CONFIG["min_component_area_px"]),
            int(overlap_pixels * float(CHANGEFORMER_CONFIG["min_component_area_ratio"])),
        )
        if not is_transformer_mode:
            min_area = max(48, int(min_area * 0.72))

        min_width = int(CHANGEFORMER_CONFIG["min_component_width_px"])
        min_height = int(CHANGEFORMER_CONFIG["min_component_height_px"])
        max_aspect_ratio = float(CHANGEFORMER_CONFIG["max_component_aspect_ratio"])
        max_area_ratio = float(CHANGEFORMER_CONFIG["max_component_area_ratio"])
        if not is_transformer_mode:
            max_area_ratio *= 1.3
        max_component_area = int(overlap_pixels * max_area_ratio)

        min_center_y_ratio = float(CHANGEFORMER_CONFIG["min_component_center_y_ratio"])
        top_region_peak_override = float(CHANGEFORMER_CONFIG["top_region_peak_override"])
        top_region_relaxed_peak_prob = float(CHANGEFORMER_CONFIG["top_region_relaxed_peak_prob"])
        top_region_relaxed_min_mean_prob = float(CHANGEFORMER_CONFIG["top_region_relaxed_min_mean_prob"])
        top_region_relaxed_max_area_ratio = float(CHANGEFORMER_CONFIG["top_region_relaxed_max_area_ratio"])
        top_region_relaxed_max_area = int(overlap_pixels * top_region_relaxed_max_area_ratio)

        min_component_mean_prob = float(CHANGEFORMER_CONFIG["min_component_mean_prob"])
        min_component_peak_prob = float(CHANGEFORMER_CONFIG["min_component_peak_prob"])
        if not is_transformer_mode:
            min_component_mean_prob *= 0.90
            min_component_peak_prob *= 0.90
            min_center_y_ratio = max(0.0, min_center_y_ratio - 0.03)
            top_region_peak_override = max(0.75, top_region_peak_override - 0.03)
            top_region_relaxed_peak_prob = max(0.65, top_region_relaxed_peak_prob - 0.03)
            top_region_relaxed_min_mean_prob *= 0.95
            top_region_relaxed_max_area = int(top_region_relaxed_max_area * 1.2)

        drop_border = bool(CHANGEFORMER_CONFIG["drop_border_components"])
        border_margin = int(CHANGEFORMER_CONFIG["border_margin_px"])
        height, width = mask.shape[:2]

        filtered = np.zeros_like(mask)
        for label_id in range(1, component_count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            left = int(stats[label_id, cv2.CC_STAT_LEFT])
            top = int(stats[label_id, cv2.CC_STAT_TOP])
            comp_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            comp_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if area < min_area:
                continue
            if max_component_area > 0 and area > max_component_area:
                continue
            if comp_width < min_width or comp_height < min_height:
                continue
            aspect_ratio = max(comp_width / max(comp_height, 1), comp_height / max(comp_width, 1))
            if aspect_ratio > max_aspect_ratio:
                continue
            if drop_border:
                right = left + comp_width
                bottom = top + comp_height
                touches_border = (
                    left <= border_margin
                    or top <= border_margin
                    or right >= (width - border_margin)
                    or bottom >= (height - border_margin)
                )
                if touches_border and area < (min_area * 3):
                    continue
            component_pixels = probability_map[labels == label_id]
            if component_pixels.size == 0:
                continue
            mean_prob = float(np.mean(component_pixels))
            peak_prob = float(np.percentile(component_pixels, 92))
            if mean_prob < min_component_mean_prob:
                continue
            if peak_prob < min_component_peak_prob:
                continue
            center_y_ratio = float((top + (0.5 * comp_height)) / max(height, 1))
            if center_y_ratio < min_center_y_ratio and peak_prob < top_region_peak_override:
                allow_relaxed_top_region = (
                    peak_prob >= top_region_relaxed_peak_prob
                    and mean_prob >= top_region_relaxed_min_mean_prob
                    and (top_region_relaxed_max_area <= 0 or area <= top_region_relaxed_max_area)
                )
                if not allow_relaxed_top_region:
                    continue
            filtered[labels == label_id] = 255
        return filtered

    def _build_probability_overlay(self, image: np.ndarray, probability_heatmap: np.ndarray) -> np.ndarray:
        alpha = float(CHANGEFORMER_CONFIG["overlay_alpha"])
        return cv2.addWeighted(image, 1.0 - alpha, probability_heatmap, alpha, 0)

    def _build_mask_overlay(self, before_img: np.ndarray, after_img: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
        base = cv2.addWeighted(before_img, 0.5, after_img, 0.5, 0)
        colored = np.zeros_like(base)
        colored[:, :, 2] = change_mask
        colored[:, :, 1] = (change_mask // 2)
        overlay = cv2.addWeighted(base, 1.0, colored, 0.72, 0)

        contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, contourIdx=-1, color=(255, 255, 255), thickness=1)
        return overlay

    def _compose_preview(
        self,
        aligned_before: np.ndarray,
        aligned_after: np.ndarray,
        probability_overlay: np.ndarray,
        mask_overlay: np.ndarray,
        alignment_mode: str,
        inference_mode: str,
    ) -> np.ndarray:
        panel_before = self._annotate_panel(aligned_before, "Before (aligned frame)")
        panel_after = self._annotate_panel(aligned_after, f"After aligned ({alignment_mode})")
        panel_prob = self._annotate_panel(probability_overlay, f"Change probability ({inference_mode})")
        panel_mask = self._annotate_panel(mask_overlay, "Binary change mask overlay")

        top = np.hstack([panel_before, panel_after])
        bottom = np.hstack([panel_prob, panel_mask])
        grid = np.vstack([top, bottom])
        return self._resize_if_too_large(
            grid,
            max_width=int(CHANGEFORMER_CONFIG["preview_max_width"]),
            max_height=int(CHANGEFORMER_CONFIG["preview_max_height"]),
        )

    def _annotate_panel(self, image: np.ndarray, title: str) -> np.ndarray:
        panel = image.copy()
        bar_h = int(CHANGEFORMER_CONFIG["panel_title_height"])
        cv2.rectangle(panel, (0, 0), (panel.shape[1], bar_h), (0, 0, 0), thickness=-1)
        cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    def _resize_if_too_large(self, image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
        if scale >= 1.0:
            return image
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _prepare_output_dir(self, before: Path, after: Path) -> Path:
        root = Path(__file__).resolve().parent.parent.parent / "results"
        pair_name = self._pair_folder_name(before, after)
        method_name = self._sanitize_folder_component(self.label)
        timestamp = datetime.now().strftime("%d.%m.%Y %H-%M")
        run_dir_name = f"{pair_name}__{method_name}__{timestamp}__{uuid4().hex[:6]}"
        output_dir = root / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _pair_folder_name(self, before: Path, after: Path) -> str:
        before_parent = before.parent.name.strip() or "pair"
        after_parent = after.parent.name.strip() or "pair"
        if before.parent == after.parent:
            return self._sanitize_folder_component(before_parent)
        if before_parent == after_parent:
            return self._sanitize_folder_component(before_parent)
        return self._sanitize_folder_component(f"{before_parent}_and_{after_parent}")

    def _sanitize_folder_component(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_text = ascii_text.strip().replace("/", "_").replace("\\", "_").replace(":", "-")
        ascii_text = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text)
        ascii_text = re.sub(r"_+", "_", ascii_text).strip("._-")
        return ascii_text or "pair"

    def _save_artifacts(
        self,
        output_dir: Path,
        aligned_before: np.ndarray,
        aligned_after: np.ndarray,
        overlap_mask: np.ndarray,
        probability_map: np.ndarray,
        probability_overlay: np.ndarray,
        change_mask: np.ndarray,
        preview: np.ndarray,
    ) -> dict[str, Path]:
        paths = {
            "aligned_before": output_dir / "aligned_before.png",
            "aligned_after": output_dir / "aligned_after.png",
            "overlap_mask": output_dir / "overlap_mask.png",
            "change_probability": output_dir / "change_probability.png",
            "change_probability_overlay": output_dir / "change_probability_overlay.png",
            "change_mask": output_dir / "change_mask.png",
            "preview": output_dir / "preview.png",
        }

        self._write_image(paths["aligned_before"], aligned_before)
        self._write_image(paths["aligned_after"], aligned_after)
        self._write_image(paths["overlap_mask"], overlap_mask)
        self._write_image(paths["change_probability"], probability_map)
        self._write_image(paths["change_probability_overlay"], probability_overlay)
        self._write_image(paths["change_mask"], change_mask)
        self._write_image(paths["preview"], preview)
        return paths

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Failed to encode image: {path}")
        path.write_bytes(encoded.tobytes())
