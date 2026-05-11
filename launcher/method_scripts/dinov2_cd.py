from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

try:
    from PIL import Image  # noqa: F401
except Exception:
    Image = None

try:
    import torch
    TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:
    torch = None
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


DINOV2_CD_CONFIG = {
    "backbone_name": "vit_base_patch14_dinov2",
    "backbone_candidates": (
        "vit_base_patch14_dinov2",
        "dinov2_vitb14",
        "dinov2_vits14",
    ),
    "input_size": 518,
    "semantic_threshold_percentile": 85.0,
    "colormap": cv2.COLORMAP_TURBO,
    "overlay_alpha": 0.60,
    "panel_title_height": 44,
    "preview_max_width": 3400,
    "preview_max_height": 2400,
    "ecc_max_iterations": 100,
    "ecc_eps": 1e-6,
    "postproc_median_ksize": 7,
    "postproc_min_component_area": 75,
    "border_sum_threshold": 16,
}


class DinoV2CdRunner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        backbone_name: str | None = None,
        input_size: int | None = None,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = str(device)
        self.force_cpu = bool(force_cpu)
        self.backbone_name = str(backbone_name or DINOV2_CD_CONFIG["backbone_name"])
        self.input_size = int(input_size or DINOV2_CD_CONFIG["input_size"])

        self._feature_model = None
        self._feature_model_source = None

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        before_img = self._read_color_image(before)
        after_img = self._read_color_image(after)
        aligned_before, aligned_after, overlap_mask, alignment_mode = self._align_after_to_before(before_img, after_img)

        probability_map, inference_ms, model_source, device_used, fallback_reason = self._predict_semantic_change_map(
            aligned_before,
            aligned_after,
        )

        # Post-process probability map: median denoise, remove small components, ignore padded borders
        cleaned_prob_map, cleaned_mask = self._postprocess_probability_map(probability_map, overlap_mask, aligned_after)

        scene_consistency = 1.0 - self._alignment_residual(aligned_before, aligned_after, overlap_mask)
        threshold = float(np.percentile(cleaned_prob_map[overlap_mask > 0], DINOV2_CD_CONFIG["semantic_threshold_percentile"])) if int(np.count_nonzero(overlap_mask)) > 0 else 1.0
        semantic_mask = ((cleaned_prob_map >= threshold) & (overlap_mask > 0)).astype(np.uint8) * 255

        overlap_pixels = int(np.count_nonzero(overlap_mask))
        change_pixels = int(np.count_nonzero(semantic_mask))
        semantic_change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)
        localized_change_mass = round(float(np.sum(probability_map * (overlap_mask > 0)) / max(overlap_pixels, 1)), 6)

        # A directional cleanup proxy: if the changed regions become less texture-dense after cleanup,
        # the pair likely reflects successful garbage removal.
        before_texture = self._texture_energy(aligned_before, semantic_mask)
        after_texture = self._texture_energy(aligned_after, semantic_mask)
        cleanup_delta = float(np.clip((before_texture - after_texture) / max(before_texture + 1e-6, 1e-6), -1.0, 1.0))

        alignment_quality = float(np.clip(scene_consistency, 0.0, 1.0))
        reliability = float(np.clip(0.55 * alignment_quality + 0.45 * (1.0 - localized_change_mass), 0.0, 1.0))
        base_cleanup_percent = float(np.clip(50.0 + 50.0 * cleanup_delta, 0.0, 100.0))
        cleanup_percent = float(np.clip(base_cleanup_percent * (0.60 + 0.40 * reliability), 0.0, 100.0))

        if reliability >= 0.72:
            confidence = "high"
        elif reliability >= 0.48:
            confidence = "medium"
        else:
            confidence = "low"

        manual_review_recommended = confidence == "low"

        probability_u8 = (np.clip(probability_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        # also save cleaned probability and mask artifacts
        cleaned_prob_u8 = (np.clip(cleaned_prob_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        cleaned_mask_u8 = (cleaned_mask > 0).astype(np.uint8) * 255
        heatmap = cv2.applyColorMap(probability_u8, int(DINOV2_CD_CONFIG["colormap"]))
        overlay = self._blend_overlay(aligned_after, heatmap, float(DINOV2_CD_CONFIG["overlay_alpha"]))
        preview = self._compose_preview(aligned_before, aligned_after, heatmap, overlay)

        output_dir = self._prepare_output_dir(before, after)
        artifacts = self._save_artifacts(
            output_dir=output_dir,
            aligned_before=aligned_before,
            aligned_after=aligned_after,
            overlap_mask=overlap_mask,
            probability_map=probability_u8,
            heatmap=heatmap,
            overlay=overlay,
            preview=preview,
        )

        # save cleaned artifacts
        cv2.imwrite(str(output_dir / "semantic_probability_cleaned.png"), cleaned_prob_u8)
        cv2.imwrite(str(output_dir / "semantic_mask_cleaned.png"), cleaned_mask_u8)

        metrics = {
            "analysis_mode": "dinov2_cd_pair",
            "model_name": self.backbone_name,
            "model_source": model_source,
            "device_requested": self.device,
            "device_used": device_used,
            "force_cpu": bool(self.force_cpu),
            "torch_import_ok": TORCH_IMPORT_ERROR is None,
            "timm_import_ok": TIMM_IMPORT_ERROR is None,
            "torch_import_error": TORCH_IMPORT_ERROR,
            "timm_import_error": TIMM_IMPORT_ERROR,
            "alignment_mode": alignment_mode,
            "scene_consistency": round(scene_consistency, 6),
            "alignment_quality": round(alignment_quality, 6),
            "semantic_threshold": round(threshold, 6),
            "semantic_change_ratio": semantic_change_ratio,
            "localized_change_mass": localized_change_mass,
            "cleanup_delta": round(cleanup_delta, 6),
            "cleanup_percent": round(cleanup_percent, 3),
            "cleanup_confidence": confidence,
            "manual_review_recommended": bool(manual_review_recommended),
            "inference_ms": round(inference_ms, 3),
            "fallback_reason": fallback_reason,
        }

        summary = (
            "DINOv2 semantic change detection on a before/after pair with overlap-aware alignment and "
            "cleanup degree estimation (0-100)."
        )
        preview_text = (
            f"Mode: dinov2_cd\n"
            f"Backbone: {self.backbone_name}\n"
            f"Device: {device_used}\n"
            f"Cleanup percent: {cleanup_percent:.1f}\n"
            f"Confidence: {confidence}"
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
        # First try ECC affine alignment (fast, good for small view changes)
        try:
            before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(DINOV2_CD_CONFIG["ecc_max_iterations"]),
                float(DINOV2_CD_CONFIG["ecc_eps"]),
            )
            cv2.findTransformECC(before_gray, after_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
            aligned_after = cv2.warpAffine(
                after_img,
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            valid_before = np.full((before_h, before_w), 255, dtype=np.uint8)
            valid_after = cv2.warpAffine(
                np.full((before_h, before_w), 255, dtype=np.uint8),
                warp_matrix,
                (before_w, before_h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            overlap_mask = cv2.bitwise_and(valid_before, valid_after)
            overlap_mask = cv2.erode(overlap_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

            # Quick residual check: if residual is high, fall back to feature-based homography
            residual = float(np.mean(cv2.absdiff(cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255.0))
            if residual < 0.035:
                return before_img, aligned_after, overlap_mask, "ecc_affine"
            # else fall through to feature based alignment
        except Exception:
            pass

        # Feature-based homography fallback (SIFT/ORB + RANSAC)
        try:
            aligned_after, homography, valid_mask = self._feature_align_homography(before_img, after_img)
            if aligned_after is not None:
                overlap_mask = (valid_mask * 255).astype(np.uint8)
                overlap_mask = cv2.erode(overlap_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
                return before_img, aligned_after, overlap_mask, "homography_ransac"
        except Exception:
            pass

        # Final fallback: no alignment beyond resize
        overlap_mask = np.full((before_h, before_w), 255, dtype=np.uint8)
        return before_img, after_img, overlap_mask, "resize_fallback"

    def _feature_align_homography(self, before_img: np.ndarray, after_img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        # Try SIFT then ORB for keypoints/descriptors
        gray1 = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

        detector = None
        try:
            detector = cv2.SIFT_create()
        except Exception:
            try:
                detector = cv2.ORB_create(2000)
            except Exception:
                detector = None

        if detector is None:
            return None, None, np.ones(gray1.shape, dtype=np.uint8)

        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, np.ones(gray1.shape, dtype=np.uint8)

        # Match descriptors
        if hasattr(cv2, 'SIFT_create') and isinstance(detector, type(cv2.SIFT_create())):
            # FLANN for SIFT
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return None, None, np.ones(gray1.shape, dtype=np.uint8)

        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.8 * n.distance:
                good.append(m)

        if len(good) < 8:
            return None, None, np.ones(gray1.shape, dtype=np.uint8)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, None, np.ones(gray1.shape, dtype=np.uint8)

        aligned_after = cv2.warpPerspective(after_img, H, (before_img.shape[1], before_img.shape[0]), flags=cv2.INTER_LINEAR)
        h_mask = cv2.warpPerspective(np.ones_like(gray2, dtype=np.uint8), H, (before_img.shape[1], before_img.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = (h_mask > 0).astype(np.uint8)
        return aligned_after, H, valid_mask

    def _predict_semantic_change_map(self, before_img: np.ndarray, after_img: np.ndarray) -> tuple[np.ndarray, float, str, str, str | None]:
        if torch is None or timm is None:
            diff_map, inference_ms = self._predict_cv_fallback(before_img, after_img)
            reason = f"torch_or_timm_unavailable; torch_error={TORCH_IMPORT_ERROR}; timm_error={TIMM_IMPORT_ERROR}"
            return diff_map, inference_ms, "cv_fallback", "cpu", reason

        try:
            model = self._load_feature_model()
            device_obj, device_used = self._resolve_device()
            model = model.to(device_obj)
            start = perf_counter()
            before_features = self._extract_feature_grid(before_img, model, device_obj)
            after_features = self._extract_feature_grid(after_img, model, device_obj)
            distance_small = self._cosine_distance_map(before_features, after_features)
            distance_map = cv2.resize(distance_small, (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            distance_map = cv2.GaussianBlur(distance_map.astype(np.float32), (5, 5), 0)
            probability_map = self._robust_normalize(distance_map)
            inference_ms = (perf_counter() - start) * 1000.0
            return probability_map, inference_ms, str(self._feature_model_source), device_used, None
        except Exception as exc:
            diff_map, inference_ms = self._predict_cv_fallback(before_img, after_img)
            reason = f"dinov2_runtime_fallback: {type(exc).__name__}: {exc}"
            return diff_map, inference_ms, "cv_fallback", "cpu", reason

    def _predict_cv_fallback(self, before_img: np.ndarray, after_img: np.ndarray) -> tuple[np.ndarray, float]:
        start = perf_counter()
        diff = cv2.absdiff(before_img, after_img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        probability_map = self._robust_normalize(blur.astype(np.float32))
        inference_ms = (perf_counter() - start) * 1000.0
        return probability_map, inference_ms

    def _load_feature_model(self):
        if self._feature_model is not None:
            return self._feature_model

        candidates = (self.backbone_name,) + tuple(
            candidate for candidate in DINOV2_CD_CONFIG["backbone_candidates"] if candidate != self.backbone_name
        )

        model = None
        last_error: Exception | None = None

        for candidate in candidates:
            try:
                model = timm.create_model(candidate, pretrained=True, num_classes=0, global_pool="")
                self.backbone_name = candidate
                break
            except Exception as exc:
                last_error = exc

        if model is None:
            hub_error: Exception | None = None
            for candidate in candidates:
                try:
                    model = torch.hub.load("facebookresearch/dinov2", candidate)
                    self.backbone_name = candidate
                    break
                except Exception as exc:
                    hub_error = exc

            if model is None:
                if last_error is not None:
                    raise RuntimeError(
                        f"Could not load DINOv2 backbone '{self.backbone_name}' via timm or torch.hub"
                    ) from last_error
                raise RuntimeError(
                    f"Could not load DINOv2 backbone '{self.backbone_name}' via timm or torch.hub"
                ) from hub_error

        self._feature_model = model.eval()
        self._feature_model_source = self.backbone_name
        return self._feature_model

    def _resolve_device(self):
        if self.force_cpu:
            return torch.device("cpu"), "cpu"
        if self.device == "cpu":
            return torch.device("cpu"), "cpu"
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if self.device == "auto" and torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    def _extract_feature_grid(self, image: np.ndarray, model, device_obj) -> np.ndarray:
        tensor = self._image_to_tensor(image, device_obj)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                features = model.forward_features(tensor)
            else:
                features = model(tensor)

        if isinstance(features, dict):
            for key in (
                "x_norm_patchtokens",
                "patchtokens",
                "x_prenorm",
                "patch_tokens",
                "tokens",
            ):
                if key in features:
                    features = features[key]
                    break

        if isinstance(features, (list, tuple)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise RuntimeError("Unexpected feature output type from backbone")

        if features.ndim == 4:
            feature_grid = features[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            return self._l2_normalize_grid(feature_grid)

        if features.ndim == 3:
            tokens = features[0].detach().cpu().numpy().astype(np.float32)
            token_count = tokens.shape[0]
            # DINO-like token output may include CLS token.
            if token_count > 1:
                maybe_side = int(round(np.sqrt(token_count - 1)))
                if maybe_side * maybe_side == token_count - 1:
                    tokens = tokens[1:, :]
                    token_count = tokens.shape[0]

            side = int(np.floor(np.sqrt(token_count)))
            if side <= 0:
                raise RuntimeError("Could not reshape token grid")
            usable = side * side
            if usable != token_count:
                tokens = tokens[:usable, :]

            feature_grid = tokens.reshape(side, side, -1)
            return self._l2_normalize_grid(feature_grid)

        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(features.shape)}")

    def _image_to_tensor(self, image: np.ndarray, device_obj):
        resized = cv2.resize(image, (int(self.input_size), int(self.input_size)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (rgb - mean) / std
        tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))).unsqueeze(0).to(device_obj)
        return tensor

    def _l2_normalize_grid(self, grid: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(grid, axis=2, keepdims=True)
        return grid / np.maximum(norms, 1e-6)

    def _cosine_distance_map(self, before_features: np.ndarray, after_features: np.ndarray) -> np.ndarray:
        if before_features.shape != after_features.shape:
            raise RuntimeError("Feature shapes do not match")
        cosine_similarity = np.sum(before_features * after_features, axis=2)
        distance = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)
        return distance.astype(np.float32)

    def _robust_normalize(self, map_data: np.ndarray) -> np.ndarray:
        low = float(np.percentile(map_data, 3.0))
        high = float(np.percentile(map_data, 97.0))
        if high <= low + 1e-8:
            return np.zeros_like(map_data, dtype=np.float32)
        normalized = (map_data - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _postprocess_probability_map(self, prob_map: np.ndarray, overlap_mask: np.ndarray, aligned_after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Denoise and remove small components; ignore border-padded areas from warping.

        Returns cleaned probability map and binary mask (0/1).
        """
        # Median denoise
        k = int(DINOV2_CD_CONFIG.get("postproc_median_ksize", 5))
        if k % 2 == 0:
            k += 1
        denoised = cv2.medianBlur((prob_map * 255.0).astype(np.uint8), k)
        denoised = denoised.astype(np.float32) / 255.0

        # Identify padded border areas in aligned image (near-black)
        border_thresh = int(DINOV2_CD_CONFIG.get("border_sum_threshold", 16))
        pad_mask = (np.sum(aligned_after.astype(np.int32), axis=2) <= border_thresh).astype(np.uint8)

        # Convert overlap_mask to binary
        overlap_bin = (overlap_mask > 0).astype(np.uint8)

        # Initial binary mask by percentiles inside overlap
        valid_pixels = (overlap_bin > 0) & (pad_mask == 0)
        if not np.any(valid_pixels):
            return prob_map, np.zeros_like(overlap_bin)

        try:
            thresh = float(np.percentile(denoised[valid_pixels], DINOV2_CD_CONFIG.get("semantic_threshold_percentile", 85.0)))
        except Exception:
            thresh = 0.5

        binary = ((denoised >= thresh) & (overlap_bin > 0) & (pad_mask == 0)).astype(np.uint8)

        # Remove small components
        min_area = int(DINOV2_CD_CONFIG.get("postproc_min_component_area", 300))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary, dtype=np.uint8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == i] = 1

        # Zero-out small component pixels in the probability map
        cleaned_prob = denoised.copy()
        cleaned_prob[cleaned == 0] = cleaned_prob[cleaned == 0] * 0.0

        return cleaned_prob.astype(np.float32), cleaned.astype(np.uint8)

    def _alignment_residual(self, before_img: np.ndarray, after_img: np.ndarray, overlap_mask: np.ndarray) -> float:
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(before_gray, after_gray).astype(np.float32) / 255.0
        valid = overlap_mask > 0
        if not np.any(valid):
            return 1.0
        return float(np.mean(diff[valid]))

    def _texture_energy(self, image: np.ndarray, focus_mask: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        energy = np.abs(lap)
        valid = focus_mask > 0
        if int(np.count_nonzero(valid)) < 64:
            valid = np.full(gray.shape, True, dtype=bool)
        return float(np.mean(energy[valid]))

    def _blend_overlay(self, image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float) -> np.ndarray:
        return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)

    def _compose_preview(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        panel_h = max(before_img.shape[0], after_img.shape[0], heatmap.shape[0], overlay.shape[0])
        panel_w = max(before_img.shape[1], after_img.shape[1], heatmap.shape[1], overlay.shape[1])

        def make_panel(image: np.ndarray, title: str) -> np.ndarray:
            canvas = np.full((panel_h + int(DINOV2_CD_CONFIG["panel_title_height"]), panel_w, 3), 18, dtype=np.uint8)
            y0 = int(DINOV2_CD_CONFIG["panel_title_height"])
            canvas[y0 : y0 + image.shape[0], : image.shape[1]] = image
            cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
            return canvas

        row_top = np.hstack([make_panel(before_img, "Before"), make_panel(after_img, "After")])
        row_bottom = np.hstack([make_panel(heatmap, "Semantic heatmap"), make_panel(overlay, "Heatmap overlay")])
        preview = np.vstack([row_top, row_bottom])

        max_w = int(DINOV2_CD_CONFIG["preview_max_width"])
        max_h = int(DINOV2_CD_CONFIG["preview_max_height"])
        scale = min(max_w / max(preview.shape[1], 1), max_h / max(preview.shape[0], 1), 1.0)
        if scale < 1.0:
            preview = cv2.resize(
                preview,
                (max(1, int(preview.shape[1] * scale)), max(1, int(preview.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return preview

    def _prepare_output_dir(self, before_path: Path, after_path: Path) -> Path:
        results_root = Path(__file__).resolve().parent.parent.parent / "results"
        method_slug = self._slugify(self.label)
        timestamp = datetime.now().strftime("%d.%m.%Y %H-%M")
        run_id = uuid4().hex[:6]
        folder_name = f"{self.method_id}__{method_slug}__{timestamp}__{run_id}"
        output_dir = results_root / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_artifacts(
        self,
        output_dir: Path,
        aligned_before: np.ndarray,
        aligned_after: np.ndarray,
        overlap_mask: np.ndarray,
        probability_map: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        preview: np.ndarray,
    ) -> dict[str, Path]:
        artifacts = {
            "aligned_before": output_dir / "aligned_before.png",
            "aligned_after": output_dir / "aligned_after.png",
            "overlap_mask": output_dir / "overlap_mask.png",
            "semantic_probability": output_dir / "semantic_probability.png",
            "semantic_heatmap": output_dir / "semantic_heatmap.png",
            "semantic_overlay": output_dir / "semantic_overlay.png",
            "preview": output_dir / "preview.png",
        }

        cv2.imwrite(str(artifacts["aligned_before"]), aligned_before)
        cv2.imwrite(str(artifacts["aligned_after"]), aligned_after)
        cv2.imwrite(str(artifacts["overlap_mask"]), overlap_mask)
        cv2.imwrite(str(artifacts["semantic_probability"]), probability_map)
        cv2.imwrite(str(artifacts["semantic_heatmap"]), heatmap)
        cv2.imwrite(str(artifacts["semantic_overlay"]), overlay)
        cv2.imwrite(str(artifacts["preview"]), preview)
        return artifacts

    def _slugify(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
        return slug or "method"
