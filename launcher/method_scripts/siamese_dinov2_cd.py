from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ..core import AnalysisResult
    from ..methods import get_method_spec
    from .dinov2_cd import DinoV2CdRunner
    from .siamese_unet_cd import SiameseUnetCdRunner
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec
    from method_scripts.dinov2_cd import DinoV2CdRunner
    from method_scripts.siamese_unet_cd import SiameseUnetCdRunner


class SiameseDinoV2Runner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        input_size: int | None = None,
        threshold: float | None = None,
        weights_path: str | Path | None = None,
        dino_backbone_name: str | None = None,
        dino_input_size: int | None = None,
        fusion_alpha_siamese: float = 0.62,
        hint_gamma: float = 1.6,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = str(device)
        self.force_cpu = bool(force_cpu)
        self.input_size = input_size
        self.threshold = threshold
        self.weights_path = weights_path
        self.dino_backbone_name = dino_backbone_name
        self.dino_input_size = dino_input_size
        self.fusion_alpha_siamese = float(np.clip(fusion_alpha_siamese, 0.0, 1.0))
        self.hint_gamma = float(max(hint_gamma, 0.1))

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        dino_runner = DinoV2CdRunner(
            "dinov2_cd",
            device=self.device,
            force_cpu=self.force_cpu,
            backbone_name=self.dino_backbone_name,
            input_size=self.dino_input_size,
        )
        dinov2_map, dino_meta = dino_runner.predict_probability_map(before, after)

        siamese_runner = SiameseUnetCdRunner(
            "siamese_unet_cd",
            device=self.device,
            force_cpu=self.force_cpu,
            input_size=self.input_size,
            threshold=self.threshold,
            weights_path=self.weights_path,
        )
        siamese_result = siamese_runner.analyze(before, after, dinov2_map=dinov2_map)

        si_prob_path = siamese_result.artifacts.get("probability_map")
        if si_prob_path is None or not Path(si_prob_path).exists():
            raise RuntimeError("Siamese artifacts do not contain probability_map")

        si_prob_u8 = cv2.imread(str(si_prob_path), cv2.IMREAD_GRAYSCALE)
        if si_prob_u8 is None:
            raise RuntimeError(f"Failed to read Siamese probability map: {si_prob_path}")
        si_prob = np.clip(si_prob_u8.astype(np.float32) / 255.0, 0.0, 1.0)

        dino_resized = cv2.resize(dinov2_map.astype(np.float32), (si_prob.shape[1], si_prob.shape[0]), interpolation=cv2.INTER_LINEAR)
        dino_resized = np.clip(dino_resized, 0.0, 1.0)
        dino_boost = np.power(dino_resized, self.hint_gamma)

        alpha = float(self.fusion_alpha_siamese)
        fused_prob = np.clip((alpha * si_prob) + ((1.0 - alpha) * dino_boost), 0.0, 1.0)
        fused_prob_u8 = (fused_prob * 255.0).astype(np.uint8)

        overlap_path = siamese_result.artifacts.get("overlap_mask")
        overlap_mask = None
        if overlap_path is not None and Path(overlap_path).exists():
            overlap_mask = cv2.imread(str(overlap_path), cv2.IMREAD_GRAYSCALE)
            if overlap_mask is not None and overlap_mask.shape != si_prob.shape:
                overlap_mask = cv2.resize(overlap_mask, (si_prob.shape[1], si_prob.shape[0]), interpolation=cv2.INTER_NEAREST)

        fused_threshold = float(np.percentile(fused_prob, 65.0))
        fused_mask = (fused_prob >= fused_threshold).astype(np.uint8) * 255
        if overlap_mask is not None:
            fused_mask = cv2.bitwise_and(fused_mask, (overlap_mask > 0).astype(np.uint8) * 255)

        fused_overlay_path = siamese_result.preview_image_path.parent / "siamese_dinov2_fused_overlay.png"
        aligned_after_path = siamese_result.artifacts.get("aligned_after")
        aligned_after = None
        if aligned_after_path is not None and Path(aligned_after_path).exists():
            aligned_after = cv2.imread(str(aligned_after_path), cv2.IMREAD_COLOR)
        if aligned_after is not None and aligned_after.shape[:2] == fused_prob.shape:
            heatmap = cv2.applyColorMap(fused_prob_u8, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(aligned_after, 0.48, heatmap, 0.52, 0.0)
            cv2.imwrite(str(fused_overlay_path), overlay)
            preview_path = fused_overlay_path
        else:
            preview_path = siamese_result.preview_image_path

        overlap_pixels = int(np.count_nonzero(overlap_mask)) if overlap_mask is not None else int(fused_mask.size)
        change_pixels = int(np.count_nonzero(fused_mask))
        fused_change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)

        metrics = dict(siamese_result.metrics)
        metrics.update(
            {
                "analysis_mode": "siamese_dinov2_pair",
                "fusion_alpha_siamese": alpha,
                "fusion_alpha_dinov2": round(1.0 - alpha, 6),
                "fused_threshold": round(fused_threshold, 6),
                "fused_change_ratio": fused_change_ratio,
                "dino_model_source": dino_meta.get("model_source"),
                "dino_device_used": dino_meta.get("device_used"),
                "dino_alignment_mode": dino_meta.get("alignment_mode"),
                "dino_inference_ms": dino_meta.get("inference_ms"),
                "dino_fallback_reason": dino_meta.get("fallback_reason"),
                "dino_torch_import_ok": dino_meta.get("torch_import_ok"),
                "dino_timm_import_ok": dino_meta.get("timm_import_ok"),
            }
        )

        artifacts = dict(siamese_result.artifacts)
        artifacts.update(
            {
                "fused_probability_map": siamese_result.preview_image_path.parent / "siamese_dinov2_fused_probability_map.png",
                "fused_change_mask": siamese_result.preview_image_path.parent / "siamese_dinov2_fused_change_mask.png",
                "fused_overlay": fused_overlay_path,
            }
        )
        cv2.imwrite(str(artifacts["fused_probability_map"]), fused_prob_u8)
        cv2.imwrite(str(artifacts["fused_change_mask"]), fused_mask)

        preview_text = (
            f"Mode: siamese_dinov2\n"
            f"Fusion: siamese={alpha:.2f}, dino={1.0 - alpha:.2f}\n"
            f"Fused change ratio: {fused_change_ratio:.6f}\n"
            f"DINO model: {metrics.get('dino_model_source', 'unknown')}\n"
            f"Change ratio: {metrics.get('change_ratio', 'N/A')}"
        )

        return AnalysisResult(
            method_id=self.method_id,
            method_name=self.label,
            summary="Siamese U-Net change detection conditioned by DINOv2 semantic hints.",
            metrics=metrics,
            before_path=before,
            after_path=after,
            preview_text=preview_text,
            preview_image_path=Path(preview_path),
            artifacts=artifacts,
        )
