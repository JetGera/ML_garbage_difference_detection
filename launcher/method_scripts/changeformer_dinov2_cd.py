from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ..core import AnalysisResult
    from ..methods import get_method_spec
    from .changeformer import ChangeformerRunner
    from .dinov2_cd import DinoV2CdRunner
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec
    from method_scripts.changeformer import ChangeformerRunner
    from method_scripts.dinov2_cd import DinoV2CdRunner


class ChangeformerDinoV2Runner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        backbone_name: str | None = None,
        input_size: int | None = None,
        threshold_percentile: float | None = None,
        weights_path: str | Path | None = None,
        dino_backbone_name: str | None = None,
        dino_input_size: int | None = None,
        fusion_alpha_changeformer: float = 0.82,
        dino_gate_percentile: float = 78.0,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = str(device)
        self.force_cpu = bool(force_cpu)
        self.backbone_name = backbone_name
        self.input_size = input_size
        self.threshold_percentile = threshold_percentile
        self.weights_path = weights_path
        self.dino_backbone_name = dino_backbone_name
        self.dino_input_size = dino_input_size
        self.fusion_alpha_changeformer = float(np.clip(fusion_alpha_changeformer, 0.0, 1.0))
        self.dino_gate_percentile = float(np.clip(dino_gate_percentile, 50.0, 99.5))

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)

        changeformer_runner = ChangeformerRunner(
            "changeformer",
            device=self.device,
            force_cpu=self.force_cpu,
            backbone_name=self.backbone_name,
            input_size=self.input_size,
            threshold_percentile=self.threshold_percentile,
            weights_path=self.weights_path,
        )
        cf_result = changeformer_runner.analyze(before, after)

        dino_runner = DinoV2CdRunner(
            "dinov2_cd",
            device=self.device,
            force_cpu=self.force_cpu,
            backbone_name=self.dino_backbone_name,
            input_size=self.dino_input_size,
        )
        dino_map, dino_meta = dino_runner.predict_probability_map(before, after)

        cf_prob_path = cf_result.artifacts.get("change_probability")
        if cf_prob_path is None or not Path(cf_prob_path).exists():
            raise RuntimeError("ChangeFormer artifacts do not contain change_probability")

        cf_prob_u8 = cv2.imread(str(cf_prob_path), cv2.IMREAD_GRAYSCALE)
        if cf_prob_u8 is None:
            raise RuntimeError(f"Failed to read ChangeFormer probability map: {cf_prob_path}")
        cf_prob = np.clip(cf_prob_u8.astype(np.float32) / 255.0, 0.0, 1.0)

        dino_resized = cv2.resize(dino_map.astype(np.float32), (cf_prob.shape[1], cf_prob.shape[0]), interpolation=cv2.INTER_LINEAR)
        dino_resized = np.clip(dino_resized, 0.0, 1.0)
        dino_resized = cv2.medianBlur((dino_resized * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0

        if dino_resized.size:
            low = float(np.percentile(dino_resized, 5.0))
            high = float(np.percentile(dino_resized, self.dino_gate_percentile))
            if high > low + 1e-6:
                dino_resized = np.clip((dino_resized - low) / (high - low), 0.0, 1.0)
            gate = float(np.percentile(dino_resized, self.dino_gate_percentile))
            dino_resized = np.where(dino_resized >= gate, dino_resized, dino_resized * 0.35)

        a = float(self.fusion_alpha_changeformer)
        fused_prob = np.clip((a * cf_prob) + ((1.0 - a) * dino_resized), 0.0, 1.0)
        fused_prob = np.clip(0.72 * fused_prob + 0.28 * cf_prob, 0.0, 1.0)

        overlap_path = cf_result.artifacts.get("overlap_mask")
        overlap_mask = None
        if overlap_path is not None and Path(overlap_path).exists():
            overlap_mask = cv2.imread(str(overlap_path), cv2.IMREAD_GRAYSCALE)
            if overlap_mask is not None and overlap_mask.shape != cf_prob.shape:
                overlap_mask = cv2.resize(overlap_mask, (cf_prob.shape[1], cf_prob.shape[0]), interpolation=cv2.INTER_NEAREST)

        threshold = float(np.percentile(fused_prob, 93.0))
        fused_mask = (fused_prob >= threshold).astype(np.uint8) * 255
        if overlap_mask is not None:
            fused_mask = cv2.bitwise_and(fused_mask, (overlap_mask > 0).astype(np.uint8) * 255)

        artifact_dir = None
        if cf_result.preview_image_path is not None:
            artifact_dir = Path(cf_result.preview_image_path).parent
        if artifact_dir is None or not artifact_dir.exists():
            artifact_dir = Path(__file__).resolve().parent.parent.parent / "results"
            artifact_dir.mkdir(parents=True, exist_ok=True)

        fused_prob_u8 = (fused_prob * 255.0).astype(np.uint8)
        fused_prob_path = artifact_dir / "fused_probability_map.png"
        fused_mask_path = artifact_dir / "fused_change_mask.png"
        fused_overlay_path = artifact_dir / "fused_overlay.png"
        cv2.imwrite(str(fused_prob_path), fused_prob_u8)
        cv2.imwrite(str(fused_mask_path), fused_mask)

        aligned_after_path = cf_result.artifacts.get("aligned_after")
        aligned_after = None
        if aligned_after_path is not None and Path(aligned_after_path).exists():
            aligned_after = cv2.imread(str(aligned_after_path), cv2.IMREAD_COLOR)
        if aligned_after is not None and aligned_after.shape[:2] == fused_prob.shape:
            heatmap = cv2.applyColorMap(fused_prob_u8, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(aligned_after, 0.45, heatmap, 0.55, 0.0)
            cv2.imwrite(str(fused_overlay_path), overlay)
            preview_path = fused_overlay_path
        else:
            preview_path = cf_result.preview_image_path

        overlap_pixels = int(np.count_nonzero(overlap_mask)) if overlap_mask is not None else int(fused_mask.size)
        change_pixels = int(np.count_nonzero(fused_mask))
        fused_change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)

        metrics = dict(cf_result.metrics)
        metrics.update(
            {
                "analysis_mode": "changeformer_dinov2_pair",
                "fusion_alpha_changeformer": a,
                "fusion_alpha_dinov2": round(1.0 - a, 6),
                "dino_gate_percentile": float(self.dino_gate_percentile),
                "fused_threshold": round(threshold, 6),
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

        artifacts = dict(cf_result.artifacts)
        artifacts.update(
            {
                "fused_probability_map": fused_prob_path,
                "fused_change_mask": fused_mask_path,
            }
        )
        if preview_path is not None and Path(preview_path).exists():
            artifacts["fused_preview"] = Path(preview_path)
        if fused_overlay_path.exists():
            artifacts["fused_overlay"] = fused_overlay_path

        preview_text = (
            f"Mode: changeformer_dinov2\n"
            f"Fusion: cf={a:.2f}, dino={1.0 - a:.2f}\n"
            f"DINO gate: p{self.dino_gate_percentile:.0f}\n"
            f"Fused change ratio: {fused_change_ratio:.6f}\n"
            f"DINO model: {metrics.get('dino_model_source', 'unknown')}"
        )

        return AnalysisResult(
            method_id=self.method_id,
            method_name=self.label,
            summary="ChangeFormer change map fused with DINOv2 semantic map.",
            metrics=metrics,
            before_path=before,
            after_path=after,
            preview_text=preview_text,
            preview_image_path=Path(preview_path) if preview_path is not None else cf_result.preview_image_path,
            artifacts=artifacts,
        )
