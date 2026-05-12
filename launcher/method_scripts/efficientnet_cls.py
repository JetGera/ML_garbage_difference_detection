from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import timm
except Exception:
    timm = None

try:
    from torchvision.models import EfficientNet_B0_Weights
except Exception:
    EfficientNet_B0_Weights = None

try:
    from ..core import AnalysisResult
    from ..methods import get_method_spec
    from ..utils.io_utils import prepare_pair_output_dir, save_artifact_images
    from ..utils.viz_utils import annotate_panel, compose_panel_grid
except ImportError:
    from core import AnalysisResult
    from methods import get_method_spec
    from utils.io_utils import prepare_pair_output_dir, save_artifact_images
    from utils.viz_utils import annotate_panel, compose_panel_grid


EFFICIENTNET_CLS_CONFIG = {
    # Model architecture name passed to timm.create_model.
    "model_name": "efficientnet_b0",
    # Optional environment variable with path to a trained binary checkpoint.
    "weights_env_var": "EFFICIENTNET_CLS_WEIGHTS",
    # Input side used for inference preprocessing.
    "input_size": 224,
    # ImageNet normalization constants.
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    # Dirty probability threshold for class label.
    "dirty_threshold": 0.42,
    # If no trained binary checkpoint is provided, dirty score is approximated
    # from ImageNet class probabilities matching these keywords.
    "dirty_keywords": (
        "garbage",
        "trash",
        "litter",
        "waste",
        "dumpster",
        "refuse",
        "ashcan",
        "garbage truck",
    ),
    # Entropy term keeps proxy predictions from collapsing to 0 when no keyword
    # class receives non-trivial mass.
    "proxy_keyword_weight": 0.75,
    "proxy_entropy_weight": 0.25,
    # Multi-crop aggregation for local trash evidence.
    "score_crop_scale": 0.58,
    "score_crop_min_side": 112,
    "score_crop_max_count": 6,
    "score_crop_max_weight": 0.72,
    # Difference heatmap tuning.
    "difference_focus_bottom_weight": 1.35,
    "difference_focus_center_weight": 1.10,
    "difference_threshold_percentile": 84,
    "difference_overlay_alpha": 0.72,
    # Preview rendering limits.
    "preview_max_width": 2600,
    "preview_max_height": 1900,
    "panel_title_height": 42,
}


class EfficientNetClsRunner:
    def __init__(
        self,
        method_id: str,
        device: str = "auto",
        force_cpu: bool = False,
        model_name: str | None = None,
        weights_path: str | Path | None = None,
        dirty_threshold: float | None = None,
    ):
        self.method_id = method_id
        self.spec = get_method_spec(method_id)
        self.device = device
        self.force_cpu = force_cpu
        self.model_name = model_name or str(EFFICIENTNET_CLS_CONFIG["model_name"])
        env_weights_path = os.environ.get(str(EFFICIENTNET_CLS_CONFIG["weights_env_var"]))
        if weights_path is not None:
            self.weights_path = Path(weights_path)
        elif env_weights_path:
            self.weights_path = Path(env_weights_path)
        else:
            self.weights_path = None
        self.dirty_threshold = float(dirty_threshold or EFFICIENTNET_CLS_CONFIG["dirty_threshold"])

        self._model = None
        self._model_source = None
        self._inference_mode = "imagenet_proxy"
        self._imagenet_labels = self._load_imagenet_labels()
        self._dirty_class_indices = self._resolve_dirty_class_indices(self._imagenet_labels)
        self._torch_import_available = torch is not None and timm is not None

    @property
    def label(self) -> str:
        return self.spec.label

    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)
        before_img = self._read_color_image(before)
        after_img = self._read_color_image(after)

        if self._torch_import_available:
            try:
                model = self._load_model()
                device_obj, device_used = self._resolve_device()
                model = model.to(device_obj)
                before_pred = self._predict_with_crops(model, before_img, device_obj)
                after_pred = self._predict_with_crops(model, after_img, device_obj)
            except Exception:
                self._torch_import_available = False
                self._inference_mode = "heuristic_fallback"
                device_used = "cpu"
                before_pred = self._predict_heuristic_with_crops(before_img)
                after_pred = self._predict_heuristic_with_crops(after_img)
        else:
            self._inference_mode = "heuristic_fallback"
            device_used = "cpu"
            before_pred = self._predict_heuristic_with_crops(before_img)
            after_pred = self._predict_heuristic_with_crops(after_img)

        diff_preview = self._build_diff_preview(before_img, after_img)
        before_overlay = self._build_score_overlay(before_img, before_pred, title="Before")
        after_overlay = self._build_score_overlay(after_img, after_pred, title="After")
        preview = self._compose_preview(before_img, after_img, before_overlay, after_overlay, diff_preview)

        output_dir = prepare_pair_output_dir(Path(__file__).resolve().parent.parent.parent / "results", before, after, self.label)
        artifacts = save_artifact_images(
            output_dir,
            {
                "before_overlay": before_overlay,
                "after_overlay": after_overlay,
                "difference_heatmap": diff_preview,
                "preview": preview,
            },
        )

        cleanup_delta = float(before_pred["dirty_prob"] - after_pred["dirty_prob"])
        cleanup_score = float(np.clip(cleanup_delta, 0.0, 1.0))

        metrics = {
            "analysis_mode": "efficientnet_cls_pair",
            "inference_mode": self._inference_mode,
            "model_name": self.model_name,
            "model_source": str(self._model_source),
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "device_requested": self.device,
            "device_used": device_used,
            "force_cpu": bool(self.force_cpu),
            "cuda_available": bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()),
            "input_size": int(EFFICIENTNET_CLS_CONFIG["input_size"]),
            "dirty_threshold": float(self.dirty_threshold),
            "dirty_keyword_classes": int(len(self._dirty_class_indices)),
            "before_crop_count": int(before_pred.get("crop_count", 1)),
            "after_crop_count": int(after_pred.get("crop_count", 1)),
            "before_crop_dirty_prob_mean": round(float(before_pred.get("crop_dirty_prob_mean", before_pred["dirty_prob"])), 6),
            "after_crop_dirty_prob_mean": round(float(after_pred.get("crop_dirty_prob_mean", after_pred["dirty_prob"])), 6),
            "before_crop_dirty_prob_max": round(float(before_pred.get("crop_dirty_prob_max", before_pred["dirty_prob"])), 6),
            "after_crop_dirty_prob_max": round(float(after_pred.get("crop_dirty_prob_max", after_pred["dirty_prob"])), 6),
            "before_crop_support_ratio": round(float(before_pred.get("crop_support_ratio", 1.0)), 6),
            "after_crop_support_ratio": round(float(after_pred.get("crop_support_ratio", 1.0)), 6),
            "before_dirty_prob": round(float(before_pred["dirty_prob"]), 6),
            "after_dirty_prob": round(float(after_pred["dirty_prob"]), 6),
            "before_clean_prob": round(float(before_pred["clean_prob"]), 6),
            "after_clean_prob": round(float(after_pred["clean_prob"]), 6),
            "cleanup_delta": round(cleanup_delta, 6),
            "cleanup_score": round(cleanup_score, 6),
            "dirty_prob_delta": round(float(after_pred["dirty_prob"] - before_pred["dirty_prob"]), 6),
            "dirty_prob_abs_delta": round(float(abs(after_pred["dirty_prob"] - before_pred["dirty_prob"])), 6),
            "before_pred_label": before_pred["label"],
            "after_pred_label": after_pred["label"],
            "before_top1_label": before_pred["top1_label"],
            "after_top1_label": after_pred["top1_label"],
            "before_top1_prob": round(float(before_pred["top1_prob"]), 6),
            "after_top1_prob": round(float(after_pred["top1_prob"]), 6),
            "before_entropy_norm": round(float(before_pred["entropy_norm"]), 6),
            "after_entropy_norm": round(float(after_pred["entropy_norm"]), 6),
            "before_inference_ms": round(float(before_pred["inference_ms"]), 3),
            "after_inference_ms": round(float(after_pred["inference_ms"]), 3),
            "inference_ms_total": round(float(before_pred["inference_ms"] + after_pred["inference_ms"]), 3),
        }

        summary = (
            "EfficientNet pair classification for clean/dirty auxiliary scoring. "
            "Use a trained binary checkpoint when available; otherwise the runner uses an ImageNet proxy score or a heuristic fallback if torch cannot load."
        )
        preview_text = (
            f"Mode: {self._inference_mode}\n"
            f"Before dirty prob: {metrics['before_dirty_prob']:.4f} ({metrics['before_pred_label']})\n"
            f"After dirty prob: {metrics['after_dirty_prob']:.4f} ({metrics['after_pred_label']})\n"
            f"Cleanup delta (before - after): {metrics['cleanup_delta']:+.4f}\n"
            f"Cleanup score: {metrics['cleanup_score']:.4f}\n"
            f"Before crops: {metrics['before_crop_count']}, after crops: {metrics['after_crop_count']}\n"
            f"Device: {device_used}"
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

    def _load_model(self):
        model_source = self._resolve_model_source()
        if self._model is not None and self._model_source == model_source:
            return self._model

        checkpoint_path = self.weights_path or self._discover_latest_training_checkpoint()

        if checkpoint_path is not None:
            if not checkpoint_path.exists():
                raise RuntimeError(f"Weights file does not exist: {checkpoint_path}")
            model = timm.create_model(self.model_name, pretrained=False, num_classes=2)
            try:
                payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(str(checkpoint_path), map_location="cpu")
            state_dict = self._extract_state_dict(payload)
            load_result = model.load_state_dict(state_dict, strict=False)
            if not self._has_any_loaded_parameters(load_result):
                raise RuntimeError(
                    "Checkpoint did not match model parameters. "
                    "Please verify model_name/num_classes/checkpoint format."
                )
            self._model = model.eval()
            self._model_source = model_source
            self._inference_mode = "binary_checkpoint"
            return self._model

        try:
            model = timm.create_model(self.model_name, pretrained=True)
        except Exception:
            model = timm.create_model(self.model_name, pretrained=False)

        self._model = model.eval()
        self._model_source = model_source
        self._inference_mode = "imagenet_proxy"
        return self._model

    def _resolve_model_source(self) -> str:
        if self.weights_path is not None and self.weights_path.exists():
            return self._checkpoint_source_token(self.weights_path)

        checkpoint_path = self._discover_latest_training_checkpoint()
        if checkpoint_path is not None:
            return self._checkpoint_source_token(checkpoint_path)

        return f"{self.model_name}:imagenet_proxy"

    def _checkpoint_source_token(self, checkpoint_path: Path) -> str:
        try:
            mtime_ns = checkpoint_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        return f"checkpoint:{checkpoint_path.resolve()}:{mtime_ns}"

    def _discover_latest_training_checkpoint(self) -> Path | None:
        # First try central weights location
        central_weights = Path(__file__).resolve().parent.parent.parent / "weights" / "efficientnet_best.pt"
        if central_weights.exists():
            return central_weights
        
        training_root = Path(__file__).resolve().parent.parent.parent / "results" / "training"
        if not training_root.exists():
            return None

        candidates = [path for path in training_root.rglob("best.pt") if path.is_file()]
        candidates.extend(path for path in training_root.rglob("last.pt") if path.is_file())
        if not candidates:
            return None

        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def _extract_state_dict(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model", "model_state_dict", "net"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return self._strip_module_prefix(value)
            if payload and all(isinstance(key, str) for key in payload.keys()):
                return self._strip_module_prefix(payload)
        raise RuntimeError("Unsupported checkpoint payload format")

    def _strip_module_prefix(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                normalized[key[len("module.") :]] = value
            else:
                normalized[key] = value
        return normalized

    def _has_any_loaded_parameters(self, load_result: Any) -> bool:
        missing = getattr(load_result, "missing_keys", []) or []
        unexpected = getattr(load_result, "unexpected_keys", []) or []
        # If both sides are fully mismatched, almost nothing can be used.
        return not (len(missing) > 0 and len(unexpected) > 0 and len(missing) <= len(unexpected))

    def _resolve_device(self) -> tuple[Any, str]:
        if torch is None:
            raise RuntimeError("torch is not installed in the active environment")

        if self.force_cpu:
            return torch.device("cpu"), "cpu"

        requested = str(self.device).strip().lower()
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

    def _predict_image(self, model, image: np.ndarray, device_obj) -> dict[str, Any]:
        tensor = self._preprocess_image(image).to(device_obj)

        start = perf_counter()
        with torch.no_grad():
            output = model(tensor)
            logits = self._extract_logits(output)
        inference_ms = (perf_counter() - start) * 1000.0

        logits = logits.detach().float().cpu()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if self._inference_mode == "binary_checkpoint":
            return self._predict_from_binary_logits(logits, inference_ms)
        return self._predict_from_imagenet_proxy(logits, inference_ms)

    def _predict_with_crops(self, model, image: np.ndarray, device_obj) -> dict[str, Any]:
        crops = self._build_scoring_crops(image)
        crop_predictions: list[dict[str, Any]] = []
        for crop in crops:
            crop_predictions.append(self._predict_image(model, crop, device_obj))
        return self._aggregate_crop_predictions(crop_predictions)

    def _predict_heuristic_with_crops(self, image: np.ndarray) -> dict[str, Any]:
        crops = self._build_scoring_crops(image)
        crop_predictions = [self._predict_heuristic(crop) for crop in crops]
        return self._aggregate_crop_predictions(crop_predictions)

    def _extract_logits(self, output):
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("Model output is empty")
            return output[0]
        if isinstance(output, dict):
            for key in ("logits", "output", "pred"):
                if key in output:
                    return output[key]
            first_value = next(iter(output.values()), None)
            if first_value is not None:
                return first_value
            raise RuntimeError("Model output dict is empty")
        return output

    def _preprocess_image(self, image_bgr: np.ndarray):
        size = int(EFFICIENTNET_CLS_CONFIG["input_size"])
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0

        mean = np.array(EFFICIENTNET_CLS_CONFIG["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(EFFICIENTNET_CLS_CONFIG["std"], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std

        chw = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor

    def _predict_from_binary_logits(self, logits, inference_ms: float) -> dict[str, Any]:
        if logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)[0]
            clean_prob = float(probs[0].item())
            dirty_prob = float(probs[1].item())
            top1_index = int(torch.argmax(probs).item())
            top1_prob = float(probs[top1_index].item())
            top1_label = "dirty" if top1_index == 1 else "clean"
            entropy_norm = self._normalized_entropy(probs.numpy())
        elif logits.shape[1] == 1:
            dirty_prob = float(torch.sigmoid(logits[0, 0]).item())
            clean_prob = float(1.0 - dirty_prob)
            top1_label = "dirty" if dirty_prob >= 0.5 else "clean"
            top1_prob = dirty_prob if top1_label == "dirty" else clean_prob
            entropy_norm = self._binary_entropy(dirty_prob)
        else:
            raise RuntimeError("Unexpected logits shape for binary inference")

        label = "dirty" if dirty_prob >= self.dirty_threshold else "clean"
        return {
            "dirty_prob": float(np.clip(dirty_prob, 0.0, 1.0)),
            "clean_prob": float(np.clip(clean_prob, 0.0, 1.0)),
            "label": label,
            "top1_label": top1_label,
            "top1_prob": float(np.clip(top1_prob, 0.0, 1.0)),
            "entropy_norm": float(np.clip(entropy_norm, 0.0, 1.0)),
            "inference_ms": float(inference_ms),
        }

    def _predict_from_imagenet_proxy(self, logits, inference_ms: float) -> dict[str, Any]:
        probs = torch.softmax(logits, dim=1)[0].numpy().astype(np.float64)
        top1_index = int(np.argmax(probs))
        top1_prob = float(probs[top1_index])
        top1_label = self._imagenet_labels[top1_index] if top1_index < len(self._imagenet_labels) else f"class_{top1_index}"

        keyword_mass = float(np.sum(probs[self._dirty_class_indices])) if self._dirty_class_indices else 0.0
        entropy_norm = self._normalized_entropy(probs)
        dirty_prob = (
            float(EFFICIENTNET_CLS_CONFIG["proxy_keyword_weight"]) * keyword_mass
            + float(EFFICIENTNET_CLS_CONFIG["proxy_entropy_weight"]) * entropy_norm
        )
        dirty_prob = float(np.clip(dirty_prob, 0.0, 1.0))
        clean_prob = float(1.0 - dirty_prob)

        label = "dirty" if dirty_prob >= self.dirty_threshold else "clean"
        return {
            "dirty_prob": dirty_prob,
            "clean_prob": clean_prob,
            "label": label,
            "top1_label": top1_label,
            "top1_prob": float(np.clip(top1_prob, 0.0, 1.0)),
            "entropy_norm": float(np.clip(entropy_norm, 0.0, 1.0)),
            "inference_ms": float(inference_ms),
        }

    def _predict_heuristic(self, image: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        brightness = float(np.mean(gray) / 255.0)
        darkness = 1.0 - brightness
        edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        texture = float(np.std(blur) / 64.0)

        dirty_score = (
            0.34 * np.clip(darkness, 0.0, 1.0)
            + 0.30 * np.clip(edge_density * 4.0, 0.0, 1.0)
            + 0.18 * np.clip(saturation, 0.0, 1.0)
            + 0.18 * np.clip(texture, 0.0, 1.0)
        )
        dirty_prob = float(np.clip(dirty_score, 0.0, 1.0))
        clean_prob = float(1.0 - dirty_prob)
        label = "dirty" if dirty_prob >= self.dirty_threshold else "clean"

        return {
            "dirty_prob": dirty_prob,
            "clean_prob": clean_prob,
            "label": label,
            "top1_label": "heuristic_dirty" if label == "dirty" else "heuristic_clean",
            "top1_prob": float(max(dirty_prob, clean_prob)),
            "entropy_norm": self._binary_entropy(dirty_prob),
            "inference_ms": 0.0,
        }

    def _build_scoring_crops(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return [image]

        crop_side = max(
            int(EFFICIENTNET_CLS_CONFIG["score_crop_min_side"]),
            int(round(min(height, width) * float(EFFICIENTNET_CLS_CONFIG["score_crop_scale"]))),
        )
        crop_side = min(crop_side, min(height, width))

        if crop_side <= 0:
            return [image]

        half_w = max(0, width - crop_side)
        half_h = max(0, height - crop_side)
        positions = [
            (0, 0),
            (half_w, 0),
            (0, half_h),
            (half_w, half_h),
            (half_w // 2, half_h // 2),
        ]

        crops: list[np.ndarray] = [image]
        seen: set[tuple[int, int]] = set()
        for x0, y0 in positions:
            x0 = int(np.clip(x0, 0, half_w))
            y0 = int(np.clip(y0, 0, half_h))
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            crop = image[y0 : y0 + crop_side, x0 : x0 + crop_side]
            if crop.size == 0:
                continue
            crops.append(crop)
            if len(crops) >= int(EFFICIENTNET_CLS_CONFIG["score_crop_max_count"]):
                break
        return crops

    def _aggregate_crop_predictions(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        if not predictions:
            return {
                "dirty_prob": 0.0,
                "clean_prob": 1.0,
                "label": "clean",
                "top1_label": "no_crops",
                "top1_prob": 0.0,
                "entropy_norm": 0.0,
                "inference_ms": 0.0,
                "crop_count": 0,
                "crop_dirty_prob_mean": 0.0,
                "crop_dirty_prob_max": 0.0,
                "crop_dirty_prob_min": 0.0,
                "crop_dirty_prob_std": 0.0,
                "crop_support_count": 0,
                "crop_support_ratio": 0.0,
            }

        dirty_probs = np.asarray([float(item["dirty_prob"]) for item in predictions], dtype=np.float32)
        clean_probs = np.asarray([float(item["clean_prob"]) for item in predictions], dtype=np.float32)
        entropies = np.asarray([float(item["entropy_norm"]) for item in predictions], dtype=np.float32)
        top1_probs = np.asarray([float(item["top1_prob"]) for item in predictions], dtype=np.float32)
        inference_ms = float(sum(float(item["inference_ms"]) for item in predictions))

        crop_mean = float(np.mean(dirty_probs))
        crop_max = float(np.max(dirty_probs))
        crop_min = float(np.min(dirty_probs))
        crop_std = float(np.std(dirty_probs))
        support_count = int(np.sum(dirty_probs >= float(self.dirty_threshold)))
        support_ratio = support_count / max(len(dirty_probs), 1)

        blend_weight = float(EFFICIENTNET_CLS_CONFIG["score_crop_max_weight"])
        dirty_prob = float(np.clip((blend_weight * crop_max) + ((1.0 - blend_weight) * crop_mean), 0.0, 1.0))
        clean_prob = float(1.0 - dirty_prob)
        label = "dirty" if dirty_prob >= self.dirty_threshold else "clean"
        top1_label = "dirty" if crop_max >= 0.5 else "clean"

        return {
            "dirty_prob": dirty_prob,
            "clean_prob": clean_prob,
            "label": label,
            "top1_label": top1_label,
            "top1_prob": float(np.clip(float(np.max(top1_probs)), 0.0, 1.0)),
            "entropy_norm": float(np.clip(float(np.mean(entropies)), 0.0, 1.0)),
            "inference_ms": inference_ms,
            "crop_count": int(len(predictions)),
            "crop_dirty_prob_mean": crop_mean,
            "crop_dirty_prob_max": crop_max,
            "crop_dirty_prob_min": crop_min,
            "crop_dirty_prob_std": crop_std,
            "crop_support_count": support_count,
            "crop_support_ratio": support_ratio,
        }

    def _binary_entropy(self, p: float) -> float:
        p_clamped = float(np.clip(p, 1e-8, 1.0 - 1e-8))
        entropy = -(p_clamped * np.log(p_clamped) + (1.0 - p_clamped) * np.log(1.0 - p_clamped))
        return float(entropy / np.log(2.0))

    def _normalized_entropy(self, probs: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs)
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(entropy / np.log(float(len(probs))))

    def _load_imagenet_labels(self) -> list[str]:
        if EfficientNet_B0_Weights is None:
            return []
        try:
            categories = EfficientNet_B0_Weights.IMAGENET1K_V1.meta.get("categories", [])
            return [str(item) for item in categories]
        except Exception:
            return []

    def _resolve_dirty_class_indices(self, labels: list[str]) -> np.ndarray:
        keywords = tuple(str(item).casefold() for item in EFFICIENTNET_CLS_CONFIG["dirty_keywords"])
        selected: list[int] = []
        for index, label in enumerate(labels):
            text = str(label).casefold()
            if any(keyword in text for keyword in keywords):
                selected.append(index)
        return np.array(selected, dtype=np.int64)

    def _build_diff_preview(self, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
        if before_img.shape[:2] != after_img.shape[:2]:
            before_img = cv2.resize(before_img, (after_img.shape[1], after_img.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(before_img, after_img)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        lab_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2LAB)
        lab_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2LAB)
        lab_delta = cv2.absdiff(lab_before, lab_after)
        lab_gray = cv2.cvtColor(lab_delta, cv2.COLOR_BGR2GRAY)

        combined = cv2.addWeighted(diff_gray, 0.58, lab_gray, 0.42, 0)
        combined = cv2.GaussianBlur(combined, (5, 5), 0)

        height, width = combined.shape[:2]
        y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
        center_bias = 0.55 + (1.0 - np.abs(y_coords - 0.68) * 1.55)
        bottom_bias = 0.70 + (y_coords * float(EFFICIENTNET_CLS_CONFIG["difference_focus_bottom_weight"]))
        spatial_weight = np.clip(center_bias * bottom_bias * float(EFFICIENTNET_CLS_CONFIG["difference_focus_center_weight"]), 0.45, 2.8)

        weighted = combined.astype(np.float32) * spatial_weight
        weighted = cv2.normalize(weighted, None, 0, 255, cv2.NORM_MINMAX)
        weighted = weighted.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(weighted)
        enhanced = cv2.medianBlur(enhanced, 3)

        threshold_value = int(max(10, np.percentile(enhanced, float(EFFICIENTNET_CLS_CONFIG["difference_threshold_percentile"])) ))
        _, mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

        heat = cv2.applyColorMap(enhanced, cv2.COLORMAP_TURBO)
        strong = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

        heat = cv2.addWeighted(heat, 0.62, strong, 0.38, 0)
        heat = cv2.GaussianBlur(heat, (3, 3), 0)

        base = after_img.copy()
        overlay_alpha = float(EFFICIENTNET_CLS_CONFIG["difference_overlay_alpha"])
        preview = cv2.addWeighted(base, 1.0 - overlay_alpha, heat, overlay_alpha, 0)

        highlight = cv2.addWeighted(preview, 0.86, strong, 0.14, 0)
        return highlight

    def _build_score_overlay(self, image: np.ndarray, pred: dict[str, Any], title: str) -> np.ndarray:
        canvas = image.copy()
        panel_h = 112
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), thickness=-1)

        dirty_prob = float(pred["dirty_prob"])
        clean_prob = float(pred["clean_prob"])
        label = str(pred["label"])
        top1_label = str(pred["top1_label"])
        top1_prob = float(pred["top1_prob"])

        cv2.putText(canvas, f"{title}: {label.upper()}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"dirty={dirty_prob:.3f} clean={clean_prob:.3f} top1={top1_label} ({top1_prob:.3f})",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        bar_x, bar_y = 12, 78
        bar_w, bar_h = max(120, canvas.shape[1] - 24), 20
        fill_w = int(round(bar_w * np.clip(dirty_prob, 0.0, 1.0)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), thickness=-1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (40, 80, 235), thickness=-1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), thickness=1)
        cv2.putText(canvas, "dirty probability", (bar_x + 6, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

        return canvas

    def _compose_preview(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        before_overlay: np.ndarray,
        after_overlay: np.ndarray,
        diff_preview: np.ndarray,
    ) -> np.ndarray:
        cell_w = max(1, int(EFFICIENTNET_CLS_CONFIG["preview_max_width"]) // 2)
        cell_h = max(1, int(EFFICIENTNET_CLS_CONFIG["preview_max_height"]) // 2)

        panels = [
            annotate_panel(before_img, "Before"),
            annotate_panel(after_img, "After"),
            annotate_panel(before_overlay, "Before score overlay"),
            annotate_panel(diff_preview, "Difference heatmap"),
        ]

        return compose_panel_grid(
            panels,
            cell_width=cell_w,
            cell_height=cell_h,
            max_width=int(EFFICIENTNET_CLS_CONFIG["preview_max_width"]),
            max_height=int(EFFICIENTNET_CLS_CONFIG["preview_max_height"]),
            columns=2,
        )
