from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .core import AnalysisResult
except ImportError:
    from core import AnalysisResult


BASE_DIR = Path(__file__).resolve().parent.parent
CONDA_ENV_DIR = BASE_DIR / "conda_envs"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    env_name: str
    env_file: Path


METHOD_SPECS = {
    "sift_ransac": MethodSpec("SIFT + RANSAC + difference map", "projekt-sift-ransac", CONDA_ENV_DIR / "sift_ransac.yml"),
    "orb_ransac": MethodSpec("ORB + RANSAC + difference map", "projekt-orb-ransac", CONDA_ENV_DIR / "orb_ransac.yml"),
    "yolov8_det": MethodSpec("YOLOv8-detection", "projekt-yolov8-det", CONDA_ENV_DIR / "yolov8_det.yml"),
    "faster_rcnn": MethodSpec("Faster R-CNN", "projekt-faster-rcnn", CONDA_ENV_DIR / "faster_rcnn.yml"),
    "mask_rcnn": MethodSpec("Mask R-CNN", "projekt-mask-rcnn", CONDA_ENV_DIR / "mask_rcnn.yml"),
    "yolov8_seg": MethodSpec("YOLOv8-seg", "projekt-yolov8-seg", CONDA_ENV_DIR / "yolov8_seg.yml"),
    "unet_seg": MethodSpec("U-Net segmentation", "projekt-unet-seg", CONDA_ENV_DIR / "unet_seg.yml"),
    "deeplabv3plus_seg": MethodSpec("DeepLabV3+ segmentation", "projekt-deeplabv3plus-seg", CONDA_ENV_DIR / "deeplabv3plus_seg.yml"),
    "segformer_seg": MethodSpec("SegFormer segmentation", "projekt-segformer-seg", CONDA_ENV_DIR / "segformer_seg.yml"),
    "siamese_unet_cd": MethodSpec("Siamese U-Net change detection", "projekt-siamese-unet-cd", CONDA_ENV_DIR / "siamese_unet_cd.yml"),
    "bit_like_cd": MethodSpec("BIT-like change detection model", "projekt-bit-like-cd", CONDA_ENV_DIR / "bit_like_cd.yml"),
    "changeformer": MethodSpec("ChangeFormer", "projekt-changeformer", CONDA_ENV_DIR / "changeformer.yml"),
    "dinov2_cd": MethodSpec("DINOv2 change detection", "projekt-dinov2-cd", CONDA_ENV_DIR / "dinov2_cd.yml"),
    "open_cd": MethodSpec("Open-CD baseline models", "projekt-open-cd", CONDA_ENV_DIR / "open_cd.yml"),
    "resnet_cls": MethodSpec("ResNet classifier for clean/dirty", "projekt-resnet-cls", CONDA_ENV_DIR / "resnet_cls.yml"),
    "efficientnet_cls": MethodSpec("EfficientNet classifier for clean/dirty", "projekt-efficientnet-cls", CONDA_ENV_DIR / "efficientnet_cls.yml"),
    "hybrid_score": MethodSpec("Hybrid score", "projekt-hybrid-score", CONDA_ENV_DIR / "hybrid_score.yml"),
}

METHODS = list(METHOD_SPECS)
METHOD_LABELS = {method_id: spec.label for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV = {method_id: spec.env_name for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV_FILES = {method_id: spec.env_file for method_id, spec in METHOD_SPECS.items()}


def get_method_spec(method_id: str) -> MethodSpec:
    try:
        return METHOD_SPECS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown method: {method_id}") from exc


class AlgorithmRunner(ABC):
    method_id: str

    def __init__(self, method_id: str):
        self.method_id = method_id

    @property
    def spec(self) -> MethodSpec:
        return get_method_spec(self.method_id)

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def env_name(self) -> str:
        return self.spec.env_name

    @property
    def env_file(self) -> Path:
        return self.spec.env_file

    @abstractmethod
    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        raise NotImplementedError


class DifferenceMapRunner(AlgorithmRunner):
    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)
        before_size = before.stat().st_size
        after_size = after.stat().st_size
        size_delta = abs(before_size - after_size)
        ratio = round(size_delta / max(before_size, after_size, 1), 4)
        summary = "Text-only preview without graphical dependencies."
        preview_text = (
            "This method has not been implemented yet.\n"
            f"Before file: {before.name}\n"
            f"After file: {after.name}\n"
            f"Before size: {before_size} bytes\n"
            f"After size: {after_size} bytes\n"
            f"Placeholder change_ratio: {ratio}"
        )
        metrics = {
            "change_ratio": ratio,
            "env": self.env_name,
            "mode": "classical baseline",
            "size_delta_bytes": size_delta,
        }
        return AnalysisResult(self.method_id, self.label, summary, metrics, before, after, preview_text)


class ScoredPlaceholderRunner(AlgorithmRunner):
    def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
        before = Path(before_path)
        after = Path(after_path)
        before_size = before.stat().st_size
        after_size = after.stat().st_size
        ratio = round(abs(before_size - after_size) / max(before_size, after_size, 1), 4)
        summary = "This method has not been implemented yet."
        preview_text = (
            f"Method: {self.label}\n"
            f"Before file: {before.name}\n"
            f"After file: {after.name}\n"
            "The real model has not been connected yet.\n"
            f"Placeholder change_ratio: {ratio}"
        )
        metrics = {
            "change_ratio": ratio,
            "env": self.env_name,
            "mode": "placeholder scaffold",
            "size_delta_bytes": abs(before_size - after_size),
        }
        return AnalysisResult(self.method_id, self.label, summary, metrics, before, after, preview_text)


def create_runner(method_id: str, **kwargs: Any) -> AlgorithmRunner:
    if method_id == "sift_ransac":
        try:
            from .method_scripts.sift_ransac import SiftRansacRunner
        except ImportError as exc:
            # Re-raise real dependency import errors (e.g. numpy/cv2) instead of masking them.
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "sift_ransac",
                "launcher.sift_ransac",
                "method_scripts.sift_ransac",
                "launcher.method_scripts.sift_ransac",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.sift_ransac import SiftRansacRunner

        return SiftRansacRunner(method_id)
    if method_id == "yolov8_seg":
        try:
            from .method_scripts.yolov8_seg import YoloV8SegRunner
        except ImportError as exc:
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "yolov8_seg",
                "launcher.yolov8_seg",
                "method_scripts.yolov8_seg",
                "launcher.method_scripts.yolov8_seg",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.yolov8_seg import YoloV8SegRunner

        return YoloV8SegRunner(method_id, **kwargs)
    if method_id == "efficientnet_cls":
        weights_path = _resolve_latest_efficientnet_checkpoint()
        if weights_path is not None and "weights_path" not in kwargs:
            kwargs["weights_path"] = weights_path
        try:
            from .method_scripts.efficientnet_cls import EfficientNetClsRunner
        except ImportError as exc:
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "efficientnet_cls",
                "launcher.efficientnet_cls",
                "method_scripts.efficientnet_cls",
                "launcher.method_scripts.efficientnet_cls",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.efficientnet_cls import EfficientNetClsRunner

        return EfficientNetClsRunner(method_id, **kwargs)
    if method_id == "changeformer":
        try:
            from .method_scripts.changeformer import ChangeformerRunner
        except ImportError as exc:
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "changeformer",
                "launcher.changeformer",
                "method_scripts.changeformer",
                "launcher.method_scripts.changeformer",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.changeformer import ChangeformerRunner

        return ChangeformerRunner(method_id, **kwargs)
    if method_id == "dinov2_cd":
        try:
            from .method_scripts.dinov2_cd import DinoV2CdRunner
        except ImportError as exc:
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "dinov2_cd",
                "launcher.dinov2_cd",
                "method_scripts.dinov2_cd",
                "launcher.method_scripts.dinov2_cd",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.dinov2_cd import DinoV2CdRunner

        return DinoV2CdRunner(method_id, **kwargs)
    if method_id == "siamese_unet_cd":
        try:
            from .method_scripts.siamese_unet_cd import SiameseUnetCdRunner
        except ImportError as exc:
            is_relative_import_context_error = "attempted relative import" in str(exc)
            missing_target_module = getattr(exc, "name", None) in {
                "siamese_unet_cd",
                "launcher.siamese_unet_cd",
                "method_scripts.siamese_unet_cd",
                "launcher.method_scripts.siamese_unet_cd",
            }
            if not (is_relative_import_context_error or missing_target_module):
                raise
            from method_scripts.siamese_unet_cd import SiameseUnetCdRunner

        return SiameseUnetCdRunner(method_id, **kwargs)
    if method_id == "orb_ransac":
        return DifferenceMapRunner(method_id)
    if method_id not in METHODS:
        raise KeyError(f"Unknown method: {method_id}")
    return ScoredPlaceholderRunner(method_id)


def _resolve_latest_efficientnet_checkpoint() -> Path | None:
    training_root = BASE_DIR / "results" / "training"
    if not training_root.exists():
        return None

    preferred_checkpoint = training_root / "20260419_195422__efficientnet_taco__110a7f2e" / "best.pt"
    if preferred_checkpoint.exists():
        return preferred_checkpoint

    canonical_checkpoint = BASE_DIR / "results" / "models" / "efficientnet" / "best.pt"
    if canonical_checkpoint.exists():
        return canonical_checkpoint

    best_candidates = [path for path in training_root.rglob("best.pt") if path.is_file()]
    if best_candidates:
        best_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return best_candidates[0]

    last_candidates = [path for path in training_root.rglob("last.pt") if path.is_file()]
    if last_candidates:
        last_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return last_candidates[0]

    return None
