from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONDA_ENV_DIR = BASE_DIR / "conda_envs"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    env_name: str
    env_file: Path


METHOD_SPECS = {
    "changeformer": MethodSpec("ChangeFormer", "projekt-changeformer", CONDA_ENV_DIR / "base.yml"),
    "changeformer_dinov2": MethodSpec("ChangeFormer + DINOv2", "projekt-changeformer", CONDA_ENV_DIR / "base.yml"),
    "dinov2": MethodSpec("DINOv2", "projekt-base", CONDA_ENV_DIR / "base.yml"),
    "efficientnet": MethodSpec("EfficientNet", "projekt-efficientnet-cls", CONDA_ENV_DIR / "base.yml"),
    "sift_ransac": MethodSpec("SIFT + RANSAC + difference map", "projekt-base", CONDA_ENV_DIR / "base.yml"),
    "siamese_unet": MethodSpec("Siamese U-Net", "projekt-base", CONDA_ENV_DIR / "base.yml"),
    "siamese_dinov2": MethodSpec("Siamese U-Net + DINOv2", "projekt-base", CONDA_ENV_DIR / "base.yml"),
    "yolov8_seg": MethodSpec("YOLOv8-seg", "projekt-base", CONDA_ENV_DIR / "base.yml"),
    # "orb_ransac": MethodSpec("ORB + RANSAC + difference map", "projekt-orb-ransac", CONDA_ENV_DIR / "orb_ransac.yml"),
    # "yolov8_det": MethodSpec("YOLOv8-detection", "projekt-yolov8-det", CONDA_ENV_DIR / "yolov8_det.yml"),
    # "faster_rcnn": MethodSpec("Faster R-CNN", "projekt-faster-rcnn", CONDA_ENV_DIR / "faster_rcnn.yml"),
    # "mask_rcnn": MethodSpec("Mask R-CNN", "projekt-mask-rcnn", CONDA_ENV_DIR / "mask_rcnn.yml"),
    # "unet_seg": MethodSpec("U-Net segmentation", "projekt-unet-seg", CONDA_ENV_DIR / "unet_seg.yml"),
    # "deeplabv3plus_seg": MethodSpec("DeepLabV3+ segmentation", "projekt-deeplabv3plus-seg", CONDA_ENV_DIR / "deeplabv3plus_seg.yml"),
    # "segformer_seg": MethodSpec("SegFormer segmentation", "projekt-segformer-seg", CONDA_ENV_DIR / "segformer_seg.yml"),
    # "siamese_unet_cd": MethodSpec("Siamese U-Net change detection", "projekt-siamese-unet-cd", CONDA_ENV_DIR / "siamese_unet_cd.yml"),
    # "bit_like_cd": MethodSpec("BIT-like change detection model", "projekt-bit-like-cd", CONDA_ENV_DIR / "bit_like_cd.yml"),
    # "open_cd": MethodSpec("Open-CD baseline models", "projekt-open-cd", CONDA_ENV_DIR / "open_cd.yml"),
    # "resnet_cls": MethodSpec("ResNet classifier for clean/dirty", "projekt-resnet-cls", CONDA_ENV_DIR / "resnet_cls.yml"),
    # "hybrid_score": MethodSpec("Hybrid score", "projekt-hybrid-score", CONDA_ENV_DIR / "hybrid_score.yml"),
}

METHODS = list(METHOD_SPECS)
METHOD_LABELS = {method_id: spec.label for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV = {method_id: spec.env_name for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV_FILES = {method_id: spec.env_file for method_id, spec in METHOD_SPECS.items()}

METHOD_ALIASES = {
    "efficientnet_cls": "efficientnet",
    "dinov2_cd": "dinov2",
    "siamese_unet_cd": "siamese_unet",
    "siamese_unet_dinov2_cd": "siamese_dinov2",
    "changeformer_dinov2_cd": "changeformer_dinov2",
}


def get_method_spec(method_id: str) -> MethodSpec:
    method_id = METHOD_ALIASES.get(method_id, method_id)
    try:
        return METHOD_SPECS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown method: {method_id}") from exc


def normalize_method_id(method_id: str) -> str:
    return METHOD_ALIASES.get(method_id, method_id)