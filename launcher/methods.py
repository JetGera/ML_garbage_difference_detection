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
    "sift_ransac": MethodSpec("SIFT + RANSAC + difference map", "01_sift", CONDA_ENV_DIR / "sift_ransac.yml"),
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