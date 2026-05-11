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
    "changeformer": MethodSpec("Changeformer", "projekt-changeformer", CONDA_ENV_DIR / "changeformer.yml"),
    "dinov2": MethodSpec("DINOv2", "projekt-dinov2-cd", CONDA_ENV_DIR / "dinov2_cd.yml"),
    "efficientnet": MethodSpec("EfficientNet", "projekt-efficientnet-cls", CONDA_ENV_DIR / "efficientnet_cls.yml"),
    "sift_ransac": MethodSpec("SIFT + RANSAC + difference map", "01_sift", CONDA_ENV_DIR / "sift_ransac.yml"),
    "siamese_unet": MethodSpec("Siamese U-Net", "projekt-siamese-unet-cd", CONDA_ENV_DIR / "siamese_unet_cd.yml"),
    "yolov8_seg": MethodSpec("YOLOv8 segmentation", "projekt-yolov8-seg", CONDA_ENV_DIR / "yolov8_seg.yml"),
}

METHODS = list(METHOD_SPECS)
METHOD_LABELS = {method_id: spec.label for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV = {method_id: spec.env_name for method_id, spec in METHOD_SPECS.items()}
METHOD_ENV_FILES = {method_id: spec.env_file for method_id, spec in METHOD_SPECS.items()}
METHOD_ALIASES = {
    "dinov2_cd": "dinov2",
    "efficientnet_cls": "efficientnet",
    "siamese_unet_cd": "siamese_unet",
}


def get_method_spec(method_id: str) -> MethodSpec:
    method_id = METHOD_ALIASES.get(method_id, method_id)
    try:
        return METHOD_SPECS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown method: {method_id}") from exc