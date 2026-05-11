from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

YOLOV8_SEG_CONFIG = {
	# Default pretrained checkpoint for inference and training starts.
	"model_name": "yolov8m_trash_finetuned.pt",
	# Default fine-tuned checkpoint for trash segmentation.
	"weights_path": Path(__file__).resolve().parent.parent.parent / "runs" / "segment" / "datasets" / "TACO" / "yolo_seg" / "runs" / "taco_mseg_restart_70ep_best" / "weights" / "best.pt",
	
	# Root folder with the TACO dataset and its exported YOLO data.
	"dataset_root": Path(__file__).resolve().parent.parent.parent / "datasets" / "TACO",
	# Subfolder inside the dataset root where YOLO artifacts are written.
	"dataset_export_subdir": "yolo_seg",
	# Preferred source annotation files used to build the YOLO export.
	"dataset_annotations_names": ["annotations.json", "annotations_unofficial.json"],
	# Single class name used for the exported YOLO segmentation dataset.
	"dataset_single_class_name": "trash",
	# Whether the runner should build the YOLO export automatically.
	"prepare_taco_export": True,
	# Fraction of images assigned to the training split.
	"split_train_ratio": 0.8,
	# Fraction of images assigned to the validation split.
	"split_val_ratio": 0.1,
	# Minimum confidence score for detections to keep.
	"confidence_threshold": 0.20,
	# IoU threshold used by prediction-side suppression and merge logic.
	"iou_threshold": 0.40,
	# Input image size used for regular full-frame inference.
	"image_size": 1536,
	# Maximum number of detections retained per image.
	"max_det": 1800,
	# Requested runtime device: auto, cuda, or cpu.
	"device": "auto",
	# Force CPU execution even when CUDA is available.
	"force_cpu": False,
	# Optional class name/id filter for predictions.
	"class_filter": None,
	# Tile inference policy: off, auto, or always.
	"tile_inference_mode": "auto",
	# Minimum image side length before tile inference can activate in auto mode.
	"tile_min_image_side": 768,
	# Tile edge length used for sliding-window inference.
	"tile_size": 640,
	# Fractional overlap between neighboring tiles.
	"tile_overlap_ratio": 0.5,
	# Prediction resolution used for each tile crop.
	"tile_predict_image_size": 960,
	# Maximum number of tiles allowed per image to avoid runaway inference.
	"tile_max_tiles": 196,
	# IoU threshold used when merging tile detections.
	"tile_nms_iou": 0.6,
	# Minimum mask area in pixels for tile detections to survive filtering.
	"tile_min_mask_area_px": 100,
	# Default number of training epochs for the fine-tuning run.
	"train_epochs": 90,
	# Default training batch size.
	"train_batch": 2,
	# Training image size used for the main fine-tuning run.
	"train_image_size": 768,
	# Early-stopping patience during training.
	"train_patience": 20,
	# Data-loader worker count during training.
	"train_workers": 2,
	# Dataset caching mode for Ultralytics training.
	"train_cache": "disk",
	# Enable AMP mixed-precision training when supported.
	"train_amp": True,
	# Training optimizer setting passed to Ultralytics.
	"train_optimizer": "auto",
	# Initial learning rate for training.
	"train_lr0": 0.01,
	# Weight decay used during training.
	"train_weight_decay": 0.0005,
	# Extra model checkpoints to try if the main model hits OOM.
	"train_model_fallbacks": ["yolov8s-seg.pt"],
	# Transparency of the mask overlay in previe Оw images.
	"mask_alpha": 0.67,
	# Maximum preview width before the composed image is scaled down.
	"preview_max_width": 4000,
	# Maximum preview height before the composed image is scaled down.
	"preview_max_height": 3000,
	# Height of the title bar drawn on each preview panel.
	"panel_title_height": 44,
	# Minimum mask area in pixels when summarizing detections.
	"min_mask_area_px": 200,
	# Heatmap palette used for difference previews.
	"colormap": cv2.COLORMAP_TURBO,
}


class _SimpleBoxes:
	def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
		self.xyxy = xyxy
		self.conf = conf
		self.cls = cls


class _SimpleMasks:
	def __init__(self, data: np.ndarray):
		self.data = data


class _SimpleResult:
	def __init__(self, boxes: _SimpleBoxes, masks: _SimpleMasks):
		self.boxes = boxes
		self.masks = masks

try:
	from ..core import AnalysisResult
	from ..methods import get_method_spec
except ImportError:
	from core import AnalysisResult
	from methods import get_method_spec


class YoloV8SegRunner:
	def __init__(
		self,
		method_id: str,
		device: str = "auto",
		force_cpu: bool = False,
		model_name: str | None = None,
		weights_path: str | Path | None = None,
		class_filter: list[int | str] | None = None,
		dataset_root: str | Path | None = None,
		prepare_taco_export: bool = bool(YOLOV8_SEG_CONFIG["prepare_taco_export"]),
	):
		self.method_id = method_id
		self.spec = get_method_spec(method_id)
		self.device = device
		self.force_cpu = force_cpu
		self.model_name = model_name or str(YOLOV8_SEG_CONFIG["model_name"])
		self.weights_path = Path(weights_path) if weights_path is not None else Path(YOLOV8_SEG_CONFIG["weights_path"])
		self.class_filter = class_filter if class_filter is not None else YOLOV8_SEG_CONFIG["class_filter"]
		self.dataset_root = Path(dataset_root) if dataset_root is not None else Path(YOLOV8_SEG_CONFIG["dataset_root"])
		self.dataset_export_root = self.dataset_root / str(YOLOV8_SEG_CONFIG["dataset_export_subdir"])
		self.dataset_annotations_path = self._resolve_taco_annotations_path()
		self.dataset_single_class_name = str(YOLOV8_SEG_CONFIG["dataset_single_class_name"])
		self.prepare_taco_export = bool(prepare_taco_export)
		self.dataset_yaml_path = self.dataset_export_root / "data.yaml"
		self._model = None
		self._model_source = None
		self._taco_export_state: dict[str, Any] | None = None

	@property
	def label(self) -> str:
		return self.spec.label

	def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
		if YOLO is None:
			raise RuntimeError("ultralytics is not installed in the active environment")

		taco_state = self._ensure_taco_yolo_export() if self.prepare_taco_export else self._build_taco_export_state(dry_run=True)

		before = Path(before_path)
		after = Path(after_path)
		before_img = self._read_color_image(before)
		after_img = self._read_color_image(after)

		model = self._load_model()
		device_used = self._resolve_device()
		class_ids = self._resolve_class_ids(model)

		before_result, before_ms, before_tile_count = self._predict(model, before_img, device_used, class_ids)
		after_result, after_ms, after_tile_count = self._predict(model, after_img, device_used, class_ids)

		before_summary = self._summarize_prediction(before_img, before_result, before_ms, side="before")
		after_summary = self._summarize_prediction(after_img, after_result, after_ms, side="after")
		has_any_detections = bool(before_summary["instance_count"] or after_summary["instance_count"])
		diff_preview = self._build_diff_preview(before_img, after_img)

		preview = self._compose_preview(
			before_img=before_img,
			after_img=after_img,
			before_overlay=before_summary["overlay"],
			after_overlay=after_summary["overlay"],
			diff_preview=diff_preview,
			has_any_detections=has_any_detections,
		)

		output_dir = self._prepare_output_dir(before, after)
		artifacts = self._save_artifacts(
			output_dir=output_dir,
			before_overlay=before_summary["overlay"],
			after_overlay=after_summary["overlay"],
			before_mask_canvas=before_summary["mask_canvas"],
			after_mask_canvas=after_summary["mask_canvas"],
			preview=preview,
		)

		before_mask_area = int(before_summary["mask_area_px"])
		after_mask_area = int(after_summary["mask_area_px"])
		before_area = int(before_img.shape[0] * before_img.shape[1])
		after_area = int(after_img.shape[0] * after_img.shape[1])
		before_ratio = before_mask_area / max(before_area, 1)
		after_ratio = after_mask_area / max(after_area, 1)

		metrics = {
			"analysis_mode": "yolov8_seg_pair",
			"model_name": self.model_name,
			"model_source": str(self._resolve_model_source()),
			"device_requested": self.device,
			"device_used": device_used,
			"force_cpu": bool(self.force_cpu),
			"cuda_available": bool(torch is not None and torch.cuda.is_available()),
			"image_size": int(YOLOV8_SEG_CONFIG["image_size"]),
			"confidence_threshold": float(YOLOV8_SEG_CONFIG["confidence_threshold"]),
			"iou_threshold": float(YOLOV8_SEG_CONFIG["iou_threshold"]),
			"tile_inference_mode": str(YOLOV8_SEG_CONFIG["tile_inference_mode"]),
			"tile_size": int(YOLOV8_SEG_CONFIG["tile_size"]),
			"tile_overlap_ratio": float(YOLOV8_SEG_CONFIG["tile_overlap_ratio"]),
			"before_tile_count": int(before_tile_count),
			"after_tile_count": int(after_tile_count),
			"selected_class_count": len(class_ids or []),
			"selected_class_ids": class_ids or [],
			"taco_dataset_root": str(self.dataset_root),
			"taco_annotations_path": str(self.dataset_annotations_path),
			"taco_export_root": str(self.dataset_export_root),
			"taco_data_yaml": str(self.dataset_yaml_path),
			"taco_source_images": int(taco_state.get("source_images", 0)),
			"taco_exported_images": int(taco_state.get("exported_images", 0)),
			"taco_exported_labels": int(taco_state.get("exported_labels", 0)),
			"taco_skipped_missing_images": int(taco_state.get("missing_images", 0)),
			"taco_skipped_invalid_polygons": int(taco_state.get("invalid_polygons", 0)),
			"before_instance_count": int(before_summary["instance_count"]),
			"after_instance_count": int(after_summary["instance_count"]),
			"instance_count_total": int(before_summary["instance_count"] + after_summary["instance_count"]),
			"before_mask_area_px": before_mask_area,
			"after_mask_area_px": after_mask_area,
			"before_mask_area_ratio": round(before_ratio, 6),
			"after_mask_area_ratio": round(after_ratio, 6),
			"mask_area_delta_px": after_mask_area - before_mask_area,
			"mask_area_delta_ratio": round(after_ratio - before_ratio, 6),
			"before_bbox_area_px": int(before_summary["bbox_area_px"]),
			"after_bbox_area_px": int(after_summary["bbox_area_px"]),
			"before_mean_confidence": round(float(before_summary["mean_confidence"]), 6),
			"after_mean_confidence": round(float(after_summary["mean_confidence"]), 6),
			"before_inference_ms": round(float(before_ms), 3),
			"after_inference_ms": round(float(after_ms), 3),
			"inference_ms_total": round(float(before_ms + after_ms), 3),
		}

		summary = "YOLOv8 segmentation inference on a before/after pair with device fallback and mask aggregation."
		preview_text = (
			f"Device: {device_used} (requested: {self.device})\n"
			f"TACO export: {self.dataset_yaml_path if self.dataset_yaml_path.exists() else 'not prepared'}\n"
			f"Tile inference mode: {metrics['tile_inference_mode']} (before tiles: {before_tile_count}, after tiles: {after_tile_count})\n"
			f"Before instances: {metrics['before_instance_count']}, mask area: {before_mask_area}px\n"
			f"After instances: {metrics['after_instance_count']}, mask area: {after_mask_area}px\n"
			f"Mask area delta ratio: {metrics['mask_area_delta_ratio']:.6f}\n"
			f"Empty detection fallback: {'diff preview' if not has_any_detections else 'not used'}"
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
		if self._model is None or self._model_source != model_source:
			self._model = YOLO(str(model_source))
			self._model_source = model_source
		return self._model

	def _resolve_model_source(self) -> Path | str:
		if self.weights_path is not None and self.weights_path.exists():
			return self.weights_path

		candidate_weights = [
			self.dataset_export_root / "weights" / "best.pt",
			self.dataset_export_root / "weights" / "last.pt",
			self.dataset_export_root / "runs" / "train" / "weights" / "best.pt",
			self.dataset_export_root / "runs" / "train" / "weights" / "last.pt",
		]
		repo_run_root = self.dataset_root.parent.parent / "runs" / "segment" / "datasets" / "TACO" / "yolo_seg" / "runs"
		run_roots = [repo_run_root, self.dataset_export_root / "runs"]
		for runs_root in run_roots:
			if runs_root.exists():
				run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
				run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
				for run_dir in run_dirs:
					candidate_weights.append(run_dir / "weights" / "best.pt")
					candidate_weights.append(run_dir / "weights" / "last.pt")
		for candidate in candidate_weights:
			if candidate.exists():
				return candidate

		return self.model_name

	def train_on_taco(
		self,
		epochs: int | None = None,
		batch: int | None = None,
		project_name: str = "yolov8m_seg_taco",
		imgsz: int | None = None,
		device: str | None = None,
		patience: int | None = None,
		workers: int | None = None,
		cache: str | bool | None = None,
		amp: bool | None = None,
		optimizer: str | None = None,
		lr0: float | None = None,
		weight_decay: float | None = None,
		allow_oom_fallback: bool = True,
	) -> dict[str, Any]:
		if YOLO is None:
			raise RuntimeError("ultralytics is not installed in the active environment")

		export_state = self._ensure_taco_yolo_export()
		if not self.dataset_yaml_path.exists():
			raise RuntimeError(f"TACO YOLO export is missing: {self.dataset_yaml_path}")

		training_device = device or self._resolve_device()
		training_epochs = int(epochs or YOLOV8_SEG_CONFIG["train_epochs"])
		training_batch = max(1, int(batch or YOLOV8_SEG_CONFIG["train_batch"]))
		training_imgsz = max(320, int(imgsz or YOLOV8_SEG_CONFIG["train_image_size"]))
		training_patience = int(patience or YOLOV8_SEG_CONFIG["train_patience"])
		training_workers = max(0, int(workers or YOLOV8_SEG_CONFIG["train_workers"]))
		training_cache = YOLOV8_SEG_CONFIG["train_cache"] if cache is None else cache
		training_amp = bool(YOLOV8_SEG_CONFIG["train_amp"] if amp is None else amp)
		training_optimizer = str(optimizer or YOLOV8_SEG_CONFIG["train_optimizer"])
		training_lr0 = float(lr0 if lr0 is not None else YOLOV8_SEG_CONFIG["train_lr0"])
		training_weight_decay = float(weight_decay if weight_decay is not None else YOLOV8_SEG_CONFIG["train_weight_decay"])
		project_dir = self.dataset_export_root / "runs"
		project_dir.mkdir(parents=True, exist_ok=True)

		model_sources: list[str] = []
		if self.weights_path is not None and self.weights_path.exists():
			model_sources.append(str(self.weights_path))
		else:
			resolved_source = self._resolve_model_source()
			if isinstance(resolved_source, Path) and resolved_source.exists():
				model_sources.append(str(resolved_source))
			else:
				model_sources.append(str(self.model_name))
			if allow_oom_fallback:
				for fallback_model in YOLOV8_SEG_CONFIG.get("train_model_fallbacks", []):
					candidate = str(fallback_model)
					if candidate and candidate not in model_sources:
						model_sources.append(candidate)

		attempts = self._build_train_attempts(model_sources, training_batch, training_imgsz, allow_oom_fallback)
		results = None
		attempt_used: dict[str, Any] | None = None
		last_oom_error: RuntimeError | None = None

		for attempt in attempts:
			model = YOLO(str(attempt["model_source"]))
			try:
				results = model.train(
					data=str(self.dataset_yaml_path),
					epochs=training_epochs,
					batch=int(attempt["batch"]),
					imgsz=int(attempt["imgsz"]),
					device=training_device,
					project=str(project_dir),
					name=project_name,
					patience=training_patience,
					workers=training_workers,
					cache=training_cache,
					amp=training_amp,
					optimizer=training_optimizer,
					lr0=training_lr0,
					weight_decay=training_weight_decay,
					exist_ok=True,
					verbose=False,
				)
				attempt_used = attempt
				break
			except RuntimeError as exc:
				if "out of memory" not in str(exc).lower():
					raise
				last_oom_error = exc
				continue

		if results is None:
			if last_oom_error is not None:
				raise RuntimeError("Training failed after all OOM fallback attempts") from last_oom_error
			raise RuntimeError("Training did not produce results")

		best_weight = None
		for candidate in [
			project_dir / project_name / "weights" / "best.pt",
			project_dir / project_name / "weights" / "last.pt",
		]:
			if candidate.exists():
				best_weight = candidate
				break

		return {
			"data_yaml": str(self.dataset_yaml_path),
			"project_dir": str(project_dir),
			"best_weight": str(best_weight) if best_weight else None,
			"train_results": str(results),
			"attempt_used": attempt_used or {},
			"taco_exported_images": int(export_state.get("exported_images", 0)),
			"taco_exported_labels": int(export_state.get("exported_labels", 0)),
		}

	def _build_train_attempts(
		self,
		model_sources: list[str],
		batch: int,
		imgsz: int,
		allow_oom_fallback: bool,
	) -> list[dict[str, Any]]:
		attempts: list[dict[str, Any]] = []
		seen: set[tuple[str, int, int]] = set()

		for model_source in model_sources:
			candidates = [(batch, imgsz)]
			if allow_oom_fallback:
				if batch > 1:
					candidates.append((1, imgsz))
				if imgsz > 640:
					candidates.append((min(batch, 2), 640))
					candidates.append((1, 640))

			for candidate_batch, candidate_imgsz in candidates:
				key = (model_source, int(candidate_batch), int(candidate_imgsz))
				if key in seen:
					continue
				seen.add(key)
				attempts.append(
					{
						"model_source": model_source,
						"batch": int(candidate_batch),
						"imgsz": int(candidate_imgsz),
					}
				)

		return attempts

	def _ensure_taco_yolo_export(self) -> dict[str, Any]:
		if self._taco_export_state is None:
			self._taco_export_state = self._build_taco_export_state(dry_run=False)
		return self._taco_export_state

	def _resolve_taco_annotations_path(self) -> Path:
		annotations_names = YOLOV8_SEG_CONFIG.get("dataset_annotations_names", ["annotations.json"])
		if isinstance(annotations_names, str):
			annotations_names = [annotations_names]

		preferred_path: Path | None = None
		for annotation_name in annotations_names:
			candidate = self.dataset_root / "data" / str(annotation_name)
			if preferred_path is None:
				preferred_path = candidate
			if candidate.exists():
				return candidate

		return preferred_path or (self.dataset_root / "data" / "annotations.json")

	def _build_taco_export_state(self, dry_run: bool) -> dict[str, Any]:
		state: dict[str, Any] = {
			"source_images": 0,
			"exported_images": 0,
			"exported_labels": 0,
			"missing_images": 0,
			"invalid_polygons": 0,
			"splits": {"train": 0, "val": 0, "test": 0},
		}
		self.dataset_annotations_path = self._resolve_taco_annotations_path()
		if not self.dataset_annotations_path.exists():
			return state

		try:
			payload = json.loads(self.dataset_annotations_path.read_text(encoding="utf-8"))
		except Exception:
			return state

		images = payload.get("images", []) or []
		annotations = payload.get("annotations", []) or []
		annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
		for annotation in annotations:
			image_id = annotation.get("image_id")
			if image_id is None:
				continue
			annotations_by_image[int(image_id)].append(annotation)

		state["source_images"] = len(images)
		if dry_run:
			return state

		self.dataset_export_root.mkdir(parents=True, exist_ok=True)
		images_root = self.dataset_export_root / "images"
		labels_root = self.dataset_export_root / "labels"
		for split in ("train", "val", "test"):
			(images_root / split).mkdir(parents=True, exist_ok=True)
			(labels_root / split).mkdir(parents=True, exist_ok=True)

		for image_info in images:
			relative_image = Path(str(image_info.get("file_name", "")))
			if not relative_image.parts:
				continue

			image_id = int(image_info.get("id", -1))
			width = int(image_info.get("width", 0))
			height = int(image_info.get("height", 0))
			source_image = self.dataset_root / "data" / relative_image
			if not source_image.exists():
				state["missing_images"] += 1
				continue

			split_name = self._resolve_taco_split(relative_image.as_posix(), image_id)
			target_image = images_root / split_name / relative_image
			target_label = labels_root / split_name / relative_image.with_suffix(".txt")
			target_image.parent.mkdir(parents=True, exist_ok=True)
			target_label.parent.mkdir(parents=True, exist_ok=True)

			self._link_or_copy_file(source_image, target_image)
			label_lines = self._build_taco_label_lines(annotations_by_image.get(image_id, []), width, height, state)
			target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
			state["exported_images"] += 1
			state["exported_labels"] += 1
			state["splits"][split_name] += 1

		self.dataset_yaml_path.write_text(self._build_taco_data_yaml_text(), encoding="utf-8")
		(self.dataset_export_root / "metadata.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
		return state

	def _build_taco_data_yaml_text(self) -> str:
		return (
			f"path: {self.dataset_export_root.as_posix()}\n"
			"train: images/train\n"
			"val: images/val\n"
			"test: images/test\n"
			"nc: 1\n"
			"names:\n"
			f"  0: {self.dataset_single_class_name}\n"
		)

	def _resolve_taco_split(self, relative_image_path: str, image_id: int) -> str:
		split_seed = f"{image_id}:{relative_image_path}"
		hash_value = int.from_bytes(hashlib.md5(split_seed.encode("utf-8")).digest(), "big")
		ratio = (hash_value % 1000) / 1000.0
		train_cutoff = float(YOLOV8_SEG_CONFIG["split_train_ratio"])
		val_cutoff = train_cutoff + float(YOLOV8_SEG_CONFIG["split_val_ratio"])
		if ratio < train_cutoff:
			return "train"
		if ratio < val_cutoff:
			return "val"
		return "test"

	def _link_or_copy_file(self, source: Path, target: Path) -> None:
		if target.exists():
			return
		try:
			os.link(source, target)
		except OSError:
			shutil.copy2(source, target)

	def _build_taco_label_lines(
		self,
		annotations: list[dict[str, Any]],
		width: int,
		height: int,
		state: dict[str, Any],
	) -> list[str]:
		label_lines: list[str] = []
		for annotation in annotations:
			if int(annotation.get("iscrowd", 0)):
				continue
			segments = annotation.get("segmentation", [])
			if not isinstance(segments, list):
				state["invalid_polygons"] += 1
				continue
			for polygon in segments:
				if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
					state["invalid_polygons"] += 1
					continue
				normalized_points: list[str] = []
				for index in range(0, len(polygon), 2):
					x = float(polygon[index]) / max(width, 1)
					y = float(polygon[index + 1]) / max(height, 1)
					x = min(max(x, 0.0), 1.0)
					y = min(max(y, 0.0), 1.0)
					normalized_points.append(f"{x:.6f}")
					normalized_points.append(f"{y:.6f}")
				label_lines.append("0 " + " ".join(normalized_points))
		return label_lines

	def _resolve_device(self) -> str:
		if self.force_cpu:
			return "cpu"
		requested = (self.device or "auto").strip().lower()
		cuda_available = bool(torch is not None and torch.cuda.is_available())
		if requested in {"cpu", "cpu:0"}:
			return "cpu"
		if requested.startswith("cuda"):
			return requested if cuda_available else "cpu"
		if requested == "auto":
			return "cuda" if cuda_available else "cpu"
		return requested

	def _resolve_class_ids(self, model) -> list[int] | None:
		if not self.class_filter:
			return None

		names = getattr(model, "names", {})
		resolved: list[int] = []
		for item in self.class_filter:
			if isinstance(item, int):
				resolved.append(item)
				continue

			needle = str(item).casefold()
			if isinstance(names, dict):
				for class_id, class_name in names.items():
					if str(class_name).casefold() == needle:
						resolved.append(int(class_id))
						break
			else:
				for class_id, class_name in enumerate(names):
					if str(class_name).casefold() == needle:
						resolved.append(int(class_id))
						break

		return sorted(set(resolved)) or None

	def _predict(self, model, image: np.ndarray, device: str, class_ids: list[int] | None) -> tuple[Any, float, int]:
		start = perf_counter()
		tile_count = 1
		if self._should_use_tile_inference(image):
			result, tile_count = self._predict_tiled(model, image, device, class_ids)
		else:
			results = self._run_model_predict(model, image, device, class_ids, imgsz=int(YOLOV8_SEG_CONFIG["image_size"]))
			if not results:
				raise RuntimeError("YOLO prediction returned no results")
			result = results[0]

		elapsed_ms = (perf_counter() - start) * 1000.0
		return result, elapsed_ms, tile_count

	def _run_model_predict(
		self,
		model,
		image: np.ndarray,
		device: str,
		class_ids: list[int] | None,
		imgsz: int,
	) -> list[Any]:
		return model.predict(
			source=image,
			device=device,
			imgsz=int(imgsz),
			conf=float(YOLOV8_SEG_CONFIG["confidence_threshold"]),
			iou=float(YOLOV8_SEG_CONFIG["iou_threshold"]),
			max_det=int(YOLOV8_SEG_CONFIG["max_det"]),
			classes=class_ids,
			verbose=False,
		)

	def _should_use_tile_inference(self, image: np.ndarray) -> bool:
		mode = str(YOLOV8_SEG_CONFIG.get("tile_inference_mode", "auto")).strip().lower()
		if mode in {"off", "false", "0"}:
			return False
		if mode in {"always", "on", "true", "1"}:
			return True
		min_side = int(YOLOV8_SEG_CONFIG.get("tile_min_image_side", 1400))
		return max(image.shape[:2]) >= max(1, min_side)

	def _predict_tiled(self, model, image: np.ndarray, device: str, class_ids: list[int] | None) -> tuple[Any, int]:
		height, width = image.shape[:2]
		tile_size = max(128, int(YOLOV8_SEG_CONFIG.get("tile_size", 768)))
		overlap_ratio = float(YOLOV8_SEG_CONFIG.get("tile_overlap_ratio", 0.2))
		overlap_px = max(0, int(tile_size * overlap_ratio))
		stride = max(32, tile_size - overlap_px)
		tile_predict_size = max(128, int(YOLOV8_SEG_CONFIG.get("tile_predict_image_size", tile_size)))
		max_tiles = max(1, int(YOLOV8_SEG_CONFIG.get("tile_max_tiles", 144)))
		min_mask_area = max(0, int(YOLOV8_SEG_CONFIG.get("tile_min_mask_area_px", YOLOV8_SEG_CONFIG["min_mask_area_px"])))

		y_starts = self._build_tile_starts(height, tile_size, stride)
		x_starts = self._build_tile_starts(width, tile_size, stride)

		all_boxes: list[np.ndarray] = []
		all_confidences: list[float] = []
		all_class_ids: list[float] = []
		all_masks: list[np.ndarray] = []
		tile_counter = 0

		for y0 in y_starts:
			for x0 in x_starts:
				tile_counter += 1
				if tile_counter > max_tiles:
					raise RuntimeError(f"Tile inference aborted: tile limit exceeded ({max_tiles})")

				y1 = min(height, y0 + tile_size)
				x1 = min(width, x0 + tile_size)
				tile_image = image[y0:y1, x0:x1]
				if tile_image.size == 0:
					continue

				tile_results = self._run_model_predict(
					model,
					tile_image,
					device,
					class_ids,
					imgsz=min(tile_predict_size, max(tile_image.shape[:2])),
				)
				if not tile_results:
					continue

				tile_result = tile_results[0]
				tile_boxes, tile_confidences, tile_class_ids = self._extract_boxes_data(tile_result)
				tile_masks = self._extract_mask_array(getattr(tile_result, "masks", None), tile_image.shape[:2])

				if tile_boxes.shape[0] == 0 or tile_masks.shape[0] == 0:
					continue

				instance_count = min(tile_boxes.shape[0], tile_masks.shape[0])
				for index in range(instance_count):
					mask = tile_masks[index]
					if int(np.count_nonzero(mask)) < min_mask_area:
						continue

					box = tile_boxes[index].astype(np.float32, copy=True)
					box[0] += float(x0)
					box[2] += float(x0)
					box[1] += float(y0)
					box[3] += float(y0)
					box[0] = float(np.clip(box[0], 0, width - 1))
					box[2] = float(np.clip(box[2], 0, width - 1))
					box[1] = float(np.clip(box[1], 0, height - 1))
					box[3] = float(np.clip(box[3], 0, height - 1))

					full_mask = np.zeros((height, width), dtype=bool)
					full_mask[y0:y1, x0:x1] = mask[: y1 - y0, : x1 - x0]

					all_boxes.append(box)
					all_confidences.append(float(tile_confidences[index]))
					all_class_ids.append(float(tile_class_ids[index]))
					all_masks.append(full_mask)

		if not all_boxes:
			empty_boxes = np.empty((0, 4), dtype=np.float32)
			empty_scores = np.empty((0,), dtype=np.float32)
			empty_classes = np.empty((0,), dtype=np.float32)
			empty_masks = np.zeros((0, height, width), dtype=bool)
			return _SimpleResult(_SimpleBoxes(empty_boxes, empty_scores, empty_classes), _SimpleMasks(empty_masks)), tile_counter

		boxes = np.vstack(all_boxes).astype(np.float32)
		confidences = np.asarray(all_confidences, dtype=np.float32)
		class_ids_array = np.asarray(all_class_ids, dtype=np.float32)
		masks = np.stack(all_masks).astype(bool)

		keep = self._nms_indices(boxes, confidences, float(YOLOV8_SEG_CONFIG.get("tile_nms_iou", 0.5)))
		if keep.size:
			boxes = boxes[keep]
			confidences = confidences[keep]
			class_ids_array = class_ids_array[keep]
			masks = masks[keep]

		max_det = max(1, int(YOLOV8_SEG_CONFIG["max_det"]))
		if boxes.shape[0] > max_det:
			order = np.argsort(confidences)[::-1][:max_det]
			boxes = boxes[order]
			confidences = confidences[order]
			class_ids_array = class_ids_array[order]
			masks = masks[order]

		return _SimpleResult(_SimpleBoxes(boxes, confidences, class_ids_array), _SimpleMasks(masks)), tile_counter

	def _build_tile_starts(self, size: int, tile_size: int, stride: int) -> list[int]:
		if size <= tile_size:
			return [0]

		starts = list(range(0, max(1, size - tile_size + 1), max(1, stride)))
		last = max(0, size - tile_size)
		if not starts or starts[-1] != last:
			starts.append(last)
		return starts

	def _extract_boxes_data(self, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		boxes = getattr(result, "boxes", None)
		if boxes is None:
			return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

		box_array = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
		confidences = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
		class_ids = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
		return box_array.astype(np.float32), confidences.astype(np.float32), class_ids.astype(np.float32)

	def _nms_indices(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
		if boxes.size == 0:
			return np.empty((0,), dtype=np.int32)

		order = np.argsort(scores)[::-1]
		keep: list[int] = []

		while order.size > 0:
			index = int(order[0])
			keep.append(index)
			if order.size == 1:
				break

			rest = order[1:]
			ious = self._pairwise_iou(boxes[index], boxes[rest])
			order = rest[ious < iou_threshold]

		return np.asarray(keep, dtype=np.int32)

	def _pairwise_iou(self, box: np.ndarray, others: np.ndarray) -> np.ndarray:
		if others.size == 0:
			return np.empty((0,), dtype=np.float32)

		x1 = np.maximum(box[0], others[:, 0])
		y1 = np.maximum(box[1], others[:, 1])
		x2 = np.minimum(box[2], others[:, 2])
		y2 = np.minimum(box[3], others[:, 3])

		inter_w = np.maximum(0.0, x2 - x1)
		inter_h = np.maximum(0.0, y2 - y1)
		intersection = inter_w * inter_h

		box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
		other_areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
		union = box_area + other_areas - intersection
		union = np.maximum(union, 1e-9)
		return (intersection / union).astype(np.float32)

	def _summarize_prediction(self, image: np.ndarray, result, inference_ms: float, side: str) -> dict[str, Any]:
		boxes = getattr(result, "boxes", None)
		masks = getattr(result, "masks", None)
		mask_array = self._extract_mask_array(masks, image.shape[:2])
		mask_canvas = self._build_mask_canvas(image.shape, mask_array)
		overlay = self._build_overlay(result, image, mask_canvas)

		if boxes is None:
			box_array = np.empty((0, 4), dtype=np.float32)
			confidences = np.empty((0,), dtype=np.float32)
			class_ids = np.empty((0,), dtype=np.int32)
		else:
			box_array = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
			confidences = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
			class_ids = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)

		mask_area_px = int(np.count_nonzero(np.any(mask_array, axis=0))) if mask_array.size else 0
		bbox_area_px = int(self._sum_box_areas(box_array))
		mean_confidence = float(confidences.mean()) if confidences.size else 0.0

		return {
			"side": side,
			"inference_ms": float(inference_ms),
			"instance_count": int(len(class_ids)),
			"mask_area_px": mask_area_px,
			"bbox_area_px": bbox_area_px,
			"mean_confidence": mean_confidence,
			"overlay": overlay,
			"mask_canvas": mask_canvas,
		}

	def _extract_mask_array(self, masks, image_shape: tuple[int, int]) -> np.ndarray:
		if masks is None or getattr(masks, "data", None) is None:
			return np.zeros((0, image_shape[0], image_shape[1]), dtype=bool)

		target_h, target_w = image_shape
		mask_data = masks.data
		if hasattr(mask_data, "detach"):
			mask_data = mask_data.detach()
		if hasattr(mask_data, "cpu"):
			mask_data = mask_data.cpu()
		mask_array = mask_data.numpy() if hasattr(mask_data, "numpy") else np.asarray(mask_data)
		if mask_array.ndim == 2:
			mask_array = mask_array[None, ...]
		if mask_array.ndim != 3:
			return np.zeros((0, target_h, target_w), dtype=bool)

		if mask_array.shape[1] != target_h or mask_array.shape[2] != target_w:
			resized_masks = np.zeros((mask_array.shape[0], target_h, target_w), dtype=bool)
			for index, mask in enumerate(mask_array):
				resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
				resized_masks[index] = resized > 0.5
			return resized_masks

		return (mask_array > 0.5).astype(bool)

	def _sum_box_areas(self, box_array: np.ndarray) -> float:
		if box_array.size == 0:
			return 0.0
		widths = np.maximum(0.0, box_array[:, 2] - box_array[:, 0])
		heights = np.maximum(0.0, box_array[:, 3] - box_array[:, 1])
		return float(np.sum(widths * heights))

	def _build_mask_canvas(self, image_shape: tuple[int, int, int], mask_array: np.ndarray) -> np.ndarray:
		height, width = image_shape[:2]
		canvas = np.zeros((height, width, 3), dtype=np.uint8)
		if mask_array.size == 0:
			return canvas

		for index, mask in enumerate(mask_array):
			color = self._palette_color(index)
			canvas[mask] = color

		return canvas

	def _palette_color(self, index: int) -> np.ndarray:
		return np.array(
			[
				(index * 53) % 255,
				(index * 97 + 80) % 255,
				(index * 193 + 40) % 255,
			],
			dtype=np.uint8,
		)

	def _build_overlay(self, result, image: np.ndarray, mask_canvas: np.ndarray) -> np.ndarray:
		try:
			annotated = result.plot()
			if annotated is not None:
				return annotated
		except Exception:
			pass
		base = image.copy()
		if mask_canvas.size == 0:
			return base
		return cv2.addWeighted(base, 1.0 - float(YOLOV8_SEG_CONFIG["mask_alpha"]), mask_canvas, float(YOLOV8_SEG_CONFIG["mask_alpha"]), 0)

	def _build_diff_preview(self, before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
		if before_img.shape[:2] != after_img.shape[:2]:
			before_img = cv2.resize(before_img, (after_img.shape[1], after_img.shape[0]), interpolation=cv2.INTER_AREA)
		diff = cv2.absdiff(before_img, after_img)
		diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		heatmap = cv2.applyColorMap(diff_gray, int(YOLOV8_SEG_CONFIG["colormap"]))
		return cv2.addWeighted(after_img, 0.55, heatmap, 0.45, 0)

	def _compose_preview(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		before_overlay: np.ndarray,
		after_overlay: np.ndarray,
		diff_preview: np.ndarray,
		has_any_detections: bool,
	) -> np.ndarray:
		cell_w = max(1, int(YOLOV8_SEG_CONFIG["preview_max_width"]) // 2)
		cell_h = max(1, int(YOLOV8_SEG_CONFIG["preview_max_height"]) // 2)
		fallback_title = "Diff fallback" if not has_any_detections else "Overlay note"
		fallback_panel = self._annotate_panel(self._fit_to_box(diff_preview, cell_w, cell_h), fallback_title)

		panels = [
			self._annotate_panel(self._fit_to_box(before_img, cell_w, cell_h), "Before"),
			self._annotate_panel(self._fit_to_box(after_img, cell_w, cell_h), "After"),
			self._annotate_panel(self._fit_to_box(before_overlay, cell_w, cell_h), "Before overlay"),
			self._annotate_panel(self._fit_to_box(after_overlay, cell_w, cell_h), "After overlay"),
		]
		if not has_any_detections:
			panels[2] = self._annotate_panel(self._fit_to_box(fallback_panel, cell_w, cell_h), "No masks detected")
			panels[3] = self._annotate_panel(self._fit_to_box(fallback_panel, cell_w, cell_h), "No masks detected")
		panels = [self._pad_to_size(panel, cell_w, cell_h) for panel in panels]
		top = np.hstack([panels[0], panels[1]])
		bottom = np.hstack([panels[2], panels[3]])
		grid = np.vstack([top, bottom])
		return self._resize_if_too_large(
			grid,
			max_width=int(YOLOV8_SEG_CONFIG["preview_max_width"]),
			max_height=int(YOLOV8_SEG_CONFIG["preview_max_height"]),
		)

	def _fit_to_box(self, image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
		height, width = image.shape[:2]
		scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
		if scale >= 1.0:
			return image
		new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
		return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

	def _pad_to_size(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
		height, width = image.shape[:2]
		if height == target_height and width == target_width:
			return image
		canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)
		x_offset = max(0, (target_width - width) // 2)
		y_offset = max(0, (target_height - height) // 2)
		canvas[y_offset : y_offset + height, x_offset : x_offset + width] = image[: target_height - y_offset, : target_width - x_offset]
		return canvas

	def _annotate_panel(self, image: np.ndarray, title: str) -> np.ndarray:
		panel = image.copy()
		panel_height = int(YOLOV8_SEG_CONFIG["panel_title_height"])
		cv2.rectangle(panel, (0, 0), (panel.shape[1], panel_height), (0, 0, 0), -1)
		cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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
		safe = value.strip().replace("/", "_").replace("\\", "_").replace(":", "-")
		return safe or "pair"

	def _save_artifacts(
		self,
		output_dir: Path,
		before_overlay: np.ndarray,
		after_overlay: np.ndarray,
		before_mask_canvas: np.ndarray,
		after_mask_canvas: np.ndarray,
		preview: np.ndarray,
	) -> dict[str, Path]:
		paths = {
			"before_overlay": output_dir / "before_overlay.png",
			"after_overlay": output_dir / "after_overlay.png",
			"before_masks": output_dir / "before_masks.png",
			"after_masks": output_dir / "after_masks.png",
			"preview": output_dir / "preview.png",
		}

		self._write_image(paths["before_overlay"], before_overlay)
		self._write_image(paths["after_overlay"], after_overlay)
		self._write_image(paths["before_masks"], before_mask_canvas)
		self._write_image(paths["after_masks"], after_mask_canvas)
		self._write_image(paths["preview"], preview)

		return paths

	def _write_image(self, path: Path, image: np.ndarray) -> None:
		ok = cv2.imwrite(str(path), image)
		if not ok:
			raise RuntimeError(f"Failed to write image: {path}")