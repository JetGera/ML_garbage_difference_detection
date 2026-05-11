from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import re
import unicodedata


def prepare_output_dir(root: Path, pair_name: str, method_label: str) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%d.%m.%Y %H-%M")
	run_dir_name = f"{pair_name}__{sanitize_folder_component(method_label)}__{timestamp}__{uuid4().hex[:6]}"
	output_dir = root / run_dir_name
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir


def pair_folder_name(before: Path, after: Path) -> str:
	before_parent = before.parent.name.strip() or "pair"
	after_parent = after.parent.name.strip() or "pair"
	if before.parent == after.parent:
		return sanitize_folder_component(before_parent)
	if before_parent == after_parent:
		return sanitize_folder_component(before_parent)
	return sanitize_folder_component(f"{before_parent}_and_{after_parent}")


def prepare_pair_output_dir(root: Path, before: Path, after: Path, method_label: str) -> Path:
	return prepare_output_dir(root, pair_folder_name(before, after), method_label)


def write_image(path: Path, image: np.ndarray) -> None:
	ok = cv2.imwrite(str(path), image)
	if not ok:
		raise RuntimeError(f"Failed to write image: {path}")


def save_artifact_images(output_dir: Path, images: dict[str, np.ndarray]) -> dict[str, Path]:
	paths: dict[str, Path] = {}
	for name, image in images.items():
		path = output_dir / f"{name}.png"
		write_image(path, image)
		paths[name] = path
	return paths


def sanitize_folder_component(value: str) -> str:
	normalized = unicodedata.normalize("NFKD", value)
	ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
	ascii_text = ascii_text.strip().replace("/", "_").replace("\\", "_").replace(":", "-")
	ascii_text = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text)
	ascii_text = re.sub(r"_+", "_", ascii_text).strip("._-")
	return ascii_text or "pair"
