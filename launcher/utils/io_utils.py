from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def prepare_output_dir(root: Path, pair_name: str, method_label: str) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%d.%m.%Y %H-%M")
	run_dir_name = f"{pair_name}__{sanitize_folder_component(method_label)}__{timestamp}__{uuid4().hex[:6]}"
	output_dir = root / run_dir_name
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir


def write_image(path: Path, image: np.ndarray) -> None:
	ok = cv2.imwrite(str(path), image)
	if not ok:
		raise RuntimeError(f"Failed to write image: {path}")


def sanitize_folder_component(value: str) -> str:
	safe = value.strip().replace("/", "_").replace("\\", "_").replace(":", "-")
	return safe or "pair"
