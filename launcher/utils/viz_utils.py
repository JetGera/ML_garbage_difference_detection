from __future__ import annotations

import cv2
import numpy as np


def annotate_panel(image: np.ndarray, title: str, bar_height: int = 44) -> np.ndarray:
	panel = image.copy()
	cv2.rectangle(panel, (0, 0), (panel.shape[1], bar_height), (0, 0, 0), -1)
	cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
	return panel


def resize_if_too_large(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
	height, width = image.shape[:2]
	scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
	if scale >= 1.0:
		return image
	new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
	return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def blend_with_alpha(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
	return cv2.addWeighted(base, 1.0, overlay, float(alpha), 0)
