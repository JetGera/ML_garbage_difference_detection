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


def pad_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
	height, width = image.shape[:2]
	if height == target_height and width == target_width:
		return image
	canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)
	x_offset = max(0, (target_width - width) // 2)
	y_offset = max(0, (target_height - height) // 2)
	canvas[y_offset : y_offset + height, x_offset : x_offset + width] = image[: target_height - y_offset, : target_width - x_offset]
	return canvas


def blend_with_alpha(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
	return cv2.addWeighted(base, 1.0, overlay, float(alpha), 0)


def compose_panel_grid(
	panels: list[np.ndarray],
	cell_width: int,
	cell_height: int,
	max_width: int,
	max_height: int,
	columns: int = 2,
) -> np.ndarray:
	if not panels:
		raise ValueError("At least one panel is required")
	if columns <= 0:
		raise ValueError("columns must be positive")

	prepared_panels = [pad_to_size(resize_if_too_large(panel, cell_width, cell_height), cell_width, cell_height) for panel in panels]
	blank_panel = np.zeros_like(prepared_panels[0])
	rows: list[np.ndarray] = []
	for index in range(0, len(prepared_panels), columns):
		row_panels = prepared_panels[index : index + columns]
		if len(row_panels) < columns:
			row_panels = row_panels + [blank_panel.copy() for _ in range(columns - len(row_panels))]
		rows.append(np.hstack(row_panels))

	grid = rows[0] if len(rows) == 1 else np.vstack(rows)
	return resize_if_too_large(grid, max_width, max_height)
