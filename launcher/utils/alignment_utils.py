from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def build_validity_mask(shape: tuple[int, int]) -> np.ndarray:
	h, w = shape[:2]
	return np.full((h, w), 255, dtype=np.uint8)


def compute_overlap_mask(valid_before: np.ndarray, valid_after: np.ndarray, erode_kernel: tuple[int, int] = (5, 5)) -> np.ndarray:
	overlap_mask = cv2.bitwise_and(valid_before, valid_after)
	if erode_kernel[0] > 1 or erode_kernel[1] > 1:
		kernel = np.ones(erode_kernel, dtype=np.uint8)
		overlap_mask = cv2.erode(overlap_mask, kernel, iterations=1)
	return overlap_mask


def warp_validity_mask(
	shape: tuple[int, int],
	warp_matrix: np.ndarray,
	output_size: tuple[int, int],
	motion_type: str,
	border_value: int = 0,
) -> np.ndarray:
	valid = build_validity_mask(shape)
	width, height = output_size
	if motion_type == "homography":
		warped = cv2.warpPerspective(
			valid,
			warp_matrix,
			(width, height),
			flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=border_value,
		)
	else:
		warped = cv2.warpAffine(
			valid,
			warp_matrix,
			(width, height),
			flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=border_value,
		)
	return warped


def try_ecc_alignment(
	before_img: np.ndarray,
	after_img: np.ndarray,
	*,
	motion_type: int = cv2.MOTION_AFFINE,
	max_iterations: int = 80,
	eps: float = 1e-5,
	gauss_filter_size: int = 5,
	min_residual: float = 0.0,
	downscale_max_side: int | None = None,
	residual_fn: Any | None = None,
	erode_kernel: tuple[int, int] = (5, 5),
	allow_crop: bool = True,
) -> dict[str, Any] | None:
	before_h, before_w = before_img.shape[:2]
	after_h, after_w = after_img.shape[:2]
	if (before_h, before_w) != (after_h, after_w):
		after_img = cv2.resize(after_img, (before_w, before_h), interpolation=cv2.INTER_LINEAR)

	work_before = before_img
	work_after = after_img
	work_w, work_h = before_w, before_h
	if downscale_max_side is not None:
		scale = min(float(downscale_max_side) / max(before_h, before_w, 1), 1.0)
		if scale < 1.0:
			work_w = max(64, int(before_w * scale))
			work_h = max(64, int(before_h * scale))
			work_size = (work_w, work_h)
			work_before = cv2.resize(before_img, work_size, interpolation=cv2.INTER_AREA)
			work_after = cv2.resize(after_img, work_size, interpolation=cv2.INTER_AREA)

	if motion_type == cv2.MOTION_HOMOGRAPHY:
		warp_matrix = np.eye(3, 3, dtype=np.float32)
		motion_name = "homography"
	else:
		warp_matrix = np.eye(2, 3, dtype=np.float32)
		motion_name = "affine" if motion_type == cv2.MOTION_AFFINE else "euclidean"

	before_gray = cv2.cvtColor(work_before, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
	after_gray = cv2.cvtColor(work_after, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
	criteria = (
		cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
		int(max_iterations),
		float(eps),
	)

	try:
		ecc_score, warp_matrix = cv2.findTransformECC(
			templateImage=before_gray,
			inputImage=after_gray,
			warpMatrix=warp_matrix,
			motionType=motion_type,
			criteria=criteria,
			inputMask=None,
			gaussFiltSize=int(gauss_filter_size),
		)
	except cv2.error:
		return None

	if motion_type == cv2.MOTION_HOMOGRAPHY:
		aligned_after = cv2.warpPerspective(
			after_img,
			warp_matrix,
			(before_w, before_h),
			flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_REFLECT,
		)
	else:
		aligned_after = cv2.warpAffine(
			after_img,
			warp_matrix,
			(before_w, before_h),
			flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_REFLECT,
		)

	valid_after = warp_validity_mask(
		(before_h, before_w),
		warp_matrix,
		(before_w, before_h),
		motion_name,
	)
	overlap_mask = compute_overlap_mask(build_validity_mask((before_h, before_w)), valid_after, erode_kernel=erode_kernel)
	overlap_ratio = float(np.count_nonzero(overlap_mask) / max(overlap_mask.size, 1))
	if overlap_ratio <= 0.05:
		return None

	if residual_fn is None:
		before_gray_full = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
		after_gray_full = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY)
		residual = float(np.mean(cv2.absdiff(before_gray_full, after_gray_full).astype(np.float32) / 255.0))
	else:
		residual = float(residual_fn(before_img, aligned_after, overlap_mask))

	if residual < min_residual:
		return None

	quality = float((float(ecc_score) + 1.0) * overlap_ratio * (1.0 - residual))
	result: dict[str, Any] = {
		"aligned_after": aligned_after,
		"overlap_mask": overlap_mask,
		"quality": quality,
		"ecc_score": float(ecc_score),
		"residual": residual,
		"warp_matrix": warp_matrix,
		"motion_name": motion_name,
	}
	if allow_crop:
		union_mask = cv2.bitwise_or(build_validity_mask((before_h, before_w)), valid_after)
		non_zero_points = cv2.findNonZero(union_mask)
		if non_zero_points is not None:
			x, y, w, h = cv2.boundingRect(non_zero_points)
			result["aligned_after"] = aligned_after[y : y + h, x : x + w]
			result["overlap_mask"] = overlap_mask[y : y + h, x : x + w]
	return result
