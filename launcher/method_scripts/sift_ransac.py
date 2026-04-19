from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

# Method 01 (SIFT + RANSAC) config block.
# `nfeatures`: maximum number of SIFT keypoints to keep.
# `contrast_threshold`: lower values make SIFT more sensitive to weak features.
# `clahe_clip_limit`: contrast enhancement strength before feature detection.
# `clahe_tile_grid`: CLAHE grid size used to normalize local contrast.
SIFT_CONFIG = {
	"nfeatures": 75000,
	"contrast_threshold": 0.02,
	"clahe_clip_limit": 2.0,
	"clahe_tile_grid": (8, 8),
}

# `ratio_test`: Lower ratio test threshold for keeping descriptor matches.
# `ransac_reproj_threshold`: RANSAC pixel tolerance for homography / affine fitting.
# `min_matches_for_homography`: minimum good matches needed before estimating homography.
# `min_overlap_ratio`: minimum overlap area needed to accept aligned canvas.
# `max_canvas_side`: hard limit for the shared canvas size.
# `max_perspective_shear`: maximum allowed perspective distortion in homography.
# `min_homography_area_ratio`: minimum warped area vs. source area accepted as sane.
# `max_homography_area_ratio`: maximum warped area vs. source area accepted as sane.
ALIGNMENT_CONFIG = {
	"ratio_test": 0.75,
	"ransac_reproj_threshold": 4.0,
	"min_matches_for_homography": 8,
	"min_overlap_ratio": 0.2,
	"max_canvas_side": 8000,
	"max_perspective_shear": 0.0025,
	"min_homography_area_ratio": 0.35,
	"max_homography_area_ratio": 3.0,
}

# `gaussian_kernel`: blur kernel for smoothing the gray difference map.
# `morph_kernel`: kernel used for cleanup after thresholding.
# `morph_open_iterations`: number of opening passes to remove noise.
# `morph_close_iterations`: number of closing passes to fill small gaps.
# `min_component_area_px`: minimum absolute connected-component area to keep.
# `min_component_area_ratio`: minimum component area relative to overlap area.
# `colormap`: OpenCV colormap used to render the heatmap preview.
# `min_component_width_px` / `min_component_height_px`: minimum bounding-box
# dimensions (in pixels) to keep a detected component. Use these to ignore
# long thin objects like grass/branches by setting e.g. width>=X or height>=Y.
CHANGE_MAP_CONFIG = {
	"gaussian_kernel": (5, 5),
	"morph_kernel": (5, 5),
	"morph_open_iterations": 1,
	"morph_close_iterations": 1,
	"min_component_area_px": 32,
	"min_component_area_ratio": 0.0005,
	"min_component_width_px": 0,
	"min_component_height_px": 0,
	"colormap": cv2.COLORMAP_INFERNO,
}

# `max_width`: maximum width of the composed preview image.
# `max_height`: maximum height of the composed preview image.
PREVIEW_CONFIG = {
	# "max_width": 1800,
	# "max_height": 1200,
	"max_width": 18000,
	"max_height": 12000,
}

try:
	from ..core import AnalysisResult
	from ..methods import get_method_spec
except ImportError:
	from core import AnalysisResult
	from methods import get_method_spec


class SiftRansacRunner:
	RATIO_TEST = ALIGNMENT_CONFIG["ratio_test"]
	RANSAC_REPROJ_THRESHOLD = ALIGNMENT_CONFIG["ransac_reproj_threshold"]
	MIN_MATCHES_FOR_HOMOGRAPHY = ALIGNMENT_CONFIG["min_matches_for_homography"]
	MIN_OVERLAP_RATIO = ALIGNMENT_CONFIG["min_overlap_ratio"]
	MAX_CANVAS_SIDE = ALIGNMENT_CONFIG["max_canvas_side"]
	MAX_PERSPECTIVE_SHEAR = ALIGNMENT_CONFIG["max_perspective_shear"]
	MIN_HOMOGRAPHY_AREA_RATIO = ALIGNMENT_CONFIG["min_homography_area_ratio"]
	MAX_HOMOGRAPHY_AREA_RATIO = ALIGNMENT_CONFIG["max_homography_area_ratio"]

	def __init__(self, method_id: str):
		self.method_id = method_id
		self.spec = get_method_spec(method_id)

	@property
	def label(self) -> str:
		return self.spec.label

	def analyze(self, before_path: str | Path, after_path: str | Path) -> AnalysisResult:
		before = Path(before_path)
		after = Path(after_path)

		before_img = self._read_color_image(before)
		after_img = self._read_color_image(after)

		aligned_before, aligned_after, overlap_mask, align_info, match_vis = self._align_after_to_before(before_img, after_img)
		diff_gray, change_mask, otsu_threshold = self._build_change_map(aligned_before, aligned_after, overlap_mask)

		preview = self._compose_preview(
			before_img=aligned_before,
			aligned_after=aligned_after,
			diff_gray=diff_gray,
			change_mask=change_mask,
			alignment_mode=align_info["alignment_mode"],
		)

		output_dir = self._prepare_output_dir(before, after)
		artifacts = self._save_artifacts(
			output_dir=output_dir,
			aligned_before=aligned_before,
			aligned_after=aligned_after,
			diff_gray=diff_gray,
			change_mask=change_mask,
			overlap_mask=overlap_mask,
			preview=preview,
			match_vis=match_vis,
		)

		change_pixels = int(np.count_nonzero(change_mask))
		overlap_pixels = int(np.count_nonzero(overlap_mask))
		canvas_pixels = int(overlap_mask.size)
		change_ratio = round(change_pixels / max(overlap_pixels, 1), 6)
		good_matches = int(align_info["good_matches"])
		inliers = int(align_info["inliers"])

		metrics = {
			"alignment_mode": align_info["alignment_mode"],
			"keypoints_before": int(align_info["keypoints_before"]),
			"keypoints_after": int(align_info["keypoints_after"]),
			"raw_matches": int(align_info["raw_matches"]),
			"good_matches": good_matches,
			"inliers": inliers,
			"inlier_ratio": round(inliers / max(good_matches, 1), 4),
			"change_pixels": change_pixels,
			"overlap_pixels": overlap_pixels,
			"canvas_pixels": canvas_pixels,
			"overlap_ratio": round(overlap_pixels / max(canvas_pixels, 1), 4),
			"change_ratio": change_ratio,
			"otsu_threshold": round(float(otsu_threshold), 2),
		}

		summary = "SIFT matching + RANSAC alignment on a shared canvas, then difference map in overlap area only."
		preview_text = (
			f"Alignment: {metrics['alignment_mode']}\n"
			f"Good matches: {good_matches}, inliers: {inliers}\n"
			f"Overlap ratio: {metrics['overlap_ratio']:.4f}, change ratio: {change_ratio:.4f}"
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

	def _align_after_to_before(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]:
		sift = self._create_sift()

		before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
		after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
		before_feat = self._enhance_for_features(before_gray)
		after_feat = self._enhance_for_features(after_gray)

		kp_before, desc_before = sift.detectAndCompute(before_feat, None)
		kp_after, desc_after = sift.detectAndCompute(after_feat, None)

		info: dict[str, Any] = {
			"alignment_mode": "identity_fallback",
			"keypoints_before": len(kp_before),
			"keypoints_after": len(kp_after),
			"raw_matches": 0,
			"good_matches": 0,
			"inliers": 0,
		}

		if desc_before is None or desc_after is None:
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, None

		matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)
		info["raw_matches"] = len(raw_matches)

		good_matches: list[cv2.DMatch] = []
		for pair in raw_matches:
			if len(pair) < 2:
				continue
			first, second = pair
			if first.distance < self.RATIO_TEST * second.distance:
				good_matches.append(first)

		info["good_matches"] = len(good_matches)
		if len(good_matches) < self.MIN_MATCHES_FOR_HOMOGRAPHY:
			match_vis = self._draw_matches(before_img, after_img, kp_before, kp_after, good_matches)
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		src_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

		homography, inlier_mask = cv2.findHomography(
			src_pts,
			dst_pts,
			method=cv2.RANSAC,
			ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
		)

		match_vis = self._draw_matches(before_img, after_img, kp_before, kp_after, good_matches, inlier_mask)
		affine_result = self._align_with_affine(before_img, after_img, good_matches, kp_before, kp_after)
		if affine_result is not None:
			aligned_before, aligned_after, overlap_mask, affine_info = affine_result
			info["alignment_mode"] = affine_info["alignment_mode"]
			info["inliers"] = affine_info["inliers"]
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		if homography is None:
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		if not self._is_homography_sane(before_img, after_img, homography):
			affine_result = self._align_with_affine(before_img, after_img, good_matches, kp_before, kp_after)
			if affine_result is not None:
				aligned_before, aligned_after, overlap_mask, affine_info = affine_result
				info["alignment_mode"] = affine_info["alignment_mode"]
				info["inliers"] = affine_info["inliers"]
				return aligned_before, aligned_after, overlap_mask, info, match_vis
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		try:
			aligned_before, aligned_after, overlap_mask = self._warp_to_shared_canvas(before_img, after_img, homography)
		except ValueError:
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		overlap_pixels = int(np.count_nonzero(overlap_mask))
		min_overlap = int(before_img.shape[0] * before_img.shape[1] * self.MIN_OVERLAP_RATIO)
		if overlap_pixels < max(1, min_overlap):
			info["alignment_mode"] = "low_overlap_fallback"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
		info["alignment_mode"] = "homography_ransac"
		info["inliers"] = inliers
		return aligned_before, aligned_after, overlap_mask, info, match_vis

	def _is_homography_sane(self, before_img: np.ndarray, after_img: np.ndarray, homography: np.ndarray) -> bool:
		if homography.shape != (3, 3):
			return False
		if not np.isfinite(homography).all():
			return False

		projective_terms = float(abs(homography[2, 0]) + abs(homography[2, 1]))
		if projective_terms > self.MAX_PERSPECTIVE_SHEAR:
			return False

		before_h, before_w = before_img.shape[:2]
		after_h, after_w = after_img.shape[:2]
		before_corners = np.float32([[0, 0], [before_w, 0], [before_w, before_h], [0, before_h]]).reshape(-1, 1, 2)
		after_corners = np.float32([[0, 0], [after_w, 0], [after_w, after_h], [0, after_h]]).reshape(-1, 1, 2)
		warped_after_corners = cv2.perspectiveTransform(after_corners, homography)
		before_area = float(before_w * before_h)
		warped_area = float(abs(cv2.contourArea(warped_after_corners.astype(np.float32))))
		if warped_area <= 1.0:
			return False

		area_ratio = warped_area / max(before_area, 1.0)
		if area_ratio < self.MIN_HOMOGRAPHY_AREA_RATIO or area_ratio > self.MAX_HOMOGRAPHY_AREA_RATIO:
			return False

		return True

	def _align_with_affine(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		good_matches: list[cv2.DMatch],
		kp_before,
		kp_after,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
		if len(good_matches) < 3:
			return None

		src_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches])
		dst_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches])
		affine, inlier_mask = cv2.estimateAffinePartial2D(
			src_pts,
			dst_pts,
			method=cv2.RANSAC,
			ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
		)
		if affine is None:
			return None

		before_h, before_w = before_img.shape[:2]
		after_h, after_w = after_img.shape[:2]
		after_corners = np.float32([[0, 0], [after_w, 0], [after_w, after_h], [0, after_h]]).reshape(-1, 1, 2)
		affine_3x3 = np.array(
			[
				[affine[0, 0], affine[0, 1], affine[0, 2]],
				[affine[1, 0], affine[1, 1], affine[1, 2]],
				[0.0, 0.0, 1.0],
			],
			dtype=np.float64,
		)
		warped_after_corners = cv2.perspectiveTransform(after_corners, affine_3x3)
		before_corners = np.float32([[0, 0], [before_w, 0], [before_w, before_h], [0, before_h]]).reshape(-1, 1, 2)
		all_corners = np.vstack([before_corners, warped_after_corners]).reshape(-1, 2)
		min_x = int(np.floor(all_corners[:, 0].min()))
		min_y = int(np.floor(all_corners[:, 1].min()))
		max_x = int(np.ceil(all_corners[:, 0].max()))
		max_y = int(np.ceil(all_corners[:, 1].max()))
		canvas_w = max(1, max_x - min_x)
		canvas_h = max(1, max_y - min_y)
		if canvas_w > self.MAX_CANVAS_SIDE or canvas_h > self.MAX_CANVAS_SIDE:
			return None

		translation = np.array(
			[
				[1.0, 0.0, -float(min_x)],
				[0.0, 1.0, -float(min_y)],
				[0.0, 0.0, 1.0],
			],
			dtype=np.float64,
		)
		affine_full = translation @ affine_3x3
		aligned_before = cv2.warpPerspective(
			before_img,
			translation,
			(canvas_w, canvas_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)
		aligned_after = cv2.warpPerspective(
			after_img,
			affine_full,
			(canvas_w, canvas_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)

		before_valid = cv2.warpPerspective(
			np.full((before_h, before_w), 255, dtype=np.uint8),
			translation,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		after_valid = cv2.warpPerspective(
			np.full((after_h, after_w), 255, dtype=np.uint8),
			affine_full,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		overlap_mask = cv2.bitwise_and(before_valid, after_valid)
		union_mask = cv2.bitwise_or(before_valid, after_valid)
		non_zero_points = cv2.findNonZero(union_mask)
		if non_zero_points is not None:
			x, y, w, h = cv2.boundingRect(non_zero_points)
			aligned_before = aligned_before[y : y + h, x : x + w]
			aligned_after = aligned_after[y : y + h, x : x + w]
			overlap_mask = overlap_mask[y : y + h, x : x + w]

		affine_info = {
			"alignment_mode": "affine_ransac",
			"inliers": int(inlier_mask.sum()) if inlier_mask is not None else 0,
		}
		return aligned_before, aligned_after, overlap_mask, affine_info

	def _create_sift(self):
		if hasattr(cv2, "SIFT_create"):
			return cv2.SIFT_create(
				nfeatures=int(SIFT_CONFIG["nfeatures"]),
				contrastThreshold=float(SIFT_CONFIG["contrast_threshold"]),
			)
		raise RuntimeError(
			"OpenCV SIFT is unavailable. Install an OpenCV build with SIFT support (for example opencv-contrib-python)."
		)

	def _enhance_for_features(self, gray: np.ndarray) -> np.ndarray:
		clahe = cv2.createCLAHE(
			clipLimit=float(SIFT_CONFIG["clahe_clip_limit"]),
			tileGridSize=tuple(SIFT_CONFIG["clahe_tile_grid"]),
		)
		return clahe.apply(gray)

	def _fallback_alignment_pair(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		info: dict[str, Any],
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		before_shape = before_img.shape[:2]
		after_shape = after_img.shape[:2]
		if before_shape != after_shape:
			info["alignment_mode"] = "resize_fallback"
			aligned_after = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_LINEAR)
		else:
			info["alignment_mode"] = "identity_fallback"
			aligned_after = after_img.copy()
		overlap_mask = np.full(before_shape, 255, dtype=np.uint8)
		return before_img.copy(), aligned_after, overlap_mask

	def _warp_to_shared_canvas(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		homography: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		before_h, before_w = before_img.shape[:2]
		after_h, after_w = after_img.shape[:2]

		before_corners = np.float32([[0, 0], [before_w, 0], [before_w, before_h], [0, before_h]]).reshape(-1, 1, 2)
		after_corners = np.float32([[0, 0], [after_w, 0], [after_w, after_h], [0, after_h]]).reshape(-1, 1, 2)
		warped_after_corners = cv2.perspectiveTransform(after_corners, homography)

		all_corners = np.vstack([before_corners, warped_after_corners]).reshape(-1, 2)
		min_x = int(np.floor(all_corners[:, 0].min()))
		min_y = int(np.floor(all_corners[:, 1].min()))
		max_x = int(np.ceil(all_corners[:, 0].max()))
		max_y = int(np.ceil(all_corners[:, 1].max()))

		canvas_w = max(1, max_x - min_x)
		canvas_h = max(1, max_y - min_y)
		if canvas_w > self.MAX_CANVAS_SIDE or canvas_h > self.MAX_CANVAS_SIDE:
			raise ValueError("Homography produced an unreasonably large canvas")

		translation = np.array(
			[
				[1.0, 0.0, -float(min_x)],
				[0.0, 1.0, -float(min_y)],
				[0.0, 0.0, 1.0],
			],
			dtype=np.float64,
		)

		before_homography = translation
		after_homography = translation @ homography

		aligned_before = cv2.warpPerspective(
			before_img,
			before_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)
		aligned_after = cv2.warpPerspective(
			after_img,
			after_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)

		before_valid = cv2.warpPerspective(
			np.full((before_h, before_w), 255, dtype=np.uint8),
			before_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		after_valid = cv2.warpPerspective(
			np.full((after_h, after_w), 255, dtype=np.uint8),
			after_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		overlap_mask = cv2.bitwise_and(before_valid, after_valid)

		union_mask = cv2.bitwise_or(before_valid, after_valid)
		non_zero_points = cv2.findNonZero(union_mask)
		if non_zero_points is not None:
			x, y, w, h = cv2.boundingRect(non_zero_points)
			aligned_before = aligned_before[y : y + h, x : x + w]
			aligned_after = aligned_after[y : y + h, x : x + w]
			overlap_mask = overlap_mask[y : y + h, x : x + w]

		return aligned_before, aligned_after, overlap_mask

	def _draw_matches(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		kp_before,
		kp_after,
		good_matches,
		inlier_mask: np.ndarray | None = None,
	) -> np.ndarray | None:
		if not good_matches:
			return None
		matches_mask = None
		if inlier_mask is not None:
			matches_mask = inlier_mask.ravel().astype(int).tolist()
		return cv2.drawMatches(
			before_img,
			kp_before,
			after_img,
			kp_after,
			good_matches,
			None,
			matchesMask=matches_mask,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
		)

	def _build_change_map(
		self,
		before_img: np.ndarray,
		aligned_after: np.ndarray,
		overlap_mask: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, float]:
		diff = cv2.absdiff(before_img, aligned_after)
		diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		diff_gray = cv2.bitwise_and(diff_gray, diff_gray, mask=overlap_mask)
		diff_blur = cv2.GaussianBlur(diff_gray, tuple(CHANGE_MAP_CONFIG["gaussian_kernel"]), 0)

		otsu_threshold, raw_mask = cv2.threshold(
			diff_blur,
			0,
			255,
			cv2.THRESH_BINARY + cv2.THRESH_OTSU,
		)
		raw_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=overlap_mask)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(CHANGE_MAP_CONFIG["morph_kernel"]))
		opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=int(CHANGE_MAP_CONFIG["morph_open_iterations"]))
		closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=int(CHANGE_MAP_CONFIG["morph_close_iterations"]))
		cleaned = self._remove_small_connected_components(closed, overlap_mask)
		cleaned = cv2.bitwise_and(cleaned, cleaned, mask=overlap_mask)
		return diff_gray, cleaned, float(otsu_threshold)

	def _remove_small_connected_components(self, mask: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
		components = cv2.connectedComponentsWithStats(mask, connectivity=8)
		num_labels, labels, stats, _ = components
		valid_pixels = int(np.count_nonzero(overlap_mask))
		min_area = max(
			int(CHANGE_MAP_CONFIG["min_component_area_px"]),
			int(valid_pixels * float(CHANGE_MAP_CONFIG["min_component_area_ratio"])),
		)
		min_width = int(CHANGE_MAP_CONFIG.get("min_component_width_px", 0))
		min_height = int(CHANGE_MAP_CONFIG.get("min_component_height_px", 0))

		filtered = np.zeros_like(mask)
		for label in range(1, num_labels):
			area = int(stats[label, cv2.CC_STAT_AREA])
			width = int(stats[label, cv2.CC_STAT_WIDTH])
			height = int(stats[label, cv2.CC_STAT_HEIGHT])
			# Keep component only if it satisfies area and bounding-box size
			if area >= min_area and width >= min_width and height >= min_height:
				filtered[labels == label] = 255
		return filtered

	def _compose_preview(
		self,
		before_img: np.ndarray,
		aligned_after: np.ndarray,
		diff_gray: np.ndarray,
		change_mask: np.ndarray,
		alignment_mode: str,
	) -> np.ndarray:
		diff_heat = cv2.applyColorMap(diff_gray, int(CHANGE_MAP_CONFIG["colormap"]))
		change_overlay = self._make_change_overlay(before_img, aligned_after, change_mask)

		panel_before = self._annotate_panel(before_img, "Before (aligned frame)")
		panel_after = self._annotate_panel(aligned_after, f"After aligned ({alignment_mode})")
		panel_diff = self._annotate_panel(diff_heat, "Difference heatmap")
		panel_mask = self._annotate_panel(change_overlay, "Detected changes (Before/After blend)")

		top = np.hstack([panel_before, panel_after])
		bottom = np.hstack([panel_diff, panel_mask])
		grid = np.vstack([top, bottom])
		return self._resize_if_too_large(
			grid,
			max_width=int(PREVIEW_CONFIG["max_width"]),
			max_height=int(PREVIEW_CONFIG["max_height"]),
		)

	def _make_change_overlay(
		self,
		before_img: np.ndarray,
		aligned_after: np.ndarray,
		change_mask: np.ndarray,
	) -> np.ndarray:
		base = cv2.addWeighted(before_img, 0.5, aligned_after, 0.5, 0)
		color_mask = np.zeros_like(base)
		color_mask[:, :, 2] = change_mask
		color_mask[:, :, 1] = (change_mask // 2)
		overlay = cv2.addWeighted(base, 1.0, color_mask, 0.7, 0)

		contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cv2.drawContours(overlay, contours, contourIdx=-1, color=(255, 255, 255), thickness=1)
		return overlay

	def _annotate_panel(self, image: np.ndarray, title: str) -> np.ndarray:
		panel = image.copy()
		cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (0, 0, 0), -1)
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
		aligned_before: np.ndarray,
		aligned_after: np.ndarray,
		diff_gray: np.ndarray,
		change_mask: np.ndarray,
		overlap_mask: np.ndarray,
		preview: np.ndarray,
		match_vis: np.ndarray | None,
	) -> dict[str, Path]:
		paths = {
			"aligned_before": output_dir / "aligned_before.png",
			"aligned_after": output_dir / "aligned_after.png",
			"difference_gray": output_dir / "difference_gray.png",
			"change_mask": output_dir / "change_mask.png",
			"overlap_mask": output_dir / "overlap_mask.png",
			"preview": output_dir / "preview.png",
		}

		self._write_image(paths["aligned_before"], aligned_before)
		self._write_image(paths["aligned_after"], aligned_after)
		self._write_image(paths["difference_gray"], diff_gray)
		self._write_image(paths["change_mask"], change_mask)
		self._write_image(paths["overlap_mask"], overlap_mask)
		self._write_image(paths["preview"], preview)

		if match_vis is not None:
			match_path = output_dir / "feature_matches.png"
			self._write_image(match_path, match_vis)
			paths["feature_matches"] = match_path

		return paths

	def _write_image(self, path: Path, image: np.ndarray) -> None:
		ok = cv2.imwrite(str(path), image)
		if not ok:
			raise RuntimeError(f"Failed to write image: {path}")
