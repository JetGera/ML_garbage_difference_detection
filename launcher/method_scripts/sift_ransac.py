from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
	from ..utils.io_utils import pair_folder_name, prepare_output_dir, write_image
	from ..utils.alignment_utils import build_validity_mask, compute_overlap_mask
	from ..utils.viz_utils import annotate_panel, resize_if_too_large
except ImportError:
	from utils.io_utils import pair_folder_name, prepare_output_dir, write_image
	from utils.alignment_utils import build_validity_mask, compute_overlap_mask
	from utils.viz_utils import annotate_panel, resize_if_too_large

# Method 01 (SIFT + RANSAC) config block.
# `nfeatures`: maximum number of SIFT keypoints to keep.
# `contrast_threshold`: lower values make SIFT more sensitive to weak features.
# `clahe_clip_limit`: contrast enhancement strength before feature detection.
# `clahe_tile_grid`: CLAHE grid size used to normalize local contrast.
SIFT_CONFIG = {
	"nfeatures": 30000,
	"contrast_threshold": 0.04,
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
	"ratio_test": 0.70,  # Balanced default for moderate view-angle changes
	"ratio_test_relaxed": 0.82,  # Secondary pass when strict matching is insufficient
	"ratio_test_emergency": 0.90,  # Last-resort pass for hard pairs; RANSAC gates bad geometry
	"ransac_reproj_threshold": 3.5,  # More tolerant for handheld capture drift/rotation
	"min_matches_for_homography": 10,
	"min_inliers_for_homography": 10,
	"min_inlier_ratio": 0.22,
	"feature_roi_enabled": True,
	"feature_roi_top_fraction": 0.03,
	"feature_roi_bottom_fraction": 0.00,
	"feature_roi_left_fraction": 0.00,
	"feature_roi_right_fraction": 0.18,
	"ecc_refine_enabled": True,
	"ecc_iterations": 80,
	"ecc_eps": 1e-5,
	"ecc_gauss_size": 5,
	"ecc_min_correlation": 0.55,
	"min_overlap_ratio": 0.2,
	"max_canvas_side": 8000,
	"max_perspective_shear": 0.0001,  # Extremely strict in config (override below)
	"min_homography_area_ratio": 0.35,
	"max_homography_area_ratio": 3.0,
	"max_affine_rotation_degrees": 18.0,
	"min_affine_scale_ratio": 0.85,
	"max_affine_scale_ratio": 1.15,
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
	# Ignore components that are larger than this fraction of the valid overlap area.
	# Set to 1.0 to disable (default behavior). Lower values will discard very large
	# regions (useful to ignore broad illumination/scene differences near image borders).
	"max_component_area_ratio": 0.15,
	# Fraction of the image height at the top to ignore detections from. Components
	# whose top boundary lies within this fraction will be discarded. Set to 0.0 to
	# disable. Example: 0.25 will ignore components starting within the top 25%.
	"ignore_top_fraction": 0.25,
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
	RATIO_TEST_RELAXED = ALIGNMENT_CONFIG["ratio_test_relaxed"]
	RATIO_TEST_EMERGENCY = ALIGNMENT_CONFIG["ratio_test_emergency"]
	RANSAC_REPROJ_THRESHOLD = ALIGNMENT_CONFIG["ransac_reproj_threshold"]
	MIN_MATCHES_FOR_HOMOGRAPHY = ALIGNMENT_CONFIG["min_matches_for_homography"]
	MIN_INLIERS_FOR_HOMOGRAPHY = ALIGNMENT_CONFIG["min_inliers_for_homography"]
	MIN_INLIER_RATIO = ALIGNMENT_CONFIG["min_inlier_ratio"]
	FEATURE_ROI_ENABLED = ALIGNMENT_CONFIG["feature_roi_enabled"]
	FEATURE_ROI_TOP_FRACTION = ALIGNMENT_CONFIG["feature_roi_top_fraction"]
	FEATURE_ROI_BOTTOM_FRACTION = ALIGNMENT_CONFIG["feature_roi_bottom_fraction"]
	FEATURE_ROI_LEFT_FRACTION = ALIGNMENT_CONFIG["feature_roi_left_fraction"]
	FEATURE_ROI_RIGHT_FRACTION = ALIGNMENT_CONFIG["feature_roi_right_fraction"]
	ECC_REFINE_ENABLED = ALIGNMENT_CONFIG["ecc_refine_enabled"]
	ECC_ITERATIONS = ALIGNMENT_CONFIG["ecc_iterations"]
	ECC_EPS = ALIGNMENT_CONFIG["ecc_eps"]
	ECC_GAUSS_SIZE = ALIGNMENT_CONFIG["ecc_gauss_size"]
	ECC_MIN_CORRELATION = ALIGNMENT_CONFIG["ecc_min_correlation"]
	MIN_OVERLAP_RATIO = ALIGNMENT_CONFIG["min_overlap_ratio"]
	MAX_CANVAS_SIDE = ALIGNMENT_CONFIG["max_canvas_side"]
	MAX_PERSPECTIVE_SHEAR = 0.0015  # Allow mild perspective effects from viewpoint changes
	MIN_HOMOGRAPHY_AREA_RATIO = ALIGNMENT_CONFIG["min_homography_area_ratio"]
	MAX_HOMOGRAPHY_AREA_RATIO = ALIGNMENT_CONFIG["max_homography_area_ratio"]
	MAX_AFFINE_ROTATION_DEGREES = ALIGNMENT_CONFIG["max_affine_rotation_degrees"]
	MIN_AFFINE_SCALE_RATIO = ALIGNMENT_CONFIG["min_affine_scale_ratio"]
	MAX_AFFINE_SCALE_RATIO = ALIGNMENT_CONFIG["max_affine_scale_ratio"]

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
			"fallback_reason": str(align_info.get("fallback_reason", "none")),
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
		fallback_reason = metrics["fallback_reason"]
		preview_text = (
			f"Alignment: {metrics['alignment_mode']}\n"
			f"Good matches: {good_matches}, inliers: {inliers}\n"
			f"Fallback reason: {fallback_reason}\n"
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
		before_roi = self._build_feature_roi_mask(before_gray.shape)
		after_roi = self._build_feature_roi_mask(after_gray.shape)

		kp_before, desc_before = sift.detectAndCompute(before_feat, before_roi)
		kp_after, desc_after = sift.detectAndCompute(after_feat, after_roi)

		info: dict[str, Any] = {
			"alignment_mode": "identity_fallback",
			"fallback_reason": "none",
			"keypoints_before": len(kp_before),
			"keypoints_after": len(kp_after),
			"raw_matches": 0,
			"good_matches": 0,
			"inliers": 0,
		}

		if desc_before is None or desc_after is None:
			info["fallback_reason"] = "missing_descriptors"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, None

		desc_before = self._to_rootsift(desc_before)
		desc_after = self._to_rootsift(desc_after)

		matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)
		info["raw_matches"] = len(raw_matches)

		good_matches = self._ratio_filter_matches(raw_matches, self.RATIO_TEST)
		if len(good_matches) < self.MIN_MATCHES_FOR_HOMOGRAPHY:
			relaxed_matches = self._ratio_filter_matches(raw_matches, self.RATIO_TEST_RELAXED)
			if len(relaxed_matches) > len(good_matches):
				good_matches = relaxed_matches
		if len(good_matches) < self.MIN_MATCHES_FOR_HOMOGRAPHY:
			emergency_matches = self._ratio_filter_matches(raw_matches, self.RATIO_TEST_EMERGENCY)
			if len(emergency_matches) > len(good_matches):
				good_matches = emergency_matches

		info["good_matches"] = len(good_matches)
		if len(good_matches) < self.MIN_MATCHES_FOR_HOMOGRAPHY:
			match_vis = self._draw_matches(before_img, after_img, kp_before, kp_after, good_matches)
			info["fallback_reason"] = "insufficient_good_matches"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		src_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

		homography_method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
		homography, inlier_mask = cv2.findHomography(
			src_pts,
			dst_pts,
			method=homography_method,
			ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
			maxIters=12000,
			confidence=0.999,
		)

		match_vis = self._draw_matches(before_img, after_img, kp_before, kp_after, good_matches, inlier_mask)

		if homography is None:
			affine_result = self._align_with_affine(before_img, after_img, good_matches, kp_before, kp_after)
			if affine_result is not None:
				aligned_before, aligned_after, overlap_mask, affine_info = affine_result
				info["alignment_mode"] = affine_info["alignment_mode"]
				info["inliers"] = affine_info["inliers"]
				info["fallback_reason"] = str(affine_info.get("fallback_reason", "none"))
				return aligned_before, aligned_after, overlap_mask, info, match_vis
			info["fallback_reason"] = "homography_estimation_failed"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		if not self._is_homography_sane(before_img, after_img, homography):
			affine_result = self._align_with_affine(before_img, after_img, good_matches, kp_before, kp_after)
			if affine_result is not None:
				aligned_before, aligned_after, overlap_mask, affine_info = affine_result
				info["alignment_mode"] = affine_info["alignment_mode"]
				info["inliers"] = affine_info["inliers"]
				info["fallback_reason"] = str(affine_info.get("fallback_reason", "none"))
				return aligned_before, aligned_after, overlap_mask, info, match_vis
			info["fallback_reason"] = "homography_sanity_failed"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
		inlier_ratio = float(inliers / max(len(good_matches), 1))
		if inliers < self.MIN_INLIERS_FOR_HOMOGRAPHY or inlier_ratio < self.MIN_INLIER_RATIO:
			affine_result = self._align_with_affine(before_img, after_img, good_matches, kp_before, kp_after)
			if affine_result is not None:
				aligned_before, aligned_after, overlap_mask, affine_info = affine_result
				info["alignment_mode"] = affine_info["alignment_mode"]
				info["inliers"] = affine_info["inliers"]
				info["fallback_reason"] = str(affine_info.get("fallback_reason", "none"))
				return aligned_before, aligned_after, overlap_mask, info, match_vis
			info["fallback_reason"] = "insufficient_homography_inliers"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		try:
			aligned_before, aligned_after, overlap_mask = self._warp_to_shared_canvas(before_img, after_img, homography)
		except ValueError:
			info["fallback_reason"] = "homography_canvas_invalid"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info)
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		overlap_pixels = int(np.count_nonzero(overlap_mask))
		min_overlap = int(before_img.shape[0] * before_img.shape[1] * self.MIN_OVERLAP_RATIO)
		if overlap_pixels < max(1, min_overlap):
			info["alignment_mode"] = "low_overlap_fallback"
			info["fallback_reason"] = "low_overlap_after_warp"
			aligned_before, aligned_after, overlap_mask = self._fallback_alignment_pair(before_img, after_img, info, fallback_mode="low_overlap_fallback")
			return aligned_before, aligned_after, overlap_mask, info, match_vis

		info["alignment_mode"] = "homography_ransac"
		info["fallback_reason"] = "none"
		info["inliers"] = inliers
		return aligned_before, aligned_after, overlap_mask, info, match_vis

	def _build_feature_roi_mask(self, image_shape: tuple[int, ...]) -> np.ndarray | None:
		if not self.FEATURE_ROI_ENABLED:
			return None
		h, w = image_shape[:2]
		x0 = int(np.clip(round(w * self.FEATURE_ROI_LEFT_FRACTION), 0, max(w - 1, 0)))
		x1 = int(np.clip(round(w * (1.0 - self.FEATURE_ROI_RIGHT_FRACTION)), 1, w))
		y0 = int(np.clip(round(h * self.FEATURE_ROI_TOP_FRACTION), 0, max(h - 1, 0)))
		y1 = int(np.clip(round(h * (1.0 - self.FEATURE_ROI_BOTTOM_FRACTION)), 1, h))
		if x1 - x0 < 32 or y1 - y0 < 32:
			return None
		mask = np.zeros((h, w), dtype=np.uint8)
		mask[y0:y1, x0:x1] = 255
		return mask

	def _ratio_filter_matches(self, raw_matches: list[list[cv2.DMatch]], ratio: float) -> list[cv2.DMatch]:
		filtered: list[cv2.DMatch] = []
		for pair in raw_matches:
			if len(pair) < 2:
				continue
			first, second = pair
			if first.distance < ratio * second.distance:
				filtered.append(first)
		return filtered

	def _to_rootsift(self, descriptors: np.ndarray) -> np.ndarray:
		if descriptors is None or descriptors.size == 0:
			return descriptors
		desc = descriptors.astype(np.float32, copy=False)
		l1 = np.sum(np.abs(desc), axis=1, keepdims=True)
		desc = desc / np.maximum(l1, 1e-12)
		return np.sqrt(desc)

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

		if not self._is_affine_sane(affine):
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
			build_validity_mask((before_h, before_w)),
			translation,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		after_valid = cv2.warpPerspective(
			build_validity_mask((after_h, after_w)),
			affine_full,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		overlap_mask = compute_overlap_mask(before_valid, after_valid)
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
			"fallback_reason": "none",
		}

		if self.ECC_REFINE_ENABLED:
			ecc_result = self._refine_with_ecc(aligned_before, aligned_after, overlap_mask)
			if ecc_result is not None:
				refined_after, refined_overlap, corr = ecc_result
				aligned_after = refined_after
				overlap_mask = refined_overlap
				affine_info["alignment_mode"] = "affine_ecc"
				affine_info["ecc_correlation"] = round(float(corr), 4)
		return aligned_before, aligned_after, overlap_mask, affine_info

	def _refine_with_ecc(
		self,
		before_img: np.ndarray,
		after_img: np.ndarray,
		overlap_mask: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, float] | None:
		if before_img.shape[:2] != after_img.shape[:2]:
			return None
		if int(np.count_nonzero(overlap_mask)) < 512:
			return None

		template = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
		input_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
		warp = np.eye(2, 3, dtype=np.float32)
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			int(self.ECC_ITERATIONS),
			float(self.ECC_EPS),
		)

		try:
			corr, warp = cv2.findTransformECC(
				template,
				input_gray,
				warp,
				cv2.MOTION_AFFINE,
				criteria,
				overlap_mask,
				int(self.ECC_GAUSS_SIZE),
			)
		except cv2.error:
			return None

		if not np.isfinite(corr) or corr < self.ECC_MIN_CORRELATION:
			return None

		h, w = after_img.shape[:2]
		refined_after = cv2.warpAffine(
			after_img,
			warp,
			(w, h),
			flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)
		after_valid = cv2.warpAffine(
			np.full((h, w), 255, dtype=np.uint8),
			warp,
			(w, h),
			flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		refined_overlap = cv2.bitwise_and(overlap_mask, after_valid)
		if int(np.count_nonzero(refined_overlap)) < 512:
			return None
		return refined_after, refined_overlap, float(corr)

	def _is_affine_sane(self, affine: np.ndarray) -> bool:
		if affine.shape != (2, 3):
			return False
		if not np.isfinite(affine).all():
			return False

		linear = affine[:, :2].astype(np.float64)
		u, singular_values, vt = np.linalg.svd(linear)
		rotation_matrix = u @ vt
		rotation_degrees = abs(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
		scale_ratio = float((singular_values[0] + singular_values[1]) / 2.0)

		if rotation_degrees > self.MAX_AFFINE_ROTATION_DEGREES:
			return False
		if scale_ratio < self.MIN_AFFINE_SCALE_RATIO or scale_ratio > self.MAX_AFFINE_SCALE_RATIO:
			return False

		# Reject obvious shear by checking how far the linear part deviates from a pure rotation+uniform scale.
		shear_matrix = np.linalg.inv(rotation_matrix) @ linear
		off_diag = max(abs(float(shear_matrix[0, 1])), abs(float(shear_matrix[1, 0])))
		if off_diag > 0.08:
			return False

		return True

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
		fallback_mode: str | None = None,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		before_shape = before_img.shape[:2]
		after_shape = after_img.shape[:2]
		if fallback_mode is not None:
			info["alignment_mode"] = fallback_mode
		if before_shape != after_shape:
			if fallback_mode is None:
				info["alignment_mode"] = "resize_fallback"
			aligned_after = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]), interpolation=cv2.INTER_LINEAR)
		else:
			if fallback_mode is None:
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
			build_validity_mask((before_h, before_w)),
			before_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		after_valid = cv2.warpPerspective(
			build_validity_mask((after_h, after_w)),
			after_homography,
			(canvas_w, canvas_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		overlap_mask = compute_overlap_mask(before_valid, after_valid)

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
		max_draw = 250
		sorted_good = sorted(good_matches, key=lambda m: m.distance)
		display_good = sorted_good[:max_draw]

		good_vis = cv2.drawMatches(
			before_img,
			kp_before,
			after_img,
			kp_after,
			display_good,
			None,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
		)
		if inlier_mask is None:
			return annotate_panel(good_vis, f"Good matches: {len(good_matches)}")

		mask = inlier_mask.ravel().astype(bool)
		inlier_matches = [match for match, keep in zip(good_matches, mask) if keep]
		if not inlier_matches:
			return annotate_panel(good_vis, f"Good matches: {len(good_matches)}")

		sorted_inliers = sorted(inlier_matches, key=lambda m: m.distance)
		display_inliers = sorted_inliers[:max_draw]
		inlier_vis = cv2.drawMatches(
			before_img,
			kp_before,
			after_img,
			kp_after,
			display_inliers,
			None,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
		)

		top = annotate_panel(good_vis, f"Good matches: {len(good_matches)}")
		bottom = annotate_panel(inlier_vis, f"RANSAC inliers: {len(inlier_matches)}")
		return np.vstack([top, bottom])

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
			left = int(stats[label, cv2.CC_STAT_LEFT])
			top = int(stats[label, cv2.CC_STAT_TOP])

			# Maximum area guard (relative to valid overlap pixels)
			max_area_ratio = float(CHANGE_MAP_CONFIG.get("max_component_area_ratio", 1.0))
			max_area = int(valid_pixels * max_area_ratio)

			# Top-area ignore: discard components whose top lies within the top fraction
			ignore_top_fraction = float(CHANGE_MAP_CONFIG.get("ignore_top_fraction", 0.0))
			top_threshold = int(overlap_mask.shape[0] * ignore_top_fraction)

			# Keep component only if it satisfies area and bounding-box size and isn't
			# too large or entirely starting in the top ignored band.
			if (
				area >= min_area
				and width >= min_width
				and height >= min_height
				and area <= max_area
				and not (ignore_top_fraction > 0.0 and top <= top_threshold)
			):
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

		panel_before = annotate_panel(before_img, "Before (aligned frame)")
		panel_after = annotate_panel(aligned_after, f"After aligned ({alignment_mode})")
		panel_diff = annotate_panel(diff_heat, "Difference heatmap")
		panel_mask = annotate_panel(change_overlay, "Detected changes (Before/After blend)")

		top = np.hstack([panel_before, panel_after])
		bottom = np.hstack([panel_diff, panel_mask])
		grid = np.vstack([top, bottom])
		return resize_if_too_large(
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

	def _prepare_output_dir(self, before: Path, after: Path) -> Path:
		root = Path(__file__).resolve().parent.parent.parent / "results"
		pair_name = pair_folder_name(before, after)
		return prepare_output_dir(root, pair_name, self.label)

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

		write_image(paths["aligned_before"], aligned_before)
		write_image(paths["aligned_after"], aligned_after)
		write_image(paths["difference_gray"], diff_gray)
		write_image(paths["change_mask"], change_mask)
		write_image(paths["overlap_mask"], overlap_mask)
		write_image(paths["preview"], preview)

		if match_vis is not None:
			match_path = output_dir / "feature_matches.png"
			write_image(match_path, match_vis)
			paths["feature_matches"] = match_path

		return paths

