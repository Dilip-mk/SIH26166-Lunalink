"""
Lunar Image Registration - Sub-Pixel Refinement Module
Applies OpenCV cornerSubPix on inlier correspondence coordinates to refine feature positions
to sub-pixel resolution and measures initial vs. refined reprojection error.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np


@dataclass
class RefinementResult:
    original_ref_points: np.ndarray      # Shape (N, 2), float32
    original_src_points: np.ndarray      # Shape (N, 2), float32
    refined_ref_points: np.ndarray       # Shape (N, 2), float32
    refined_src_points: np.ndarray       # Shape (N, 2), float32
    initial_rmse: float                  # Reprojection RMSE before sub-pixel refinement
    refined_rmse: float                  # Reprojection RMSE after sub-pixel refinement
    rmse_improvement: float              # Improvement in pixels (initial_rmse - refined_rmse)
    refined_homography: Optional[np.ndarray] = None


def compute_reprojection_rmse(
    src_points: np.ndarray,
    ref_points: np.ndarray,
    H: np.ndarray,
) -> float:
    """Calculate the Root Mean Square Error (RMSE) in pixels between transformed source and reference points."""
    if len(src_points) == 0 or len(ref_points) == 0 or H is None:
        return 0.0

    src_reshaped = src_points.reshape(-1, 1, 2).astype(np.float32)
    ref_reshaped = ref_points.reshape(-1, 1, 2).astype(np.float32)

    transformed = cv2.perspectiveTransform(src_reshaped, H)
    diffs = transformed - ref_reshaped
    sq_errors = np.sum(diffs**2, axis=2)
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return rmse


def refine_subpixel_correspondences(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    inlier_ref_points: np.ndarray,
    inlier_src_points: np.ndarray,
    initial_H: np.ndarray,
    win_size: Tuple[int, int] = (5, 5),
    max_allowed_shift: float = 2.0,
) -> RefinementResult:
    """Refine inlier keypoint coordinates using gradient-based sub-pixel corner interpolation.

    Parameters:
        ref_gray: Single-channel uint8 reference image.
        src_gray: Single-channel uint8 source image.
        inlier_ref_points: (N, 2) float32 inlier coordinates in reference image.
        inlier_src_points: (N, 2) float32 inlier coordinates in source image.
        initial_H: Initial 3x3 homography mapping source to reference frame.
        win_size: Search window half-size (win_size[0], win_size[1]).
        max_allowed_shift: Maximum acceptable coordinate displacement in pixels.
                           Refinements exceeding this threshold are clamped/reverted to prevent drift.

    Returns:
        RefinementResult with original vs refined coordinates and measurable RMSE metrics.
    """
    n_pts = len(inlier_ref_points)
    initial_rmse = compute_reprojection_rmse(inlier_src_points, inlier_ref_points, initial_H)

    if n_pts < 4 or initial_H is None:
        return RefinementResult(
            original_ref_points=inlier_ref_points.copy(),
            original_src_points=inlier_src_points.copy(),
            refined_ref_points=inlier_ref_points.copy(),
            refined_src_points=inlier_src_points.copy(),
            initial_rmse=initial_rmse,
            refined_rmse=initial_rmse,
            rmse_improvement=0.0,
            refined_homography=initial_H,
        )

    h_ref, w_ref = ref_gray.shape[:2]
    h_src, w_src = src_gray.shape[:2]
    wx, wy = win_size

    # Criteria: stop after 30 iterations or when change < 0.001 pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    refined_ref = inlier_ref_points.copy().astype(np.float32)
    refined_src = inlier_src_points.copy().astype(np.float32)

    # 1. Safely identify points sufficiently far from the border to refine
    valid_ref_mask = (
        (inlier_ref_points[:, 0] >= wx + 1) & (inlier_ref_points[:, 0] < w_ref - wx - 1) &
        (inlier_ref_points[:, 1] >= wy + 1) & (inlier_ref_points[:, 1] < h_ref - wy - 1)
    )
    valid_src_mask = (
        (inlier_src_points[:, 0] >= wx + 1) & (inlier_src_points[:, 0] < w_src - wx - 1) &
        (inlier_src_points[:, 1] >= wy + 1) & (inlier_src_points[:, 1] < h_src - wy - 1)
    )
    joint_mask = valid_ref_mask & valid_src_mask

    if np.any(joint_mask):
        sub_ref_pts = refined_ref[joint_mask].reshape(-1, 1, 2).copy()
        sub_src_pts = refined_src[joint_mask].reshape(-1, 1, 2).copy()

        try:
            cv2.cornerSubPix(ref_gray, sub_ref_pts, win_size, (-1, -1), criteria)
            cv2.cornerSubPix(src_gray, sub_src_pts, win_size, (-1, -1), criteria)

            sub_ref_flat = sub_ref_pts.reshape(-1, 2)
            sub_src_flat = sub_src_pts.reshape(-1, 2)
            orig_ref_flat = inlier_ref_points[joint_mask]
            orig_src_flat = inlier_src_points[joint_mask]

            # Enforce maximum displacement threshold to prevent runaway gradient shift
            shift_ref = np.linalg.norm(sub_ref_flat - orig_ref_flat, axis=1)
            shift_src = np.linalg.norm(sub_src_flat - orig_src_flat, axis=1)
            stable_mask = (shift_ref <= max_allowed_shift) & (shift_src <= max_allowed_shift)

            joint_indices = np.where(joint_mask)[0]
            accepted_indices = joint_indices[stable_mask]

            refined_ref[accepted_indices] = sub_ref_flat[stable_mask]
            refined_src[accepted_indices] = sub_src_flat[stable_mask]
        except Exception:
            # Fallback to unrefined coordinates if cornerSubPix encounters a numerical error
            refined_ref = inlier_ref_points.copy()
            refined_src = inlier_src_points.copy()

    # 2. Re-fit refined homography with RANSAC
    try:
        ref_pts_H = refined_ref.reshape(-1, 1, 2)
        src_pts_H = refined_src.reshape(-1, 1, 2)
        H_ref, _ = cv2.findHomography(src_pts_H, ref_pts_H, cv2.RANSAC, 5.0)
        if H_ref is not None:
            refined_rmse = compute_reprojection_rmse(refined_src, refined_ref, H_ref)
            final_H = H_ref
        else:
            refined_rmse = compute_reprojection_rmse(refined_src, refined_ref, initial_H)
            final_H = initial_H
    except Exception:
        refined_rmse = initial_rmse
        final_H = initial_H

    improvement = initial_rmse - refined_rmse

    return RefinementResult(
        original_ref_points=inlier_ref_points,
        original_src_points=inlier_src_points,
        refined_ref_points=refined_ref,
        refined_src_points=refined_src,
        initial_rmse=initial_rmse,
        refined_rmse=refined_rmse,
        rmse_improvement=improvement,
        refined_homography=final_H,
    )
