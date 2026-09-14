"""
Lunar Image Registration - V2 Modular Pipeline
Orchestrates preprocessing, correspondence engine matching, uniform spatial selection,
RANSAC geometric verification, sub-pixel coordinate refinement, image registration, and quality metrics.
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, Optional
import cv2
import numpy as np

from preprocessing import preprocess_image
from engine import get_correspondence_engine, MatchResult
from spatial import (
    select_uniform_correspondences,
    draw_spatial_grid_visualization,
    draw_spatial_selection_visualization,
)
from refinement import refine_subpixel_correspondences, compute_reprojection_rmse


def run_pipeline(
    source_path: str,
    reference_path: Optional[str] = None,
    method: str = "sift",
    preprocessing: str = "raw",
    spatial_strategy: str = "grid",
    max_matches_per_cell: Optional[int] = None,
    target_total_matches: Optional[int] = None,
    enable_subpixel: bool = True,
    ransac_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Execute the complete modular lunar image registration pipeline.

    Parameters:
        source_path: Filepath to the source image (to be warped/registered).
        reference_path: Filepath to the reference image (target coordinate frame).
        method: Correspondence algorithm ('sift' baseline). Default: 'sift'.
        preprocessing: Illumination strategy ('raw', 'clahe', 'normalized'). Default: 'raw'.
        spatial_strategy: Spatial selection mode ('none', 'grid', 'adaptive_grid'). Default: 'grid'.
        max_matches_per_cell: Hard cap on matches per grid cell. None lets the strategy decide.
        target_total_matches: Target total selected matches (used by 'adaptive_grid'). None uses default.
        enable_subpixel: Whether to apply gradient-based sub-pixel coordinate refinement.
        ransac_threshold: Maximum reprojection threshold in pixels for RANSAC. Default: 5.0.

    Returns:
        Dict containing all quantitative registration metrics, timing, and saved output file paths.
    """
    start_time = time.perf_counter()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Path resolution and validation (No hardcoded reference image)
    if reference_path is None or not str(reference_path).strip():
        raise ValueError("Reference image path must be specified. Hardcoded reference is disabled.")

    ref_path = os.path.abspath(reference_path)
    src_path = os.path.abspath(source_path)

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image missing at '{ref_path}'.")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source image missing at '{src_path}'.")

    # 2. Image Loading
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError(f"Failed to read reference image from '{ref_path}'. File may be corrupt or unsupported.")

    src_img = cv2.imread(src_path)
    if src_img is None:
        raise ValueError(f"Failed to read source image from '{src_path}'. File may be corrupt or unsupported.")

    h_ref, w_ref = ref_img.shape[:2]
    h_src, w_src = src_img.shape[:2]

    # 3. Robust Preprocessing
    ref_gray = preprocess_image(ref_img, mode=preprocessing)
    src_gray = preprocess_image(src_img, mode=preprocessing)

    # 4. Correspondence Engine Matching
    engine = get_correspondence_engine(method=method)
    match_result: MatchResult = engine.match(reference_gray=ref_gray, source_gray=src_gray)

    num_kp_ref = match_result.reference_keypoints_count
    num_kp_src = match_result.source_keypoints_count
    num_candidates = match_result.candidate_matches
    num_good_matches = match_result.good_matches

    if num_good_matches < 4:
        raise ValueError(
            f"Insufficient good matches found ({num_good_matches} < 4). "
            f"Cannot estimate geometric transformation between images."
        )

    # 4b. Pre-selection coverage (before spatial filtering)
    pre_spatial = select_uniform_correspondences(
        ref_points=match_result.reference_points,
        source_points=match_result.source_points,
        confidences=match_result.confidence_scores,
        image_shape=(h_ref, w_ref),
        grid_rows=4,
        grid_cols=4,
        spatial_strategy="none",
    )
    occupied_cells_before = pre_spatial.occupied_cells
    spatial_coverage_before = pre_spatial.spatial_coverage

    # 5. Spatial Selection & Distribution
    spatial_res = select_uniform_correspondences(
        ref_points=match_result.reference_points,
        source_points=match_result.source_points,
        confidences=match_result.confidence_scores,
        image_shape=(h_ref, w_ref),
        grid_rows=4,
        grid_cols=4,
        max_per_cell=max_matches_per_cell,
        target_total_matches=target_total_matches,
        spatial_strategy=spatial_strategy,
        min_required_matches=4,
    )

    selected_ref_pts = spatial_res.selected_ref_points
    selected_src_pts = spatial_res.selected_source_points
    num_selected = len(selected_ref_pts)

    # Post-selection coverage
    occupied_cells_after = spatial_res.occupied_cells
    spatial_coverage_after = spatial_res.spatial_coverage

    if num_selected < 4:
        raise ValueError(
            f"Insufficient matches remaining after spatial selection ({num_selected} < 4)."
        )

    # 6. RANSAC Geometric Verification & Homography Estimation
    src_pts_arr = selected_src_pts.reshape(-1, 1, 2)
    ref_pts_arr = selected_ref_pts.reshape(-1, 1, 2)

    H_initial, mask = cv2.findHomography(src_pts_arr, ref_pts_arr, cv2.RANSAC, ransac_threshold)

    if H_initial is None or mask is None:
        raise RuntimeError("Homography estimation failed using RANSAC.")

    mask_ravel = mask.ravel()
    num_inliers = int(np.sum(mask_ravel == 1))
    num_outliers = int(num_selected - num_inliers)

    if num_inliers == 0:
        raise RuntimeError("RANSAC geometric verification produced zero inliers.")

    inlier_ratio = (num_inliers / float(num_selected)) * 100.0

    inlier_src_pts = selected_src_pts[mask_ravel == 1]
    inlier_ref_pts = selected_ref_pts[mask_ravel == 1]

    # Initial reprojection RMSE over inliers
    initial_rmse = compute_reprojection_rmse(inlier_src_pts, inlier_ref_pts, H_initial)

    # 7. Sub-Pixel Refinement
    if enable_subpixel and num_inliers >= 4:
        refinement_res = refine_subpixel_correspondences(
            ref_gray=ref_gray,
            src_gray=src_gray,
            inlier_ref_points=inlier_ref_pts,
            inlier_src_points=inlier_src_pts,
            initial_H=H_initial,
            win_size=(5, 5),
            max_allowed_shift=2.0,
        )
        final_H = refinement_res.refined_homography
        refined_rmse = refinement_res.refined_rmse
        rmse_improvement = refinement_res.rmse_improvement
    else:
        final_H = H_initial
        refined_rmse = initial_rmse
        rmse_improvement = 0.0

    # For backward compatibility, primary rmse metric reports the final evaluated RMSE
    reported_rmse = refined_rmse if refined_rmse > 0 else initial_rmse

    # 8. Re-calculate spatial coverage based strictly on validated inliers
    inlier_spatial = select_uniform_correspondences(
        ref_points=inlier_ref_pts,
        source_points=inlier_src_pts,
        confidences=spatial_res.selected_confidences[mask_ravel == 1],
        image_shape=(h_ref, w_ref),
        grid_rows=4,
        grid_cols=4,
        spatial_strategy="none",
    )
    occupied_count = inlier_spatial.occupied_cells
    total_cells = inlier_spatial.total_cells
    spatial_coverage = inlier_spatial.spatial_coverage

    # 9. Warp Source Image & Generate Visualizations
    registered_img = cv2.warpPerspective(src_img, final_H, (w_ref, h_ref))
    overlay_img = cv2.addWeighted(ref_img, 0.5, registered_img, 0.5, 0)

    # Inliers correspondence visualization
    # Map inlier indices back to original keypoints if available
    inlier_orig_indices = spatial_res.selected_indices[mask_ravel == 1]
    if match_result.raw_dmatches is not None and match_result.keypoints_ref is not None:
        inlier_dmatches = [match_result.raw_dmatches[i] for i in inlier_orig_indices]
        inliers_vis = cv2.drawMatches(
            ref_img,
            match_result.keypoints_ref,
            src_img,
            match_result.keypoints_src,
            inlier_dmatches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        # Fallback correspondence visualization
        inliers_vis = overlay_img.copy()

    # Spatial distribution visualization (selected points with grid)
    spatial_vis = draw_spatial_grid_visualization(
        ref_image=ref_img,
        ref_points=selected_ref_pts,
        grid_rows=4,
        grid_cols=4,
        inlier_mask=mask_ravel,
    )

    # Spatial selection visualization (candidates vs selected, with per-cell counts)
    spatial_sel_vis = draw_spatial_selection_visualization(
        ref_image=ref_img,
        all_ref_points=match_result.reference_points,
        selected_ref_points=selected_ref_pts,
        grid_rows=4,
        grid_cols=4,
        cell_counts=spatial_res.cell_counts,
    )

    # 10. Save Output Images with Unique Run Stem
    src_stem = os.path.splitext(os.path.basename(src_path))[0]
    run_id = uuid.uuid4().hex[:8]
    output_stem = f"{src_stem}_{run_id}"

    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    registered_path = os.path.normpath(os.path.join(data_dir, f"{output_stem}_registered.jpg"))
    overlay_path = os.path.normpath(os.path.join(data_dir, f"{output_stem}_overlay.jpg"))
    inliers_path = os.path.normpath(os.path.join(data_dir, f"{output_stem}_ransac_inliers.jpg"))
    spatial_path = os.path.normpath(os.path.join(data_dir, f"{output_stem}_spatial_grid.jpg"))
    spatial_sel_path = os.path.normpath(os.path.join(data_dir, f"{output_stem}_spatial_matches.jpg"))

    if not cv2.imwrite(registered_path, registered_img):
        raise IOError(f"Failed to save registered image to '{registered_path}'.")
    if not cv2.imwrite(overlay_path, overlay_img):
        raise IOError(f"Failed to save overlay image to '{overlay_path}'.")
    if not cv2.imwrite(inliers_path, inliers_vis):
        raise IOError(f"Failed to save inliers visualization to '{inliers_path}'.")
    if not cv2.imwrite(spatial_path, spatial_vis):
        raise IOError(f"Failed to save spatial visualization to '{spatial_path}'.")
    if not cv2.imwrite(spatial_sel_path, spatial_sel_vis):
        raise IOError(f"Failed to save spatial selection visualization to '{spatial_sel_path}'.")

    processing_time = round(time.perf_counter() - start_time, 4)

    # Determine effective max_per_cell for reporting
    effective_max_per_cell = max_matches_per_cell
    if effective_max_per_cell is None and spatial_strategy != "none":
        # Report the computed cap from cell_counts
        if spatial_res.cell_counts:
            effective_max_per_cell = max(spatial_res.cell_counts.values())

    return {
        # Preserved V1 Metrics
        "reference_keypoints": num_kp_ref,
        "source_keypoints": num_kp_src,
        "good_matches": num_good_matches,
        "ransac_inliers": num_inliers,
        "ransac_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "rmse": reported_rmse,
        "spatial_coverage": spatial_coverage,
        "occupied_grid_cells": occupied_count,
        "total_grid_cells": total_cells,
        "registered_image": registered_path,
        "overlay_image": overlay_path,
        "ransac_visualization": inliers_path,
        # Extended V2 Metrics
        "method": method,
        "preprocessing": preprocessing,
        "candidate_matches": num_candidates,
        "selected_matches": num_selected,
        "initial_rmse": initial_rmse,
        "refined_rmse": refined_rmse,
        "rmse_improvement": rmse_improvement,
        "spatial_visualization": spatial_path,
        "processing_time": processing_time,
        # Phase 2: Spatial selection metrics
        "spatial_strategy": spatial_res.strategy,
        "max_matches_per_cell": effective_max_per_cell,
        "occupied_grid_cells_before": occupied_cells_before,
        "occupied_grid_cells_after": occupied_cells_after,
        "spatial_coverage_before": spatial_coverage_before,
        "spatial_coverage_after": spatial_coverage_after,
        "spatial_selection_visualization": spatial_sel_path,
    }


def main():
    """CLI utility for executing the modular pipeline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <reference_path> <source_path> [method] [preprocessing] [spatial_strategy]")
        print("Example: python pipeline.py ../data/reference.jpg ../data/source.jpg sift raw grid")
        sys.exit(1)

    ref_arg = sys.argv[1]
    src_arg = sys.argv[2]
    method_arg = sys.argv[3] if len(sys.argv) > 3 else "sift"
    prep_arg = sys.argv[4] if len(sys.argv) > 4 else "raw"
    strategy_arg = sys.argv[5] if len(sys.argv) > 5 else "grid"

    try:
        results = run_pipeline(
            source_path=src_arg,
            reference_path=ref_arg,
            method=method_arg,
            preprocessing=prep_arg,
            spatial_strategy=strategy_arg,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== LUNAR IMAGE REGISTRATION RESULT (V2) ===")
    print()
    print(f"Method: {results['method'].upper()} | Preprocessing: {results['preprocessing'].upper()} | Spatial: {results['spatial_strategy'].upper()}")
    print(f"Processing Time: {results['processing_time']}s")
    print(f"Reference Keypoints: {results['reference_keypoints']}")
    print(f"Source Keypoints: {results['source_keypoints']}")
    print(f"Candidate Matches: {results['candidate_matches']}")
    print(f"Good Matches (Ratio Test): {results['good_matches']}")
    print(f"Selected Matches (Spatial): {results['selected_matches']} (max/cell: {results['max_matches_per_cell']})")
    print(f"RANSAC Inliers: {results['ransac_inliers']}")
    print(f"RANSAC Outliers: {results['ransac_outliers']}")
    print(f"Inlier Ratio: {results['inlier_ratio']:.2f}%")
    print(f"Initial RMSE: {results['initial_rmse']:.4f} px")
    print(f"Refined RMSE: {results['refined_rmse']:.4f} px (delta: {results['rmse_improvement']:.4f} px)")
    print(f"Spatial Coverage (before): {results['spatial_coverage_before']:.2f}% ({results['occupied_grid_cells_before']}/16 cells)")
    print(f"Spatial Coverage (after):  {results['spatial_coverage_after']:.2f}% ({results['occupied_grid_cells_after']}/16 cells)")
    print(f"Spatial Coverage (inlier): {results['spatial_coverage']:.2f}% ({results['occupied_grid_cells']}/{results['total_grid_cells']} cells)")
    print()
    print(f"Registered Image: {results['registered_image']}")
    print(f"Overlay Image: {results['overlay_image']}")
    print(f"RANSAC Inliers Vis: {results['ransac_visualization']}")
    print(f"Spatial Grid Vis: {results['spatial_visualization']}")
    print(f"Spatial Selection Vis: {results['spatial_selection_visualization']}")


if __name__ == "__main__":
    main()
