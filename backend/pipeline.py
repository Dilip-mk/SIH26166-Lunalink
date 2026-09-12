import os
import sys
import cv2
import numpy as np


def run_pipeline(source_path):
    # Resolve script directory and reference image path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))

    if os.path.isabs(source_path):
        src_path = os.path.normpath(source_path)
    elif os.path.exists(source_path):
        src_path = os.path.normpath(source_path)
    elif os.path.dirname(source_path):
        src_path = os.path.normpath(os.path.join(script_dir, source_path))
    else:
        src_path = os.path.normpath(os.path.join(script_dir, "..", "data", source_path))

    ref_filename = os.path.basename(ref_path)
    src_filename = os.path.basename(src_path)
    src_stem = os.path.splitext(src_filename)[0]

    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
    registered_path = os.path.normpath(os.path.join(data_dir, f"{src_stem}_registered.jpg"))
    overlay_path = os.path.normpath(os.path.join(data_dir, f"{src_stem}_overlay.jpg"))
    inliers_path = os.path.normpath(os.path.join(data_dir, f"{src_stem}_ransac_inliers.jpg"))

    # 1. Error handling: Check missing image files
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image missing at '{ref_path}'.")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source image missing at '{src_path}'.")

    # Load images using OpenCV
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError(f"Failed to load reference image from '{ref_path}'.")

    src_img = cv2.imread(src_path)
    if src_img is None:
        raise ValueError(f"Failed to load source image from '{src_path}'.")

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # 2. Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_src, des_src = sift.detectAndCompute(src_gray, None)

    # Error handling: Check for missing or empty descriptors
    if des_ref is None or len(des_ref) == 0:
        raise ValueError(f"No descriptors computed for reference image '{ref_path}'.")

    if des_src is None or len(des_src) == 0:
        raise ValueError(f"No descriptors computed for source image '{src_path}'.")

    # 3. Match descriptors using BFMatcher with NORM_L2 and k=2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_src, k=2)

    # 4. Apply Lowe's ratio test with threshold 0.75
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # 5. Error handling: Check if at least 4 good matches exist
    if len(good_matches) < 4:
        raise ValueError(
            f"Insufficient good matches found ({len(good_matches)} < 4). Cannot estimate homography."
        )

    # Extract corresponding point coordinates
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 6. Estimate homography transforming SOURCE coordinates to REFERENCE coordinates using RANSAC
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

    # Error handling: Homography estimation failure
    if H is None or mask is None:
        raise RuntimeError("Homography estimation failed using RANSAC.")

    # 7. Separate RANSAC inliers and outliers
    mask_ravel = mask.ravel()
    inlier_indices = np.where(mask_ravel == 1)[0]
    inlier_matches = [good_matches[i] for i in inlier_indices]

    num_inliers = int(len(inlier_matches))
    num_outliers = int(len(good_matches) - num_inliers)

    # Error handling: No RANSAC inliers
    if num_inliers == 0:
        raise RuntimeError("RANSAC produced zero inliers.")

    # 8. Calculate statistics
    num_kp_ref = len(kp_ref)
    num_kp_src = len(kp_src)
    num_good_matches = len(good_matches)
    inlier_ratio = (num_inliers / num_good_matches) * 100.0

    # 9. Calculate reprojection RMSE using ONLY RANSAC inlier points
    inlier_src_pts = src_pts[mask_ravel == 1]
    inlier_ref_pts = ref_pts[mask_ravel == 1]

    transformed_src_pts = cv2.perspectiveTransform(inlier_src_pts, H)
    diffs = transformed_src_pts - inlier_ref_pts
    sq_errors = np.sum(diffs**2, axis=2)
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # 10. Calculate spatial coverage on a 4 x 4 grid over the reference image
    h_ref, w_ref = ref_img.shape[:2]
    cell_w = w_ref / 4.0
    cell_h = h_ref / 4.0

    occupied_cells = set()
    for pt in inlier_ref_pts:
        x, y = pt[0]
        col = min(3, max(0, int(x / cell_w)))
        row = min(3, max(0, int(y / cell_h)))
        occupied_cells.add((row, col))

    occupied_count = len(occupied_cells)
    total_cells = 16
    spatial_coverage = (occupied_count / total_cells) * 100.0

    # 11. Register source image using cv2.warpPerspective()
    registered_img = cv2.warpPerspective(src_img, H, (w_ref, h_ref))

    # 12. Create blended overlay between reference and registered image using cv2.addWeighted()
    overlay_img = cv2.addWeighted(ref_img, 0.5, registered_img, 0.5, 0)

    # Draw RANSAC inliers visualization
    inliers_vis = cv2.drawMatches(
        ref_img,
        kp_ref,
        src_img,
        kp_src,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 13. Save outputs into ../data/ using the source filename
    os.makedirs(data_dir, exist_ok=True)

    if not cv2.imwrite(registered_path, registered_img):
        raise IOError(f"Failed to save registered image to '{registered_path}'.")

    if not cv2.imwrite(overlay_path, overlay_img):
        raise IOError(f"Failed to save overlay image to '{overlay_path}'.")

    if not cv2.imwrite(inliers_path, inliers_vis):
        raise IOError(f"Failed to save inliers visualization to '{inliers_path}'.")

    return {
        "reference_keypoints": num_kp_ref,
        "source_keypoints": num_kp_src,
        "good_matches": num_good_matches,
        "ransac_inliers": num_inliers,
        "ransac_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "spatial_coverage": spatial_coverage,
        "occupied_grid_cells": occupied_count,
        "total_grid_cells": total_cells,
        "registered_image": registered_path,
        "overlay_image": overlay_path,
        "ransac_visualization": inliers_path,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine source image path from command-line argument or default to source.jpg
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
    else:
        source_arg = "source.jpg"

    try:
        results = run_pipeline(source_arg)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))
    ref_filename = os.path.basename(ref_path)

    if os.path.isabs(source_arg):
        src_path = os.path.normpath(source_arg)
    elif os.path.exists(source_arg):
        src_path = os.path.normpath(source_arg)
    elif os.path.dirname(source_arg):
        src_path = os.path.normpath(os.path.join(script_dir, source_arg))
    else:
        src_path = os.path.normpath(os.path.join(script_dir, "..", "data", source_arg))
    src_filename = os.path.basename(src_path)

    # 14. Print clean final report
    print("=== LUNAR IMAGE REGISTRATION RESULT ===")
    print()
    print(f"Reference: {ref_filename}")
    print(f"Source: {src_filename}")
    print()
    print(f"Reference keypoints: {results['reference_keypoints']}")
    print(f"Source keypoints: {results['source_keypoints']}")
    print(f"Good matches: {results['good_matches']}")
    print(f"RANSAC inliers: {results['ransac_inliers']}")
    print(f"RANSAC outliers: {results['ransac_outliers']}")
    print(f"Inlier ratio: {results['inlier_ratio']:.2f}%")
    print(f"RMSE: {results['rmse']:.2f} pixels")
    print(f"Spatial coverage: {results['spatial_coverage']:.2f}%")
    print(f"Occupied grid cells: {results['occupied_grid_cells']} / {results['total_grid_cells']}")
    print()
    print("Registered image:")
    print(results['registered_image'])
    print()
    print("Overlay image:")
    print(results['overlay_image'])
    print()
    print("RANSAC visualization:")
    print(results['ransac_visualization'])


if __name__ == "__main__":
    main()

