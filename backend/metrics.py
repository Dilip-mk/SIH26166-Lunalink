import os
import sys
import cv2
import numpy as np


def main():
    # Resolve reference image path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))

    # Determine source image path from command-line argument or default to source.jpg
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
    else:
        source_arg = "source.jpg"

    if os.path.isabs(source_arg):
        src_path = os.path.normpath(source_arg)
    elif os.path.dirname(source_arg):
        src_path = os.path.normpath(os.path.join(script_dir, source_arg))
    else:
        src_path = os.path.normpath(os.path.join(script_dir, "..", "data", source_arg))

    ref_filename = os.path.basename(ref_path)
    src_filename = os.path.basename(src_path)

    # Error handling: Missing reference image
    if not os.path.exists(ref_path):
        print(f"Error: Reference image missing at '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    # Error handling: Missing source image
    if not os.path.exists(src_path):
        print(f"Error: Source image missing at '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Error handling: Image loading failure
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Error: Failed to load reference image from '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    src_img = cv2.imread(src_path)
    if src_img is None:
        print(f"Error: Failed to load source image from '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_src, des_src = sift.detectAndCompute(src_gray, None)

    # Error handling: No descriptors
    if des_ref is None or len(des_ref) == 0:
        print(f"Error: No descriptors computed for reference image '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    if des_src is None or len(des_src) == 0:
        print(f"Error: No descriptors computed for source image '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Match descriptors using BFMatcher with NORM_L2 and k=2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_src, k=2)

    # Apply Lowe's ratio test with threshold 0.75
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Error handling: Fewer than 4 good matches
    if len(good_matches) < 4:
        print(
            f"Error: Insufficient good matches found ({len(good_matches)} < 4). Cannot estimate homography.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract corresponding point coordinates
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography from SOURCE coordinates to REFERENCE coordinates using RANSAC
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

    # Error handling: Homography estimation failure
    if H is None or mask is None:
        print("Error: Homography estimation failed using RANSAC.", file=sys.stderr)
        sys.exit(1)

    mask_ravel = mask.ravel()
    inliers = int(np.sum(mask_ravel))
    outliers = len(good_matches) - inliers

    # Error handling: Zero inliers
    if inliers == 0:
        print("Error: RANSAC produced zero inliers.", file=sys.stderr)
        sys.exit(1)

    inlier_ratio = (inliers / len(good_matches)) * 100.0

    # Calculate RMSE (reprojection error) on RANSAC inliers
    inlier_mask_bool = (mask_ravel == 1)
    inlier_src_pts = src_pts[inlier_mask_bool]  # shape (N, 1, 2)
    inlier_ref_pts = ref_pts[inlier_mask_bool]  # shape (N, 1, 2)

    transformed_src_pts = cv2.perspectiveTransform(inlier_src_pts, H)
    diffs = transformed_src_pts - inlier_ref_pts
    sq_errors = np.sum(diffs**2, axis=2)  # shape (N, 1)
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # Spatial coverage calculation over 4 x 4 grid on REFERENCE image
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

    # Print results formatted clearly as requested
    print("=== LUNAR IMAGE REGISTRATION METRICS ===")
    print(f"Reference image: {ref_filename}")
    print(f"Source image: {src_filename}")
    print()
    print(f"Reference keypoints: {len(kp_ref)}")
    print(f"Source keypoints: {len(kp_src)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"RANSAC inliers: {inliers}")
    print(f"RANSAC outliers: {outliers}")
    print(f"Inlier ratio: {inlier_ratio:.2f}%")
    print(f"RMSE: {rmse:.2f} pixels")
    print(f"Spatial coverage: {spatial_coverage:.2f}%")
    print(f"Occupied grid cells: {occupied_count} / {total_cells}")


if __name__ == "__main__":
    main()
