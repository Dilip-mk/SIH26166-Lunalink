import os
import sys
import cv2
import numpy as np


def main():
    # Resolve relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))

    # Determine source image path (CLI argument if provided, default to source.jpg)
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

    src_filename = os.path.basename(src_path)
    src_stem = os.path.splitext(src_filename)[0]
    out_filename = f"{src_stem}_ransac_inliers.jpg"
    out_path = os.path.normpath(os.path.join(script_dir, "..", "data", out_filename))

    # Error handling: check if input image files exist
    if not os.path.exists(ref_path):
        print(f"Error: Reference image missing at '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(src_path):
        print(f"Error: Source image missing at '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Load images using OpenCV
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Error: Failed to read reference image from '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    src_img = cv2.imread(src_path)
    if src_img is None:
        print(f"Error: Failed to read source image from '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_src, des_src = sift.detectAndCompute(src_gray, None)

    # Error handling: check for missing or empty descriptors
    if des_ref is None or len(des_ref) == 0:
        print(f"Error: No descriptors computed for reference image '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    if des_src is None or len(des_src) == 0:
        print(f"Error: No descriptors computed for source image '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # BFMatcher with L2 distance and KNN matching (k=2)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_src, k=2)

    # Apply Lowe's ratio test (threshold = 0.75)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Error handling: check if sufficient good matches are available (>= 4 required for homography)
    if len(good_matches) < 4:
        print(
            f"Error: Insufficient good matches found ({len(good_matches)} < 4). Cannot estimate homography.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract corresponding point coordinates
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate geometric transformation using RANSAC (reprojection threshold = 5.0 pixels)
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        print("Error: Homography estimation failed using RANSAC.", file=sys.stderr)
        sys.exit(1)

    # Separate inliers and outliers based on the RANSAC mask
    mask_ravel = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_ravel[i] == 1]
    outlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_ravel[i] == 0]

    num_inliers = len(inlier_matches)
    num_outliers = len(outlier_matches)
    inlier_ratio = (num_inliers / len(good_matches)) * 100.0

    # Print summary results
    print(f"Reference keypoints: {len(kp_ref)}")
    print(f"{src_stem.capitalize()} keypoints: {len(kp_src)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"Inliers: {num_inliers}")
    print(f"Outliers: {num_outliers}")
    print(f"Inlier ratio: {inlier_ratio:.2f}%")

    # Draw only inlier matches
    inlier_vis = cv2.drawMatches(
        ref_img,
        kp_ref,
        src_img,
        kp_src,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Save inlier matches visualization
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not cv2.imwrite(out_path, inlier_vis):
        print(f"Error: Failed to save visualization to '{out_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully saved inlier match visualization to '{out_path}'.")


if __name__ == "__main__":
    main()
