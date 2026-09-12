import os
import sys
import cv2
import numpy as np


def main():
    # Resolve script and reference image path
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

    # Derive dynamic output file paths based on source filename stem
    src_filename = os.path.basename(src_path)
    src_stem = os.path.splitext(src_filename)[0]

    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
    registered_path = os.path.normpath(os.path.join(data_dir, f"{src_stem}_registered.jpg"))
    overlay_path = os.path.normpath(os.path.join(data_dir, f"{src_stem}_overlay.jpg"))

    # Error handling: missing image files
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

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_src, des_src = sift.detectAndCompute(src_gray, None)

    # Error handling: missing or empty descriptors
    if des_ref is None or len(des_ref) == 0:
        print(f"Error: No descriptors computed for reference image '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    if des_src is None or len(des_src) == 0:
        print(f"Error: No descriptors computed for source image '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # Match descriptors using BFMatcher with L2 distance and k=2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_src, k=2)

    # Apply Lowe's ratio test with threshold 0.75
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Error handling: insufficient matches
    if len(good_matches) < 4:
        print(
            f"Error: Insufficient good matches found ({len(good_matches)} < 4). Cannot estimate homography.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract corresponding point coordinates
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography transforming SOURCE coordinates into REFERENCE coordinates
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

    # Error handling: homography estimation failure
    if H is None or mask is None:
        print("Error: Homography estimation failed using RANSAC.", file=sys.stderr)
        sys.exit(1)

    num_inliers = int(np.sum(mask))
    inlier_ratio = (num_inliers / len(good_matches)) * 100.0

    # Warp/register source image into reference image dimensions
    h_ref, w_ref = ref_img.shape[:2]
    registered_img = cv2.warpPerspective(src_img, H, (w_ref, h_ref))

    # Create blended overlay between reference image and registered image
    overlay_img = cv2.addWeighted(ref_img, 0.5, registered_img, 0.5, 0)

    # Save output images
    os.makedirs(data_dir, exist_ok=True)
    if not cv2.imwrite(registered_path, registered_img):
        print(f"Error: Failed to save registered image to '{registered_path}'.", file=sys.stderr)
        sys.exit(1)

    if not cv2.imwrite(overlay_path, overlay_img):
        print(f"Error: Failed to save overlay image to '{overlay_path}'.", file=sys.stderr)
        sys.exit(1)

    # Print summary statistics
    print(f"Reference keypoints: {len(kp_ref)}")
    print(f"Source keypoints: {len(kp_src)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"RANSAC inliers: {num_inliers}")
    print(f"Inlier ratio: {inlier_ratio:.2f}%")
    print(f"Registered image path: {registered_path}")
    print(f"Overlay image path: {overlay_path}")


if __name__ == "__main__":
    main()
