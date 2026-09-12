import os
import sys
import cv2


def main():
    # Resolve relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))

    # Determine source image path (use CLI argument if provided, default to source.jpg)
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

    out_path = os.path.normpath(os.path.join(script_dir, "..", "data", "good_matches.jpg"))

    # Error handling for missing input files
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

    # Detect keypoints and compute descriptors with SIFT
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_src, des_src = sift.detectAndCompute(src_gray, None)

    num_kp_ref = len(kp_ref) if kp_ref is not None else 0
    num_kp_src = len(kp_src) if kp_src is not None else 0

    # Error handling for missing/empty descriptors or keypoints
    if des_ref is None or len(des_ref) == 0:
        print(f"Error: No descriptors computed for reference image '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    if des_src is None or len(des_src) == 0:
        print(f"Error: No descriptors computed for source image '{src_path}'.", file=sys.stderr)
        sys.exit(1)

    # BFMatcher with L2 distance and KNN matching (k=2)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_src, k=2)

    # Apply Lowe's ratio test with ratio threshold = 0.75
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Print numbers of keypoints and good matches
    print(f"Reference image keypoints: {num_kp_ref}")
    print(f"Source image keypoints: {num_kp_src}")
    print(f"Good matches found: {len(good_matches)}")

    # Draw good matches
    match_vis = cv2.drawMatches(
        ref_img,
        kp_ref,
        src_img,
        kp_src,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Save visualization to file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not cv2.imwrite(out_path, match_vis):
        print(f"Error: Failed to save match visualization to '{out_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully saved match visualization to '{out_path}'.")


if __name__ == "__main__":
    main()
