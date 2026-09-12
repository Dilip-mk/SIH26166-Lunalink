import os
import sys
import argparse
import cv2


def process_image(image_path):
    # Verify file existence
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(
            f"Error: Failed to load image from '{image_path}'. File may be invalid or corrupted.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize OpenCV SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    num_keypoints = len(keypoints) if keypoints is not None else 0

    image_basename = os.path.basename(image_path)
    print(f"Image '{image_basename}': {num_keypoints} keypoints detected.")

    # Draw keypoints on original image
    vis_img = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Save output visualization in the same directory with '_keypoints' suffix
    dir_name, file_name = os.path.split(image_path)
    base_name, ext = os.path.splitext(file_name)
    if not ext:
        ext = ".jpg"
    out_path = os.path.join(dir_name, f"{base_name}_keypoints{ext}")

    if not cv2.imwrite(out_path, vis_img):
        print(f"Error: Failed to save visualization to '{out_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Keypoint visualization saved to '{out_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Detect SIFT keypoints and descriptors for a given image."
    )
    parser.add_argument("image_path", help="Path to the input image file.")

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    process_image(args.image_path)


if __name__ == "__main__":
    main()

