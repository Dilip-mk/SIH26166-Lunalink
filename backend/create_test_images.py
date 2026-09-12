import os
import sys
from PIL import Image, ImageEnhance


def create_test_images():
    # Determine directory relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.normpath(os.path.join(script_dir, "..", "data", "reference.jpg"))
    src_path = os.path.normpath(os.path.join(script_dir, "..", "data", "source.jpg"))

    if not os.path.exists(ref_path):
        print(f"Error: Reference image not found at '{ref_path}'.", file=sys.stderr)
        sys.exit(1)

    try:
        with Image.open(ref_path) as img:
            # Convert to RGB mode if needed for JPEG saving
            img = img.convert("RGB")

            # 1. Scale to 80%
            width, height = img.size
            new_width = max(1, int(width * 0.8))
            new_height = max(1, int(height * 0.8))
            resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
            scaled_img = img.resize((new_width, new_height), resample_filter)

            # 2. Rotate by 12 degrees
            rotated_img = scaled_img.rotate(12, expand=True)

            # 3. Slightly change brightness (factor 1.10)
            enhancer = ImageEnhance.Brightness(rotated_img)
            transformed_img = enhancer.enhance(1.10)

            # Ensure image is RGB before JPEG save
            if transformed_img.mode != "RGB":
                transformed_img = transformed_img.convert("RGB")

            # Save transformed image
            os.makedirs(os.path.dirname(src_path), exist_ok=True)
            transformed_img.save(src_path, "JPEG", quality=95)
            print(f"Successfully saved transformed image to '{src_path}'.")

    except FileNotFoundError:
        print(f"Error: Reference image not found at '{ref_path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing reference image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    create_test_images()
