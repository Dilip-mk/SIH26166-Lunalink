"""
Lunar Image Registration - Preprocessing Module
Provides deterministic, lightweight illumination and contrast enhancement for lunar optical imagery.
"""

from typing import Literal
import cv2
import numpy as np

PreprocessingMode = Literal["raw", "clahe", "normalized"]


def preprocess_image(image: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Preprocess an input lunar image to grayscale with optional illumination normalization.

    Parameters:
        image: np.ndarray
            Input image, either BGR (3 channels) or Grayscale (1 channel), uint8.
        mode: str
            Preprocessing strategy:
            - 'raw': Simple grayscale conversion without contrast modification.
            - 'clahe': Contrast Limited Adaptive Histogram Equalization to enhance local crater contrast.
            - 'normalized': Full dynamic-range min-max normalization to stretch illumination extremes.

    Returns:
        np.ndarray: Single-channel uint8 preprocessed grayscale image.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input: expected non-empty numpy ndarray.")

    if image.size == 0:
        raise ValueError("Invalid image input: image buffer is empty.")

    # 1. Ensure single-channel grayscale
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError(f"Unsupported image dimensions: shape={image.shape}")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clean_mode = str(mode).strip().lower()

    # 2. Apply deterministic enhancement
    if clean_mode == "raw":
        return gray

    elif clean_mode == "clahe":
        # Adaptive histogram equalization with moderate clip limit to avoid noise amplification
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    elif clean_mode == "normalized":
        # Min-Max stretch to span the complete [0, 255] grayscale spectrum
        norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm_img

    else:
        supported = ["raw", "clahe", "normalized"]
        raise ValueError(
            f"Unsupported preprocessing mode '{mode}'. Supported modes: {supported}"
        )
