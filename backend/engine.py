"""
Lunar Image Registration - Correspondence Engine Interface & SIFT Baseline
Provides an abstract base class for feature correspondence matching and the concrete
SIFT baseline implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class MatchResult:
    """Standardized representation of pairwise image correspondence matches."""
    reference_points: np.ndarray          # Shape (N, 2), float32: (x, y) coordinates in reference image
    source_points: np.ndarray             # Shape (N, 2), float32: (x, y) coordinates in source image
    confidence_scores: np.ndarray         # Shape (N,), float32: confidence scores in [0.0, 1.0]
    reference_keypoints_count: int        # Total detected keypoints in reference
    source_keypoints_count: int           # Total detected keypoints in source
    candidate_matches: int                # Raw matches before ratio/confidence filtering
    good_matches: int                     # Filtered matches passing ratio/threshold test
    method: str                           # Method identifier (e.g., 'sift')
    lowe_ratios: Optional[np.ndarray] = None  # Shape (N,), float32: d1 / d2 ratio values
    keypoints_ref: Optional[List[Any]] = None  # List of cv2.KeyPoint objects (reference)
    keypoints_src: Optional[List[Any]] = None  # List of cv2.KeyPoint objects (source)
    raw_dmatches: Optional[List[Any]] = None   # Filtered list of cv2.DMatch objects


class CorrespondenceEngine(ABC):
    """Abstract interface defining the contract for all lunar correspondence models."""

    @abstractmethod
    def match(self, reference_gray: np.ndarray, source_gray: np.ndarray) -> MatchResult:
        """Find feature correspondences between reference and source grayscale images.

        Parameters:
            reference_gray: Single-channel uint8 reference image.
            source_gray: Single-channel uint8 source image to be registered.

        Returns:
            MatchResult containing corresponding points, confidences, counts, and metadata.
        """
        pass


class SIFTCorrespondenceEngine(CorrespondenceEngine):
    """Baseline correspondence engine using classical SIFT, BFMatcher (NORM_L2), and Lowe's ratio test."""

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        nfeatures: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ):
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, reference_gray: np.ndarray, source_gray: np.ndarray) -> MatchResult:
        if reference_gray is None or source_gray is None:
            raise ValueError("Both reference and source images must be provided.")

        # 1. Detect SIFT keypoints and extract 128-d descriptors
        kp_ref, des_ref = self.sift.detectAndCompute(reference_gray, None)
        kp_src, des_src = self.sift.detectAndCompute(source_gray, None)

        num_kp_ref = len(kp_ref) if kp_ref else 0
        num_kp_src = len(kp_src) if kp_src else 0

        if des_ref is None or len(des_ref) == 0:
            raise ValueError("No descriptors computed for reference image. Image may be featureless.")

        if des_src is None or len(des_src) == 0:
            raise ValueError("No descriptors computed for source image. Image may be featureless.")

        # 2. k-NN Matching with k=2
        raw_knn_matches = self.matcher.knnMatch(des_ref, des_src, k=2)
        total_candidate_matches = len(raw_knn_matches)

        # 3. Apply Lowe's ratio test & compute confidence score
        # For each pair (m, n), m is best match, n is second-best match.
        # Condition: m.distance < ratio_threshold * n.distance
        # Distinctiveness confidence: 1.0 - (d1 / d2)
        good_dmatches = []
        confidences = []
        ratios = []

        for pair in raw_knn_matches:
            if len(pair) == 2:
                m, n = pair
                if n.distance > 1e-7:
                    ratio = m.distance / n.distance
                else:
                    ratio = 1.0

                if ratio < self.ratio_threshold:
                    good_dmatches.append(m)
                    ratios.append(ratio)
                    # Margin confidence: closer to 0 ratio -> confidence closer to 1.0
                    conf = max(0.0, min(1.0, 1.0 - ratio))
                    confidences.append(conf)

        good_matches_count = len(good_dmatches)

        if good_matches_count == 0:
            ref_pts = np.empty((0, 2), dtype=np.float32)
            src_pts = np.empty((0, 2), dtype=np.float32)
            conf_arr = np.empty((0,), dtype=np.float32)
            ratio_arr = np.empty((0,), dtype=np.float32)
        else:
            ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_dmatches])
            src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_dmatches])
            conf_arr = np.float32(confidences)
            ratio_arr = np.float32(ratios)

        return MatchResult(
            reference_points=ref_pts,
            source_points=src_pts,
            confidence_scores=conf_arr,
            reference_keypoints_count=num_kp_ref,
            source_keypoints_count=num_kp_src,
            candidate_matches=total_candidate_matches,
            good_matches=good_matches_count,
            method="sift",
            lowe_ratios=ratio_arr,
            keypoints_ref=kp_ref,
            keypoints_src=kp_src,
            raw_dmatches=good_dmatches,
        )


def get_correspondence_engine(method: str = "sift", **kwargs) -> CorrespondenceEngine:
    """Factory function to instantiate the desired correspondence engine."""
    clean_method = str(method).strip().lower()
    if clean_method == "sift":
        ratio = kwargs.get("ratio_threshold", 0.75)
        return SIFTCorrespondenceEngine(ratio_threshold=ratio)
    elif clean_method in ("advanced", "learned", "loftr"):
        raise NotImplementedError(
            f"Advanced model '{method}' is not enabled in Phase 1. Use 'sift' baseline."
        )
    else:
        raise ValueError(f"Unknown correspondence method: '{method}'. Supported: ['sift']")
