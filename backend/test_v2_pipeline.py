"""
V2 Phase 2 -- Comprehensive Backend Test Suite
Tests modular correspondence architecture with spatial selection strategies:
  engine.py, preprocessing.py, spatial.py, refinement.py, pipeline.py

Usage:
    python test_v2_pipeline.py
"""

import os
import sys
import time
import inspect
import traceback
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
REF_PATH = os.path.join(DATA_DIR, "uploads", "reference.jpg") if os.path.exists(os.path.join(DATA_DIR, "uploads", "reference.jpg")) else os.path.join(DATA_DIR, "reference.jpg")
SRC1_PATH = os.path.join(DATA_DIR, "uploads", "source.jpg") if os.path.exists(os.path.join(DATA_DIR, "uploads", "source.jpg")) else os.path.join(DATA_DIR, "source.jpg")
SRC2_PATH = os.path.join(DATA_DIR, "uploads", "source2.jpg") if os.path.exists(os.path.join(DATA_DIR, "uploads", "source2.jpg")) else os.path.join(DATA_DIR, "source2.jpg")

# --------------------------------------------------------------------------- #
# Result tracking
# --------------------------------------------------------------------------- #
PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: List[Tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    RESULTS.append((name, passed, detail))


# =========================================================================== #
#  1. MODULE IMPORT TESTS
# =========================================================================== #
def test_imports():
    print("\n=== 1. Module Import Tests ===")
    modules = {}

    for mod_name in ["engine", "preprocessing", "spatial", "refinement", "pipeline"]:
        try:
            modules[mod_name] = __import__(mod_name)
            report(f"import {mod_name}", True)
        except Exception as e:
            report(f"import {mod_name}", False, str(e))

    return modules


# =========================================================================== #
#  2. ENGINE TESTS
# =========================================================================== #
def test_engine(modules: dict):
    print("\n=== 2. Engine Module Tests ===")
    engine_mod = modules.get("engine")
    if not engine_mod:
        report("engine module available", False, "import failed")
        return

    has_base = hasattr(engine_mod, "CorrespondenceEngine")
    report("CorrespondenceEngine base class exists", has_base)

    has_sift = hasattr(engine_mod, "SIFTCorrespondenceEngine")
    report("SIFTCorrespondenceEngine exists", has_sift)

    has_mr = hasattr(engine_mod, "MatchResult")
    report("MatchResult dataclass exists", has_mr)

    has_factory = hasattr(engine_mod, "get_correspondence_engine")
    report("get_correspondence_engine factory exists", has_factory)

    if not (has_base and has_sift and has_mr and has_factory):
        return

    is_subclass = issubclass(engine_mod.SIFTCorrespondenceEngine, engine_mod.CorrespondenceEngine)
    report("SIFT inherits CorrespondenceEngine", is_subclass)

    sig = inspect.signature(engine_mod.SIFTCorrespondenceEngine.match)
    params = list(sig.parameters.keys())
    report("match() accepts reference_gray", "reference_gray" in params)
    report("match() accepts source_gray", "source_gray" in params)

    init_sig = inspect.signature(engine_mod.SIFTCorrespondenceEngine.__init__)
    ratio_default = init_sig.parameters.get("ratio_threshold")
    if ratio_default and ratio_default.default == 0.75:
        report("default Lowe ratio = 0.75", True)
    else:
        report("default Lowe ratio = 0.75", False,
               f"got {ratio_default.default if ratio_default else 'missing'}")

    eng = engine_mod.get_correspondence_engine("sift")
    report("factory('sift') returns SIFTCorrespondenceEngine",
           isinstance(eng, engine_mod.SIFTCorrespondenceEngine))

    try:
        engine_mod.get_correspondence_engine("unknown_method")
        report("factory rejects unknown method", False, "no exception raised")
    except (ValueError, NotImplementedError):
        report("factory rejects unknown method", True)

    if not os.path.exists(REF_PATH) or not os.path.exists(SRC1_PATH):
        report("engine matching (source.jpg)", False, "test images missing")
        return

    ref_gray = cv2.imread(REF_PATH, cv2.IMREAD_GRAYSCALE)
    src_gray = cv2.imread(SRC1_PATH, cv2.IMREAD_GRAYSCALE)
    result = eng.match(reference_gray=ref_gray, source_gray=src_gray)

    report("MatchResult.reference_points shape",
           isinstance(result.reference_points, np.ndarray) and result.reference_points.ndim == 2)
    report("MatchResult.source_points shape",
           isinstance(result.source_points, np.ndarray) and result.source_points.ndim == 2)
    report("MatchResult.confidence_scores exist",
           isinstance(result.confidence_scores, np.ndarray) and len(result.confidence_scores) > 0)
    report("confidence scores in [0, 1]",
           bool(np.all(result.confidence_scores >= 0) and np.all(result.confidence_scores <= 1)))
    report("reference_keypoints_count > 0", result.reference_keypoints_count > 0)
    report("source_keypoints_count > 0", result.source_keypoints_count > 0)
    report("candidate_matches > 0", result.candidate_matches > 0)
    report("good_matches > 0", result.good_matches > 0)
    report("lowe_ratios preserved",
           result.lowe_ratios is not None and len(result.lowe_ratios) == result.good_matches)
    report("method = 'sift'", result.method == "sift")


# =========================================================================== #
#  3. PREPROCESSING TESTS
# =========================================================================== #
def test_preprocessing(modules: dict):
    print("\n=== 3. Preprocessing Module Tests ===")
    prep_mod = modules.get("preprocessing")
    if not prep_mod:
        report("preprocessing module available", False, "import failed")
        return

    has_func = hasattr(prep_mod, "preprocess_image")
    report("preprocess_image function exists", has_func)
    if not has_func:
        return

    test_img = cv2.imread(REF_PATH)
    if test_img is None:
        report("test image loads", False, "reference.jpg not found")
        return

    for mode in ["raw", "clahe", "normalized"]:
        try:
            result = prep_mod.preprocess_image(test_img, mode=mode)
            is_gray = result.ndim == 2
            is_uint8 = result.dtype == np.uint8
            report(f"mode='{mode}' returns grayscale uint8",
                   is_gray and is_uint8,
                   f"shape={result.shape}, dtype={result.dtype}")
        except Exception as e:
            report(f"mode='{mode}' works", False, str(e))

    r1 = prep_mod.preprocess_image(test_img, mode="clahe")
    r2 = prep_mod.preprocess_image(test_img, mode="clahe")
    report("CLAHE is deterministic", bool(np.array_equal(r1, r2)))

    try:
        prep_mod.preprocess_image(test_img, mode="invalid_mode")
        report("rejects invalid mode", False, "no exception")
    except ValueError:
        report("rejects invalid mode", True)

    try:
        prep_mod.preprocess_image(None)
        report("rejects None input", False, "no exception")
    except (ValueError, TypeError):
        report("rejects None input", True)


# =========================================================================== #
#  4. SPATIAL SELECTION TESTS (Phase 2)
# =========================================================================== #
def test_spatial(modules: dict):
    print("\n=== 4. Spatial Selection Module Tests ===")
    spatial_mod = modules.get("spatial")
    if not spatial_mod:
        report("spatial module available", False, "import failed")
        return

    has_func = hasattr(spatial_mod, "select_uniform_correspondences")
    report("select_uniform_correspondences exists", has_func)
    if not has_func:
        return

    # Check strategy parameter exists
    sig = inspect.signature(spatial_mod.select_uniform_correspondences)
    params = list(sig.parameters.keys())
    report("accepts spatial_strategy param", "spatial_strategy" in params)
    report("accepts target_total_matches param", "target_total_matches" in params)
    report("accepts max_per_cell param", "max_per_cell" in params)

    # Check SpatialDistributionResult has strategy field
    has_result = hasattr(spatial_mod, "SpatialDistributionResult")
    report("SpatialDistributionResult exists", has_result)
    if has_result:
        fields = [f.name for f in spatial_mod.SpatialDistributionResult.__dataclass_fields__.values()]
        report("SpatialDistributionResult has 'strategy'", "strategy" in fields)

    # Create synthetic correspondences spread across image
    np.random.seed(42)
    n = 200
    image_shape = (400, 600)
    ref_pts = np.random.rand(n, 2).astype(np.float32)
    ref_pts[:, 0] *= image_shape[1]
    ref_pts[:, 1] *= image_shape[0]
    src_pts = ref_pts + np.random.randn(n, 2).astype(np.float32) * 5
    confidences = np.random.rand(n).astype(np.float32)

    # --- Strategy: none (pass-through) ---
    res_none = spatial_mod.select_uniform_correspondences(
        ref_points=ref_pts, source_points=src_pts, confidences=confidences,
        image_shape=image_shape, grid_rows=4, grid_cols=4,
        spatial_strategy="none",
    )
    report("strategy='none': returns all points", len(res_none.selected_ref_points) == n)
    report("strategy='none': strategy field", res_none.strategy == "none")

    # --- Strategy: grid (fixed cap) ---
    res_grid = spatial_mod.select_uniform_correspondences(
        ref_points=ref_pts, source_points=src_pts, confidences=confidences,
        image_shape=image_shape, grid_rows=4, grid_cols=4,
        max_per_cell=5, spatial_strategy="grid",
    )
    report("strategy='grid': reduces points",
           len(res_grid.selected_ref_points) < n,
           f"selected={len(res_grid.selected_ref_points)}")
    report("strategy='grid': max per cell <= 5 x 16 = 80",
           len(res_grid.selected_ref_points) <= 80)
    report("strategy='grid': strategy field", res_grid.strategy == "grid")

    # --- Strategy: grid with default cap (no max_per_cell specified) ---
    res_grid_default = spatial_mod.select_uniform_correspondences(
        ref_points=ref_pts, source_points=src_pts, confidences=confidences,
        image_shape=image_shape, grid_rows=4, grid_cols=4,
        spatial_strategy="grid",
    )
    report("strategy='grid' default cap: reduces points",
           len(res_grid_default.selected_ref_points) < n,
           f"selected={len(res_grid_default.selected_ref_points)}")

    # --- Strategy: adaptive_grid ---
    res_adaptive = spatial_mod.select_uniform_correspondences(
        ref_points=ref_pts, source_points=src_pts, confidences=confidences,
        image_shape=image_shape, grid_rows=4, grid_cols=4,
        spatial_strategy="adaptive_grid", target_total_matches=50,
    )
    report("strategy='adaptive_grid': reduces points",
           len(res_adaptive.selected_ref_points) <= n,
           f"selected={len(res_adaptive.selected_ref_points)}")
    # Should be roughly near 50 (may be less if cells have fewer points)
    report("strategy='adaptive_grid': near target (<=60)",
           len(res_adaptive.selected_ref_points) <= 60,
           f"selected={len(res_adaptive.selected_ref_points)}")
    report("strategy='adaptive_grid': strategy field", res_adaptive.strategy == "adaptive_grid")

    # --- Empty input ---
    res_empty = spatial_mod.select_uniform_correspondences(
        ref_points=np.empty((0, 2), dtype=np.float32),
        source_points=np.empty((0, 2), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        image_shape=image_shape, spatial_strategy="grid",
    )
    report("empty input handled gracefully", len(res_empty.selected_ref_points) == 0)

    # --- Confidence preference test ---
    cluster_pts = np.full((50, 2), [50.0, 50.0], dtype=np.float32)
    cluster_src = cluster_pts.copy()
    cluster_conf = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    res_conf = spatial_mod.select_uniform_correspondences(
        ref_points=cluster_pts, source_points=cluster_src, confidences=cluster_conf,
        image_shape=image_shape, grid_rows=4, grid_cols=4, max_per_cell=3,
        spatial_strategy="grid",
    )
    if len(res_conf.selected_confidences) > 0:
        report("selects high-confidence points",
               float(res_conf.selected_confidences.min()) >= 0.9,
               f"min_conf={float(res_conf.selected_confidences.min()):.3f}")
    else:
        report("selects high-confidence points", False, "no points selected")

    # --- Invalid strategy rejection ---
    try:
        spatial_mod.select_uniform_correspondences(
            ref_points=ref_pts, source_points=src_pts, confidences=confidences,
            image_shape=image_shape, spatial_strategy="invalid_strategy",
        )
        report("rejects invalid strategy", False, "no exception")
    except ValueError:
        report("rejects invalid strategy", True)

    # --- Coverage preserved after selection ---
    report("total_cells = 16 (grid)", res_grid.total_cells == 16)
    report("occupied_cells preserved (grid)",
           res_grid.occupied_cells > 0 and res_grid.occupied_cells <= res_none.occupied_cells)

    # --- draw_spatial_selection_visualization exists ---
    has_sel_vis = hasattr(spatial_mod, "draw_spatial_selection_visualization")
    report("draw_spatial_selection_visualization exists", has_sel_vis)


# =========================================================================== #
#  5. REFINEMENT TESTS
# =========================================================================== #
def test_refinement(modules: dict):
    print("\n=== 5. Refinement Module Tests ===")
    ref_mod = modules.get("refinement")
    if not ref_mod:
        report("refinement module available", False, "import failed")
        return

    has_refine = hasattr(ref_mod, "refine_subpixel_correspondences")
    has_rmse = hasattr(ref_mod, "compute_reprojection_rmse")
    report("refine_subpixel_correspondences exists", has_refine)
    report("compute_reprojection_rmse exists", has_rmse)

    if not has_refine:
        return

    has_result = hasattr(ref_mod, "RefinementResult")
    report("RefinementResult dataclass exists", has_result)

    if has_result:
        fields = [f.name for f in ref_mod.RefinementResult.__dataclass_fields__.values()]
        required_fields = ["original_ref_points", "original_src_points",
                           "refined_ref_points", "refined_src_points",
                           "initial_rmse", "refined_rmse"]
        for f in required_fields:
            report(f"RefinementResult has '{f}'", f in fields)

    ref_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    src_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    few_pts = np.float32([[30, 30], [50, 50]])
    H_identity = np.eye(3, dtype=np.float64)

    try:
        res = ref_mod.refine_subpixel_correspondences(
            ref_gray=ref_gray, src_gray=src_gray,
            inlier_ref_points=few_pts, inlier_src_points=few_pts,
            initial_H=H_identity,
        )
        report("handles < 4 points gracefully", True)
        report("returns original when refinement skipped",
               np.array_equal(res.original_ref_points, res.refined_ref_points))
    except Exception as e:
        report("handles < 4 points gracefully", False, str(e))


# =========================================================================== #
#  6. PIPELINE INTEGRATION TESTS
# =========================================================================== #
def test_pipeline_integration(modules: dict):
    print("\n=== 6. Pipeline Integration Tests ===")
    pipe_mod = modules.get("pipeline")
    if not pipe_mod:
        report("pipeline module available", False, "import failed")
        return

    has_run = hasattr(pipe_mod, "run_pipeline")
    report("run_pipeline function exists", has_run)
    if not has_run:
        return

    # Verify no hardcoded reference path in source
    src_path = os.path.join(SCRIPT_DIR, "pipeline.py")
    if os.path.exists(src_path):
        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = f.readlines()
        has_hardcoded = False
        for line in src_lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("print(") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if 'data/reference.jpg' in line or 'data\\reference.jpg' in line:
                has_hardcoded = True
                break
        report("no hardcoded reference path in pipeline.py", not has_hardcoded)
    else:
        report("pipeline.py file found", False)

    # Verify signature
    sig = inspect.signature(pipe_mod.run_pipeline)
    params = list(sig.parameters.keys())
    report("accepts reference_path param", "reference_path" in params)
    report("accepts source_path param", "source_path" in params)
    report("accepts method param", "method" in params)
    report("accepts preprocessing param", "preprocessing" in params)
    report("accepts spatial_strategy param", "spatial_strategy" in params)
    report("accepts max_matches_per_cell param", "max_matches_per_cell" in params)
    report("accepts target_total_matches param", "target_total_matches" in params)

    # Check defaults
    p = sig.parameters
    report("default method = 'sift'", p.get("method") and p["method"].default == "sift")
    report("default preprocessing = 'raw'", p.get("preprocessing") and p["preprocessing"].default == "raw")
    report("default spatial_strategy = 'grid'", p.get("spatial_strategy") and p["spatial_strategy"].default == "grid")

    # Error: missing reference path
    try:
        pipe_mod.run_pipeline(source_path=SRC1_PATH, reference_path=None)
        report("rejects None reference_path", False, "no exception")
    except (ValueError, FileNotFoundError):
        report("rejects None reference_path", True)

    # Error: nonexistent file
    try:
        pipe_mod.run_pipeline(
            source_path=SRC1_PATH,
            reference_path="nonexistent_image_xyz.jpg"
        )
        report("rejects nonexistent reference file", False, "no exception")
    except FileNotFoundError:
        report("rejects nonexistent reference file", True)

    # Error: unknown method
    try:
        pipe_mod.run_pipeline(
            source_path=SRC1_PATH,
            reference_path=REF_PATH,
            method="unknown_method_xyz"
        )
        report("rejects unknown method", False, "no exception")
    except (ValueError, NotImplementedError):
        report("rejects unknown method", True)


# =========================================================================== #
#  7. FULL PIPELINE RUNS
# =========================================================================== #

REQUIRED_METRICS = [
    "reference_keypoints", "source_keypoints", "good_matches",
    "ransac_inliers", "ransac_outliers", "inlier_ratio", "rmse",
    "spatial_coverage", "occupied_grid_cells", "total_grid_cells",
    "method", "preprocessing", "candidate_matches", "selected_matches",
    "initial_rmse", "refined_rmse", "processing_time",
    # Phase 2 metrics
    "spatial_strategy", "max_matches_per_cell",
    "occupied_grid_cells_before", "occupied_grid_cells_after",
    "spatial_coverage_before", "spatial_coverage_after",
]

REQUIRED_OUTPUTS = [
    "registered_image", "overlay_image", "ransac_visualization",
    "spatial_selection_visualization",
]


def run_full_pipeline_test(
    label: str,
    ref_path: str,
    src_path: str,
    method: str = "sift",
    preprocessing: str = "raw",
    spatial_strategy: str = "grid",
) -> Dict[str, Any]:
    """Run the full pipeline and validate outputs."""
    print(f"\n=== 7. Full Pipeline Run: {label} ===")
    pipe = __import__("pipeline")

    if not os.path.exists(ref_path):
        report(f"{label}: reference exists", False, ref_path)
        return {}
    if not os.path.exists(src_path):
        report(f"{label}: source exists", False, src_path)
        return {}

    report(f"{label}: reference exists", True)
    report(f"{label}: source exists", True)

    try:
        t0 = time.perf_counter()
        results = pipe.run_pipeline(
            source_path=src_path,
            reference_path=ref_path,
            method=method,
            preprocessing=preprocessing,
            spatial_strategy=spatial_strategy,
        )
        elapsed = time.perf_counter() - t0
        report(f"{label}: pipeline executed successfully", True, f"{elapsed:.3f}s")
    except Exception as e:
        report(f"{label}: pipeline executed successfully", False, str(e))
        traceback.print_exc()
        return {}

    # Check all required metric keys
    for key in REQUIRED_METRICS:
        present = key in results
        val = results.get(key, "MISSING")
        report(f"{label}: metric '{key}' present", present, f"value={val}")

    # Check output files exist on disk
    for key in REQUIRED_OUTPUTS:
        if key in results:
            exists = os.path.isfile(results[key])
            report(f"{label}: output '{key}' file exists", exists, results[key])
        else:
            report(f"{label}: output '{key}' key present", False)

    # Validate metric sanity
    if results:
        r = results
        report(f"{label}: reference_keypoints > 0", r.get("reference_keypoints", 0) > 0)
        report(f"{label}: source_keypoints > 0", r.get("source_keypoints", 0) > 0)
        report(f"{label}: good_matches > 0", r.get("good_matches", 0) > 0)
        report(f"{label}: ransac_inliers > 0", r.get("ransac_inliers", 0) > 0)
        report(f"{label}: inlier_ratio in (0, 100]",
               0 < r.get("inlier_ratio", 0) <= 100,
               f"{r.get('inlier_ratio', 0):.2f}%")
        report(f"{label}: rmse >= 0", r.get("rmse", -1) >= 0,
               f"{r.get('rmse', 0):.4f} px")
        report(f"{label}: initial_rmse >= 0", r.get("initial_rmse", -1) >= 0)
        report(f"{label}: refined_rmse >= 0", r.get("refined_rmse", -1) >= 0)
        report(f"{label}: spatial_coverage in [0, 100]",
               0 <= r.get("spatial_coverage", -1) <= 100)
        report(f"{label}: processing_time > 0", r.get("processing_time", 0) > 0)
        report(f"{label}: method = '{method}'", r.get("method") == method)
        report(f"{label}: preprocessing = '{preprocessing}'",
               r.get("preprocessing") == preprocessing)
        report(f"{label}: candidate_matches >= good_matches",
               r.get("candidate_matches", 0) >= r.get("good_matches", 0))

        # Phase 2: spatial selection actually reduced matches
        report(f"{label}: selected_matches < good_matches",
               r.get("selected_matches", 0) < r.get("good_matches", 0),
               f"selected={r.get('selected_matches', 0)}, good={r.get('good_matches', 0)}")

        # Phase 2: coverage preserved (not worse than before)
        report(f"{label}: coverage preserved after selection",
               r.get("spatial_coverage_after", 0) >= r.get("spatial_coverage_before", 0) - 0.01,
               f"before={r.get('spatial_coverage_before', 0):.1f}%, after={r.get('spatial_coverage_after', 0):.1f}%")

    return results


# =========================================================================== #
#  8. BACKWARD COMPATIBILITY TESTS
# =========================================================================== #
def test_backward_compatibility(modules: dict):
    print("\n=== 8. Backward Compatibility Tests ===")
    pipe_mod = modules.get("pipeline")
    if not pipe_mod:
        report("pipeline module available", False)
        return

    if not os.path.exists(REF_PATH) or not os.path.exists(SRC1_PATH):
        report("test images available", False)
        return

    # V1-style call with 'none' strategy reproduces original behavior
    try:
        results_none = pipe_mod.run_pipeline(
            source_path=SRC1_PATH,
            reference_path=REF_PATH,
            spatial_strategy="none",
        )
        report("strategy='none' succeeds", True)
        report("strategy='none' passes all matches",
               results_none.get("selected_matches") == results_none.get("good_matches"),
               f"selected={results_none.get('selected_matches')}, good={results_none.get('good_matches')}")
        report("defaults to method='sift'", results_none.get("method") == "sift")
        report("defaults to preprocessing='raw'", results_none.get("preprocessing") == "raw")

        v1_keys = [
            "reference_keypoints", "source_keypoints", "good_matches",
            "ransac_inliers", "ransac_outliers", "inlier_ratio", "rmse",
            "spatial_coverage", "occupied_grid_cells", "total_grid_cells",
            "registered_image", "overlay_image", "ransac_visualization",
        ]
        all_present = all(k in results_none for k in v1_keys)
        report("all V1 metric keys present", all_present)
    except Exception as e:
        report("strategy='none' succeeds", False, str(e))

    # Default call (uses 'grid' strategy)
    try:
        results_default = pipe_mod.run_pipeline(
            source_path=SRC1_PATH,
            reference_path=REF_PATH,
        )
        report("default call (grid strategy) succeeds", True)
        report("default spatial_strategy = 'grid'",
               results_default.get("spatial_strategy") == "grid")
    except Exception as e:
        report("default call (grid strategy) succeeds", False, str(e))


# =========================================================================== #
#  9. COMPARISON: BEFORE vs AFTER (Phase 2 key test)
# =========================================================================== #
def test_before_after_comparison(modules: dict):
    print("\n=== 9. Before/After Comparison ===")
    pipe_mod = modules.get("pipeline")
    if not pipe_mod:
        report("pipeline module available", False)
        return

    for src_name, src_path in [("source.jpg", SRC1_PATH), ("source2.jpg", SRC2_PATH)]:
        if not os.path.exists(src_path):
            report(f"{src_name}: available", False)
            continue

        try:
            before = pipe_mod.run_pipeline(
                source_path=src_path, reference_path=REF_PATH,
                spatial_strategy="none",
            )
            after = pipe_mod.run_pipeline(
                source_path=src_path, reference_path=REF_PATH,
                spatial_strategy="grid",
            )

            b_sel = before.get("selected_matches", 0)
            a_sel = after.get("selected_matches", 0)
            report(f"{src_name}: grid reduced selected matches",
                   a_sel < b_sel,
                   f"none={b_sel}, grid={a_sel}")

            b_cov = before.get("spatial_coverage_before", 0)
            a_cov = after.get("spatial_coverage_after", 0)
            report(f"{src_name}: coverage preserved",
                   a_cov >= b_cov - 0.01,
                   f"before={b_cov:.1f}%, after={a_cov:.1f}%")

            b_rmse = before.get("initial_rmse", 999)
            a_rmse = after.get("initial_rmse", 999)
            report(f"{src_name}: initial RMSE",
                   a_rmse >= 0,
                   f"none={b_rmse:.4f}px, grid={a_rmse:.4f}px")

        except Exception as e:
            report(f"{src_name}: comparison", False, str(e))


# =========================================================================== #
#  MAIN TEST RUNNER
# =========================================================================== #
def main():
    print("=" * 70)
    print("  V2 PHASE 2 -- BACKEND TEST SUITE")
    print("=" * 70)
    print(f"Python:  {sys.version}")
    print(f"OpenCV:  {cv2.__version__}")
    print(f"NumPy:   {np.__version__}")
    print(f"CWD:     {os.getcwd()}")
    print(f"Data:    {DATA_DIR}")
    print(f"Ref:     {REF_PATH}  (exists={os.path.exists(REF_PATH)})")
    print(f"Src1:    {SRC1_PATH}  (exists={os.path.exists(SRC1_PATH)})")
    print(f"Src2:    {SRC2_PATH}  (exists={os.path.exists(SRC2_PATH)})")
    print("=" * 70)

    modules = test_imports()
    test_engine(modules)
    test_preprocessing(modules)
    test_spatial(modules)
    test_refinement(modules)
    test_pipeline_integration(modules)

    # Full pipeline runs with grid strategy (new default)
    results_src1 = run_full_pipeline_test(
        "source.jpg [grid]", REF_PATH, SRC1_PATH,
        method="sift", preprocessing="raw", spatial_strategy="grid")

    results_src2 = run_full_pipeline_test(
        "source2.jpg [grid]", REF_PATH, SRC2_PATH,
        method="sift", preprocessing="raw", spatial_strategy="grid")

    test_backward_compatibility(modules)
    test_before_after_comparison(modules)

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"  PASSED: {PASS_COUNT}")
    print(f"  FAILED: {FAIL_COUNT}")
    print(f"  TOTAL:  {PASS_COUNT + FAIL_COUNT}")
    print("=" * 70)

    if FAIL_COUNT > 0:
        print("\n  FAILED TESTS:")
        for name, passed, detail in RESULTS:
            if not passed:
                print(f"    X {name}  --  {detail}")

    # ---- Metrics Summary ----
    phase2_keys = [
        "good_matches", "selected_matches", "ransac_inliers", "ransac_outliers",
        "inlier_ratio", "initial_rmse", "refined_rmse", "rmse",
        "spatial_coverage_before", "spatial_coverage_after", "spatial_coverage",
        "occupied_grid_cells_before", "occupied_grid_cells_after", "occupied_grid_cells",
        "spatial_strategy", "max_matches_per_cell", "processing_time",
    ]

    if results_src1:
        print("\n--- source.jpg [grid] Metrics ---")
        for k in phase2_keys:
            v = results_src1.get(k, "N/A")
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if results_src2:
        print("\n--- source2.jpg [grid] Metrics ---")
        for k in phase2_keys:
            v = results_src2.get(k, "N/A")
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print()
    if FAIL_COUNT == 0:
        print("All tests passed. V2 Phase 2 spatial selection is verified.")
    else:
        print(f"{FAIL_COUNT} test(s) failed. Review failures above.")

    return FAIL_COUNT


if __name__ == "__main__":
    sys.exit(main())
