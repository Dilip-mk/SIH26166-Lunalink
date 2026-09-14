"""
Lunar Image Registration - Spatially Uniform Correspondence Selection
Divides the reference image into an adaptive regular grid and selects a balanced distribution
of high-confidence correspondences to prevent local textured craters from dominating the geometry.

Supports three spatial strategies:
  - "none":           Pass all matches through unchanged.
  - "grid":           Fixed max_per_cell cap across a uniform grid.
  - "adaptive_grid":  Automatically compute a balanced per-cell budget from a target_total_matches
                      goal, distributing budget across occupied cells proportionally and capping
                      dense cells while keeping sparse cells intact.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class SpatialDistributionResult:
    selected_ref_points: np.ndarray        # Shape (K, 2), float32
    selected_source_points: np.ndarray     # Shape (K, 2), float32
    selected_confidences: np.ndarray       # Shape (K,), float32
    selected_indices: np.ndarray           # Indices relative to the input array
    occupied_cells: int                    # Count of grid cells containing >= 1 match
    total_cells: int                       # Total grid cells (rows * cols)
    spatial_coverage: float                # (occupied_cells / total_cells) * 100.0
    cell_counts: Dict[Tuple[int, int], int]  # Map of (row, col) -> match count
    strategy: str = "none"                 # Strategy used for selection


def _assign_to_grid(
    ref_points: np.ndarray,
    image_shape: Tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> Tuple[Dict[Tuple[int, int], List[int]], float, float]:
    """Assign each correspondence to a grid cell based on its reference-image coordinate.

    Returns:
        (grid_buckets, cell_w, cell_h)
    """
    h_ref, w_ref = image_shape[:2]
    cell_w = max(1.0, float(w_ref) / float(grid_cols))
    cell_h = max(1.0, float(h_ref) / float(grid_rows))

    grid_buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(ref_points):
        c = min(grid_cols - 1, max(0, int(x / cell_w)))
        r = min(grid_rows - 1, max(0, int(y / cell_h)))
        cell = (r, c)
        if cell not in grid_buckets:
            grid_buckets[cell] = []
        grid_buckets[cell].append(idx)

    return grid_buckets, cell_w, cell_h


def _build_empty_result(total_cells: int, strategy: str) -> SpatialDistributionResult:
    """Return an empty SpatialDistributionResult."""
    return SpatialDistributionResult(
        selected_ref_points=np.empty((0, 2), dtype=np.float32),
        selected_source_points=np.empty((0, 2), dtype=np.float32),
        selected_confidences=np.empty((0,), dtype=np.float32),
        selected_indices=np.empty((0,), dtype=np.int32),
        occupied_cells=0,
        total_cells=total_cells,
        spatial_coverage=0.0,
        cell_counts={},
        strategy=strategy,
    )


def _select_top_k_per_cell(
    grid_buckets: Dict[Tuple[int, int], List[int]],
    confidences: np.ndarray,
    per_cell_budgets: Dict[Tuple[int, int], int],
    min_required_matches: int,
    n_points: int,
) -> Tuple[List[int], Dict[Tuple[int, int], int]]:
    """Within each cell, pick the top-K highest-confidence indices according to per_cell_budgets.

    Returns:
        (selected_indices_list, cell_counts)
    """
    selected_indices_list: List[int] = []
    cell_counts: Dict[Tuple[int, int], int] = {}

    for cell, idxs in grid_buckets.items():
        budget = per_cell_budgets.get(cell, len(idxs))
        sorted_idxs = sorted(idxs, key=lambda i: confidences[i], reverse=True)
        chosen = sorted_idxs[:budget]
        selected_indices_list.extend(chosen)
        cell_counts[cell] = len(chosen)

    # Safety: ensure at least min_required_matches for downstream RANSAC
    if len(selected_indices_list) < min_required_matches and n_points >= min_required_matches:
        all_sorted = sorted(range(n_points), key=lambda i: confidences[i], reverse=True)
        selected_indices_list = all_sorted[:min_required_matches]

    return selected_indices_list, cell_counts


def select_uniform_correspondences(
    ref_points: np.ndarray,
    source_points: np.ndarray,
    confidences: np.ndarray,
    image_shape: Tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_per_cell: Optional[int] = None,
    target_total_matches: Optional[int] = None,
    spatial_strategy: str = "grid",
    min_required_matches: int = 4,
) -> SpatialDistributionResult:
    """Group correspondences into spatial grid cells and select top-confidence matches per cell.

    Parameters:
        ref_points: (N, 2) float32 coordinates in reference image frame.
        source_points: (N, 2) float32 coordinates in source image frame.
        confidences: (N,) float32 confidence scores for each match.
        image_shape: (height, width) of the reference image.
        grid_rows: Number of grid subdivisions vertically (default: 4).
        grid_cols: Number of grid subdivisions horizontally (default: 4).
        max_per_cell: Hard maximum matches to retain per cell.
                      Used by both "grid" and "adaptive_grid" strategies.
                      If None under "grid" strategy, a default cap is computed.
        target_total_matches: Desired total number of selected matches.
                              Used only by "adaptive_grid" strategy.
                              If None under "adaptive_grid", defaults to 50% of input matches.
        spatial_strategy: Selection strategy.
                          "none"           - pass all matches through unchanged.
                          "grid"           - fixed max_per_cell cap per grid cell.
                          "adaptive_grid"  - compute per-cell budget from target_total_matches.
        min_required_matches: Minimum number of matches needed for downstream RANSAC.

    Returns:
        SpatialDistributionResult containing selected correspondences and coverage metrics.
    """
    n_points = len(ref_points)
    total_cells = grid_rows * grid_cols
    strategy = str(spatial_strategy).strip().lower()

    if strategy not in ("none", "grid", "adaptive_grid"):
        raise ValueError(
            f"Unknown spatial_strategy '{spatial_strategy}'. "
            f"Supported: ['none', 'grid', 'adaptive_grid']"
        )

    if n_points == 0:
        return _build_empty_result(total_cells, strategy)

    # ---- Strategy: none ----
    if strategy == "none":
        grid_buckets, _, _ = _assign_to_grid(ref_points, image_shape, grid_rows, grid_cols)
        occupied_count = len(grid_buckets)
        cell_counts = {cell: len(idxs) for cell, idxs in grid_buckets.items()}
        return SpatialDistributionResult(
            selected_ref_points=ref_points,
            selected_source_points=source_points,
            selected_confidences=confidences,
            selected_indices=np.arange(n_points, dtype=np.int32),
            occupied_cells=occupied_count,
            total_cells=total_cells,
            spatial_coverage=(occupied_count / total_cells) * 100.0,
            cell_counts=cell_counts,
            strategy=strategy,
        )

    # ---- Assign correspondences to grid cells ----
    grid_buckets, _, _ = _assign_to_grid(ref_points, image_shape, grid_rows, grid_cols)
    occupied_count = len(grid_buckets)

    # ---- Strategy: grid ----
    if strategy == "grid":
        if max_per_cell is None:
            # Compute a sensible default: distribute evenly across occupied cells
            # Target retaining ~30% of matches or at least 8 per cell
            default_target = max(min_required_matches * 4, int(n_points * 0.30))
            computed_cap = max(1, default_target // max(1, occupied_count))
            effective_cap = computed_cap
        else:
            effective_cap = max(1, int(max_per_cell))

        per_cell_budgets = {cell: effective_cap for cell in grid_buckets}

    # ---- Strategy: adaptive_grid ----
    elif strategy == "adaptive_grid":
        if target_total_matches is not None:
            target = max(min_required_matches, int(target_total_matches))
        else:
            # Default: 50% of good matches, minimum 16
            target = max(min_required_matches * 4, int(n_points * 0.50))

        if occupied_count == 0:
            return _build_empty_result(total_cells, strategy)

        # Equal base budget across occupied cells
        base_budget = max(1, target // occupied_count)

        # Compute per-cell budgets, capped by actual cell population
        per_cell_budgets = {}
        remaining = target
        # First pass: assign base budget or actual count (whichever is smaller)
        underfilled_cells = []
        for cell, idxs in grid_buckets.items():
            allocated = min(base_budget, len(idxs))
            per_cell_budgets[cell] = allocated
            remaining -= allocated
            if len(idxs) > allocated:
                underfilled_cells.append(cell)

        # Second pass: redistribute remaining budget to cells that have surplus matches
        # Prioritize cells with higher average confidence (more reliable region)
        if remaining > 0 and underfilled_cells:
            cell_avg_conf = []
            for cell in underfilled_cells:
                idxs = grid_buckets[cell]
                avg_c = float(np.mean(confidences[idxs]))
                headroom = len(idxs) - per_cell_budgets[cell]
                cell_avg_conf.append((cell, avg_c, headroom))
            # Sort by avg confidence descending
            cell_avg_conf.sort(key=lambda x: x[1], reverse=True)

            for cell, _, headroom in cell_avg_conf:
                if remaining <= 0:
                    break
                extra = min(remaining, headroom)
                per_cell_budgets[cell] += extra
                remaining -= extra

        # Apply hard max_per_cell cap if specified
        if max_per_cell is not None:
            hard_cap = max(1, int(max_per_cell))
            for cell in per_cell_budgets:
                per_cell_budgets[cell] = min(per_cell_budgets[cell], hard_cap)

    # ---- Execute top-K selection per cell ----
    selected_indices_list, cell_counts = _select_top_k_per_cell(
        grid_buckets, confidences, per_cell_budgets, min_required_matches, n_points,
    )

    # De-duplicate and sort
    selected_arr = np.array(sorted(set(selected_indices_list)), dtype=np.int32)
    sel_ref = ref_points[selected_arr]
    sel_src = source_points[selected_arr]
    sel_conf = confidences[selected_arr]

    # Recount occupied cells after selection
    selected_occupied = len({cell for cell, cnt in cell_counts.items() if cnt > 0})
    selected_coverage = (selected_occupied / total_cells) * 100.0

    return SpatialDistributionResult(
        selected_ref_points=sel_ref,
        selected_source_points=sel_src,
        selected_confidences=sel_conf,
        selected_indices=selected_arr,
        occupied_cells=selected_occupied,
        total_cells=total_cells,
        spatial_coverage=selected_coverage,
        cell_counts=cell_counts,
        strategy=strategy,
    )


def draw_spatial_grid_visualization(
    ref_image: np.ndarray,
    ref_points: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    inlier_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a visual representation of spatial grid distribution and match density."""
    vis = ref_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    cell_w = w / float(grid_cols)
    cell_h = h / float(grid_rows)

    # Draw grid lines
    grid_color = (80, 80, 80)
    for c in range(1, grid_cols):
        x = int(c * cell_w)
        cv2.line(vis, (x, 0), (x, h), grid_color, 1)
    for r in range(1, grid_rows):
        y = int(r * cell_h)
        cv2.line(vis, (0, y), (w, y), grid_color, 1)

    # Draw points
    for i, pt in enumerate(ref_points):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if inlier_mask is not None and i < len(inlier_mask) and inlier_mask[i] == 1:
            color = (0, 230, 115)  # Green for inlier
            radius = 3
        else:
            color = (0, 165, 255)  # Orange for candidate/outlier
            radius = 2
        cv2.circle(vis, (x, y), radius, color, -1)

    return vis


def draw_spatial_selection_visualization(
    ref_image: np.ndarray,
    all_ref_points: np.ndarray,
    selected_ref_points: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    cell_counts: Optional[Dict[Tuple[int, int], int]] = None,
) -> np.ndarray:
    """Render a detailed visualization showing ALL candidates vs SELECTED correspondences
    on the reference image with grid overlay and per-cell counts.

    - Gray dots: candidate correspondences (not selected)
    - Cyan/green dots: selected correspondences
    - Per-cell count labels in each grid cell
    """
    vis = ref_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    cell_w_px = w / float(grid_cols)
    cell_h_px = h / float(grid_rows)

    # Semi-transparent overlay for grid
    overlay = vis.copy()

    # Draw grid lines (white, semi-visible)
    for c in range(1, grid_cols):
        x = int(c * cell_w_px)
        cv2.line(overlay, (x, 0), (x, h), (160, 160, 160), 1, cv2.LINE_AA)
    for r in range(1, grid_rows):
        y = int(r * cell_h_px)
        cv2.line(overlay, (0, y), (w, y), (160, 160, 160), 1, cv2.LINE_AA)

    # Draw ALL candidate points as small gray dots
    for pt in all_ref_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(overlay, (x, y), 1, (100, 100, 100), -1)

    # Draw SELECTED points as larger colored dots
    for pt in selected_ref_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(overlay, (x, y), 3, (0, 230, 200), -1)  # Cyan-green
        cv2.circle(overlay, (x, y), 3, (0, 180, 160), 1)    # Border

    # Per-cell count labels
    if cell_counts is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, min(0.5, cell_w_px / 400.0))
        for (r, c), count in cell_counts.items():
            cx = int((c + 0.5) * cell_w_px)
            cy = int((r + 0.15) * cell_h_px)
            label = str(count)
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            # Background rect
            cv2.rectangle(overlay, (tx - 2, ty - text_size[1] - 2),
                          (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(overlay, label, (tx, ty), font, font_scale, (0, 255, 200), 1, cv2.LINE_AA)

    # Blend
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    return vis
