import os
import shutil
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pipeline import run_pipeline

# Initialize FastAPI application
app = FastAPI(title="Lunar Image Registration Backend API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to guarantee CORS headers on uncaught errors and prevent server crashes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "*",
    }
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=cors_headers,
        )

    error_msg = str(exc) or exc.__class__.__name__
    print(f"[ERROR] Unhandled Exception: {error_msg}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {error_msg}"},
        headers=cors_headers,
    )

# Supported image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "uploads"))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))

# Serve the data/ directory as /outputs/ so the browser can load generated images
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=DATA_DIR), name="outputs")


def _validate_and_save(file: UploadFile, label: str, save_dir: str) -> str:
    """Validate extension and save an uploaded file to save_dir. Returns absolute saved path."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail=f"No {label} file provided.")

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}' for {label}. Allowed: {allowed_str}",
        )

    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.normpath(os.path.join(save_dir, filename))

    try:
        # Seek to start of file pointer in case it was read previously
        file.file.seek(0)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save {label}: {str(e)}")

    return saved_path


def _path_to_output_url(fs_path: str | None) -> str | None:
    """Convert an absolute filesystem path inside data/ to a browser-accessible /outputs/ URL."""
    if not fs_path:
        return None
    filename = os.path.basename(fs_path)
    return f"/outputs/{filename}"


@app.get("/health")
async def health_check():
    """Health check endpoint for verifying backend connectivity."""
    return {"status": "ok", "service": "Lunar Image Registration Backend API"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        saved_path = _validate_and_save(file, "image", UPLOAD_DIR)
        return {
            "message": "Image uploaded successfully",
            "filename": os.path.basename(saved_path),
            "saved_path": saved_path,
        }
    finally:
        try:
            await file.close()
        except Exception:
            pass


@app.post("/register")
async def register_image(
    reference_file: UploadFile = File(...),
    source_file: UploadFile = File(...),
    method: str = "sift",
    preprocessing: str = "raw",
    spatial_strategy: str = "grid",
):
    """
    Accept reference and source images, run the modular correspondence pipeline,
    and return metrics + browser-accessible output image URLs.
    """
    try:
        # Save both uploaded files to data/uploads/
        ref_path = _validate_and_save(reference_file, "reference_file", UPLOAD_DIR)
        src_path = _validate_and_save(source_file, "source_file", UPLOAD_DIR)

        # Run the CV pipeline with V2 parameters
        try:
            results = run_pipeline(
                source_path=src_path,
                reference_path=ref_path,
                method=method,
                preprocessing=preprocessing,
                spatial_strategy=spatial_strategy,
            )
        except HTTPException:
            raise
        except Exception as e:
            error_detail = str(e) or e.__class__.__name__
            print(f"[ERROR] Pipeline execution failed: {error_detail}")
            raise HTTPException(
                status_code=400,
                detail=f"Registration processing failed: {error_detail}",
            )

        # Return V1 + V2 metrics and browser-accessible /outputs/ URLs
        return {
            "success": True,
            "message": "Registration completed successfully",
            "reference_filename": os.path.basename(ref_path),
            "source_filename": os.path.basename(src_path),
            "metrics": {
                # Preserved V1 metrics
                "reference_keypoints": results["reference_keypoints"],
                "source_keypoints": results["source_keypoints"],
                "good_matches": results["good_matches"],
                "ransac_inliers": results["ransac_inliers"],
                "ransac_outliers": results["ransac_outliers"],
                "inlier_ratio": results["inlier_ratio"],
                "rmse": results["rmse"],
                "spatial_coverage": results["spatial_coverage"],
                "occupied_grid_cells": results["occupied_grid_cells"],
                "total_grid_cells": results["total_grid_cells"],
                # Extended V2 metrics
                "method": results["method"],
                "preprocessing": results["preprocessing"],
                "candidate_matches": results["candidate_matches"],
                "selected_matches": results["selected_matches"],
                "initial_rmse": results["initial_rmse"],
                "refined_rmse": results["refined_rmse"],
                "processing_time": results["processing_time"],
                # Phase 2 spatial metrics
                "spatial_strategy": results["spatial_strategy"],
                "max_matches_per_cell": results["max_matches_per_cell"],
                "spatial_coverage_before": results["spatial_coverage_before"],
                "spatial_coverage_after": results["spatial_coverage_after"],
            },
            "outputs": {
                "registered_image": _path_to_output_url(results.get("registered_image")),
                "overlay_image": _path_to_output_url(results.get("overlay_image")),
                "ransac_visualization": _path_to_output_url(results.get("ransac_visualization")),
                "spatial_grid_image": _path_to_output_url(results.get("spatial_visualization")),
                "spatial_matches_image": _path_to_output_url(results.get("spatial_selection_visualization")),
            },
        }
    finally:
        try:
            await reference_file.close()
        except Exception:
            pass
        try:
            await source_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # Restrict reload watching strictly to SCRIPT_DIR (backend/) so data/ file writes don't trigger server restarts
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[SCRIPT_DIR])

