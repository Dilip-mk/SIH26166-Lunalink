import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save {label}: {str(e)}")

    return saved_path


def _path_to_output_url(fs_path: str) -> str:
    """Convert an absolute filesystem path inside data/ to a browser-accessible /outputs/ URL."""
    filename = os.path.basename(fs_path)
    return f"/outputs/{filename}"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    saved_path = _validate_and_save(file, "image", UPLOAD_DIR)
    return {
        "message": "Image uploaded successfully",
        "filename": os.path.basename(saved_path),
        "saved_path": saved_path,
    }


@app.post("/register")
async def register_image(
    reference_file: UploadFile = File(...),
    source_file: UploadFile = File(...),
):
    """
    Accept reference and source images, run the full SIFT→RANSAC→Homography pipeline,
    and return metrics + browser-accessible output image URLs.
    """
    # Save both uploaded files to data/uploads/
    ref_path = _validate_and_save(reference_file, "reference_file", UPLOAD_DIR)
    src_path = _validate_and_save(source_file, "source_file", UPLOAD_DIR)

    # Run the CV pipeline with both paths
    try:
        results = run_pipeline(source_path=src_path, reference_path=ref_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Registration processing failed: {str(e)}",
        )

    # Return metrics and browser-accessible /outputs/ URLs
    return {
        "message": "Registration completed successfully",
        "reference_filename": os.path.basename(ref_path),
        "source_filename": os.path.basename(src_path),
        "metrics": {
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
        },
        "outputs": {
            "registered_image": _path_to_output_url(results["registered_image"]),
            "overlay_image": _path_to_output_url(results["overlay_image"]),
            "ransac_visualization": _path_to_output_url(results["ransac_visualization"]),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
