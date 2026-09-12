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


def _path_to_output_url(fs_path: str) -> str:
    """Convert an absolute filesystem path inside data/ to a browser-accessible /outputs/ URL."""
    filename = os.path.basename(fs_path)
    return f"/outputs/{filename}"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. Error handling: Check for missing file or filename
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # 2. Error handling: Validate file extension
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed image extensions: {allowed_str}",
        )

    # 3. Create target uploads directory if it does not exist
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create uploads directory: {str(e)}",
        )

    # 4. Save the uploaded image preserving original filename
    saved_path = os.path.normpath(os.path.join(UPLOAD_DIR, filename))
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    return {
        "message": "Image uploaded successfully",
        "filename": filename,
        "saved_path": saved_path,
    }


@app.post("/register")
async def register_image(file: UploadFile = File(...)):
    # 1. Error handling: Check for missing file or filename
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # 2. Error handling: Validate file extension
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed image extensions: {allowed_str}",
        )

    # 3. Create target uploads directory if it does not exist
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create uploads directory: {str(e)}",
        )

    # 4. Save the uploaded source image preserving original filename
    saved_path = os.path.normpath(os.path.join(UPLOAD_DIR, filename))
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    # 5. Call pipeline processing
    try:
        results = run_pipeline(saved_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Registration processing failed: {str(e)}",
        )

    # 6. Format API response — convert Windows filesystem paths to browser-accessible /outputs/ URLs
    return {
        "message": "Registration completed successfully",
        "filename": filename,
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
