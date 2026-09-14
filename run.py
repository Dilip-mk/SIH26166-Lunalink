"""
LunaLink / SIH26166 — FastAPI Server Launcher
Runs Uvicorn with auto-reload scoped exclusively to the backend/ Python source directory.
Generated output images in data/ and uploaded files in data/uploads/ are excluded from reload watching.
"""
import os
import sys
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    print(f"[LunaLink] Starting FastAPI server on http://localhost:8000 ...")
    print(f"[LunaLink] Reload watcher scoped exclusively to: {BACKEND_DIR}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[BACKEND_DIR],
        app_dir=BACKEND_DIR,
    )
