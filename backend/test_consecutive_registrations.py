import requests
import json
import os
import sys

API_BASE = "http://127.0.0.1:8000"
UPLOADS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "uploads"))

ref_file = os.path.join(UPLOADS_DIR, "reference.jpg")
src1_file = os.path.join(UPLOADS_DIR, "source.jpg")
src2_file = os.path.join(UPLOADS_DIR, "source2.jpg")
src3_file = os.path.join(UPLOADS_DIR, "src.jpg")

print(f"Checking images exist:")
print(f"  Reference: {ref_file} (exists: {os.path.exists(ref_file)})")
print(f"  Source 1:  {src1_file} (exists: {os.path.exists(src1_file)})")
print(f"  Source 2:  {src2_file} (exists: {os.path.exists(src2_file)})")
print(f"  Source 3:  {src3_file} (exists: {os.path.exists(src3_file)})")

# 1. Health check
try:
    r_health = requests.get(f"{API_BASE}/health", timeout=5)
    print(f"\n[HEALTH CHECK] Status: {r_health.status_code}, Response: {r_health.json()}")
except Exception as e:
    print(f"\n[HEALTH CHECK] FAILED: {e}")
    sys.exit(1)

# Function to run a registration
def test_reg(index, src_path):
    print(f"\n==========================================")
    print(f"RUNNING REGISTRATION #{index} with {os.path.basename(src_path)}")
    print(f"==========================================")
    
    with open(ref_file, "rb") as rf, open(src_path, "rb") as sf:
        files = {
            "reference_file": ("reference.jpg", rf, "image/jpeg"),
            "source_file": (os.path.basename(src_path), sf, "image/jpeg"),
        }
        data = {
            "method": "sift",
            "preprocessing": "raw",
            "spatial_strategy": "grid",
        }
        
        try:
            resp = requests.post(f"{API_BASE}/register", files=files, data=data, timeout=30)
            print(f"HTTP Status: {resp.status_code}")
            if resp.status_code == 200:
                res = resp.json()
                print(f"Success: {res.get('success')}")
                print(f"Good matches: {res.get('metrics', {}).get('good_matches')}")
                print(f"Inliers: {res.get('metrics', {}).get('ransac_inliers')}")
                print(f"RMSE: {res.get('metrics', {}).get('refined_rmse') or res.get('metrics', {}).get('rmse')}")
                print(f"Outputs: {json.dumps(res.get('outputs', {}), indent=2)}")
                return True, resp.status_code, None
            else:
                print(f"Response text: {resp.text}")
                return False, resp.status_code, resp.text
        except Exception as e:
            print(f"Request exception: {e}")
            return False, None, str(e)

# Run Registration #1
ok1, status1, err1 = test_reg(1, src1_file)

# Check health after #1
alive1 = False
try:
    h1 = requests.get(f"{API_BASE}/health", timeout=5)
    alive1 = (h1.status_code == 200)
except:
    alive1 = False
print(f"Backend alive after #1: {'YES' if alive1 else 'NO'}")

# Run Registration #2
ok2, status2, err2 = test_reg(2, src2_file)

# Check health after #2
alive2 = False
try:
    h2 = requests.get(f"{API_BASE}/health", timeout=5)
    alive2 = (h2.status_code == 200)
except:
    alive2 = False
print(f"Backend alive after #2: {'YES' if alive2 else 'NO'}")

# Run Registration #3
ok3, status3, err3 = test_reg(3, src3_file)

# Check health after #3
alive3 = False
try:
    h3 = requests.get(f"{API_BASE}/health", timeout=5)
    alive3 = (h3.status_code == 200)
except:
    alive3 = False
print(f"Backend alive after #3: {'YES' if alive3 else 'NO'}")

print(f"\n==========================================")
print(f"FINAL TEST SUMMARY:")
print(f"Registration #1: {'PASS' if ok1 else 'FAIL'} (HTTP {status1})")
print(f"Registration #2: {'PASS' if ok2 else 'FAIL'} (HTTP {status2})")
print(f"Registration #3: {'PASS' if ok3 else 'FAIL'} (HTTP {status3})")
print(f"Backend alive after #1: {'YES' if alive1 else 'NO'}")
print(f"Backend alive after #2: {'YES' if alive2 else 'NO'}")
print(f"Backend alive after #3: {'YES' if alive3 else 'NO'}")
print(f"==========================================")
