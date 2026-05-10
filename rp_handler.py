import os
import shutil
import subprocess
from pathlib import Path

import requests
import runpod


# ----------------------------
# Model download (runs once at worker startup)
# ----------------------------
def ensure_model():
    """
    Download the LingBot-Map checkpoint from Hugging Face if not present.
    Requires HF_TOKEN env var set in RunPod endpoint settings.
    """
    model_dir = Path(os.getenv("MODEL_DIR", "/models"))
    model_file = os.getenv("MODEL_FILE", "lingbot-map-long.pt")
    model_path = model_dir / model_file

    if model_path.exists():
        print(f"Model already present at {model_path}, skipping download.")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var is not set. "
            "Add it to your RunPod endpoint environment variables."
        )

    print(f"Downloading {model_file} from Hugging Face...")
    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="robbyant/lingbot-map",
        filename=model_file,
        token=hf_token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {model_path}")


# ----------------------------
# Helpers
# ----------------------------
def _require_input(inp: dict, key: str) -> str:
    v = inp.get(key)
    if not v or not isinstance(v, str):
        raise ValueError(f"Missing or invalid required field: input.{key}")
    return v


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _progress(job: dict, msg: str):
    try:
        runpod.serverless.progress_update(job, msg)
    except Exception:
        pass


def download_file(url: str, dst_path: Path, timeout_s: int = 600):
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def find_any_ply(root: Path) -> Path:
    candidates = list(root.rglob("*.ply"))
    if not candidates:
        raise RuntimeError(f"No .ply output found under {root}")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def get_patient_id(scan_id: str) -> str:
    """
    Look up patient_id from the scans table using service-role credentials.
    Required so we can upload to {patient_id}/{scan_id}/pointcloud.ply
    which matches the RLS policy on the scan-pointclouds bucket.
    """
    supabase_url = _require_env("SUPABASE_URL").rstrip("/")
    service_role_key = _require_env("SUPABASE_SERVICE_ROLE_KEY")

    resp = requests.get(
        f"{supabase_url}/rest/v1/scans",
        params={"id": f"eq.{scan_id}", "select": "patient_id"},
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError(f"No scan row found for scan_id={scan_id}")
    patient_id = rows[0].get("patient_id")
    if not patient_id:
        raise RuntimeError(f"scan row has no patient_id for scan_id={scan_id}")
    return patient_id


def upload_ply_to_supabase(file_path: Path, object_path: str) -> str:
    """
    Upload to Supabase Storage (PRIVATE bucket) using Service Role key.
    Returns the storage path (e.g. {patient_id}/{scan_id}/pointcloud.ply).
    """
    supabase_url = _require_env("SUPABASE_URL").rstrip("/")
    service_role_key = _require_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = _require_env("SUPABASE_BUCKET")

    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"
    headers = {
        "authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "content-type": "application/octet-stream",
        "x-upsert": "true",
    }
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=600)
    resp.raise_for_status()
    return object_path


def run_lingbot_map(job: dict, scan_id: str, scratch_root: Path, scan_type: str) -> Path:
    """
    Run lingbot-map demo.py.

    FIX 4: --mode streaming is not a valid flag in LingBot-Map.
    Default (no --mode flag) = streaming.
    Only pass --mode windowed for wand scans.
    """
    model_dir = os.getenv("MODEL_DIR", "/models")
    model_file = os.getenv("MODEL_FILE", "lingbot-map-long.pt")
    model_path = Path(model_dir) / model_file
    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found at {model_path}")

    scan_dir = scratch_root / scan_id
    video_path = scan_dir / "input.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Expected input video at {video_path}")

    windowed = (scan_type == "wand")

    cmd = [
        "python3.10",
        "/app/lingbot-map/demo.py",
        "--model_path", str(model_path),
        "--video_path", str(video_path),
        "--fps", "15",
        "--keyframe_interval", "2",
        "--camera_num_iterations", "2",
        "--conf_threshold", "1.5",
    ]
    # FIX 4: Only add --mode windowed for wand scans; streaming is the default
    if windowed:
        cmd += ["--mode", "windowed", "--window_size", "128"]

    _progress(job, f"Running LingBot-Map (windowed={windowed})...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"lingbot-map demo.py failed (exit {proc.returncode})")

    _progress(job, "Searching for .ply output...")
    return find_any_ply(scan_dir)


# ----------------------------
# Runpod Serverless handler
# ----------------------------
def handler(job: dict):
    scratch_root = Path("/scratch")
    scan_dir = None

    inp = job.get("input") or {}
    scan_id = inp.get("scan_id")

    try:
        video_url = _require_input(inp, "video_url")
        scan_id = _require_input(inp, "scan_id")
        scan_type = inp.get("scan_type", "scope")  # 'scope' | 'wand'

        # FIX 2: Look up patient_id so we can build the correct storage path
        _progress(job, "Looking up patient_id...")
        patient_id = get_patient_id(scan_id)

        scan_dir = scratch_root / scan_id
        _progress(job, "Creating scratch workspace...")
        scan_dir.mkdir(parents=True, exist_ok=True)

        video_path = scan_dir / "input.mp4"
        _progress(job, "Downloading input video...")
        download_file(video_url, video_path)

        ply_path = run_lingbot_map(job, scan_id=scan_id, scratch_root=scratch_root, scan_type=scan_type)

        # FIX 2: Upload to {patient_id}/{scan_id}/pointcloud.ply
        # This matches the RLS policy: (storage.foldername(name))[1] = patient_id
        object_path = f"{patient_id}/{scan_id}/pointcloud.ply"

        _progress(job, "Uploading .ply to Supabase Storage...")
        uploaded_path = upload_ply_to_supabase(ply_path, object_path)

        _progress(job, "Done.")

        # FIX 1: Return pointcloud_url (not ply_path) so reconstruct-scan-callback
        #         can find it with pickPointCloudPath().
        # FIX 3: Do NOT call post_callback() manually — the dispatcher already set
        #         webhook=callbackUrl in the RunPod request, so RunPod's native
        #         webhook delivers this return value automatically. A manual POST
        #         would fire the callback twice.
        return {
            "pointcloud_url": uploaded_path,
            "scan_id": scan_id,
            "status": "completed",
        }

    except Exception as e:
        err_msg = str(e)
        print(f"Handler error: {err_msg}")
        # Return error — RunPod's webhook will deliver this to the callback
        # which will mark the scan as failed.
        return {"error": err_msg}

    finally:
        if scan_dir and scan_dir.exists():
            _progress(job, f"Cleaning up {scan_dir} ...")
            shutil.rmtree(scan_dir, ignore_errors=True)


# Download model before accepting any jobs
ensure_model()

runpod.serverless.start({"handler": handler})
