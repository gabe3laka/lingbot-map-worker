import os
import sys
import shutil
from pathlib import Path

import numpy as np
import requests
import runpod
import torch

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

def unproject_depth_to_world(depth, extrinsic, intrinsic):
    """
    Unproject per-pixel depth into 3-D world-space points.

    Based on gct_base.py::_unproject_depth_to_world from lingbot-map.

    LingBot-Map's depth head outputs shape (B, S, H, W, 1) — i.e. there is a
    trailing channel dimension. After postprocess() squeezes the batch dim (B=1),
    depth arrives here as (N, H, W, 1). We squeeze the trailing 1 before
    computing to get (N, H, W).

    Args:
        depth     : (N, H, W) or (N, H, W, 1) — metric depth, numpy or torch
        extrinsic : (N, 3, 4) — camera-to-world (c2w), numpy or torch
        intrinsic : (N, 3, 3) — camera intrinsic matrix, numpy or torch

    Returns:
        world_points : (N, H, W, 3) float32 numpy array
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().cpu().float().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.detach().cpu().float().numpy()

    depth     = depth.astype(np.float32)
    extrinsic = extrinsic.astype(np.float32)
    intrinsic = intrinsic.astype(np.float32)

    # ── Normalise depth shape to (N, H, W) ──────────────────────────────
    # gct_base forward docstring: depth is [B, S, H, W, 1].
    # After _squeeze_single_batch removes B=1 → (N, H, W, 1).
    # Squeeze the trailing channel dim if present.
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]           # (N, H, W, 1) → (N, H, W)
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0, :, :]       # (N, 1, H, W) → (N, H, W)  (channels-first fallback)

    if depth.ndim != 3:
        raise RuntimeError(
            f"Cannot reduce depth to (N, H, W): got shape {depth.shape}"
        )

    N, H, W = depth.shape
    print(f"  unproject: depth={depth.shape}, extrinsic={extrinsic.shape}, intrinsic={intrinsic.shape}")

    # ── Pixel grid: (H, W, 3) homogeneous coords [u, v, 1] ─────────────
    v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([u_idx, v_idx, np.ones((H, W), dtype=np.float32)], axis=-1)  # (H, W, 3)

    # ── K^{-1} per frame: (N, 3, 3) ─────────────────────────────────────
    K_inv = np.linalg.inv(intrinsic)   # (N, 3, 3)

    # ── Camera-space rays: (N, H, W, 3) = K^{-1} @ [u, v, 1]^T ─────────
    cam_rays = np.einsum("nij,hwj->nhwi", K_inv, pix)

    # ── Scale by depth → camera-frame 3-D points: (N, H, W, 3) ─────────
    cam_pts = cam_rays * depth[..., np.newaxis]

    # ── Rotate + translate into world space: P_w = R @ P_c + t ──────────
    R = extrinsic[:, :3, :3]           # (N, 3, 3)
    t = extrinsic[:, :3, 3]            # (N, 3)
    world = (
        np.einsum("nij,nhwj->nhwi", R, cam_pts)
        + t[:, np.newaxis, np.newaxis, :]
    )

    return world.astype(np.float32)    # (N, H, W, 3)

def export_ply(predictions: dict, ply_path: Path, conf_threshold: float = 1.5):
    """
    Build a .ply point cloud from LingBot-Map model output tensors.

    Handles two cases:
      1. Model emits 'world_points' directly (future / alternate config).
      2. Model emits 'depth' + 'extrinsic' + 'intrinsic' (current streaming
         path) — we unproject depth into world space ourselves.

    Key shape notes from gct_base.py forward() docstring:
      depth          : (B, S, H, W, 1)  → after _squeeze_single_batch: (N, H, W, 1)
      depth_conf     : (B, S, H, W)     → after squeeze:               (N, H, W)
      extrinsic      : (B, S, 3, 4)     → after squeeze:               (N, 3, 4)  c2w
      intrinsic      : (B, S, 3, 3)     → after squeeze:               (N, 3, 3)
      images         : (B, S, 3, H, W)  → after squeeze:               (N, 3, H, W)
    """
    import open3d as o3d

    # ── Obtain (N, H, W, 3) world points ────────────────────────────────
    if "world_points" in predictions:
        pts = predictions["world_points"]
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().float().numpy()
        # world_points from model: (B, S, H, W, 3) → squeeze → (N, H, W, 3)
        # or channel-first (N, 3, H, W) — handle both
        if pts.ndim == 4 and pts.shape[-1] == 3:
            pass                            # already (N, H, W, 3)
        elif pts.ndim == 4 and pts.shape[1] == 3:
            pts = pts.transpose(0, 2, 3, 1) # (N, 3, H, W) → (N, H, W, 3)
        print(f"Using world_points from model, shape: {pts.shape}")
    else:
        if "depth" not in predictions:
            raise RuntimeError(
                f"predictions has neither 'world_points' nor 'depth'. "
                f"Keys present: {list(predictions.keys())}"
            )
        if "extrinsic" not in predictions or "intrinsic" not in predictions:
            raise RuntimeError(
                "predictions missing 'extrinsic' and/or 'intrinsic'. "
                "Make sure postprocess() was called before export_ply()."
            )
        print("world_points not in predictions — unprojecting from depth + extrinsic + intrinsic")
        # Log shapes to help debug any future issues
        d = predictions["depth"]
        e = predictions["extrinsic"]
        i = predictions["intrinsic"]
        print(f"  depth shape  : {d.shape if hasattr(d, 'shape') else type(d)}")
        print(f"  extrinsic    : {e.shape if hasattr(e, 'shape') else type(e)}")
        print(f"  intrinsic    : {i.shape if hasattr(i, 'shape') else type(i)}")
        pts = unproject_depth_to_world(d, e, i)   # → (N, H, W, 3)

    pts_flat = pts.reshape(-1, 3)                 # (N*H*W, 3)

    # ── Confidence mask ──────────────────────────────────────────────────
    conf_key = (
        "world_points_conf" if "world_points_conf" in predictions
        else "depth_conf"   if "depth_conf"        in predictions
        else None
    )
    if conf_key is not None:
        conf = predictions[conf_key]
        if isinstance(conf, torch.Tensor):
            conf = conf.detach().cpu().float().numpy()
        # depth_conf shape after squeeze: (N, H, W) — already flat-compatible
        print(f"  conf ({conf_key}) shape: {conf.shape}")
        mask = conf.reshape(-1) >= conf_threshold
    else:
        mask = np.ones(pts_flat.shape[0], dtype=bool)

    # Drop NaN / Inf pixels (sky, edges, textureless regions)
    finite = np.isfinite(pts_flat).all(axis=-1)
    mask   = mask & finite
    pts_flat = pts_flat[mask]

    # ── Colour from images tensor ────────────────────────────────────────
    colours = None
    if "images" in predictions:
        imgs = predictions["images"]
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu().float().numpy()
        # images shape after squeeze: (N, 3, H, W) → (N, H, W, 3)
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        colours_flat = imgs.reshape(-1, 3)[mask]
        colours = np.clip(colours_flat, 0.0, 1.0)

    print(f"Writing {len(pts_flat):,} points to {ply_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat.astype(np.float64))
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"PLY saved: {ply_path}  ({ply_path.stat().st_size / 1024:.1f} KB)")
    return ply_path

def run_lingbot_map(job: dict, scan_id: str, scan_dir: Path, scan_type: str) -> Path:
    """
    Run LingBot-Map inference directly via Python API (no subprocess),
    then export the resulting point cloud to a .ply file.
    """
    # ── Add lingbot-map to Python path ──────────────────────────────────
    lingbot_map_root = Path("/app/lingbot-map")
    if str(lingbot_map_root) not in sys.path:
        sys.path.insert(0, str(lingbot_map_root))

    model_dir  = os.getenv("MODEL_DIR",  "/models")
    model_file = os.getenv("MODEL_FILE", "lingbot-map-long.pt")
    model_path = Path(model_dir) / model_file
    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found at {model_path}")

    video_path = scan_dir / "input.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Expected input video at {video_path}")

    windowed = (scan_type == "wand")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Import demo helpers from lingbot-map ────────────────────────────
    from demo import load_images, load_model, postprocess

    # ── Load frames ──────────────────────────────────────────────────────
    _progress(job, "Extracting frames from video...")
    images, paths, resolved_folder = load_images(
        video_path=str(video_path),
        fps=15,
        image_size=518,
        patch_size=14,
    )

    # ── Build a minimal args namespace for load_model ────────────────────
    import argparse
    args = argparse.Namespace(
        mode="windowed" if windowed else "streaming",
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        num_scale_frames=8,
        kv_cache_sliding_window=64,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=False,
        camera_num_iterations=2,
        model_path=str(model_path),
        window_size=128,
        overlap_size=16,
        overlap_keyframes=None,
        keyframe_interval=2,
        offload_to_cpu=True,
        compile=False,
    )

    # ── Load model ───────────────────────────────────────────────────────
    _progress(job, "Loading model checkpoint...")
    model = load_model(args, device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)
    else:
        dtype = torch.float32

    images     = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}, mode={args.mode}")

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    # ── Inference ────────────────────────────────────────────────────────
    _progress(job, f"Running LingBot-Map inference ({num_frames} frames)...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )

    print(f"Inference done. Prediction keys: {list(predictions.keys())}")

    # ── Post-process ─────────────────────────────────────────────────────
    _progress(job, "Post-processing predictions...")
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = postprocess(predictions, images_for_post)

    # Log post-process output shapes to help diagnose any remaining issues
    for k, v in predictions.items():
        shape = v.shape if hasattr(v, "shape") else type(v)
        print(f"  post-process key={k!r}  shape={shape}")

    # Attach images for colour export
    if "images" not in predictions:
        predictions["images"] = images_cpu

    # ── Export .ply ───────────────────────────────────────────────────────
    _progress(job, "Exporting point cloud to .ply...")
    ply_path = scan_dir / "pointcloud.ply"
    export_ply(predictions, ply_path, conf_threshold=1.5)
    return ply_path

# ----------------------------
# Runpod Serverless handler
# ----------------------------
def handler(job: dict):
    scratch_root = Path("/scratch")
    scan_dir     = None

    inp     = job.get("input") or {}
    scan_id = inp.get("scan_id")

    try:
        video_url = _require_input(inp, "video_url")
        scan_id   = _require_input(inp, "scan_id")
        scan_type = inp.get("scan_type", "scope")  # 'scope' | 'wand'

        _progress(job, "Looking up patient_id...")
        patient_id = get_patient_id(scan_id)

        scan_dir = scratch_root / scan_id
        _progress(job, "Creating scratch workspace...")
        scan_dir.mkdir(parents=True, exist_ok=True)

        video_path = scan_dir / "input.mp4"
        _progress(job, "Downloading input video...")
        download_file(video_url, video_path)

        ply_path = run_lingbot_map(
            job, scan_id=scan_id, scan_dir=scan_dir, scan_type=scan_type
        )

        # Upload to {patient_id}/{scan_id}/pointcloud.ply
        # Matches RLS policy: (storage.foldername(name))[1] = patient_id
        object_path = f"{patient_id}/{scan_id}/pointcloud.ply"

        _progress(job, "Uploading .ply to Supabase Storage...")
        uploaded_path = upload_ply_to_supabase(ply_path, object_path)

        _progress(job, "Done.")

        return {
            "pointcloud_url": uploaded_path,
            "scan_id":        scan_id,
            "status":         "completed",
        }

    except Exception as e:
        err_msg = str(e)
        print(f"Handler error: {err_msg}")
        import traceback
        traceback.print_exc()
        return {"error": err_msg}

    finally:
        if scan_dir and scan_dir.exists():
            _progress(job, f"Cleaning up {scan_dir} ...")
            shutil.rmtree(scan_dir, ignore_errors=True)

# Download model before accepting any jobs
ensure_model()

runpod.serverless.start({"handler": handler})
