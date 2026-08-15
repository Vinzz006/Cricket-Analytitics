import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image 

# cricketlm model 
sys.path.insert(0, str(Path(__file__).parent / "cricklm"))
from inference import  CrickLMInference

# mediapipe tasks api
import mediapipe as mp 
from mediapipe.tasks import python as mp_python 
from mediapipe.tasks.python import vision as mp_vision

# app
app = FastAPI(title="CrickIQ API",version="2.0.0")

# cors
app.add_middleware(CORSMiddleware, 
allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Constants
GREEN = (74, 255, 92)
MAX_DIM = 1080
MODEL_PATH = Path(__file__).parent/"cricklm"/"pose_landmarker_full.task"

# MediaPipe PoseLandmark indices (Tasks API uses integers directly)
NOSE           = 0
LEFT_SHOULDER  = 11;  RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13;  RIGHT_ELBOW    = 14
LEFT_WRIST     = 15;  RIGHT_WRIST    = 16
LEFT_HIP       = 23;  RIGHT_HIP      = 24
LEFT_KNEE      = 25;  RIGHT_KNEE     = 26
LEFT_ANKLE     = 27;  RIGHT_ANKLE    = 28

# skeleton connection (index pairs)
CONNECTIONS = [
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

# joints with angle labels
ANGLE_JOINTS = { 
    "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_elbow":  (LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST),
    "right_knee":  (RIGHT_HIP,  RIGHT_KNEE,  RIGHT_ANKLE),
    "left_knee":   (LEFT_HIP,   LEFT_KNEE,   LEFT_ANKLE),
    "right_shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    "left_shoulder":  (LEFT_ELBOW,  LEFT_SHOULDER,  LEFT_HIP),
    "right_hip":   (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    "left_hip":    (LEFT_SHOULDER,  LEFT_HIP,  LEFT_KNEE),
}

# singletons
_landmarker: Optional[mp_vision.PoseLandmarker] = None
_crick_engine: Optional[CrickLMInference] = None

def get_landmarker() -> Optional[mp_vision.PoseLandmarker]:
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    if not MODEL_PATH.exists():
        print(
            f"[WARN] Pose model not found at {MODEL_PATH}\n"
            f"       Download from: https://storage.googleapis.com/mediapipe-models/"
            f"pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )
        return None 
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path = str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses = 1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    _landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
    print("[OK] MediaPipe PoseLandmarker loaded.")
    return _landmarker

def get_crick_engine() -> CrickLMInference:
    global _crick_engine
    if _crick_engine is None:
        ckpt = Path(__file__).parent/"cricklm"/"checkpoints"/"best.pt"
        tok = Path(__file__).parent/"cricklm"/"checkpoints"/"tokenizer.json"
        _crick_engine = CrickLMInference(
            checkpoint_path = str(ckpt),
            tokenizer_path = str(tok),
        )
    return _crick_engine

# Geometry helpers
def calc_angle(a,b,c) -> float:
    ax, ay = a;  bx, by = b;  cx, cy = c
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0,dot / mag))))

def resize_image(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h,w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h,w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def pil_to_bgr(pill_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pill_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()

# 1. Pose extraction
def extract_pose(img_bgr: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]
    landmarker = get_landmarker()

    # mediapipe path
    if landmarker is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return {"detected": False, "error": "No human pose detected in image."}
        
        lms = result.pose_landmarks[0]

        def px(idx):
            lm = lms[idx]
            return int(lm.x * w), int(lm.y * h)
        
        def vis(idx):
            return round(lms[idx].visibility,3)
        
    # fallback (temporary)
    else:
        print("[WARN] Using heuristic pose fallback (no MediaPipe model)")
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,30,60), (20,180,255))
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {"detected": False, "error": "No pose detected (heuristic fallback)."}
        cx_body = int(np.mean(xs))
        cy_body = int(np.mean(ys))

        def px(idx):
            offsets = {
                NOSE: (0, -h//4), LEFT_SHOULDER: (-w//8, -h//6),
                RIGHT_SHOULDER: (w//8, -h//6),
                LEFT_ELBOW: (-w//6, 0), RIGHT_ELBOW: (w//6, -h//16),
                LEFT_WRIST: (-w//5, h//8), RIGHT_WRIST: (w//5, h//16),
                LEFT_HIP: (-w//10, h//8), RIGHT_HIP: (w//10, h//8),
                LEFT_KNEE: (-w//10, h//4), RIGHT_KNEE: (w//10, h//4),
                LEFT_ANKLE: (-w//10, h//2-h//8), RIGHT_ANKLE: (w//10, h//2-h//8),
            }
            dx, dy = offsets.get(idx(0,0))
            return (cx_body + dx, cy_body + dy)
        
        def vis(idx):
            return 0.95

    # computation of joint angles
    angles = {}
    for name, (a_idx,b_idx,c_idx) in ANGLE_JOINTS.items():
        try:
            if vis(a_idx) < 0.3 or vis(b_idx) < 0.3 or vis(c_idx) < 0.3:
                angles[name] = None
                continue
            angles[name] = round(calc_angle(px(a_idx), px(b_idx), px(c_idx)), 1)
        except Exception:
            angles[name] = None
    
    # body metrics
    rs = px(RIGHT_SHOULDER); ls = px(LEFT_SHOULDER)
    rh = px(RIGHT_HIP);      lh = px(LEFT_HIP)
    ra = px(RIGHT_ANKLE);    la = px(LEFT_ANKLE)
    nose_pt = px(NOSE)
    rw = px(RIGHT_WRIST);    lw = px(LEFT_WRIST)
 
    shoulder_w = abs(rs[0] - ls[0])
    stance_w   = abs(ra[0] - la[0])
    body_cx    = (rh[0] + lh[0]) // 2
    head_off   = nose_pt[0] - body_cx
    lead_wrist = min(rw[1], lw[1])
    wrist_h    = round(1 - lead_wrist / max(h, 1), 3)
 
    shoulder_angle = math.degrees(
        math.atan2(rs[1]-ls[1], rs[0]-ls[0])
    )
    hip_angle = math.degrees(
        math.atan2(rh[1]-lh[1], rh[0]-lh[0])
    )
 
    metrics = {
        "shoulder_width_px":              shoulder_w,
        "stance_width_px":                stance_w,
        "head_offset_from_center_px":     round(head_off, 1),
        "shoulder_alignment_angle_deg":   round(shoulder_angle, 1),
        "hip_alignment_angle_deg":        round(hip_angle, 1),
        "wrist_height_normalized":        wrist_h,
        "stance_to_shoulder_ratio":       round(stance_w / max(shoulder_w, 1), 3),
    }

    # store pixels
    keypoints = {
        "nose": px(NOSE),
        "left_shoulder": px(LEFT_SHOULDER), "right_shoulder": px(RIGHT_SHOULDER),
        "left_elbow": px(LEFT_ELBOW),       "right_elbow": px(RIGHT_ELBOW),
        "left_wrist": px(LEFT_WRIST),       "right_wrist": px(RIGHT_WRIST),
        "left_hip": px(LEFT_HIP),           "right_hip": px(RIGHT_HIP),
        "left_knee": px(LEFT_KNEE),         "right_knee": px(RIGHT_KNEE),
        "left_ankle": px(LEFT_ANKLE),       "right_ankle": px(RIGHT_ANKLE),
    }
 
    return {
        "detected":       True,
        "image_size":     {"width": w, "height": h},
        "joint_angles_deg": angles,
        "body_metrics":   metrics,
        "_keypoints_px":  keypoints,   
    }

# 2 skeleton overlay
def draw_skeleton(img_bgr: np.ndarray, features: dict) -> np.ndarray:
    if not features.get("detected"):
        return img_bgr
    
    kp = features["_keypoints_px"]
    angles = features["joint_angles_deg"]
    h, w = img_bgr.shape[:2]
 
    overlay = img_bgr.copy()

    dark = np.zeros_like(overlay)
    cv2.addWeighted(overlay, 0.62, dark, 0.38, 0, overlay)

    idx_to_key = {
        NOSE: "nose",
        LEFT_SHOULDER: "left_shoulder",   RIGHT_SHOULDER: "right_shoulder",
        LEFT_ELBOW:    "left_elbow",       RIGHT_ELBOW:    "right_elbow",
        LEFT_WRIST:    "left_wrist",       RIGHT_WRIST:    "right_wrist",
        LEFT_HIP:      "left_hip",         RIGHT_HIP:      "right_hip",
        LEFT_KNEE:     "left_knee",        RIGHT_KNEE:     "right_knee",
        LEFT_ANKLE:    "left_ankle",       RIGHT_ANKLE:    "right_ankle",
    }

    # bones
    for (a_idx, b_idx) in CONNECTIONS:
        a_key = idx_to_key.get(a_idx)
        b_key = idx_to_key.get(b_idx)

        if a_key not in kp or b_key not in kp:
            continue

        pt1 = tuple(kp[a_key])
        pt2 = tuple(kp[b_key])
        cv2.line(overlay, pt1, pt2, (20, 100, 20), 8, cv2.LINE_AA)
        cv2.line(overlay, pt1, pt2, GREEN, 2, cv2.LINE_AA)

    # joints
    for key, pt in kp.items():
        pt = tuple(pt)  
        is_head = key == "nose"
        r_outer = 12 if is_head else 7
        r_inner = 7  if is_head else 4
        cv2.circle(overlay, pt, r_outer + 4, (15, 60, 15), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, r_outer,     GREEN,          -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, r_inner,     (200, 255, 200), -1, cv2.LINE_AA)   

    # angle annotations
    joint_label_map = {
        "right_elbow": "right_elbow", "left_elbow": "left_elbow",
        "right_knee":  "right_knee",  "left_knee":  "left_knee",
    }

    for angle_key, kp_key in joint_label_map.items():
        val = angles.get(angle_key)
        if val is None or kp_key not in kp:
            continue
        pt = tuple(kp[kp_key])
        label = f"{int(val)}\u00b0"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tx = pt[0] + 12
        ty = pt[1] - th - 4
        # Pill background
        cv2.rectangle(overlay,
                      (tx - 4, ty - 3),
                      (tx + tw + 4, ty + th + 3),
                      (8, 8, 8), -1)
        cv2.rectangle(overlay,
                      (tx - 4, ty - 3),
                      (tx + tw + 4, ty + th + 3),
                      GREEN, 1)
        cv2.putText(overlay, label, (tx, ty + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)
        
    # Watermark
    cv2.putText(overlay, "CrickIQ  AI Analysis",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, GREEN, 1, cv2.LINE_AA)
 
    return overlay

# 3: CrickLM analysis
def run_crick_analysis(features: dict, player_type: str) -> dict:
    engine = get_crick_engine()
    return engine.analyze(features, player_type)
 
 
# API Routes 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "CrickIQ API",
        "model": "CrickLM (local transformer)",
        "mediapipe_model": MODEL_PATH.exists(),
    }
 
 
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    player_type: str = Form(default="batsman"),
):
    t0 = time.time()
 
    # Validate 
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported (JPG, PNG, WEBP).")
 
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 15 MB.")
 
    # Decode image
    try:
        pil_img = Image.open(io.BytesIO(contents))
        img_bgr = pil_to_bgr(pil_img)
    except Exception:
        raise HTTPException(400, "Could not decode image. Please upload a valid JPG/PNG.")
 
    img_bgr = resize_image(img_bgr)

    #  Step 1: Pose extraction 
    try:
        features = extract_pose(img_bgr)
    except Exception as e:
        raise HTTPException(500, f"Pose extraction failed: {e}")
 
    if not features.get("detected"):
        raise HTTPException(422, features.get("error", "No pose detected."))
 
    # Step 2: Skeleton overlay
    try:
        annotated    = draw_skeleton(img_bgr, features)
        annotated_b64 = bgr_to_b64(annotated)
    except Exception as e:
        raise HTTPException(500, f"Skeleton overlay failed: {e}")
 
    # Step 3: CrickLM analysis 
    try:
        analysis = run_crick_analysis(features, player_type)
    except Exception as e:
        raise HTTPException(500, f"CrickLM analysis failed: {e}")
 
    elapsed = round(time.time() - t0, 2)
 
    # Strip internal keypoints from response
    public_features = {
        "joint_angles":  features["joint_angles_deg"],
        "body_metrics":  features["body_metrics"],
    }
 
    return JSONResponse({
        "success":         True,
        "elapsed_seconds": elapsed,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "features":        public_features,
        "analysis":        analysis,
    })
        