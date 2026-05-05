"""
drowsiness_inference_v4.py
Dual-mode drowsiness detection:
  1. EfficientNet model prediction (from your trained model)
  2. Eye Aspect Ratio (EAR) — directly measures eye openness using facial landmarks

Both run simultaneously. Either one triggering = DROWSY alert.

Requirements:
    pip install timm opencv-python mediapipe numpy torch torchvision

Usage:
    python drowsiness_inference_v3.py
    python drowsiness_inference_v3.py --ear_threshold 0.22   # adjust eye openness sensitivity
    python drowsiness_inference_v3.py --model_threshold 0.40  # adjust model sensitivity
    python drowsiness_inference_v3.py --source 1              # different webcam
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "drowsiness_model.pth"
CONFIG_PATH = "model_config.json"

ALERT_SECONDS = 1.5   # seconds of drowsiness before alarm triggers

DROWSY_COLOR  = (0, 0, 255)     # Red
AWAKE_COLOR   = (0, 200, 0)     # Green
WARNING_COLOR = (0, 165, 255)   # Orange

# MediaPipe landmark indices for left and right eyes
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONFIG & MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_config(path):
    if not Path(path).exists():
        print(f"⚠️  {path} not found — using default config")
        return {
            "architecture": "efficientnet_b0",
            "img_size": 96,
            "classes": ["awake", "drowsy"],
            "num_classes": 2,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std":  [0.229, 0.224, 0.225],
        }
    with open(path) as f:
        return json.load(f)


def build_model(num_classes):
    m = timm.create_model("efficientnet_b0", pretrained=False)
    in_feat = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.40),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(p=0.30),
        nn.Linear(256, num_classes),
    )
    return m


def load_model(model_path, config, device):
    if not Path(model_path).exists():
        print(f"⚠️  {model_path} not found — running EAR-only mode")
        return None
    model = build_model(config["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded")
    return model


def make_transform(config):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["normalize_mean"], config["normalize_std"]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# EAR — Eye Aspect Ratio
# EAR < threshold means eyes are closed/drowsy
# Normal open eye EAR ≈ 0.25-0.30
# Closed eye EAR ≈ 0.15-0.20
# ─────────────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def draw_eye_landmarks(frame, landmarks, eye_indices, img_w, img_h, color):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x  = int(lm.x * img_w)
        y  = int(lm.y * img_h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)
    # Draw eye outline
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PREDICT
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def model_predict(face_crop_bgr, model, transform, device, classes):
    if model is None:
        return None, 0.0
    rgb    = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    return classes[idx], probs[idx].item()


# ─────────────────────────────────────────────────────────────────────────────
# ALERT
# ─────────────────────────────────────────────────────────────────────────────
def play_alert():
    print("\a", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, is_drowsy, drowsy_reason, ear, model_label, model_conf,
             drowsy_seconds, alert_seconds, fps):
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # EAR value (top left)
    ear_color = DROWSY_COLOR if ear < 0 else (AWAKE_COLOR if ear > 0.23 else WARNING_COLOR)
    ear_text  = f"EAR: {ear:.3f}" if ear >= 0 else "EAR: --"
    cv2.putText(frame, ear_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 1)

    # Model prediction (top left)
    if model_label:
        m_color = DROWSY_COLOR if model_label == "drowsy" else AWAKE_COLOR
        cv2.putText(frame, f"Model: {model_label} {model_conf*100:.0f}%",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m_color, 1)

    # Main status (large, centre top)
    status      = "DROWSY" if is_drowsy else "AWAKE"
    status_color = DROWSY_COLOR if is_drowsy else AWAKE_COLOR
    reason_text = f"  ({drowsy_reason})" if is_drowsy and drowsy_reason else ""
    cv2.putText(frame, status + reason_text,
                (w // 2 - 120, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, status_color, 2)

    # Drowsiness timer bar
    if drowsy_seconds > 0:
        progress  = min(drowsy_seconds / alert_seconds, 1.0)
        bar_w     = int(w * 0.6)
        bar_x     = (w - bar_w) // 2
        bar_y     = h - 40
        bar_h     = 18
        fill_w    = int(bar_w * progress)
        bar_color = WARNING_COLOR if progress < 1.0 else DROWSY_COLOR

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"DROWSY  {drowsy_seconds:.1f}s",
                    (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    # ALERT banner
    if drowsy_seconds >= alert_seconds:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 35), (w, h // 2 + 35),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "!! DROWSINESS ALERT !!",
                    (w // 2 - 195, h // 2 + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION — auto-measure your personal open-eye EAR baseline
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_ear(cap, face_mesh, left_eye, right_eye, seconds=3):
    print(f"\n👁  CALIBRATION: Keep eyes OPEN for {seconds} seconds...")
    print("   This measures YOUR normal eye openness.\n")
    ear_samples = []
    start = time.time()

    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        remaining = seconds - (time.time() - start)
        cv2.putText(frame, f"CALIBRATING — keep eyes OPEN: {remaining:.1f}s",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Drowsiness Detection — press Q to quit", frame)
        cv2.waitKey(1)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            l_ear = eye_aspect_ratio(lms, left_eye,  w, h)
            r_ear = eye_aspect_ratio(lms, right_eye, w, h)
            avg   = (l_ear + r_ear) / 2.0
            ear_samples.append(avg)

    if len(ear_samples) < 5:
        print("⚠️  Could not calibrate — using default EAR threshold 0.22")
        return 0.22

    baseline = np.mean(ear_samples)
    # Drowsy threshold = 80% of open-eye baseline
    threshold = round(baseline * 0.80, 3)
    print(f"   Open-eye EAR baseline : {baseline:.3f}")
    print(f"   Drowsy threshold set  : {threshold:.3f}")
    print(f"   (will alert when EAR drops below {threshold:.3f})\n")
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(source, ear_threshold, model_threshold, alert_seconds, skip_calibration):
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        HAS_MEDIAPIPE = True
        print("✅ MediaPipe ready")
    except ImportError:
        print("⚠️  MediaPipe not found — install with: pip install mediapipe")
        print("   Running model-only mode (less accurate for eye detection)")
        HAS_MEDIAPIPE = False
        mp_face_mesh  = None

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    config    = load_config(CONFIG_PATH)
    classes   = config["classes"]
    model     = load_model(MODEL_PATH, config, device)
    transform = make_transform(config) if model else None

    # Haar cascade for face bounding box (used for model crop)
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open video source: {source}")

    # ── Calibration ──────────────────────────────────────────────────────────
    if HAS_MEDIAPIPE and not skip_calibration:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as fm:
            ear_threshold = calibrate_ear(
                cap, fm, LEFT_EYE, RIGHT_EYE, seconds=3
            )
    else:
        print(f"Using EAR threshold: {ear_threshold}")

    print(f"Model confidence threshold : {model_threshold}")
    print(f"Alert after                : {alert_seconds}s")
    print(f"\n✅ Running — press Q to quit\n")

    drowsy_start  = None
    alert_played  = False
    prev_time     = time.time()

    # EAR smoothing buffer
    ear_buffer    = []
    EAR_SMOOTH    = 5   # average over last N frames

    face_mesh_ctx = (
        mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if HAS_MEDIAPIPE else None
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror — more natural for webcam
            h, w  = frame.shape[:2]

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            ear_drowsy    = False
            model_drowsy  = False
            avg_ear       = -1.0
            model_label   = None
            model_conf    = 0.0
            drowsy_reason = ""

            # ── EAR detection via MediaPipe ───────────────────────────────
            if face_mesh_ctx is not None:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh_ctx.process(rgb)

                if result.multi_face_landmarks:
                    lms   = result.multi_face_landmarks[0].landmark
                    l_ear = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
                    r_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                    avg_ear = (l_ear + r_ear) / 2.0

                    # Smooth EAR over last N frames
                    ear_buffer.append(avg_ear)
                    if len(ear_buffer) > EAR_SMOOTH:
                        ear_buffer.pop(0)
                    smooth_ear = np.mean(ear_buffer)

                    # Draw eye landmarks
                    eye_color = DROWSY_COLOR if smooth_ear < ear_threshold else AWAKE_COLOR
                    draw_eye_landmarks(frame, lms, LEFT_EYE,  w, h, eye_color)
                    draw_eye_landmarks(frame, lms, RIGHT_EYE, w, h, eye_color)

                    if smooth_ear < ear_threshold:
                        ear_drowsy    = True
                        drowsy_reason = f"eyes closing (EAR {smooth_ear:.2f})"

            # ── Model detection via Haar + EfficientNet ───────────────────
            if model is not None:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                for (x, y, fw, fh) in faces:
                    pad  = int(fw * 0.1)
                    x1   = max(0, x - pad)
                    y1   = max(0, y - pad)
                    x2   = min(w, x + fw + pad)
                    y2   = min(h, y + fh + pad)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    model_label, model_conf = model_predict(
                        crop, model, transform, device, classes
                    )
                    box_color = (DROWSY_COLOR if model_label == "drowsy"
                                 and model_conf >= model_threshold
                                 else AWAKE_COLOR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    if model_label == "drowsy" and model_conf >= model_threshold:
                        model_drowsy  = True
                        if not drowsy_reason:
                            drowsy_reason = f"model {model_conf*100:.0f}%"

            # ── Combined decision ─────────────────────────────────────────
            is_drowsy = ear_drowsy or model_drowsy

            if is_drowsy:
                if drowsy_start is None:
                    drowsy_start = now
                drowsy_seconds = now - drowsy_start
                if drowsy_seconds >= alert_seconds and not alert_played:
                    play_alert()
                    alert_played = True
            else:
                drowsy_start   = None
                drowsy_seconds = 0.0
                alert_played   = False

            # ── Draw ─────────────────────────────────────────────────────
            frame = draw_hud(
                frame,
                is_drowsy,
                drowsy_reason,
                avg_ear,
                model_label,
                model_conf,
                drowsy_seconds if drowsy_start else 0.0,
                alert_seconds,
                fps,
            )

            cv2.imshow("Drowsiness Detection — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    finally:
        if face_mesh_ctx is not None:
            face_mesh_ctx.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",           default="0",
                        help="Webcam index or video file path")
    parser.add_argument("--ear_threshold",    type=float, default=0.22,
                        help="EAR threshold — lower = less sensitive (default 0.22)")
    parser.add_argument("--model_threshold",  type=float, default=0.40,
                        help="Model confidence threshold (default 0.40)")
    parser.add_argument("--alert",            type=float, default=1.5,
                        help="Seconds before alert triggers (default 1.5)")
    parser.add_argument("--no_calibration",   action="store_true",
                        help="Skip EAR calibration and use default threshold")
    args = parser.parse_args()

    run(
        source=args.source,
        ear_threshold=args.ear_threshold,
        model_threshold=args.model_threshold,
        alert_seconds=args.alert,
        skip_calibration=args.no_calibration,
    )                       
