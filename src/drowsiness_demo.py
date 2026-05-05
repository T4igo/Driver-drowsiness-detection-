"""
drowsiness_demo.py
Professional car-dashboard style drowsiness detection demo.
Designed for professor/presentation demos.

Install:
    pip install timm opencv-python mediapipe numpy torch torchvision pygame

Run:
    python drowsiness_demo.py
"""

import argparse, json, time, collections, threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm, torch.nn as nn

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "drowsiness_model.pth"
CONFIG_PATH = "model_config.json"

# ── MediaPipe eye landmark indices ────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Dashboard colours (BGR) ───────────────────────────────────────────────────
C_BG        = (15,  17,  20)
C_PANEL     = (22,  26,  32)
C_BORDER    = (40,  48,  58)
C_ACCENT    = (0,   200, 255)
C_GREEN     = (50,  220, 120)
C_RED       = (50,   60, 230)
C_ORANGE    = (30,  160, 255)
C_WHITE     = (220, 225, 230)
C_GREY      = (90,  100, 115)
C_DARK_RED  = (30,   30, 160)
C_YELLOW    = (0,   230, 255)

ALERT_SECS  = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ALARM  (pygame sine-wave, runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────
_alarm_active = False
_alarm_thread = None
_alarm_lock   = threading.Lock()


def _make_beep(frequency=880, duration_ms=380, volume=0.9, sample_rate=44100):
    """Return a pygame Sound object for a pure sine-wave tone."""
    import pygame
    n = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])          # pygame needs stereo
    return pygame.sndarray.make_sound(stereo)


def _alarm_loop():
    """Plays alternating hi/lo beeps until _alarm_active is False."""
    global _alarm_active
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        beep_hi = _make_beep(frequency=960, duration_ms=350)
        beep_lo = _make_beep(frequency=620, duration_ms=350)
        while True:
            with _alarm_lock:
                if not _alarm_active:
                    break
            beep_hi.play()
            time.sleep(0.40)
            with _alarm_lock:
                if not _alarm_active:
                    break
            beep_lo.play()
            time.sleep(0.40)
    except Exception:
        # Fallback to terminal bell if pygame fails
        while True:
            with _alarm_lock:
                if not _alarm_active:
                    break
            print("\a", end="", flush=True)
            time.sleep(0.80)


def start_alarm():
    global _alarm_active, _alarm_thread
    with _alarm_lock:
        if _alarm_active:
            return                                   # already running
        _alarm_active = True
    _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
    _alarm_thread.start()


def stop_alarm():
    global _alarm_active
    with _alarm_lock:
        _alarm_active = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_config(path):
    if not Path(path).exists():
        return {"architecture":"efficientnet_b0","img_size":96,
                "classes":["awake","drowsy"],"num_classes":2,
                "normalize_mean":[0.485,0.456,0.406],
                "normalize_std":[0.229,0.224,0.225]}
    with open(path) as f:
        return json.load(f)

def build_model(n):
    m = timm.create_model("efficientnet_b0", pretrained=False)
    f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.40), nn.Linear(f,256), nn.SiLU(),
        nn.Dropout(0.30), nn.Linear(256,n))
    return m

def load_model(path, cfg, dev):
    if not Path(path).exists():
        return None
    m = build_model(cfg["num_classes"])
    m.load_state_dict(torch.load(path, map_location=dev))
    m.to(dev).eval()
    return m

def make_tfm(cfg):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
    ])

@torch.no_grad()
def model_predict(crop, model, tfm, dev, classes):
    if model is None: return None, 0.0
    t = tfm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(dev)
    p = F.softmax(model(t), dim=1)[0]
    i = p.argmax().item()
    return classes[i], p[i].item()

# ─────────────────────────────────────────────────────────────────────────────
# EAR
# ─────────────────────────────────────────────────────────────────────────────
def ear(lms, idx, w, h):
    pts = [(lms[i].x*w, lms[i].y*h) for i in idx]
    A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
    return (A+B)/(2*C+1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def filled_rect(img, x, y, w, h, color, alpha=1.0):
    if alpha < 1.0:
        sub = img[y:y+h, x:x+w]
        rect = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
        img[y:y+h, x:x+w] = sub
    else:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, -1)

def text(img, s, x, y, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, s, (x,y), font, scale, color, thickness, cv2.LINE_AA)

def text_c(img, s, cx, y, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw,_),_ = cv2.getTextSize(s, font, scale, thickness)
    cv2.putText(img, s, (cx-tw//2, y), font, scale, color, thickness, cv2.LINE_AA)

def arc_progress(img, cx, cy, r, pct, color, bg_color, thickness=8):
    cv2.ellipse(img,(cx,cy),(r,r),0,0,360,bg_color,thickness,cv2.LINE_AA)
    end = int(360*pct)
    if end > 0:
        cv2.ellipse(img,(cx,cy),(r,r),-90,-0,end,color,thickness,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# ALARM SIGN
# ─────────────────────────────────────────────────────────────────────────────
def draw_alarm_sign(img, cx, cy, size, pulse):
    """Pulsing ⚠ triangle."""
    glow_size = int(size * 1.35)
    glow_pts  = np.array([[cx, cy-glow_size],
                           [cx-glow_size, cy+glow_size],
                           [cx+glow_size, cy+glow_size]], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [glow_pts], C_RED)
    cv2.addWeighted(overlay, pulse*0.45, img, 1-pulse*0.45, 0, img)

    tri_pts = np.array([[cx, cy-size],
                         [cx-size, cy+size],
                         [cx+size, cy+size]], np.int32)
    cv2.fillPoly(img, [tri_pts], C_ORANGE)
    cv2.polylines(img, [tri_pts], True, C_WHITE, 2, cv2.LINE_AA)
    text_c(img, "!", cx, cy+size-8, 1.1, (20,20,20), 3)
    text_c(img, "!", cx, cy+size-8, 1.1, C_WHITE, 1)


def draw_fullscreen_alarm(img, W, H, pulse):
    """Red border + triangle + flashing WAKE UP banner."""
    cv2.rectangle(img, (0,0), (W,H), C_RED, int(12+pulse*10))
    draw_alarm_sign(img, W-90, 115, 45, pulse)
    text_c(img, "DROWSY ALERT", W-90, 180, 0.45, C_RED, 1)
    if pulse > 0.55:
        bh = 64
        by = H//2 - bh//2
        filled_rect(img, 0, by, W, bh, (10,10,10), 0.75)
        text_c(img, "WAKE  UP!", W//2, by+46, 1.8, C_DARK_RED, 6, cv2.FONT_HERSHEY_DUPLEX)
        text_c(img, "WAKE  UP!", W//2, by+46, 1.8, C_YELLOW,   2, cv2.FONT_HERSHEY_DUPLEX)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, W=1280, H=720):
        self.W = W; self.H = H
        self.alert_log    = collections.deque(maxlen=6)
        self.ear_history  = collections.deque(maxlen=120)
        self.total_alerts = 0
        self.session_start = time.time()

    def render(self, cam_frame, state):
        canvas = np.full((self.H, self.W, 3), C_BG, dtype=np.uint8)
        is_drowsy   = state["is_drowsy"]
        drowsy_secs = state["drowsy_secs"]
        ear_val     = state["ear"]
        model_label = state["model_label"]
        model_conf  = state["model_conf"]
        fps         = state["fps"]
        ear_thresh  = state["ear_thresh"]
        self.ear_history.append(ear_val if ear_val>=0 else (ear_thresh or 0.22))

        if drowsy_secs >= ALERT_SECS:
            filled_rect(canvas, 0, 0, self.W, self.H, C_DARK_RED,
                        0.18+0.10*abs(np.sin(time.time()*4)))

        # top bar
        filled_rect(canvas, 0, 0, self.W, 52, C_PANEL)
        cv2.line(canvas,(0,52),(self.W,52),C_BORDER,1)
        text(canvas,"DRIVER MONITORING SYSTEM",24,34,0.7,C_ACCENT,1,cv2.FONT_HERSHEY_DUPLEX)
        sm,ss = divmod(int(time.time()-self.session_start),60)
        text(canvas,f"SESSION  {sm:02d}:{ss:02d}",self.W-220,34,0.55,C_GREY)
        text(canvas,f"FPS {fps:.0f}",self.W-60,34,0.45,C_GREY)

        # camera
        CAM_X,CAM_Y,CAM_W,CAM_H = 24,68,640,480
        if cam_frame is not None:
            canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cv2.resize(cam_frame,(CAM_W,CAM_H))
        cv2.rectangle(canvas,(CAM_X,CAM_Y),(CAM_X+CAM_W,CAM_Y+CAM_H),
                      C_RED if is_drowsy else C_GREEN, 2)
        filled_rect(canvas,CAM_X,CAM_Y,160,28,C_PANEL,0.85)
        text(canvas,"LIVE FEED",CAM_X+10,CAM_Y+19,0.45,C_ACCENT)

        # status badge
        SY = CAM_Y+CAM_H+14
        filled_rect(canvas,CAM_X,SY,CAM_W,56,C_PANEL)
        cv2.rectangle(canvas,(CAM_X,SY),(CAM_X+CAM_W,SY+56),C_BORDER,1)
        text_c(canvas,"DROWSINESS DETECTED" if is_drowsy else "DRIVER ALERT",
               CAM_X+CAM_W//2, SY+36, 0.85,
               C_RED if is_drowsy else C_GREEN, 2, cv2.FONT_HERSHEY_DUPLEX)

        # right panel
        RX = CAM_X+CAM_W+24
        RW = self.W-RX-24

        ARC_CX,ARC_CY,ARC_R = RX+RW//2, 200, 85
        pct = min(drowsy_secs/ALERT_SECS,1.0) if is_drowsy else 0.0
        arc_col = C_RED if pct>=1.0 else (C_ORANGE if pct>0.3 else C_GREEN)
        arc_progress(canvas,ARC_CX,ARC_CY,ARC_R,pct,arc_col,C_BORDER,thickness=10)
        text_c(canvas,f"{drowsy_secs:.1f}s",ARC_CX,ARC_CY+8,1.0,arc_col,2)
        text_c(canvas,"DROWSY TIMER",ARC_CX,ARC_CY+30,0.38,C_GREY)
        if pct>=1.0:
            ra = 0.5+0.5*abs(np.sin(time.time()*5))
            cv2.ellipse(canvas,(ARC_CX,ARC_CY),(ARC_R+14,ARC_R+14),
                        0,0,360,C_RED,int(ra*6+1),cv2.LINE_AA)

        def stat_box(x,y,w,h,label,value,vc):
            filled_rect(canvas,x,y,w,h,C_PANEL)
            cv2.rectangle(canvas,(x,y),(x+w,y+h),C_BORDER,1)
            text_c(canvas,label,x+w//2,y+h-28,0.35,C_GREY)
            text_c(canvas,value,x+w//2,y+h-8, 0.52,vc,1)

        BY = ARC_CY+ARC_R+18; BW = (RW-10)//3
        stat_box(RX,BY,BW,58,"EAR",
                 f"{ear_val:.3f}" if ear_val>=0 else "---",
                 C_RED if (ear_val>=0 and ear_thresh and ear_val<ear_thresh) else C_GREEN)
        stat_box(RX+BW+5,BY,BW,58,"MODEL",
                 f"{model_conf*100:.0f}%" if model_label else "---",
                 C_RED if model_label=="drowsy" else C_GREEN)
        stat_box(RX+2*(BW+5),BY,BW,58,"ALERTS",str(self.total_alerts),C_ORANGE)

        GY,GH = BY+70, 90
        filled_rect(canvas,RX,GY,RW,GH,C_PANEL)
        cv2.rectangle(canvas,(RX,GY),(RX+RW,GY+GH),C_BORDER,1)
        text(canvas,"EAR HISTORY",RX+8,GY+14,0.38,C_GREY)
        if len(self.ear_history)>2:
            pts=[]; mn,mx=0.10,0.40
            for i,v in enumerate(self.ear_history): pts.append(v)
            for i in range(1,len(pts)):
                x1=RX+int((i-1)/120*RW); x2=RX+int(i/120*RW)
                y1=GY+GH-int((pts[i-1]-mn)/(mx-mn)*GH)
                y2=GY+GH-int((pts[i]-mn)/(mx-mn)*GH)
                y1=max(GY+2,min(GY+GH-2,y1)); y2=max(GY+2,min(GY+GH-2,y2))
                cv2.line(canvas,(x1,y1),(x2,y2),
                         C_RED if pts[i]<(ear_thresh or 0.22) else C_GREEN,1,cv2.LINE_AA)
        if ear_thresh:
            ty=GY+GH-int((ear_thresh-0.10)/0.30*GH)
            ty=max(GY+2,min(GY+GH-2,ty))
            cv2.line(canvas,(RX,ty),(RX+RW,ty),C_ORANGE,1)
            text(canvas,"THRESH",RX+4,ty-3,0.3,C_ORANGE)

        LY=GY+GH+14; LH=self.H-LY-24
        filled_rect(canvas,RX,LY,RW,LH,C_PANEL)
        cv2.rectangle(canvas,(RX,LY),(RX+RW,LY+LH),C_BORDER,1)
        text(canvas,"ALERT LOG",RX+8,LY+16,0.40,C_GREY)
        for i,entry in enumerate(reversed(list(self.alert_log))):
            ty2=LY+32+i*20
            if ty2+16>LY+LH: break
            text(canvas,entry,RX+8,ty2,0.36,C_ORANGE if i==0 else C_GREY)

        # alarm sign — drawn last, always on top
        if drowsy_secs >= ALERT_SECS:
            pulse = abs(np.sin(time.time()*4))
            draw_fullscreen_alarm(canvas, self.W, self.H, pulse)

        # bottom bar
        filled_rect(canvas,0,self.H-30,self.W,30,C_PANEL)
        cv2.line(canvas,(0,self.H-30),(self.W,self.H-30),C_BORDER,1)
        text(canvas,"EfficientNet-B0  |  Val Accuracy 89%  |  Drowsy Recall 94%",
             24,self.H-10,0.38,C_GREY)
        text(canvas,"Press Q to quit",self.W-160,self.H-10,0.38,C_GREY)
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(cap, face_mesh, W, H, secs=3):
    print(f"\n👁  Keep eyes OPEN for {secs}s — calibrating...\n")
    samples, start = [], time.time()
    while time.time()-start < secs:
        ret,frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w   = frame.shape[:2]
        res   = face_mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        rem   = secs-(time.time()-start)
        cal   = np.full((H,W,3),C_BG,dtype=np.uint8)
        filled_rect(cal,W//2-300,H//2-60,600,120,C_PANEL)
        cv2.rectangle(cal,(W//2-300,H//2-60),(W//2+300,H//2+60),C_ACCENT,2)
        text_c(cal,"CALIBRATION",W//2,H//2-20,0.9,C_ACCENT,2,cv2.FONT_HERSHEY_DUPLEX)
        text_c(cal,f"Keep eyes OPEN — {rem:.1f}s",W//2,H//2+20,0.6,C_WHITE)
        filled_rect(cal,W//2-300,H//2+40,int(600*(1-rem/secs)),10,C_ACCENT)
        cv2.imshow("Drowsiness Detection",cal); cv2.waitKey(1)
        if res.multi_face_landmarks:
            lms=res.multi_face_landmarks[0].landmark
            samples.append((ear(lms,LEFT_EYE,w,h)+ear(lms,RIGHT_EYE,w,h))/2)
    if len(samples)<5:
        print("⚠️  Calibration failed — using default 0.22"); return 0.22
    baseline  = np.mean(samples)
    threshold = round(baseline*0.78,3)
    print(f"   Baseline EAR : {baseline:.3f}\n   Threshold    : {threshold:.3f}\n")
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(source, no_cal, model_thresh):
    try:
        import mediapipe as mp
        mp_fm=mp.solutions.face_mesh; HAS_MP=True
    except ImportError:
        print("⚠️  pip install mediapipe"); HAS_MP=False; mp_fm=None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config(CONFIG_PATH)
    model  = load_model(MODEL_PATH, cfg, device)
    tfm    = make_tfm(cfg) if model else None
    haar   = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    W,H  = 1280,720
    dash = Dashboard(W,H)

    ear_thresh = 0.22
    face_mesh  = None
    if HAS_MP:
        face_mesh = mp_fm.FaceMesh(max_num_faces=1,refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
        if not no_cal:
            ear_thresh = calibrate(cap,face_mesh,W,H)

    cv2.namedWindow("Drowsiness Detection",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drowsiness Detection",W,H)

    drowsy_start   = None
    alert_logged   = False
    ear_buf        = collections.deque(maxlen=6)
    prev_t         = time.time()

    print("✅ Running — press Q to quit\n")

    while True:
        ret,frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        fh,fw = frame.shape[:2]

        now=time.time(); fps=1.0/max(now-prev_t,1e-6); prev_t=now

        ear_val=ear_drowsy=-1.0 if False else -1.0
        ear_drowsy=model_drowsy=False
        model_label=None; model_conf=0.0
        ear_val=-1.0

        if face_mesh:
            res=face_mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lms=res.multi_face_landmarks[0].landmark
                l=ear(lms,LEFT_EYE,fw,fh); r=ear(lms,RIGHT_EYE,fw,fh)
                ear_val=(l+r)/2; ear_buf.append(ear_val)
                smooth=np.mean(ear_buf)
                eye_col=C_RED if smooth<ear_thresh else C_GREEN
                for idx in LEFT_EYE+RIGHT_EYE:
                    lm=lms[idx]
                    cv2.circle(frame,(int(lm.x*fw),int(lm.y*fh)),2,eye_col,-1)
                if smooth<ear_thresh: ear_drowsy=True

        if model:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=haar.detectMultiScale(gray,1.1,5,minSize=(60,60))
            for (x,y,bw,bh) in faces:
                pad=int(bw*0.1)
                crop=frame[max(0,y-pad):min(fh,y+bh+pad),max(0,x-pad):min(fw,x+bw+pad)]
                if crop.size==0: continue
                model_label,model_conf=model_predict(crop,model,tfm,device,cfg["classes"])
                box_col=(C_RED if model_label=="drowsy" and model_conf>=model_thresh else C_GREEN)
                cv2.rectangle(frame,(x,y),(x+bw,y+bh),box_col,2)
                if model_label=="drowsy" and model_conf>=model_thresh: model_drowsy=True

        is_drowsy = ear_drowsy or model_drowsy

        if is_drowsy:
            if drowsy_start is None:
                drowsy_start = now
                alert_logged = False
            drowsy_secs = now - drowsy_start

            if drowsy_secs >= ALERT_SECS:
                start_alarm()                          # starts beeping (no-op if already on)
                if not alert_logged:
                    alert_logged = True
                    dash.total_alerts += 1
                    reason = "EAR" if ear_drowsy else "MODEL"
                    dash.alert_log.append(
                        f"[{time.strftime('%H:%M:%S')}]  Alert #{dash.total_alerts}  ({reason})")
        else:
            if drowsy_start is not None:
                stop_alarm()                           # stop beeping immediately
            drowsy_start = None
            drowsy_secs  = 0.0
            alert_logged = False

        state = dict(
            is_drowsy   = is_drowsy,
            drowsy_secs = drowsy_secs if drowsy_start else 0.0,
            ear         = float(np.mean(ear_buf)) if ear_buf else -1.0,
            ear_thresh  = ear_thresh,
            model_label = model_label,
            model_conf  = model_conf,
            fps         = fps,
        )

        canvas = dash.render(frame, state)
        cv2.imshow("Drowsiness Detection", canvas)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    stop_alarm()
    if face_mesh: face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session ended. Total alerts: {dash.total_alerts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",          default="0")
    p.add_argument("--model_threshold", type=float, default=0.40)
    p.add_argument("--no_calibration",  action="store_true")
    a = p.parse_args()
    run(a.source, a.no_calibration, a.model_threshold)
