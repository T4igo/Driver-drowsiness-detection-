# ◈ Driver Drowsiness Detection System

A professional, real-time dual-mode driver drowsiness detection system designed to prevent accidents by monitoring driver alertness. 

This project utilizes a custom-trained **EfficientNet-B0** model for image-based drowsiness classification, paired with **MediaPipe FaceMesh** for precise Eye Aspect Ratio (EAR) calculations. 

## Features
- **Dual-Mode Detection**: Combines Deep Learning (EfficientNet-B0) and facial landmark geometry (EAR) for robust performance.
- **Auto-Calibration**: Automatically measures your personal baseline "open eye" state for highly accurate drowsiness thresholds.
- **Car Dashboard GUI**: A premium `OpenCV`-based dashboard showing real-time stats, EAR history, and visual alerts (`drowsiness_demo.py`).
- **Web Interface**: A lightweight, standalone HTML/JS version (`drowsiness_web.html`) that uses MediaPipe FaceMesh to run entirely in the browser.
- **Audio Alarms**: High-contrast visual alerts and audio alarms that trigger when drowsiness is detected for a sustained duration.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/T4igo/Driver-drowsiness-detection.git
   cd Driver-drowsiness-detection
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Professional Dashboard (Python)
Run the fully-featured desktop application with the car-dashboard UI:

```bash
python drowsiness_demo.py
```
**Flags:**
- `--source 1` : Use a different webcam (default is `0`).
- `--model_threshold 0.40` : Adjust model sensitivity.
- `--no_calibration` : Skip the initial 3-second auto-calibration.

### 2. Standalone Inference Script
Run the standard inference script for testing or integrating into other pipelines:

```bash
python drowsiness_inference_v3.py
```
**Flags:**
- `--ear_threshold 0.22` : Manually set the EAR threshold.
- `--alert 1.5` : Seconds of continuous drowsiness before triggering the alarm.

### 3. Web Application
If you prefer running the detector in your browser without installing heavy dependencies:

1. Start a local server:
   ```bash
   python -m http.server
   ```
2. Open your browser and navigate to `http://localhost:8000/drowsiness_web.html`

## 🧠 How it Works

1. **Eye Aspect Ratio (EAR)**: Uses 468 facial landmarks from MediaPipe to compute the distance between the eyelids. If the ratio falls below the calibrated threshold, it flags the driver as drowsy.
2. **EfficientNet-B0**: The webcam frame is cropped around the face using Haar Cascades and fed into a fine-tuned EfficientNet-B0 model (`drowsiness_model.pth`).
3. **Alert Logic**: If either the EAR or the deep learning model detects drowsiness consistently for a set duration (e.g., 1.5 to 2.0 seconds), an alarm is triggered.
