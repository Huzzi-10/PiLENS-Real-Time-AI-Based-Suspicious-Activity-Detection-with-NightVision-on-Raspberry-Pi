# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0
# See LICENSE file in project root for full license information.


import sys, os, time, threading, subprocess
from datetime import datetime
from pathlib import Path
from collections import deque

# --- CV / ML ---
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# --- GUI ---
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QListWidget, QListWidgetItem, QMessageBox, QDialog, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen

# --- Flask (MJPEG) ---
from flask import Flask, Response

# --- GPIO optional ---
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

# --- Email libs (kept from your code) ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ------------------------- YOUR ORIGINAL CONFIG -------------------------
USE_VIDEO = False
VIDEO_PATH = "/home/huzaifa/Desktop/Main Project VigilantEye/Camera.mp4"
CLASSIFIER_MODEL_PATH = "/home/huzaifa/Desktop/Main Project VigilantEye/action_model.pth"
SEQUENCE_LEN = 16
IMG_SIZE = 112
CLASSES = ["Normal", "Suspicious"]
PROCESS_FPS = 5
BUFFER_SEC = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LED_PIN = 17
BUZZER_PIN = 18

MODEL_PATH = "yolo11n.pt"  # YOLO model file (your original)
FRAME_SIZE = (1280, 720)
FPS = 20

MIN_RECORD_SECONDS = 5.0
CONF_THRESHOLD = 0.60

# Email credentials (as you provided — keep secure)
SENDER_EMAIL = "huzaifaasad17805@gmail.com"
SENDER_PASSWORD = "xuzb lfuo puqi xona"
RECEIVER_EMAIL = "huzaifaasadlm10@gmail.com"

# Folders
INTRUDER_FOLDER = "Intruders"
RECORDINGS_FOLDER = "recordings"
Path(INTRUDER_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RECORDINGS_FOLDER).mkdir(parents=True, exist_ok=True)

# ------------------------- SAFE GPIO SETUP -------------------------
if _HAS_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# ------------------------- EMAIL FUNCTION (your logic preserved) -------------------------
def send_email_alert(video_path, label):
    try:
        sender_email = SENDER_EMAIL
        sender_password = SENDER_PASSWORD
        receiver_email = RECEIVER_EMAIL
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"⚠ VigilantEye Alert - {label} detected"

        body = f"Suspicious activity detected!\n\nTime: {timestamp}\nActivity: {label}"
        msg.attach(MIMEText(body, 'plain'))

        with open(video_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(video_path)}')
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"✓ Email sent with video: {video_path}")
    except Exception as e:
        print("Email send failed:", e)

# ------------------------- CNN-LSTM (kept same) -------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, num_classes=len(CLASSES)):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load classifier
classifier = CNN_LSTM().to(DEVICE)
try:
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()
except Exception as e:
    print("Warning: could not load classifier model:", e)

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

# YOLO
yolo = YOLO(MODEL_PATH)

# ------------------------- VIDEO / CAMERA CHOICE -------------------------
USE_CAMERA = not USE_VIDEO

# ------------------------- helpers -------------------------
def start_video_writer(path, fps=FPS, size=FRAME_SIZE):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, size)

# ------------------------- MJPEG Flask -------------------------
flask_app = Flask(__name__)
latest_stream_frame = None
_stream_lock = threading.Lock()
_FLASK_RUNNING = False

@flask_app.route('/video')
def video_feed():
    def generate():
        while True:
            with _stream_lock:
                frame = latest_stream_frame
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_server(host="0.0.0.0", port=8000):
    global _FLASK_RUNNING
    if _FLASK_RUNNING:
        return
    _FLASK_RUNNING = True
    flask_app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)

# ------------------------- ToggleSwitch UI widget -------------------------
class ToggleSwitch(QWidget):
    toggled = pyqtSignal(bool)
    def __init__(self, parent=None, width=60, height=32):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._checked = True
        self._offset = 1.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(160)
        self._bg_off = QColor("#bbbbbb")
        self._bg_on = QColor("#4caf50")
        self._knob = QColor("#ffffff")
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        bg_color = self._bg_on if self._checked else self._bg_off
        p.setBrush(QBrush(bg_color)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, r.height()/2, r.height()/2)
        margin = 3
        knob_d = r.height() - 2*margin
        x_left = margin
        x_right = r.width() - margin - knob_d
        x = x_left + (x_right - x_left) * self._offset
        knob_rect = QRectF(x, margin, knob_d, knob_d)
        p.setBrush(QBrush(self._knob)); p.setPen(QPen(QColor(180,180,180), 0.5))
        p.drawEllipse(knob_rect)
        p.end()
    def mouseReleaseEvent(self, ev):
        self.setChecked(not self._checked)
        super().mouseReleaseEvent(ev)
    def isChecked(self): return self._checked
    def setChecked(self, val: bool):
        if self._checked == val: return
        self._checked = bool(val)
        start = self._offset; end = 1.0 if self._checked else 0.0
        self._anim.stop(); self._anim.setStartValue(start); self._anim.setEndValue(end); self._anim.start()
        self.toggled.emit(self._checked); self.update()
    def toggle(self): self.setChecked(not self._checked)
    def getOffset(self): return self._offset
    def setOffset(self, v): self._offset = float(v); self.update()
    offset = pyqtProperty(float, fget=getOffset, fset=setOffset)

# ------------------------- CameraThread (wraps your main loop) -------------------------
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)
    alert_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._picam2 = None
        self._cap = None

        # local copies of state
        self.frame_buffer = deque(maxlen=BUFFER_SEC * PROCESS_FPS)
        self.person_sequences = {}
        self.person_labels = {}
        self.last_processed = 0
        self.processing_interval = 1.0 / PROCESS_FPS
        self.prev_time = time.time()

        # manual recording
        self._manual_recording = False
        self._manual_writer = None
        self._manual_path = None

    def setup(self):
        try:
            if USE_VIDEO:
                self._cap = cv2.VideoCapture(VIDEO_PATH)
                self.log.emit("Using video file: " + VIDEO_PATH)
            else:
                self._picam2 = Picamera2()
                cfg = self._picam2.create_preview_configuration(main={"size": FRAME_SIZE})
                self._picam2.configure(cfg)
                self._picam2.start()
                time.sleep(0.4)
                self.log.emit("Picamera2 started")
        except Exception as e:
            self.log.emit(f"Error setting up camera: {e}")

    def start_manual_recording(self):
        if self._manual_recording:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(RECORDINGS_FOLDER, f"manual_{timestamp}.mp4")
        self._manual_writer = start_video_writer(path, fps=PROCESS_FPS, size=(FRAME_SIZE[0], FRAME_SIZE[1]))
        self._manual_path = path
        self._manual_recording = True
        self.log.emit(f"Manual recording started: {path}")

    def stop_manual_recording(self):
        if not self._manual_recording:
            return
        try:
            if self._manual_writer:
                self._manual_writer.release()
        except Exception:
            pass
        self._manual_writer = None
        self._manual_recording = False
        self.log.emit(f"Manual recording stopped: {self._manual_path}")

    def run(self):
        global latest_stream_frame, _stream_lock
        self._running = True
        self.setup()

        # warm start: convert classifier to eval already done earlier
        while self._running:
            try:
                # read frame
                if USE_VIDEO:
                    ret, frame = self._cap.read()
                    if not ret:
                        self.log.emit("Video finished or cannot read frame")
                        break
                else:
                    frame_rgb = self._picam2.capture_array()  # RGB
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                now = time.time()
                if now - self.last_processed < self.processing_interval:
                    # still need to emit/display frame for GUI and stream
                    # but skip heavy processing
                    pass
                self.last_processed = now

                # maintain buffer for suspicious clip saving
                self.frame_buffer.append(frame.copy())

                # YOLO detections
                results = yolo(frame, verbose=False)[0]
                current_pids = []

                for box in results.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # sanitize box
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    pid = f"{x1}_{y1}_{x2}_{y2}"
                    current_pids.append(pid)
                    if pid not in self.person_sequences:
                        self.person_sequences[pid] = []

                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    img_t = transform(rgb)
                    self.person_sequences[pid].append(img_t)

                    if len(self.person_sequences[pid]) == SEQUENCE_LEN:
                        seq_tensor = torch.stack(self.person_sequences[pid]).unsqueeze(0).to(DEVICE).float()
                        with torch.no_grad():
                            try:
                                out = classifier(seq_tensor)
                                pred = torch.argmax(out, dim=1).item()
                                label = CLASSES[pred]
                            except Exception as e:
                                label = "Normal"
                                self.log.emit(f"Classifier error: {e}")

                        self.person_labels[pid] = label
                        self.person_sequences[pid] = []

                        if label == "Suspicious":
                            try:
                                if _HAS_GPIO:
                                    GPIO.output(LED_PIN, GPIO.HIGH); GPIO.output(BUZZER_PIN, GPIO.HIGH)
                            except Exception:
                                pass
                            # save buffered clip
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            video_name = os.path.join(INTRUDER_FOLDER, f"suspicious_{timestamp}.mp4")
                            writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), PROCESS_FPS,
                                                     (frame.shape[1], frame.shape[0]))
                            for bf in self.frame_buffer:
                                writer.write(bf)
                            writer.release()
                            # send email (non-blocking)
                            threading.Thread(target=send_email_alert, args=(video_name, label), daemon=True).start()
                        else:
                            try:
                                if _HAS_GPIO:
                                    GPIO.output(LED_PIN, GPIO.LOW); GPIO.output(BUZZER_PIN, GPIO.LOW)
                            except Exception:
                                pass

                    draw_label = self.person_labels.get(pid, "Person")
                    color = (0, 255, 0) if draw_label == "Normal" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, draw_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # remove old person keys
                for pid in list(self.person_sequences.keys()):
                    if pid not in current_pids:
                        self.person_sequences.pop(pid, None)
                        self.person_labels.pop(pid, None)

                # FPS overlay
                new_time = time.time()
                fps = 1.0 / max(1e-6, (new_time - self.prev_time))
                self.prev_time = new_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # prepare MJPEG frame
                try:
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        with _stream_lock:
                            latest_stream_frame = jpeg.tobytes()
                except Exception:
                    pass

                # manual recording write
                if self._manual_recording and self._manual_writer is not None:
                    try:
                        self._manual_writer.write(cv2.resize(frame, FRAME_SIZE))
                    except Exception:
                        pass

                # emit frame to GUI
                self.frame_ready.emit(frame)

                # tiny sleep
                time.sleep(0.01)

            except Exception as e:
                self.log.emit(f"Camera runtime error: {e}")
                time.sleep(0.2)

        # cleanup
        try:
            if self._picam2:
                self._picam2.stop()
            if self._cap:
                self._cap.release()
            if self._manual_writer:
                self._manual_writer.release()
        except Exception:
            pass
        self.log.emit("Camera thread stopped")

    def stop(self):
        self._running = False
        self.wait()

# ------------------------- Intruder list dialog -------------------------
class IntruderListDialog(QDialog):
    def __init__(self, parent=None, folder=INTRUDER_FOLDER):
        super().__init__(parent)
        self.setWindowTitle("Intruders")
        self.resize(820, 420)
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Refresh"); open_btn = QPushButton("Open Selected"); remove_btn = QPushButton("Delete Selected")
        btn_layout.addWidget(load_btn); btn_layout.addWidget(open_btn); btn_layout.addWidget(remove_btn)
        layout.addWidget(self.list_widget); layout.addLayout(btn_layout); self.setLayout(layout)
        load_btn.clicked.connect(self.load_files); open_btn.clicked.connect(self.open_selected); remove_btn.clicked.connect(self.delete_selected)
        self.folder = folder
        self.load_files()
    def load_files(self):
        self.list_widget.clear()
        files = sorted(os.listdir(self.folder), reverse=True)
        for f in files:
            self.list_widget.addItem(QListWidgetItem(f))
    def open_selected(self):
        sel = self.list_widget.currentItem()
        if not sel: return
        path = os.path.join(self.folder, sel.text())
        try:
            if sys.platform.startswith('linux'): subprocess.Popen(['xdg-open', path])
            elif sys.platform == 'darwin': subprocess.Popen(['open', path])
            elif sys.platform == 'win32': os.startfile(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")
    def delete_selected(self):
        sel = self.list_widget.currentItem()
        if not sel: return
        path = os.path.join(self.folder, sel.text())
        try:
            os.remove(path); self.load_files()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not delete file: {e}")

# ------------------------- Tailscale wrapper -------------------------
class TailscaleManager:
    def is_installed(self):
        try:
            subprocess.run(["tailscale", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False
    def status(self):
        try:
            res = subprocess.run(["tailscale", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
            return res.stdout.strip()
        except Exception as e:
            return f"Error: {e}"
    def up(self):
        try:
            res = subprocess.run(["sudo", "tailscale", "up"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.stdout + ("\n" + res.stderr if res.stderr else "")
        except Exception as e:
            return f"Error: {e}"
    def down(self):
        try:
            res = subprocess.run(["sudo", "tailscale", "down"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return res.stdout + ("\n" + res.stderr if res.stderr else "")
        except Exception as e:
            return f"Error: {e}"

# ------------------------- Main Window -------------------------
APP_STYLESHEET = """
QWidget { background-color: #0d0f12; color: #e8e8e8; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
QFrame#videoFrame { background-color: #08090b; border: 2px solid #101214; border-radius: 10px; }
QLabel#videoLabel { border-radius: 8px; }
QFrame#alertBar { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #0f3b21, stop:1 #143e28); border-radius: 6px; min-height: 14px; }
QPushButton { background-color: #13161a; border: 2px solid #1f242b; padding: 10px 16px; border-radius: 10px; color: #d6d6d6; font-weight: 500; }
QPushButton#startBtn { background-color: #0f3b21; border: 2px solid #1a6f3e; color: #81ffba; }
QPushButton#stopBtn { background-color: #3b0f0f; border: 2px solid #6f1a1a; color: #ff8585; }
QTextEdit, QListWidget { background-color: #121417; border: 1px solid #22272e; border-radius: 8px; padding: 8px; color: #c7c7c7; }
QLabel#statusNormal { background: #07220f; color: #7ef7b8; padding: 6px 12px; border-radius: 12px; border: 1px solid #1a4b2d; }
QLabel#statusAlert { background: #2a0a0a; color: #ffb0b0; padding: 6px 12px; border-radius: 12px; border: 1px solid #5a1a1a; }
"""

LIGHT_THEME = """
QWidget { background-color: #e8e8e8; color: #111111; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
QFrame#videoFrame { background-color: #f0f0f0; border: 2px solid #ccc; border-radius: 10px; }
"""

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VigilantEye — Final GUI")
        self.resize(1200, 780)
        self.setStyleSheet(APP_STYLESHEET)
        self.current_theme = 'dark'

        # left: video area
        self.video_frame = QFrame(); self.video_frame.setObjectName('videoFrame')
        vbox = QVBoxLayout(self.video_frame)
        self.video_label = QLabel(); self.video_label.setObjectName('videoLabel')
        self.video_label.setMinimumSize(980, 560); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.video_label)
        self.alert_bar = QFrame(); self.alert_bar.setObjectName('alertBar'); self.alert_bar.setFixedHeight(14)
        vbox.addWidget(self.alert_bar)

        # right: controls
        self.start_btn = QPushButton("Start Monitoring"); self.start_btn.setObjectName('startBtn')
        self.stop_btn = QPushButton("Stop Monitoring"); self.stop_btn.setObjectName('stopBtn')
        self.start_manual_btn = QPushButton("Start Recording (Manual)")
        self.stop_manual_btn = QPushButton("Stop Recording (Manual)"); self.stop_manual_btn.setEnabled(False)
        self.view_intruders_btn = QPushButton("View Intruders")
        self.view_recordings_btn = QPushButton("View Recordings")
        self.open_intruder_folder_btn = QPushButton("Open Intruders Folder")
        self.open_recordings_folder_btn = QPushButton("Open Recordings Folder")
        self.toggle_buzzer_btn = QPushButton("Toggle Buzzer (Test)")

        self.status_chip = QLabel("NORMAL"); self.status_chip.setObjectName('statusNormal')
        self.theme_switch = ToggleSwitch(); self.theme_switch.setChecked(True); self.theme_switch.toggled.connect(self.on_theme_toggled)

        # tailscale / remote
        self.tailscale = TailscaleManager()
        self.ts_status = QLabel("Tailscale: Unknown")
        self.ts_enable_btn = QPushButton("Enable Tailscale")
        self.ts_disable_btn = QPushButton("Disable Tailscale")
        self.remote_btn = QPushButton("Start Remote Stream (port 8000)")

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)

        right_layout = QVBoxLayout()
        for w in [self.start_btn, self.stop_btn, self.start_manual_btn, self.stop_manual_btn,
                  self.view_intruders_btn, self.view_recordings_btn, self.open_intruder_folder_btn, self.open_recordings_folder_btn,
                  self.toggle_buzzer_btn]:
            right_layout.addWidget(w)

        right_layout.addSpacing(8)
        theme_row = QHBoxLayout(); theme_row.addWidget(QLabel("Theme:")); theme_row.addWidget(self.theme_switch); theme_row.addStretch()
        right_layout.addLayout(theme_row)
        right_layout.addSpacing(6)

        ts_row = QHBoxLayout(); ts_row.addWidget(self.ts_enable_btn); ts_row.addWidget(self.ts_disable_btn)
        right_layout.addWidget(self.ts_status); right_layout.addLayout(ts_row); right_layout.addWidget(self.remote_btn)
        right_layout.addSpacing(8)

        status_layout = QHBoxLayout(); status_layout.addWidget(QLabel("Status:")); status_layout.addWidget(self.status_chip); status_layout.addStretch()
        right_layout.addLayout(status_layout)
        right_layout.addWidget(QLabel("Logs:")); right_layout.addWidget(self.log_text)

        layout = QHBoxLayout(); layout.addWidget(self.video_frame, stretch=3); layout.addLayout(right_layout, stretch=1)
        self.setLayout(layout)

        # camera thread
        self.cam_thread = CameraThread()
        self.cam_thread.frame_ready.connect(self.update_frame)
        self.cam_thread.log.connect(self.log)
        self.cam_thread.alert_changed.connect(self.on_alert_changed)

        # connect signals
        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.start_manual_btn.clicked.connect(self.start_manual_recording)
        self.stop_manual_btn.clicked.connect(self.stop_manual_recording)
        self.view_intruders_btn.clicked.connect(self.view_intruders)
        self.view_recordings_btn.clicked.connect(self.view_recordings)
        self.open_intruder_folder_btn.clicked.connect(lambda: self.open_folder(INTRUDER_FOLDER))
        self.open_recordings_folder_btn.clicked.connect(lambda: self.open_folder(RECORDINGS_FOLDER))
        self.toggle_buzzer_btn.clicked.connect(self.toggle_buzzer)

        self.ts_enable_btn.clicked.connect(self.enable_tailscale)
        self.ts_disable_btn.clicked.connect(self.disable_tailscale)
        self.remote_btn.clicked.connect(self.start_remote_stream)

        # UI initial
        self.stop_btn.setEnabled(False)
        self.alert_pulse_timer = QTimer(); self.alert_pulse_timer.timeout.connect(self._pulse_alert_bar); self._pulse_state = 0
        # update tailscale label
        if not self.tailscale.is_installed():
            self.ts_status.setText("Tailscale: NOT INSTALLED"); self.ts_enable_btn.setEnabled(False); self.ts_disable_btn.setEnabled(False)
        else:
            self.update_tailscale_status_label()

        self._flask_thread = None

    # actions
    def start_monitoring(self):
        self.log("Starting monitoring...")
        self.cam_thread.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def stop_monitoring(self):
        self.log("Stopping monitoring...")
        self.cam_thread.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.on_alert_changed(False)

    def start_manual_recording(self):
        self.cam_thread.start_manual_recording()
        self.start_manual_btn.setEnabled(False); self.stop_manual_btn.setEnabled(True)

    def stop_manual_recording(self):
        self.cam_thread.stop_manual_recording()
        self.start_manual_btn.setEnabled(True); self.stop_manual_btn.setEnabled(False)

    def view_intruders(self):
        dlg = IntruderListDialog(self); dlg.exec()

    def view_recordings(self):
        dlg = IntruderListDialog(self, folder=RECORDINGS_FOLDER); dlg.exec()

    def open_folder(self, folder):
        try:
            p = os.path.abspath(folder)
            if sys.platform.startswith('linux'): subprocess.Popen(['xdg-open', p])
            elif sys.platform == 'darwin': subprocess.Popen(['open', p])
            elif sys.platform == 'win32': os.startfile(p)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")

    def toggle_buzzer(self):
        try:
            if _HAS_GPIO:
                GPIO.output(BUZZER_PIN, GPIO.HIGH); time.sleep(0.35); GPIO.output(BUZZER_PIN, GPIO.LOW); self.log("Buzzer toggled")
            else:
                self.log("GPIO not available on this system")
        except Exception as e:
            self.log(f"GPIO error: {e}")

    # tailscale
    def update_tailscale_status_label(self):
        try:
            s = self.tailscale.status()
            first = s.splitlines()[0] if s else "No status"
            self.ts_status.setText(f"Tailscale: {first}")
        except Exception:
            self.ts_status.setText("Tailscale: Error")

    def enable_tailscale(self):
        if not self.tailscale.is_installed():
            QMessageBox.warning(self, "Tailscale missing", "Tailscale is not installed.")
            return
        self.log("Enabling Tailscale (may prompt for sudo/auth)...")
        def _up(): out = self.tailscale.up(); self.log(f"Tailscale up: {out.strip()[:300]}"); self.update_tailscale_status_label()
        threading.Thread(target=_up, daemon=True).start()

    def disable_tailscale(self):
        if not self.tailscale.is_installed():
            QMessageBox.warning(self, "Tailscale missing", "Tailscale is not installed.")
            return
        self.log("Disabling Tailscale...")
        def _down(): out = self.tailscale.down(); self.log(f"Tailscale down: {out.strip()[:300]}"); self.update_tailscale_status_label()
        threading.Thread(target=_down, daemon=True).start()

    def start_remote_stream(self):
        global _FLASK_RUNNING
        if _FLASK_RUNNING:
            self.log("Remote stream already running (http://<IP>:8000/video)")
            return
        try:
            self._flask_thread = threading.Thread(target=start_flask_server, daemon=True)
            self._flask_thread.start()
            self.log("Remote stream started on port 8000 (http://<IP>:8000/video). Use Tailscale IP to view remotely.")
            self.update_tailscale_status_label()
        except Exception as e:
            self.log(f"Error starting remote stream: {e}")

    # theme toggle
    def on_theme_toggled(self, checked: bool):
        if checked:
            self.setStyleSheet(APP_STYLESHEET); self.current_theme = 'dark'
        else:
            self.setStyleSheet(LIGHT_THEME); self.current_theme = 'light'

    # frame update
    def update_frame(self, frame_bgr):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pix)
        except Exception as e:
            self.log(f"Frame update error: {e}")

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S"); self.log_text.append(f"[{ts}] {text}")

    def on_alert_changed(self, active: bool):
        if active:
            self.status_chip.setText('ALERT'); self.status_chip.setStyleSheet("background: #2a0a0a; color: #ffb0b0; padding:6px 12px; border-radius:12px; border:1px solid #5a1a1a;")
            self.alert_pulse_timer.start(350); self.video_frame.setStyleSheet('QFrame#videoFrame { border: 3px solid #a12a2a; }')
        else:
            self.status_chip.setText('NORMAL'); self.status_chip.setStyleSheet(''); self.alert_pulse_timer.stop(); self.alert_bar.setStyleSheet(''); self.setStyleSheet(APP_STYLESHEET if self.current_theme=='dark' else LIGHT_THEME)

    def _pulse_alert_bar(self):
        self._pulse_state = (self._pulse_state + 1) % 4
        if self._pulse_state == 0:
            style = 'background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0f3b21, stop:1 #143e28);'
        elif self._pulse_state == 1:
            style = 'background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #143e28, stop:1 #1aa95d);'
        elif self._pulse_state == 2:
            style = 'background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1aa95d, stop:1 #143e28);'
        else:
            style = 'background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #143e28, stop:1 #0f3b21);'
        self.alert_bar.setStyleSheet(style)

    def closeEvent(self, event):
        try: self.cam_thread.stop()
        except Exception: pass
        try:
            if _HAS_GPIO: GPIO.cleanup()
        except Exception: pass
        event.accept()

# ------------------------- MAIN -------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    # start flask server in background too (optional)
    # threading.Thread(target=start_flask_server, daemon=True).start()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
