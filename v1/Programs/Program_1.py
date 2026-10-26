# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0
# See LICENSE file in project root for full license information.


import sys, os, time, threading, subprocess
from datetime import datetime
from pathlib import Path

# ML / Camera / CV
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QListWidget, QListWidgetItem, QMessageBox, QDialog, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen

# Flask for remote MJPEG stream
from flask import Flask, Response

# GPIO (optional)
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

# Email libs
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from config import get_setting, project_path

# ---------------- CONFIG (edit as needed) ----------------
SENDER_EMAIL = get_setting("SENDER_EMAIL")
SENDER_PASSWORD = get_setting("SENDER_PASSWORD")
RECEIVER_EMAIL = get_setting("RECEIVER_EMAIL")

INTRUDER_FOLDER = "Intruders"
MODEL_PATH = get_setting("YOLO_MODEL_PATH", project_path("Models", "yolo11n.pt"))
FRAME_SIZE = (1280, 720)
FPS = 20

MIN_RECORD_SECONDS = 5.0
SNAPSHOT_DELAY = 2.0
NO_PERSON_TIMEOUT = 1.5
CONF_THRESHOLD = 0.60

# GPIO pins
GREEN_LED = 27
RED_LED = 17
BUZZER_PIN = 18

Path(INTRUDER_FOLDER).mkdir(parents=True, exist_ok=True)

# Setup GPIO safely
if _HAS_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GREEN_LED, GPIO.OUT)
    GPIO.setup(RED_LED, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(GREEN_LED, GPIO.HIGH)
    GPIO.output(RED_LED, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# ---------------- THEMES / STYLES ----------------
APP_STYLESHEET = """
/* SECURITY SYSTEM PRO THEME */
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
DARK_THEME = APP_STYLESHEET
LIGHT_THEME = """
QWidget { background-color: #e8e8e8; color: #111111; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
QFrame#videoFrame { background-color: #f0f0f0; border: 2px solid #ccc; border-radius: 10px; }
QLabel#videoLabel { border-radius: 8px; }
QFrame#alertBar { background: #88c999; border-radius: 6px; min-height: 14px; }
QPushButton { background-color: #d9d9d9; border: 2px solid #bbb; padding: 10px 16px; border-radius: 10px; color: #111; font-weight: 500; }
QPushButton#startBtn { background-color: #4caf50; border: 2px solid #388e3c; color: #fff; }
QPushButton#stopBtn { background-color: #f44336; border: 2px solid #d32f2f; color: #fff; }
QTextEdit, QListWidget { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 8px; padding: 8px; color: #111; }
QLabel#statusNormal { background: #c3e6cb; color: #155724; padding: 6px 12px; border-radius: 12px; border: 1px solid #8fd19e; }
QLabel#statusAlert { background: #f8d7da; color: #721c24; padding: 6px 12px; border-radius: 12px; border: 1px solid #f5c6cb; }
"""

# ---------------- Email helper ----------------
def send_email_async(subject, body, attachment_path):
    def _send():
        try:
            if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
                print("[Email] Skipped: set SENDER_EMAIL, SENDER_PASSWORD, and RECEIVER_EMAIL in .env")
                return

            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            if attachment_path and os.path.exists(attachment_path):
                part = MIMEBase('application', 'octet-stream')
                with open(attachment_path, 'rb') as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
                msg.attach(part)
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            server.quit()
            print("[Email] Sent")
        except Exception as e:
            print("[Email] Error:", e)
    threading.Thread(target=_send, daemon=True).start()

# ---------------- Video writer helper ----------------
def start_video_writer(path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)

# ---------------- Shared MJPEG stream state ----------------
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
    # run Flask without the reloader
    flask_app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)

# ---------------- Custom ToggleSwitch Widget (Option A) ----------------
class ToggleSwitch(QWidget):
    toggled = pyqtSignal(bool)
    def __init__(self, parent=None, width=60, height=32):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._checked = True
        self._offset = 1.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(180)
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

# ---------------- Camera / Detection Thread ----------------
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)
    alert_changed = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._model = None
        self._picam2 = None
        self.event_active = False
        self.event_start_time = None
        self.video_writer = None
        self.video_path = None
        self.snapshot_taken = False
        self.email_sent_for_snapshot = False
        self.last_person_seen_time = None
    def setup(self):
        try:
            self._model = YOLO(MODEL_PATH)
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(main={"format":"RGB888","size":FRAME_SIZE})
            self._picam2.configure(config)
            self._picam2.start()
            time.sleep(0.4)
            self.log.emit("Model and camera initialized")
        except Exception as e:
            self.log.emit(f"Error initializing model/camera: {e}")
    def run(self):
        global latest_stream_frame, _stream_lock
        self._running = True
        self.setup()
        while self._running:
            try:
                frame = self._picam2.capture_array()            # RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                now = time.time()
                # detection
                results = self._model(frame_bgr, stream=True)
                person_in_frame = False
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0]); conf = float(box.conf[0])
                        if cls == 0 and conf >= CONF_THRESHOLD:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            if (x2-x1)<40 or (y2-y1)<40: continue
                            cv2.rectangle(frame_bgr, (x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(frame_bgr, f"Person {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            person_in_frame = True
                # person present handling
                if person_in_frame:
                    self.last_person_seen_time = now
                    if not self.event_active:
                        self.event_active = True
                        self.event_start_time = now
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        self.video_path = os.path.join(INTRUDER_FOLDER, f"intruder_{timestamp}.avi")
                        self.video_writer = start_video_writer(self.video_path)
                        self.snapshot_taken = False
                        self.email_sent_for_snapshot = False
                        self.log.emit("Intruder detected — recording started")
                        self.alert_changed.emit(True)
                    # GPIO indicators
                    try:
                        if _HAS_GPIO:
                            GPIO.output(GREEN_LED, GPIO.LOW); GPIO.output(RED_LED, GPIO.HIGH); GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    except Exception:
                        pass
                    if self.video_writer:
                        self.video_writer.write(cv2.resize(frame_bgr, FRAME_SIZE))
                    if not self.snapshot_taken and (now - self.event_start_time >= SNAPSHOT_DELAY):
                        snap_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                        snapshot_path = os.path.join(INTRUDER_FOLDER, snap_name)
                        cv2.imwrite(snapshot_path, frame_bgr)
                        self.snapshot_taken = True
                        self.log.emit(f"Snapshot saved: {snapshot_path}")
                        if not self.email_sent_for_snapshot:
                            send_email_async("⚠ VigilantEye Alert", "Intruder detected. Snapshot attached.", snapshot_path)
                            self.email_sent_for_snapshot = True
                else:
                    if self.last_person_seen_time and (now - self.last_person_seen_time >= NO_PERSON_TIMEOUT):
                        if self.event_active:
                            rec_time = now - self.event_start_time
                            if rec_time < MIN_RECORD_SECONDS: time.sleep(MIN_RECORD_SECONDS - rec_time)
                            self.event_active = False
                            if self.video_writer:
                                self.video_writer.release(); self.video_writer = None
                                self.log.emit(f"Recording saved: {self.video_path}")
                            self.alert_changed.emit(False)
                        try:
                            if _HAS_GPIO:
                                GPIO.output(GREEN_LED, GPIO.HIGH); GPIO.output(RED_LED, GPIO.LOW); GPIO.output(BUZZER_PIN, GPIO.LOW)
                        except Exception:
                            pass
                # prepare MJPEG frame for remote stream (thread-safe)
                try:
                    ret, jpeg = cv2.imencode('.jpg', frame_bgr)
                    if ret:
                        with _stream_lock:
                            latest_stream_frame = jpeg.tobytes()
                except Exception:
                    pass
                # emit frame to GUI
                self.frame_ready.emit(frame_bgr)
                time.sleep(0.01)
            except Exception as e:
                self.log.emit(f"Runtime error: {e}")
                time.sleep(0.5)
        # cleanup
        try:
            if self._picam2: self._picam2.stop()
            if self.video_writer: self.video_writer.release()
        except Exception:
            pass
        self.log.emit("Camera thread stopped")
    def stop(self):
        self._running = False
        self.wait()

# ---------------- Intruder Browser Dialog ----------------
class IntruderListDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intruders")
        self.resize(820,420)
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Refresh"); open_btn = QPushButton("Open Selected"); remove_btn = QPushButton("Delete Selected")
        btn_layout.addWidget(load_btn); btn_layout.addWidget(open_btn); btn_layout.addWidget(remove_btn)
        layout.addWidget(self.list_widget); layout.addLayout(btn_layout); self.setLayout(layout)
        load_btn.clicked.connect(self.load_files); open_btn.clicked.connect(self.open_selected); remove_btn.clicked.connect(self.delete_selected)
        self.load_files()
    def load_files(self):
        self.list_widget.clear()
        files = sorted(os.listdir(INTRUDER_FOLDER), reverse=True)
        for f in files: self.list_widget.addItem(QListWidgetItem(f))
    def open_selected(self):
        sel = self.list_widget.currentItem()
        if not sel: return
        path = os.path.join(INTRUDER_FOLDER, sel.text())
        try:
            if sys.platform.startswith('linux'): subprocess.Popen(['xdg-open', path])
            elif sys.platform == 'darwin': subprocess.Popen(['open', path])
            elif sys.platform == 'win32': os.startfile(path)
        except Exception as e: QMessageBox.warning(self, "Error", f"Could not open file: {e}")
    def delete_selected(self):
        sel = self.list_widget.currentItem()
        if not sel: return
        path = os.path.join(INTRUDER_FOLDER, sel.text())
        try: os.remove(path); self.load_files()
        except Exception as e: QMessageBox.warning(self, "Error", f"Could not delete file: {e}")

# ---------------- TAILSCALE MANAGER (simple wrapper) ----------------
class TailscaleManager:
    def is_installed(self):
        try:
            subprocess.run(["tailscale", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False
    def status(self):
        try:
            result = subprocess.run(["tailscale", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
            return result.stdout.strip()
        except Exception as e:
            return f"Error getting status: {e}"
    def up(self):
        try:
            result = subprocess.run(["sudo", "tailscale", "up"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout + ("\n" + result.stderr if result.stderr else "")
        except Exception as e:
            return f"Error starting tailscale: {e}"
    def down(self):
        try:
            result = subprocess.run(["sudo", "tailscale", "down"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout + ("\n" + result.stderr if result.stderr else "")
        except Exception as e:
            return f"Error stopping tailscale: {e}"

# ---------------- Main Window (merged) ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VigilantEye — GUI (Remote Enabled)")
        self.resize(1150, 760)
        self.setStyleSheet(APP_STYLESHEET)
        self.current_theme = 'dark'
        # Left video frame + alert bar
        self.video_frame = QFrame(); self.video_frame.setObjectName('videoFrame')
        video_layout = QVBoxLayout(self.video_frame)
        self.video_label = QLabel(); self.video_label.setObjectName('videoLabel')
        self.video_label.setMinimumSize(980,580); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        self.alert_bar = QFrame(); self.alert_bar.setObjectName('alertBar'); self.alert_bar.setFixedHeight(16)
        video_layout.addWidget(self.alert_bar)
        # Right controls
        self.start_btn = QPushButton("Start Monitoring"); self.start_btn.setObjectName('startBtn')
        self.stop_btn  = QPushButton("Stop Monitoring"); self.stop_btn.setObjectName('stopBtn')
        self.open_folder_btn = QPushButton("Open Intruders Folder"); self.view_intruders_btn = QPushButton("View Intruders")
        self.toggle_buzzer_btn = QPushButton("Toggle Buzzer (Test)")
        self.status_label = QLabel("Status: "); self.status_chip = QLabel("NORMAL"); self.status_chip.setObjectName('statusNormal')
        # Theme switch
        self.theme_switch = ToggleSwitch(); self.theme_switch.setChecked(True); self.theme_switch.toggled.connect(self.on_theme_toggled)
        # Remote access / Tailscale UI
        self.tailscale = TailscaleManager()
        self.ts_label = QLabel("Remote Access (Tailscale / MJPEG):")
        self.ts_status = QLabel(self._tailscale_status_short())
        self.ts_enable_btn = QPushButton("Enable Tailscale"); self.ts_disable_btn = QPushButton("Disable Tailscale")
        self.ts_enable_btn.setStyleSheet("background-color:#0a66c2; color:white; padding:8px; border-radius:8px;")
        self.ts_disable_btn.setStyleSheet("background-color:#c21807; color:white; padding:8px; border-radius:8px;")
        # Remote stream control
        self.remote_btn = QPushButton("Start Remote Stream (http://<IP>:8000/video)")
        self.remote_btn.setStyleSheet("background-color:#00695c; color:white; padding:8px; border-radius:8px;")
        # logs
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        # layout right
        right_layout = QVBoxLayout()
        for w in [self.start_btn,self.stop_btn,self.open_folder_btn,self.view_intruders_btn,self.toggle_buzzer_btn]:
            right_layout.addWidget(w)
        right_layout.addSpacing(8)
        theme_row = QHBoxLayout(); theme_row.addWidget(QLabel("Theme:")); theme_row.addWidget(self.theme_switch); theme_row.addStretch()
        right_layout.addLayout(theme_row); right_layout.addSpacing(8)
        # tailscale section
        right_layout.addWidget(self.ts_label)
        ts_row = QHBoxLayout(); ts_row.addWidget(self.ts_enable_btn); ts_row.addWidget(self.ts_disable_btn)
        right_layout.addLayout(ts_row); right_layout.addWidget(self.ts_status)
        right_layout.addWidget(self.remote_btn); right_layout.addSpacing(8)
        # status row
        status_layout = QHBoxLayout(); status_layout.addWidget(self.status_label); status_layout.addWidget(self.status_chip); status_layout.addStretch()
        right_layout.addLayout(status_layout); right_layout.addSpacing(6)
        right_layout.addWidget(QLabel("Logs:")); right_layout.addWidget(self.log_text)
        main_layout = QHBoxLayout(); main_layout.addWidget(self.video_frame, stretch=3); main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)
        # camera thread
        self.cam_thread = CameraThread(); self.cam_thread.frame_ready.connect(self.update_frame); self.cam_thread.log.connect(self.log); self.cam_thread.alert_changed.connect(self.on_alert_changed)
        # signals
        self.start_btn.clicked.connect(self.start_monitoring); self.stop_btn.clicked.connect(self.stop_monitoring)
        self.open_folder_btn.clicked.connect(self.open_intruder_folder); self.view_intruders_btn.clicked.connect(self.view_intruders)
        self.toggle_buzzer_btn.clicked.connect(self.toggle_buzzer)
        self.ts_enable_btn.clicked.connect(self.enable_tailscale); self.ts_disable_btn.clicked.connect(self.disable_tailscale)
        self.remote_btn.clicked.connect(self.start_remote_stream)
        # initial tailscale state
        if not self.tailscale.is_installed():
            self.ts_status.setText("Tailscale: NOT INSTALLED"); self.ts_enable_btn.setEnabled(False); self.ts_disable_btn.setEnabled(False)
        else:
            self.update_tailscale_status_label()
        # UI initial state
        self.stop_btn.setEnabled(False)
        self.alert_pulse_timer = QTimer(); self.alert_pulse_timer.timeout.connect(self._pulse_alert_bar); self._pulse_state = 0
        # flask thread placeholder
        self._flask_thread = None
    # --- actions ---
    def start_monitoring(self):
        if not Path(MODEL_PATH).exists():
            QMessageBox.warning(self, "Model not found", f"Model file not found at {MODEL_PATH}")
            return
        self.log("Starting monitoring...")
        self.cam_thread.start(); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
    def stop_monitoring(self):
        self.log("Stopping monitoring..."); self.cam_thread.stop(); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.on_alert_changed(False)
    def open_intruder_folder(self):
        try:
            p = os.path.abspath(INTRUDER_FOLDER)
            if sys.platform.startswith('linux'): subprocess.Popen(['xdg-open', p])
            elif sys.platform == 'darwin': subprocess.Popen(['open', p])
            elif sys.platform == 'win32': os.startfile(p)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    def view_intruders(self):
        dlg = IntruderListDialog(self); dlg.exec()
    def toggle_buzzer(self):
        try:
            if _HAS_GPIO:
                GPIO.output(BUZZER_PIN, GPIO.HIGH); time.sleep(0.35); GPIO.output(BUZZER_PIN, GPIO.LOW); self.log("Buzzer toggled")
            else:
                self.log("GPIO not available on this system")
        except Exception as e:
            self.log(f"GPIO error: {e}")
    # Tailscale helpers
    def _tailscale_status_short(self):
        if not self.tailscale.is_installed(): return "Tailscale: NOT INSTALLED"
        try:
            s = self.tailscale.status(); first_line = s.splitlines()[0] if s else "No status"
            return f"Tailscale: {first_line}"
        except Exception: return "Tailscale: Error"
    def update_tailscale_status_label(self):
        try: self.ts_status.setText(self._tailscale_status_short())
        except Exception: self.ts_status.setText("Tailscale: Error")
    def enable_tailscale(self):
        if not self.tailscale.is_installed(): QMessageBox.warning(self, "Tailscale missing", "Tailscale is not installed."); return
        self.log("Enabling Tailscale (may prompt for sudo/auth)...")
        def _up(): out = self.tailscale.up(); self.log(f"Tailscale up: {out.strip()[:300]}"); self.update_tailscale_status_label()
        threading.Thread(target=_up, daemon=True).start()
    def disable_tailscale(self):
        if not self.tailscale.is_installed(): QMessageBox.warning(self, "Tailscale missing", "Tailscale is not installed."); return
        self.log("Disabling Tailscale...")
        def _down(): out = self.tailscale.down(); self.log(f"Tailscale down: {out.strip()[:300]}"); self.update_tailscale_status_label()
        threading.Thread(target=_down, daemon=True).start()
    # Remote stream control
    def start_remote_stream(self):
        # starts Flask MJPEG server in background (if not already)
        global _FLASK_RUNNING
        if _FLASK_RUNNING:
            self.log("Remote stream already running (http://<IP>:8000/video)")
            return
        try:
            self._flask_thread = threading.Thread(target=start_flask_server, daemon=True)
            self._flask_thread.start()
            self.log("Remote stream started on port 8000 (http://<IP>:8000/video). Use Tailscale IP to view remotely.")
            # update status label if tailscale installed
            self.update_tailscale_status_label()
        except Exception as e:
            self.log(f"Error starting remote stream: {e}")
    # Theme switch
    def on_theme_toggled(self, checked: bool):
        if checked:
            self.setStyleSheet(DARK_THEME); self.current_theme = 'dark'
        else:
            self.setStyleSheet(LIGHT_THEME); self.current_theme = 'light'
    # display frame
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
            self.status_chip.setText('NORMAL'); self.status_chip.setStyleSheet(''); self.alert_pulse_timer.stop(); self.alert_bar.setStyleSheet(''); self.setStyleSheet(DARK_THEME if self.current_theme=='dark' else LIGHT_THEME)
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

# ---------------- MAIN ----------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
