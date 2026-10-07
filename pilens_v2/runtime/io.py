# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Hardware / network side of the runtime: settings, camera, GPIO, clip
saving, email and the MJPEG stream. All of it degrades gracefully off-Pi."""

import os
import smtplib
import threading
import time
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- settings (.env, same file and keys as v1) ----------------------------------

def _load_env(path=PROJECT_ROOT / ".env"):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setting(name, default=""):
    _load_env()
    return os.getenv(name, default)


# --- frame sources ----------------------------------------------------------------

class PiCamera:
    """Picamera2 with FIXED exposure and gain (auto-exposure causes flicker)."""

    def __init__(self, size=(640, 480), fps=30, exposure_us=None, gain=None):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": tuple(size), "format": "RGB888"},  # RGB888 = BGR byte order (OpenCV)
            controls={"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(0.5)
        exposure_us = exposure_us or int(setting("CAMERA_EXPOSURE_US", "20000"))
        gain = gain or float(setting("CAMERA_GAIN", "4.0"))
        self.cam.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us), "AnalogueGain": float(gain)})
        self.fps = fps

    def read(self):
        return True, self.cam.capture_array("main")

    def release(self):
        self.cam.stop()


class VideoSource:
    """Video file or USB webcam via OpenCV. realtime=True paces a file at its fps."""

    def __init__(self, src, realtime=True):
        import cv2
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
        if not self.cap.isOpened():
            raise IOError(f"cannot open {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_file = not str(src).isdigit()
        self.realtime = realtime
        self._t0, self._n = None, 0

    def read(self):
        ok, frame = self.cap.read()
        if ok and self.is_file and self.realtime:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            self._n += 1
            delay = self._t0 + self._n / self.fps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        return ok, frame

    def release(self):
        self.cap.release()


# --- GPIO ------------------------------------------------------------------------

class Signals:
    """LED + buzzer. gpiozero (Pi 5) -> RPi.GPIO (rpi-lgpio) -> no-op."""

    def __init__(self, led_pin=17, buzzer_pin=18):
        self.led = self.buzzer = None
        self._gpio = None
        try:
            from gpiozero import LED, Buzzer
            self.led, self.buzzer = LED(led_pin), Buzzer(buzzer_pin)
        except Exception:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(led_pin, GPIO.OUT)
                GPIO.setup(buzzer_pin, GPIO.OUT)
                self._gpio, self._pins = GPIO, (led_pin, buzzer_pin)
            except Exception:
                pass

    @property
    def available(self):
        return bool(self.led or self._gpio)

    def set(self, on):
        if self.led:
            (self.led.on if on else self.led.off)()
            (self.buzzer.on if on else self.buzzer.off)()
        elif self._gpio:
            for p in self._pins:
                self._gpio.output(p, self._gpio.HIGH if on else self._gpio.LOW)

    def pulse(self, seconds):
        self.set(True)
        time.sleep(seconds)
        self.set(False)


# --- clip saving + email -----------------------------------------------------------

def save_clip(frames_with_t, folder, label):
    """Write (t, frame) pairs to an mp4 at their measured fps; returns the path."""
    import cv2
    if not frames_with_t:
        return None
    Path(folder).mkdir(parents=True, exist_ok=True)
    ts = [t for t, _ in frames_with_t]
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 15.0
    h, w = frames_with_t[0][1].shape[:2]
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)
    path = str(Path(folder) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{safe}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, f in frames_with_t:
        writer.write(f)
    writer.release()
    return path


def send_email(video_path, label, details):
    sender, password, receiver = setting("SENDER_EMAIL"), setting("SENDER_PASSWORD"), setting("RECEIVER_EMAIL")
    if not all([sender, password, receiver]):
        return "skipped: set SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL in .env"
    msg = MIMEMultipart()
    msg["From"], msg["To"] = sender, receiver
    msg["Subject"] = f"PiLENS Alert - {label}"
    msg.attach(MIMEText(f"Suspicious activity detected.\n\nTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                        f"Activity: {label}\n{details}", "plain"))
    if video_path and Path(video_path).exists():
        part = MIMEBase("application", "octet-stream")
        part.set_payload(Path(video_path).read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={Path(video_path).name}")
        msg.attach(part)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    return "sent"


# --- MJPEG stream -------------------------------------------------------------

def start_stream(get_jpeg, host=None, port=None):
    """Flask MJPEG on /video. Default host 127.0.0.1: set STREAM_HOST to the Pi's
    Tailscale IP (100.x.y.z) for remote viewing; never expose it publicly."""
    from flask import Flask, Response
    host = host or setting("STREAM_HOST", "127.0.0.1")
    port = int(port or setting("STREAM_PORT", "8000"))
    app = Flask("pilens_v2")

    @app.route("/video")
    def video():
        def gen():
            while True:
                jpg = get_jpeg()
                if jpg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1 / 15)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    th = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, threaded=True,
                                                 use_reloader=False), daemon=True)
    th.start()
    return f"http://{host}:{port}/video"
