# VigilantEye AI-Based Suspicious Activity Detection System
**An IoT-based AI system for real-time suspicious activity detection using YOLO, CNN, LSTM on Raspberry Pi with nightvision camera. Features multi-mode alerts, secure remote access, and selective storage**
</br>
</br>

## Overview
VigilantEye is an AI-powered security system designed for realtime monitoring and detection of suspicious activities in low-light or nighttime environments. Built on Raspberry Pi, it uses computer vision and machine learning models to analyze video feeds from a nightvision camera. The system detects persons or intrusions and triggers alerts, making it ideal for military, border security, or civil applications.

Key components include:
- Hardware: Raspberry Pi 5 (8GB RAM) as the core processor, Raspberry Pi Camera Module 3 (12MP NOIR with IR LEDs for nightvision), LEDs and buzzer for local alerts.
- Software: Ultralytics YOLO for person detection, CNN for feature extraction, LSTM for sequence analysis, and Flask for remote MJPEG streaming.
- Features: Multi-mode detection (basic person alert, intruder face recognition, advanced suspicious analysis), selective storage of suspicious clips, and secure remote access via Tailscale.
- Performance: Runs at 5 FPS to balance heavy ML processing on low-power hardware.

This project aims to provide a low-cost, portable solution for enhanced security, with potential extensions for audio detection and long-range wireless connectivity.

</br>
![Motivational Scenario](Figures/Motivational%20Scenario.jpg)
</br>

## Key Features
- **Multi-Mode Detection**: Three modes for flexible security:
  - Mode 1: Basic person alert with immediate email (time, location, pic/video clip).
  - Mode 2: Intruder detection using face recognition, with red LED and buzzer alerts.
  - Mode 3: Advanced suspicious activity analysis via YOLO + CNN + LSTM.
- **Nightvision Capability**: 12MP NOIR camera with IR LEDs for low-light environments.
- **Secure Remote Access**: Tailscale for hack-proof worldwide viewing, with Flask-based MJPEG streaming[](http://[IP]:8000/video).
- **Selective Storage**: Only saves suspicious clips to optimize storage.
- **Integrated Alerts**: Email notifications with attachments, LED/buzzer feedback, and GUI for monitoring.
- **GUI Interface**: PyQt6-based desktop app for start/stop, logs, intruder folder browser, and theme toggle (dark/light).
- **GPIO Integration**: LED and buzzer control for physical alerts (optional on non-RPi systems).

</br>

## Hardware Requirements
- Raspberry Pi 5 (8GB RAM)
- Raspberry Pi Camera Module 3 (12MP NOIR with IR LEDs + LDR illuminator)
- Red/Green LEDs + Buzzer (GPIO pins: 27 green, 17 red, 18 buzzer)
- Breadboard & Wires for connections
- Optional: Wired microphone for audio extensions

</br>

## Software Requirements
- Python 3.12
- Libraries: ultralytics (YOLO), picamera2, cv2 (OpenCV), numpy, PyQt6, flask, smtplib, RPi.GPIO (if on RPi)
- Install: `pip install ultralytics picamera2 opencv-python numpy pyqt6 flask RPi.GPIO`
- Model: Download "yolo11n.pt" from Ultralytics.

</br>

## Installation & Setup
1. Clone the repository: `git clone https://github.com/Huzzi-10/VigilantEye.git`
2. Navigate to the directory: `cd VigilantEye`
3. Install dependencies: `pip install -r requirements.txt` (or above command).
4. Configure email in code: Edit SENDER_EMAIL, PASSWORD, RECEIVER_EMAIL.
5. For Tailscale: Install Tailscale on RPi, run `sudo tailscale up` for secure IP.
6. Run the program: `python VigilantEye.py`

</br>

## Usage
- Launch the GUI and click "Start Monitoring" to begin detection.
- Person detected: Triggers alerts (email with snapshot/video, LED/buzzer).
- View logs in GUI, open Intruders folder for saved clips.
- Remote view: Use Tailscale IP + port 8000/video in browser.
- Stop monitoring to end the session.

</br>

## Future Work
- Integrate audio detection for suspicious sounds (e.g., glass break).
- Add long-distance wireless with NRF antenna.
- Implement continual learning for model adaptation.
- Deploy on multiple RPi devices for distributed surveillance.