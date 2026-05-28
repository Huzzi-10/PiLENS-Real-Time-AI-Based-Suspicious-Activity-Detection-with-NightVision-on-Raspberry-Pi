# PiLENS: Raspberry Pi-Based AI Night Vision Surveillance System

*An IoT-based AI surveillance system for real-time suspicious activity detection using YOLO, CNN, and LSTM on Raspberry Pi with night vision support.*

## Overview

PiLENS is a Raspberry Pi-based smart surveillance system designed for real-time suspicious activity detection in low-light and night-time environments. It combines Raspberry Pi edge computing, computer vision, deep learning, infrared imaging, and automated alerts to provide a low-cost intelligent security solution.

Unlike conventional CCTV systems that depend heavily on human monitoring or basic motion detection, PiLENS analyzes video feeds automatically and can alert users when a person, intruder, or suspicious activity is detected. The system is designed for academic institutions, residential areas, warehouses, restricted zones, and research projects.

Key components include:

- **Hardware:** Raspberry Pi 5, Raspberry Pi Camera Module 3 NOIR, IR LEDs, LEDs, buzzer, and optional sensors.
- **Software:** Python, OpenCV, Ultralytics YOLO, CNN, LSTM, PyQt6, Flask, and Tailscale.
- **Detection modes:** Basic person alert, intruder face recognition, and advanced suspicious activity analysis.
- **Storage:** Selective saving of suspicious clips to reduce unnecessary storage usage.
- **Performance target:** 5 FPS processing on Raspberry Pi hardware.

## Technical Highlights

- Edge AI pipeline running on Raspberry Pi
- YOLO-based real-time person detection
- CNN-LSTM suspicious activity classification
- Face recognition mode for authorized/intruder detection
- PyQt6 desktop dashboard for monitoring and logs
- Flask MJPEG stream for remote viewing
- Tailscale support for secure private access
- GPIO-based LED and buzzer alerts
- Environment-based configuration for credentials and model paths

## Demo

Add a short demo video or GIF here before applying for internships. A strong demo should show:

- GUI launch and live camera feed
- Detection event with bounding box or alert state
- Email or local alert behavior
- Saved suspicious clip or screenshot

## Proposed Solution

PiLENS combines hardware and software modules into a portable surveillance platform.

### Hardware Components

- Raspberry Pi 5 as the main processing unit
- Raspberry Pi Camera Module 3 NOIR for night vision
- IR LEDs for low-light illumination
- Red/green LEDs and buzzer for local alerts
- Optional motion, LDR, or microphone sensors for future extensions

### Software and AI Components

- YOLO for real-time person detection
- CNN for visual feature extraction
- LSTM for temporal activity analysis
- OpenCV for video processing
- PyQt6 GUI for local monitoring
- Flask MJPEG stream for remote viewing
- Tailscale for secure remote access

### Core Capabilities

- Detects humans and suspicious behaviors in real time
- Operates in low-light and night conditions
- Sends alerts through email, GUI logs, LEDs, and buzzer
- Saves only suspicious clips for efficient storage
- Supports future expansion to audio detection, cloud storage, and multi-camera monitoring

## Target Users and Applications

- University campuses and dormitories
- Warehouses and storage facilities
- Private homes
- Public or restricted areas requiring intelligent monitoring
- AI and computer vision research projects

## Motivational Scenario

### The Problem: Growing Security Concerns

In institutions such as universities, libraries, laboratories, and campus entry points, traditional surveillance systems require security staff to continuously monitor multiple camera feeds. This approach is tiring, error-prone, and often leads to delayed response.

Security personnel often face challenges such as:

- Continuous monitoring of multiple camera streams
- Blurred or low-visibility feeds during nighttime
- Delayed identification of suspicious activities
- Inability to respond instantly to threats

### Limitations of Manual Monitoring

1. **Time-consuming observation:** Monitoring multiple screens simultaneously is mentally exhausting.
2. **Human error:** Fatigue can cause missed detections.
3. **Delayed response:** Suspicious activity may not be noticed immediately.
4. **Poor night visibility:** Standard cameras struggle in low-light conditions.

### The Need for a Smart Surveillance Solution

An ideal system should:

- Automatically detect unusual activities
- Work efficiently in low-light or night conditions
- Send real-time alerts
- Reduce dependency on constant human supervision
- Improve response time

### Our Solution: PiLENS

PiLENS continuously analyzes live camera feeds, detects suspicious activity, and notifies the concerned authority when a potential threat is identified.

![Motivational Scenario](assets/Motivational_Scenario.jpg)

## Problem Statement

Traditional surveillance systems in universities and public institutions rely heavily on continuous human monitoring of CCTV footage. This manual approach is inefficient, prone to human error, and can result in delayed responses to suspicious activity.

Most existing surveillance systems only record footage. They do not actively identify abnormal behavior or generate intelligent alerts. This creates a gap between incident occurrence and response time.

PiLENS addresses this problem by integrating AI-based detection, night vision hardware, automated alerts, and selective storage into a real-time embedded surveillance system.

### Current Situation

Universities rely on CCTV systems that require constant human supervision.

### Problem

Human monitoring is inefficient and error-prone.

### Gap

Existing systems lack AI-based real-time suspicious activity detection.

### Impact

Delayed detection may lead to security threats and property damage.

### Proposed Need

An automated AI-powered surveillance system is required.

## System Architecture

The PiLENS architecture is divided into three main layers: input, processing, and output.

### Input Layer

The input layer collects video from the Raspberry Pi Camera Module 3 NOIR. IR LEDs support night vision in low-light environments. Optional future inputs may include microphone data for detecting suspicious sounds such as shouting or glass breaking.

Secure remote access can be handled through Tailscale, allowing monitoring from outside the local network without exposing the device directly to the public internet.

### Processing Layer

The Raspberry Pi 5 acts as the central edge device. The system processes video using a multi-stage AI pipeline:

- **Person/Object Detection:** YOLO detects people and generates bounding boxes.
- **Feature Extraction:** CNN-based processing extracts visual features.
- **Temporal Analysis:** LSTM analyzes sequences to identify suspicious behavior.
- **Selective Storage:** Only suspicious clips are saved.
- **Multi-mode Operation:** Users can run different programs depending on the required detection mode.

### Output Layer

When suspicious activity is detected, the system can generate alerts through:

- Email notifications with timestamps, snapshots, and video clips
- GPIO-connected LEDs and buzzer
- GUI logs and intruder file browser
- Flask-based remote MJPEG stream

![Workflow Diagram](assets/Work_Flow.png)

### Circuit Diagram

![Circuit Diagram](assets/Circuit_Digram.jpg)

### Phase-wise Flow

![Phase-wise Flow](assets/Phases_Wise.jpg)

## Detection Modes

### Mode 1: Basic Person Alert

Detects a person and triggers an alert immediately.

![Program 1 Flowchart](assets/Program_1_Flowchart.jpg)

Run:

```bash
python Programs/Program_1.py
```

### Mode 2: Intruder Face Recognition

Uses face recognition to check whether a detected person is authorized.

![Program 2 Flowchart](assets/Program_2_Flowchart.jpg)

Run:

```bash
python Programs/Program_2.py
```

### Mode 3: Advanced Suspicious Activity Detection

Uses YOLO, CNN, and LSTM for suspicious activity analysis.

![Program 3 Flowchart](assets/Program_3_Flowchart.jpg)

Run:

```bash
python Programs/Program_3.py
```

## Key Features

- Multi-mode AI detection
- Night vision support through NOIR camera and IR LEDs
- Secure remote viewing with Tailscale and Flask
- Selective suspicious clip storage
- Email notifications with media attachments
- PyQt6 desktop GUI
- Optional GPIO-based LED and buzzer alerts

## Hardware Requirements

- Raspberry Pi 5, 8 GB RAM recommended
- Raspberry Pi Camera Module 3 NOIR
- IR LEDs or IR illuminator
- Red LED, green LED, and buzzer
- Breadboard and jumper wires
- Optional wired microphone for future audio detection

## Software Requirements

- Python 3.11
- Raspberry Pi OS recommended for camera and GPIO support
- Python packages listed in `requirements.txt`
- Tailscale for secure remote access

Core libraries:

- `ultralytics`
- `picamera2`
- `opencv-python`
- `numpy`
- `PyQt6`
- `Flask`
- `face-recognition`
- `torch`
- `torchvision`
- `RPi.GPIO`

![Feature Image](assets/Featural_Image.png)

## Hardware Gallery

| Prototype | Prototype | Prototype |
| --- | --- | --- |
| ![Hardware 1](assets/Hardware_1.jpeg) | ![Hardware 2](assets/Hardware_2.jpeg) | ![Hardware 3](assets/Hardware_3.jpeg) |
| ![Hardware 4](assets/Hardware_4.jpeg) | ![Hardware 5](assets/Hardware_5.jpg) | ![Hardware 6](assets/Hardware_6.jpeg) |
| ![Hardware 7](assets/Hardware_7.jpeg) | ![Hardware 8](assets/Hardware_8.jpeg) | ![Hardware 9](assets/Hardware_9.jpeg) |
| ![Hardware 10](assets/Hardware_10.jpeg) |  |  |

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi.git
   ```

2. Open the project folder:

   ```bash
   cd PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On Raspberry Pi/Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables:

   ```bash
   copy .env.example .env
   ```

   On Raspberry Pi/Linux:

   ```bash
   cp .env.example .env
   ```

   Update `.env` with your Gmail app password, receiver email, and optional model paths.

6. Check model paths:

   - YOLO model is stored at `Models/yolo11n.pt`.
   - Action model is stored at `Models/action_model.pth`.
   - Update `YOLO_MODEL_PATH` and `CLASSIFIER_MODEL_PATH` in `.env` if needed.

7. Configure email settings:

   Update `SENDER_EMAIL`, `SENDER_PASSWORD`, and `RECEIVER_EMAIL` in `.env`. Use a Gmail app password instead of your normal Gmail password.

8. Start Tailscale on Raspberry Pi if remote access is required:

   ```bash
   sudo tailscale up
   ```

9. Run one of the detection modes:

   ```bash
   python Programs/Program_1.py
   python Programs/Program_2.py
   python Programs/Program_3.py
   ```

## Usage

- Launch the selected program.
- Click **Start Monitoring** in the GUI.
- View logs and saved suspicious clips from the GUI.
- Use the Flask stream for remote viewing:

  ```text
  http://<TAILSCALE_OR_LOCAL_IP>:8000/video
  ```

- Click **Stop Monitoring** to end the session.

## Application Screenshots

| GUI Screenshot | GUI Screenshot | GUI Screenshot |
| --- | --- | --- |
| ![Screenshot 1](assets/SS-1.png) | ![Screenshot 2](assets/SS-2.png) | ![Screenshot 3](assets/SS-3.png) |

## Documentation

- [Hardware Setup](docs/hardware_setup.md)
- [Raspberry Pi Setup](docs/raspberry_pi_setup.md)
- [Model Training](docs/model_training.md)
- [Troubleshooting](docs/troubleshooting.md)

## Known Limitations

- Raspberry Pi camera and GPIO features require Raspberry Pi hardware.
- Email alerts require a valid Gmail app password.
- Face recognition requires authorized face images in the `Authorized` folder.
- Real-time performance depends on lighting, camera resolution, model size, and Raspberry Pi load.
- Program 3 requires the trained `Models/action_model.pth` classifier.

## CV Highlights

- Built an AI-powered Raspberry Pi surveillance system using YOLO, CNN-LSTM, OpenCV, PyQt6, and Flask.
- Implemented real-time person detection, suspicious activity classification, face recognition, email alerts, GPIO alerts, and selective video storage.
- Designed a night vision monitoring workflow with secure remote access through Tailscale and MJPEG streaming.

## Future Work

- Add audio anomaly detection
- Add long-distance wireless support using NRF modules
- Add cloud storage for logs and video clips
- Build a mobile app for live monitoring
- Deploy multiple Raspberry Pi devices for distributed surveillance
- Improve model learning and adaptation over time

## Project Status

This project is under active development. Planned improvements include mobile application support, audio-based suspicious activity detection, and expanded wireless monitoring.
