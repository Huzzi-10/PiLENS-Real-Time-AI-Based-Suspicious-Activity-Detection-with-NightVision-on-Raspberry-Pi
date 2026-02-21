# PiLENS: Raspberry Pi–Based Deep Learning Enhanced Nightvision Surveillance System
*An IoT-based AI system for real-time suspicious activity detection using YOLO, CNN, LSTM on Raspberry Pi with nightvision camera. Features multi-mode alerts, secure remote access, and selective storage*
</br>
</br>

## Overview

PiLENS is a Raspberry Pi-based deep learning enhanced night vision surveillance system I developed to provide intelligent, real-time suspicious activity detection in low-light and night-time environments. It combines the edge computing power of the Raspberry Pi with advanced computer vision and deep learning algorithms to create a smart, cost-effective security solution. Unlike conventional CCTV systems that rely on human monitoring or basic motion detection, PiLENS uses AI to identify potentially suspicious behavior with better accuracy and fewer false alarms. By integrating infrared imaging and night vision enhancement, it works reliably even in tough lighting conditions where regular cameras struggle.
<br>
The main goal of PiLENS is to make high-tech surveillance affordable and accessible for places like academic institutions, residential areas, warehouses, and other security-sensitive spots. It bridges the gap between cost and intelligence by offering portable, scalable monitoring that minimizes manual effort. With real-time video processing, automated detection, and alert generation, it boosts situational awareness. Its modular design allows for expansions like audio anomaly detection, wireless modules, cloud integration, and multi-camera support. Overall, PiLENS is a practical step toward smarter surveillance that's both user-friendly and technologically solid.
<br>
What makes PiLENS stand out is its multi-mode detection system, where it handles basic person alerts, intruder face recognition, and advanced suspicious activity analysis all in one. It also has selective storage to save only suspicious clips, saving space, and secure remote access via Tailscale for hack-proof viewing from anywhere. Running at 5 FPS on low-cost hardware, it's optimized for efficiency without compromising performance.
</br>
Key components include:
- Hardware: Raspberry Pi 5 (8GB RAM) as the core processor, Raspberry Pi Camera Module 3 (12MP NOIR with IR LEDs for nightvision), LEDs and buzzer for local alerts.
- Software: Ultralytics YOLO for person detection, CNN for feature extraction, LSTM for sequence analysis, and Flask for remote MJPEG streaming.
- Features: Multi-mode detection (basic person alert, intruder face recognition, advanced suspicious analysis), selective storage of suspicious clips, and secure remote access via Tailscale.
- Performance: Runs at 5 FPS to balance heavy ML processing on low-power hardware.

This project aims to provide a low-cost, portable solution for enhanced security, with potential extensions for audio detection and long-range wireless connectivity.



### Proposed Solution:

PiLENS combines hardware and software solutions:

**1. Hardware Components:**

- Raspberry Pi 5 as the main processing unit
- IR camera for night vision
- IR LEDs for low-light illumination
- Optional motion or LDR sensors for enhanced detection

**2. Software & AI Components:**

- Deep Learning models (e.g., YOLO, CNN, LSTM) for real-time suspicious activity detection
- Python and OpenCV for video processing and preprocessing
- Optional Flask/Dashboard for remote monitoring
- Alert system via buzzer, email, or mobile notification

**3. Key Capabilities:**

- Detects humans and suspicious behaviors in real-time
- Operates reliably in low-light/night conditions
- Minimal false alarms due to AI-based detection
- Portable, low-cost setup suitable for universities, offices, homes, and restricted areas


### Core Features:

- Real-time AI-based detection of suspicious activities
- Night vision enabled through IR camera and LEDs
- Low-cost, compact, and portable design
- Automated alert system (optional integration with mobile or cloud)
- Expandable to multi-camera and wireless setups
- Easy setup and lightweight software requirements


### Target Users / Applications:

- University campuses and dormitories
- Warehouses and storage facilities
- Private homes for enhanced security
- Public areas requiring intelligent night surveillance
- Experimental research setups in AI and computer vision


### Future Extensions / Scalability:

- Integration of audio anomaly detection for enhanced security
- Long-range wireless connectivity for remote monitoring
- Cloud storage for video logs and alerts
- Mobile applications for live monitoring
- Multi-camera AI fusion for comprehensive coverage
- Real-time analytics dashboards


### Key Takeaways:

PiLENS is not just a surveillance tool; it is a smart, AI-powered security system designed to provide reliable detection, low-cost implementation, and scalability. Its combination of Raspberry Pi hardware, deep learning models, and night vision makes it ideal for modern security challenges, research applications, and real-life deployment scenarios.

</br>

## Motivational Scenario
### The Problem: Growing Security Concerns

In modern institutional environments such as universities, libraries, laboratories, and campus entry points, security threats are becoming increasingly complex. Traditional surveillance systems rely heavily on human operators to continuously monitor multiple camera feeds. However, manual monitoring is prone to human fatigue, delayed reaction time, and oversight errors.

- Security personnel often face challenges such as:
- Continuous monitoring of multiple camera streams
- Blurred or low-visibility feeds during nighttime
- Delayed identification of suspicious activities
- Inability to respond instantly to threats

As a result, many incidents are detected too late, reducing the overall effectiveness of surveillance infrastructure.

### Limitations of Manual Monitoring

Manual surveillance systems suffer from several practical limitations:

1. Time-Consuming Observation – Monitoring multiple screens simultaneously is mentally exhausting.
2. Human Error – Fatigue can cause missed detections.
3. Delayed Response – Suspicious activity may not be noticed immediately.
4. Poor Night Visibility – Standard cameras struggle in low-light conditions.

These limitations clearly indicate the need for a smarter and more automated surveillance approach.


### The Need for a Smart Surveillance Solution

With the advancement of Artificial Intelligence and embedded systems, there is a strong need to shift from passive monitoring to intelligent monitoring.

An ideal system should:
- Automatically detect unusual activities
- Operate efficiently in low-light or night conditions
- Send real-time alerts
- Reduce dependency on constant human supervision
- Provide faster response time

This gap between traditional systems and intelligent automation forms the core motivation behind PiLENS.


### Our Solution: PiLENS (Vigilant Eye)

PiLENS is an AI-powered smart surveillance system designed to enhance security monitoring using modern embedded hardware and intelligent algorithms.

The system integrates:
- Infrared (IR) Camera for night vision capability
- Raspberry Pi 5 as the central processing unit
- AI-based detection algorithms
- Real-time alert mechanism via Wi-Fi
- Automated threat notification system

Unlike traditional systems, PiLENS does not rely solely on human observation. Instead, it continuously analyzes live feeds, detects anomalies, and instantly notifies the concerned authority when suspicious activity is identified.


![Motivational Scenario](Figures/Motivational_Scenario.jpg)

</br>

## Problem Statement

Traditional surveillance systems in universities and public institutions rely heavily on continuous human monitoring of CCTV footage. This manual monitoring approach is inefficient, prone to human error, and often results in delayed response to suspicious activities. Security personnel cannot effectively observe multiple camera feeds simultaneously for extended periods, which increases the risk of unnoticed incidents.</br>

Furthermore, most existing surveillance systems lack intelligent behavior analysis and automated alert mechanisms. They only record footage without actively identifying abnormal or suspicious patterns. This creates a significant gap between incident occurrence and response time.</br>

There is a need for an intelligent, automated, and real-time surveillance system that can monitor environments continuously, detect suspicious activities using AI techniques, and instantly alert authorities. The system should also function effectively during low-light or nighttime conditions.</br>

VigilantEye aims to address these challenges by integrating artificial intelligence with hardware-based surveillance to provide real-time detection, automated alerts, and enhanced security monitoring.</br>

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

<br>

## System Architecture

The VigilantEye system is designed as an integrated IoT-based surveillance platform that combines hardware components, software modules, and AI-driven processing to enable real-time suspicious activity detection. The architecture is modular, allowing for scalability and future expansions such as audio integration or long-range wireless connectivity. It leverages the edge computing capabilities of the Raspberry Pi to perform on-device analysis, reducing latency and enhancing privacy by minimizing data transmission to external servers. The system is divided into three main layers: Input Layer, Processing Layer, and Output Layer.

### Input Layer

This layer handles data acquisition from IoT devices and sensors. The primary input is the video stream from the Raspberry Pi Camera Module 3 (12MP NOIR with IR LEDs and LDR illuminator), which enables nightvision functionality for low-light environments. The camera captures high-resolution footage at 5 FPS to balance performance and resource usage on the Raspberry Pi 5 (8GB RAM). Additional inputs include potential microphone data for audio detection (e.g., suspicious sounds like glass breaking or shouting), though this is planned for future extensions. Secure remote access is facilitated through Tailscale, which provides a VPN-like IP for worldwide monitoring without compromising security. All inputs are fed directly to the core processing unit, ensuring realtime data flow.

### Processing Layer

At the heart of the system is the Raspberry Pi 5, serving as the central microcontroller and edge device. It processes the input streams using a multi-stage AI pipeline:

- Person/Object Detection: Utilizes the Ultralytics YOLO model to identify persons and generate bounding boxes in the video feed, with a confidence threshold of 0.60 to reduce false positives.
- Feature Extraction and Analysis: The bounding box data is passed to a Convolutional Neural Network (CNN) for feature extraction (e.g., motion patterns or unusual behaviors). This is followed by Long Short-Term Memory (LSTM) networks via PyTorch for temporal sequence analysis, determining if the activity is normal or suspicious.
- Multi-Mode Operation: The system operates in three modes:
  - Mode 1 (Basic Person Alert): Immediate detection of any person triggers an alert without deep analysis.
  - Mode 2 (Intruder Face Recognition): Uses face recognition to check if the person is authorized; unauthorized triggers specific alerts.
  - Mode 3 (Advanced Suspicious Detection): Full YOLO-CNN-LSTM pipeline for detailed behavior analysis.
- Selective Storage: Only suspicious clips are saved to the "Intruders" folder, optimizing storage on the high-resolution camera output.
- The processing is optimized for 5 FPS to prevent overload on the low-power Raspberry Pi, with all computations performed on-device for edge efficiency.

### Output Layer

This layer manages alerts and user interaction. Upon detection, the system generates multi-modal alerts:

- Email notifications with timestamps, location details, attached snapshots, and video clips (using smtplib for async sending).
- Local indicators via GPIO-connected red/green LEDs and buzzer for immediate feedback (e.g., red LED and buzzer for intruders).
- Voice TTS alerts (using pyttsx3 for spoken warnings).
- The PyQt6-based desktop GUI provides a user interface for starting/stopping monitoring, viewing logs, browsing intruder files, toggling themes (dark/light), and controlling Tailscale/Flask for remote MJPEG streaming (accessible at http://[IP]:8000/video).
- Logging and notifications are stored securely, with options for future cloud integration.

The architecture emphasizes low-cost, portability, and security, making it suitable for military or civil applications. It uses Flask for streaming and Tailscale for encrypted remote access, ensuring hack-proof operation. Future enhancements include audio anomaly detection and NRF-based long-distance wireless connectivity to extend the system's range.

![Work_Flow_Digram](Figures/Work_Flow.png)




## Key Features
- **Multi-Mode Detection**: Three modes for flexible security:
  - Mode 1: Basic person alert with immediate email (time, location, pic/video clip).
  - Mode 2: Intruder detection using face recognition, with red LED and buzzer alerts.
  - Mode 3: Advanced suspicious activity analysis via YOLO + CNN + LSTM.
</br>

![Program_1_Flowchart](Figures/Program_1_Flowchart.jpg)

### This is the Mode 1 

<br>

![Program_2_Flowchart](Figures/Program_2_Flowchart.jpg)

### This is the Mode 2

<br>


- **Nightvision Capability**: 12MP NOIR camera with IR LEDs for low-light environments.
- **Secure Remote Access**: Tailscale for hack-proof worldwide viewing, with Flask-based MJPEG streaming[](http://[IP]:8000/video).
- **Selective Storage**: Only saves suspicious clips to optimize storage.
- **Integrated Alerts**: Email notifications with attachments, LED/buzzer feedback, and GUI for monitoring.
- **GUI Interface**: PyQt6-based desktop app for start/stop, logs, intruder folder browser, and theme toggle (dark/light).
- **GPIO Integration**: LED and buzzer control for physical alerts (optional on non-RPi systems).

</br>



</br>

## Hardware Requirements
- Raspberry Pi 5 (8GB RAM)
- Raspberry Pi Camera Module 3 (12MP NOIR with IR LEDs + LDR illuminator)
- Red/Green LEDs + Buzzer (GPIO pins: 27 green, 17 red, 18 buzzer)
- Breadboard & Wires for connections
- Optional: Wired microphone for audio extensions

</br>

## Software Requirements
- Python 3.11
- Libraries: ultralytics (YOLO), picamera2, cv2 (OpenCV), numpy, PyQt6, flask, smtplib, RPi.GPIO (if on RPi)
- Install: `pip install ultralytics picamera2 opencv-python numpy pyqt6 flask RPi.GPIO`
- Model: Download "yolo11n.pt" from Ultralytics.

</br>

![Featural_Image](Figures/Featural_Image.png)

</br>

## Installation & Setup
1. Clone the repository: `git clone https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi.git`
2. Navigate to the directory: `cd [Folder Name]`
3. Install dependencies: `pip install -r requirements.txt` (or above command).
4. Configure email in code: Edit SENDER_EMAIL, PASSWORD, RECEIVER_EMAIL.
5. For Tailscale: Install Tailscale on RPi, run `sudo tailscale up` for secure IP.
6. Run the program: `python [Program_File].py`

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

This Project is also under development on new features like Mobile Application Development and secondly audio detection...