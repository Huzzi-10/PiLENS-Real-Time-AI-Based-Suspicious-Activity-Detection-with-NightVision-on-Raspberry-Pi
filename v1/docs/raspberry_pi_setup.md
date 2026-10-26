# Raspberry Pi Setup

## Recommended Environment

- Raspberry Pi OS
- Python 3.11
- Camera interface enabled
- Internet access for dependency installation and Tailscale setup

## Setup Steps

1. Clone the repository.
2. Create a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Copy `.env.example` to `.env` and update email settings.
5. Confirm model files exist in the `Models` folder.
6. Run the selected program from the project root.

## Camera Check

Before running PiLENS, verify that the Raspberry Pi camera works with the standard Raspberry Pi camera tools or a minimal Picamera2 script.

## Remote Access

Tailscale can be used to access the Flask MJPEG stream securely from another device:

```bash
sudo tailscale up
```

Then open:

```text
http://<TAILSCALE_OR_LOCAL_IP>:8000/video
```
