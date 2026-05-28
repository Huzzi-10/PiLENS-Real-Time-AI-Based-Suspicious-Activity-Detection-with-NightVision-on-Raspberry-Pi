# Troubleshooting

## Images Do Not Show on GitHub

GitHub paths are case-sensitive. The README uses lowercase `assets/...` paths because the files are tracked by Git under `assets`.

## Email Alerts Do Not Send

- Copy `.env.example` to `.env`.
- Set `SENDER_EMAIL`, `SENDER_PASSWORD`, and `RECEIVER_EMAIL`.
- Use a Gmail app password, not your normal Gmail password.
- Confirm that internet access is available.

## Model Not Found

Confirm these files exist:

```text
Models/yolo11n.pt
Models/action_model.pth
```

If the files are stored somewhere else, update `.env`.

## Raspberry Pi Camera Error

- Confirm the camera is connected correctly.
- Test Picamera2 outside the full application.
- Run on Raspberry Pi OS for best camera support.

## GPIO Not Available

GPIO alerts only work on Raspberry Pi hardware. On non-Raspberry Pi systems, the app can still run without physical LED/buzzer output if the camera dependencies are available.
