# Hardware Setup

This document summarizes the recommended hardware setup for PiLENS.

## Required Components

- Raspberry Pi 5, 8 GB RAM recommended
- Raspberry Pi Camera Module 3 NOIR
- IR LEDs or IR illuminator for night vision
- Red LED, green LED, and buzzer
- Breadboard and jumper wires
- Stable 5V power supply for Raspberry Pi

## GPIO Pins

| Component | GPIO Pin |
| --- | --- |
| Green LED | GPIO 27 |
| Red LED | GPIO 17 |
| Buzzer | GPIO 18 |

## Notes

- Use suitable resistors with LEDs.
- Confirm GPIO pin numbering uses BCM mode.
- Keep IR LEDs powered safely and avoid drawing too much current from Raspberry Pi GPIO pins.
- Test the camera separately before running the full AI pipeline.
