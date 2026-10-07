# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""PiLENS v2: X3D-S two-stage cascade (binary MIL + 14-class) for Raspberry Pi 5.

v1 (ResNet18 + LSTM, Programs/Program_3.py) is frozen as the baseline.
Every module here reads its constants from ``pilens_v2.spec`` so training,
export and the Pi runtime always use the same clip and preprocessing.
"""

__version__ = "2.0.0.dev0"
