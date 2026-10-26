#!/usr/bin/env bash
# PiLENS v2: one-time Raspberry Pi 5 setup (Raspberry Pi OS Bookworm 64-bit).
#   bash scripts/pi_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-flask python3-venv
python3 -m venv --system-site-packages "$HOME/pilens-venv"
# shellcheck disable=SC1091
source "$HOME/pilens-venv/bin/activate"
pip install -r requirements-v2-pi.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> .env bani: email, STREAM_HOST (Tailscale IP) aur camera exposure/gain bharo"
fi

python - <<'PY'
import importlib
for m in ("onnxruntime", "cv2", "numpy", "flask", "picamera2", "gpiozero", "psutil"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  MISSING {m}: {e}")
PY

if ls exports/*.onnx >/dev/null 2>&1; then
  echo ">> exports/ mila: $(ls exports)"
else
  echo ">> exports/ nahi mila: Kaggle ki pilens_outputs.zip se exports/ folder yahan copy karo"
fi
vcgencmd measure_temp || true
echo ">> setup done. Har naye terminal mein: source ~/pilens-venv/bin/activate"
