# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Raspberry Pi health readings (temperature, clock, throttling) and versions.

Every reading degrades to None off-Pi, so the same scripts run on a laptop.
"""

import os
import platform
import subprocess
import threading
import time
from pathlib import Path


def cpu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return None


def cpu_freq_mhz():
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) / 1000.0
    except Exception:
        return None


def throttled():
    """`vcgencmd get_throttled` as an int (0 = never throttled), None if unavailable."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return None


def cpu_ram():
    try:
        import psutil
        return psutil.cpu_percent(interval=None), psutil.virtual_memory().used / 2**20
    except Exception:
        return None, None


def system_info(threads=None):
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "kernel": platform.release(), "cpu_count": os.cpu_count(), "ort_threads": threads}
    for key, path in (("board", "/proc/device-tree/model"),):
        try:
            info[key] = Path(path).read_text().strip("\x00\n ")
        except Exception:
            info[key] = None
    try:
        rel = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        info["os"] = rel.get("PRETTY_NAME", "").strip('"')
    except Exception:
        info["os"] = None
    for mod in ("onnxruntime", "numpy", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


class HealthLogger(threading.Thread):
    """Samples temperature / clock / CPU / RAM every ``period`` s in the background."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._halt = threading.Event()
        self.t0 = time.time()

    def run(self):
        cpu_ram()  # prime psutil's cpu_percent
        while not self._halt.is_set():
            cpu, ram = cpu_ram()
            self.samples.append({"t": round(time.time() - self.t0, 2), "temp_c": cpu_temp_c(),
                                 "freq_mhz": cpu_freq_mhz(), "cpu_pct": cpu, "ram_mb": ram})
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def summary(self):
        def col(k):
            return [s[k] for s in self.samples if s[k] is not None]
        out = {}
        for k in ("temp_c", "freq_mhz", "cpu_pct", "ram_mb"):
            v = col(k)
            if v:
                out[k] = {"start": v[0], "end": v[-1], "min": min(v), "max": max(v), "mean": sum(v) / len(v)}
        out["throttled_flags"] = throttled()
        return out
