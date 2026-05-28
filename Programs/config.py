# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path=PROJECT_ROOT / ".env"):
    """Load simple KEY=VALUE entries from .env without requiring extra packages."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_setting(name, default=""):
    load_env_file()
    return os.getenv(name, default)


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))
