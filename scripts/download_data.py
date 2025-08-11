#!/usr/bin/env python
import os
import sys
import hashlib
import urllib.request
from pathlib import Path

DATA_URL = os.environ.get("KINDLE_DATA_URL", "")  # set your dataset URL here
TARGET_PATH = Path("data/raw/kindle_reviews.csv")
TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATA_URL:
    print("[WARN] KINDLE_DATA_URL not set. Provide URL via env var.")
    sys.exit(0)

if TARGET_PATH.exists():
    size = TARGET_PATH.stat().st_size
    print(f"[INFO] File already exists: {TARGET_PATH} ({size/1024/1024:.2f} MB)")
    sys.exit(0)

print(f"[INFO] Downloading from {DATA_URL} -> {TARGET_PATH}")
try:
    with urllib.request.urlopen(DATA_URL) as resp, open(TARGET_PATH, 'wb') as f:
        f.write(resp.read())
    print("[OK] Download complete")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)
