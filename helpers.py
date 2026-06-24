"""
============================================================
AI Road Damage Detection — utils/helpers.py
Common helper functions
============================================================
"""

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_image(path: str) -> Optional[np.ndarray]:
    """Load image from path, return BGR numpy array or None."""
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot load image: {path}")
    return img


def save_image(img: np.ndarray, path: str, quality: int = 95) -> bool:
    """Save BGR numpy array to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_json(data: dict, path: str) -> None:
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    """Load JSON file as dictionary."""
    with open(path) as f:
        return json.load(f)


def timestamp() -> str:
    """Return current timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def get_video_info(path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(path)
    info = {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) /
                       max(cap.get(cv2.CAP_PROP_FPS),1)),
    }
    cap.release()
    return info


def draw_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: Tuple[int,int],
    font_scale: float = 0.6,
    color: Tuple[int,int,int] = (255,255,255),
    bg_color: Tuple[int,int,int] = (0,0,0),
    thickness: int = 1,
    padding: int = 4,
) -> np.ndarray:
    """Draw text with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img,
                  (x - padding, y - th - bl - padding),
                  (x + tw + padding, y + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y - bl), font, font_scale, color,
                thickness, cv2.LINE_AA)
    return img


def resize_keep_aspect(
    img: np.ndarray,
    max_size: int = 640
) -> np.ndarray:
    """Resize image keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_size/w, max_size/h)
    return cv2.resize(img, (int(w*scale), int(h*scale)))


def count_files(directory: str, extensions: List[str] = None) -> int:
    """Count files in directory with given extensions."""
    p = Path(directory)
    if not p.exists():
        return 0
    if extensions is None:
        return len(list(p.iterdir()))
    return sum(len(list(p.glob(f"*{ext}"))) for ext in extensions)


def format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"