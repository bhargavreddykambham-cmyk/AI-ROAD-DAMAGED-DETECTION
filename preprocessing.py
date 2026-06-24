"""
============================================================
AI Road Damage Detection — preprocessing.py
Image and video preprocessing pipeline
============================================================
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class PreprocessConfig:
    target_size:       Tuple[int,int] = (640, 640)
    enhance_contrast:  bool           = True
    reduce_noise:      bool           = True
    clahe_clip:        float          = 2.0
    clahe_tile:        Tuple[int,int] = (8, 8)
    bilateral_d:       int            = 9
    bilateral_sc:      float          = 75.0
    bilateral_ss:      float          = 75.0


class ImagePreprocessor:
    """Complete image preprocessing for road damage detection."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=self.cfg.clahe_tile
        )
        logger.info(f"Preprocessor ready: target={self.cfg.target_size}")

    def resize(self, img: np.ndarray,
               size: Optional[Tuple[int,int]] = None) -> Tuple[np.ndarray, tuple]:
        """Letterbox resize preserving aspect ratio."""
        size = size or self.cfg.target_size
        h, w = img.shape[:2]
        tw, th = size
        scale  = min(tw/w, th/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
        padded  = np.full((th,tw,3), 114, dtype=np.uint8)
        px, py  = (tw-nw)//2, (th-nh)//2
        padded[py:py+nh, px:px+nw] = resized
        return padded, (scale, px, py)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement on L channel of LAB."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    def reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """Bilateral filter — preserves edges while removing noise."""
        return cv2.bilateralFilter(img, self.cfg.bilateral_d,
                                   self.cfg.bilateral_sc, self.cfg.bilateral_ss)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Full pipeline: denoise → enhance → resize."""
        meta = {"original_shape": img.shape}
        if self.cfg.reduce_noise:
            img = self.reduce_noise(img)
        if self.cfg.enhance_contrast:
            img = self.enhance_contrast(img)
        img, tp = self.resize(img)
        meta.update({"transform": tp, "output_shape": img.shape})
        return img, meta

    def batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.preprocess(im)[0] for im in images]


class VideoFrameExtractor:
    """Extract frames from video files for dataset creation."""

    def __init__(self, fps_target: int = 5, blur_thresh: float = 50.0):
        self.fps_target  = fps_target
        self.blur_thresh = blur_thresh
        self.prep        = ImagePreprocessor()

    def extract(self, video_path: str, output_dir: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        fps      = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps / self.fps_target))
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(output_dir, exist_ok=True)

        saved = 0; fn = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if fn % interval == 0 and not self._blurry(frame):
                processed, _ = self.prep.preprocess(frame)
                cv2.imwrite(os.path.join(output_dir, f"frame_{fn:06d}.jpg"), processed)
                saved += 1
            fn += 1

        cap.release()
        logger.info(f"Extracted {saved} frames from {video_path}")
        return saved

    def _blurry(self, img: np.ndarray) -> bool:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(g, cv2.CV_64F).var() < self.blur_thresh


class DatasetBuilder:
    """Build & validate YOLO dataset structure for road damage."""

    SPLITS = ["train","valid","test"]

    def __init__(self, root: str):
        self.root = Path(root)

    def create_dirs(self):
        for s in self.SPLITS:
            (self.root/s/"images").mkdir(parents=True, exist_ok=True)
            (self.root/s/"labels").mkdir(parents=True, exist_ok=True)
        logger.info("Dataset dirs created")

    def validate(self) -> dict:
        stats = {"splits":{}, "issues":[]}
        for s in self.SPLITS:
            imgs = list((self.root/s/"images").glob("*.jpg")) + \
                   list((self.root/s/"images").glob("*.png"))
            lbls = list((self.root/s/"labels").glob("*.txt"))
            stats["splits"][s] = {"images":len(imgs),"labels":len(lbls)}
            for img in imgs:
                if not (self.root/s/"labels"/(img.stem+".txt")).exists():
                    stats["issues"].append(f"Missing label: {s}/{img.name}")
        return stats

    def generate_synthetic(self, n_per_split: dict = None):
        """Generate synthetic road damage images for demo."""
        n_per_split = n_per_split or {"train":60,"valid":15,"test":10}
        self.create_dirs()
        for split, n in n_per_split.items():
            img_d = self.root/split/"images"
            lbl_d = self.root/split/"labels"
            for i in range(n):
                np.random.seed(i + {"train":0,"valid":5000,"test":9000}[split])
                img = self._make_road_img()
                dets= self._make_labels(img.shape)
                cv2.imwrite(str(img_d/f"road_{split}_{i:04d}.jpg"), img)
                with open(lbl_d/f"road_{split}_{i:04d}.txt","w") as f:
                    f.write("\n".join(dets))
            logger.info(f"{split}: {n} synthetic images")

    def _make_road_img(self) -> np.ndarray:
        img   = np.random.randint(50,85,(640,640,3),dtype=np.uint8)
        noise = np.random.randint(0,18,(640,640,3),dtype=np.uint8)
        img   = cv2.add(img, noise)
        n = np.random.randint(1,6)
        for _ in range(n):
            cls = np.random.randint(0,4)
            x1  = np.random.randint(20,500)
            y1  = np.random.randint(200,560)
            x2  = x1 + np.random.randint(40,180)
            y2  = y1 + np.random.randint(20,90)
            col = [(25,25,25),(20,20,20),(175,165,155),(45,45,45)][cls]
            cv2.rectangle(img,(x1,y1),(min(x2,639),min(y2,639)),col,-1)
        return img

    def _make_labels(self, shape) -> List[str]:
        n = np.random.randint(1,5)
        out = []
        for _ in range(n):
            cls = np.random.randint(0,4)
            cx  = round(np.random.uniform(0.15,0.85),6)
            cy  = round(np.random.uniform(0.40,0.90),6)
            bw  = round(np.random.uniform(0.06,0.28),6)
            bh  = round(np.random.uniform(0.03,0.14),6)
            out.append(f"{cls} {cx} {cy} {bw} {bh}")
        return out


if __name__ == "__main__":
    db = DatasetBuilder("./data")
    db.generate_synthetic()
    stats = db.validate()
    print("\nDataset validation:")
    for s,v in stats["splits"].items():
        print(f"  {s}: images={v['images']}, labels={v['labels']}")
    if stats["issues"]:
        print(f"  Issues: {len(stats['issues'])}")