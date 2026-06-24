"""
============================================================
AI Road Damage Detection — train.py
Train YOLOv8 on road damage dataset (potholes, cracks, patches)
Usage: python train.py --epochs 100 --model yolov8n.pt
============================================================
"""

import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path

import yaml
from loguru import logger

logger.remove()
logger.add("logs/training.log", rotation="10 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add(sys.stderr, level="INFO")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 Road Damage Detector")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="Base model: yolov8n/s/m/l/x.pt")
    p.add_argument("--data",    default="data.yaml")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--lr0",     type=float, default=0.01)
    p.add_argument("--device",  default="auto",
                   help="auto | cpu | cuda | 0 | 0,1")
    p.add_argument("--project", default="outputs/runs")
    p.add_argument("--name",    default="road_damage_yolov8")
    p.add_argument("--resume",  action="store_true")
    return p.parse_args()


def detect_device(device_arg: str) -> str:
    import torch
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.info("No GPU found — using CPU")
    return "cpu"


def validate_data(yaml_path: str):
    """Check dataset exists; generate synthetic demo data if not."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("path", "."))
    train_imgs = data_root / cfg.get("train", "train/images")

    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        logger.warning(f"No training images found at {train_imgs}")
        logger.info("Generating synthetic demo dataset…")
        _make_synthetic_dataset(data_root)
    else:
        n = len(list(train_imgs.glob("*.jpg")) + list(train_imgs.glob("*.png")))
        logger.info(f"Found {n} training images")


def _make_synthetic_dataset(root: Path):
    """Generate simple synthetic road damage images for demo."""
    import numpy as np
    import cv2

    for split, n in [("train",60),("valid",15),("test",10)]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            np.random.seed(i * 100 + {"train":0,"valid":1,"test":2}[split])
            # Road-like grey image
            img = np.random.randint(55, 90, (640, 640, 3), dtype=np.uint8)
            noise = np.random.randint(0, 15, (640, 640, 3), dtype=np.uint8)
            img  = cv2.add(img, noise)

            labels = []
            n_obj = np.random.randint(1, 5)
            for _ in range(n_obj):
                cls = np.random.randint(0, 4)
                cx  = np.random.uniform(0.15, 0.85)
                cy  = np.random.uniform(0.40, 0.90)   # road zone
                bw  = np.random.uniform(0.08, 0.30)
                bh  = np.random.uniform(0.04, 0.15)
                # Draw damage region
                x1 = int((cx - bw/2) * 640)
                y1 = int((cy - bh/2) * 640)
                x2 = int((cx + bw/2) * 640)
                y2 = int((cy + bh/2) * 640)
                colors = [(30,30,30),(20,20,20),(180,170,160),(50,50,50)]
                cv2.rectangle(img,(x1,y1),(x2,y2),colors[cls],-1)
                labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            cv2.imwrite(str(img_dir / f"road_{split}_{i:04d}.jpg"), img)
            with open(lbl_dir / f"road_{split}_{i:04d}.txt","w") as f:
                f.write("\n".join(labels))

    logger.info("Synthetic dataset generated")


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.project, exist_ok=True)

    validate_data(args.data)
    device = detect_device(args.device)

    logger.info("=" * 55)
    logger.info("YOLOv8 Road Damage Training")
    logger.info("=" * 55)
    logger.info(f"  Model  : {args.model}")
    logger.info(f"  Data   : {args.data}")
    logger.info(f"  Epochs : {args.epochs}")
    logger.info(f"  Batch  : {args.batch}")
    logger.info(f"  ImgSz  : {args.imgsz}")
    logger.info(f"  Device : {device}")

    model = YOLO(args.model)
    t0    = time.time()

    results = model.train(
        data          = args.data,
        epochs        = args.epochs,
        batch         = args.batch,
        imgsz         = args.imgsz,
        lr0           = args.lr0,
        device        = device,
        project       = args.project,
        name          = args.name,
        exist_ok      = True,
        resume        = args.resume,
        patience      = 30,
        save          = True,
        plots         = True,
        verbose       = True,
        # Road-damage specific augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 3.0,
        translate     = 0.1,
        scale         = 0.5,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.10,
        close_mosaic  = 10,
    )

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/60:.1f} min")

    # Copy best model
    best_pt_candidates = [
        Path(f"{args.project}/{args.name}/weights/best.pt"),
        Path(f"runs/detect/{args.name}/weights/best.pt"),
    ]
    for cand in best_pt_candidates:
        if cand.exists():
            dest = Path("models/best_road_damage.pt")
            shutil.copy2(cand, dest)
            logger.info(f"Best model → {dest}")
            break

    # Save training report
    try:
        rd = results.results_dict
        report = {
            "mAP50":     round(float(rd.get("metrics/mAP50(B)",     0)), 4),
            "mAP50_95":  round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
            "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
            "recall":    round(float(rd.get("metrics/recall(B)",    0)), 4),
            "training_time_min": round(elapsed / 60, 2),
        }
    except Exception:
        report = {"training_time_min": round(elapsed / 60, 2)}

    report_path = Path("outputs/training_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved: {report_path}")
    print("\n" + "="*55)
    print("TRAINING COMPLETE")
    print("="*55)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("="*55)
    return report


if __name__ == "__main__":
    args = parse_args()
    train(args)