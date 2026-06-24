"""
============================================================
AI Road Damage Detection — predict.py
Run inference on image, video, or camera stream
Usage:
  python predict.py --source road.jpg
  python predict.py --source road_video.mp4 --save
  python predict.py --source 0  (webcam)
============================================================
"""

import os
import cv2
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

from loguru import logger

logger.remove()
logger.add("logs/predict.log", rotation="5 MB", level="INFO")
logger.add(sys.stderr, level="INFO")


# ─────────────────────────────────────────────────────────
# Road Damage Detector (OpenCV — no vehicle detection)
# ─────────────────────────────────────────────────────────
class RoadDamageDetector:
    COLOURS = {
        "Pothole":     (0,   0,   255),
        "Crack":       (0,   165, 255),
        "Patch":       (0,   200, 255),
        "Road Damage": (128,  0,  128),
    }

    def __init__(self, conf: float = 0.30, yolo_model: Optional[str] = None):
        self.conf = conf
        self.yolo = None
        if yolo_model and Path(yolo_model).exists():
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_model)
                logger.info(f"YOLO model loaded: {yolo_model}")
            except ImportError:
                logger.warning("ultralytics not installed — falling back to OpenCV detector")

    def detect(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        if self.yolo:
            return self._yolo_detect(bgr)
        return self._cv_detect(bgr)

    # ── OpenCV detection ─────────────────────────────────
    def _cv_detect(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        annotated = bgr.copy()
        h, w = bgr.shape[:2]
        road_y  = int(h * 0.28)
        roi     = bgr[road_y:, :]

        dets = (self._potholes(roi, road_y)
              + self._cracks(roi, road_y)
              + self._patches(roi, road_y))
        dets = self._nms(dets)

        for d in dets:
            self._draw(annotated, d)
        annotated = self._panel(annotated, dets, w, h)
        return annotated, dets

    def _potholes(self, roi, yo):
        dets = []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11,11), 0)
        _, dark = cv2.threshold(blur, 68, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        for c in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 1200: continue
            x,y,bw,bh = cv2.boundingRect(c)
            if bw/max(bh,1) > 4.5 or y < 5: continue
            cf = min(0.96, 0.42 + a/75000)
            if cf < self.conf: continue
            dets.append(dict(label="Pothole",conf=cf,x1=x,y1=y+yo,x2=x+bw,y2=y+bh+yo,area=int(a)))
        return dets

    def _cracks(self, roi, yo):
        dets = []
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 35, 105)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
        for c in cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 250: continue
            x,y,bw,bh = cv2.boundingRect(c)
            if max(bw,bh)/max(min(bw,bh),1) < 2.5 or y < 5: continue
            cf = min(0.93, 0.38 + a/48000)
            if cf < self.conf: continue
            p=4
            dets.append(dict(label="Crack",conf=cf,x1=max(0,x-p),y1=max(0,y+yo-p),
                             x2=x+bw+p,y2=y+bh+yo+p,area=int(a)))
        return dets

    def _patches(self, roi, yo):
        dets = []
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv,(0,0,100),(30,50,200)),
            cv2.inRange(hsv,(0,0,180),(180,30,255))
        )
        k    = cv2.getStructuringElement(cv2.MORPH_RECT,(20,10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
               cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 2200: continue
            x,y,bw,bh = cv2.boundingRect(c)
            asp = bw/max(bh,1)
            if asp>6 or asp<0.3 or y<5: continue
            cf = min(0.89, 0.36 + a/88000)
            if cf < self.conf: continue
            dets.append(dict(label="Patch",conf=cf,x1=x,y1=y+yo,x2=x+bw,y2=y+bh+yo,area=int(a)))
        return dets

    # ── YOLO detection (if model provided) ───────────────
    def _yolo_detect(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        CLASS_MAP = {0:"Pothole",1:"Crack",2:"Patch",3:"Road Damage"}
        results = self.yolo.predict(bgr, conf=self.conf, verbose=False)
        r       = results[0]
        annotated = r.plot()
        dets = []
        if r.boxes is not None:
            for i in range(len(r.boxes)):
                cls  = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])
                x1,y1,x2,y2 = map(int, r.boxes.xyxy[i].tolist())
                dets.append(dict(label=CLASS_MAP.get(cls,f"class_{cls}"),
                                 conf=conf,x1=x1,y1=y1,x2=x2,y2=y2,
                                 area=(x2-x1)*(y2-y1)))
        return annotated, dets

    @staticmethod
    def _nms(dets, iou=0.38):
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        kept = []
        for d in dets:
            if not any(RoadDamageDetector._iou(d,k)>iou for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _iou(a,b):
        ix1=max(a["x1"],b["x1"]); iy1=max(a["y1"],b["y1"])
        ix2=min(a["x2"],b["x2"]); iy2=min(a["y2"],b["y2"])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(a["x2"]-a["x1"])*(a["y2"]-a["y1"])
        ub=(b["x2"]-b["x1"])*(b["y2"]-b["y1"])
        return inter/max(ua+ub-inter,1)

    def _draw(self, img, det):
        x1,y1,x2,y2=det["x1"],det["y1"],det["x2"],det["y2"]
        col=self.COLOURS.get(det["label"],(0,255,0))
        lbl=f"{det['label']}  {det['conf']:.2f}"
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        f=cv2.FONT_HERSHEY_SIMPLEX; fs=0.56
        (tw,th),bl=cv2.getTextSize(lbl,f,fs,1)
        cv2.rectangle(img,(x1,y1-th-bl-4),(x1+tw+4,y1),col,-1)
        cv2.putText(img,lbl,(x1+2,y1-bl-2),f,fs,(255,255,255),1,cv2.LINE_AA)

    def _panel(self, img, dets, w, h):
        ov=img.copy(); cv2.rectangle(ov,(0,0),(w,110),(0,0,0),-1)
        cv2.addWeighted(ov,0.65,img,0.35,0,img)
        counts={}
        for d in dets: counts[d["label"]]=counts.get(d["label"],0)+1
        n=len(dets)
        sc=(0,255,100) if n==0 else (0,255,100) if n<=3 else (0,165,255) if n<=7 else (0,0,255)
        sev="GOOD" if n==0 else "LOW" if n<=3 else "MODERATE" if n<=7 else "HIGH"
        f=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,"ROAD DAMAGE ANALYSIS",(10,27),f,0.74,(255,255,255),2)
        cv2.putText(img,f"Severity: {sev}",(10,56),f,0.63,sc,1)
        cv2.putText(img,f"Detections: {n}",(10,80),f,0.56,(200,200,200),1)
        s="  |  ".join([f"{k}: {v}" for k,v in counts.items()]) or "No damage"
        cv2.putText(img,s,(10,102),f,0.46,(160,160,160),1)
        return img


# ─────────────────────────────────────────────────────────
# Inference runners
# ─────────────────────────────────────────────────────────
def run_image(path: str, detector: RoadDamageDetector, save: bool, output_dir: str):
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Cannot read image: {path}")
        return

    t0 = time.time()
    ann, dets = detector.detect(img)
    ms = (time.time()-t0)*1000

    counts = {}
    for d in dets: counts[d["label"]] = counts.get(d["label"],0)+1

    print(f"\n{'='*50}")
    print(f"  File     : {path}")
    print(f"  Detected : {len(dets)}")
    print(f"  Time     : {ms:.0f} ms")
    for k,v in counts.items():
        print(f"    {k}: {v}")
    print(f"{'='*50}")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_img = os.path.join(output_dir, Path(path).stem + "_detected.jpg")
        cv2.imwrite(out_img, ann)

        out_json = os.path.join(output_dir, Path(path).stem + "_result.json")
        with open(out_json,"w") as f:
            json.dump({"detections":[{k:v for k,v in d.items() if k!="area"}
                        for d in dets], "counts":counts, "ms":round(ms,1)}, f, indent=2)
        logger.info(f"Saved: {out_img}")
        logger.info(f"JSON : {out_json}")

    cv2.imshow("Road Damage Detection", ann)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path: str, detector: RoadDamageDetector, save: bool,
              output_dir: str, skip: int = 1, display: bool = True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {w}x{h} @ {fps:.0f}fps, {tot} frames")

    writer = None
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_v = os.path.join(output_dir, Path(path).stem + "_detected.mp4")
        writer = cv2.VideoWriter(out_v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    agg = {}; fn = 0; fps_timer = time.time(); fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if fn % skip == 0:
            ann, dets = detector.detect(frame)
            for d in dets: agg[d["label"]] = agg.get(d["label"],0)+1
        else:
            ann = frame

        fn += 1
        if fn % 30 == 0:
            fps_display = 30 / (time.time()-fps_timer); fps_timer = time.time()
        cv2.putText(ann,f"FPS:{fps_display:.0f} Frame:{fn}/{tot}",
                    (w-220,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(200,200,200),1)

        if writer: writer.write(ann)
        if display:
            cv2.imshow("Road Damage — Video", ann)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if writer: writer.release()
    if display: cv2.destroyAllWindows()

    print(f"\n{'='*50}\nVIDEO SUMMARY\n{'='*50}")
    print(f"  Frames   : {fn}")
    for k,v in agg.items(): print(f"  {k}: {v}")
    print(f"{'='*50}")


def run_realtime(source: int, detector: RoadDamageDetector):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {source}")
        return
    logger.info(f"Real-time detection on camera {source} — press Q to quit")

    fps_t = time.time(); fps_n = 0; fps = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        ann, _ = detector.detect(frame)
        fps_n += 1
        if fps_n % 30 == 0: fps = 30/(time.time()-fps_t); fps_t = time.time()
        cv2.putText(ann,f"FPS: {fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Road Damage — Real-Time", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Road Damage Inference")
    p.add_argument("--source",  required=True, help="Image path, video path, or camera index")
    p.add_argument("--conf",    type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--model",   default=None, help="Path to trained .pt YOLO model (optional)")
    p.add_argument("--save",    action="store_true", default=True, help="Save annotated output")
    p.add_argument("--output",  default="outputs/detections", help="Output directory")
    p.add_argument("--skip",    type=int, default=1, help="Process every N frames (video)")
    p.add_argument("--no-display", action="store_true", help="Skip display window")
    args = p.parse_args()

    detector = RoadDamageDetector(conf=args.conf, yolo_model=args.model)

    src = args.source
    display = not args.no_display

    if src.isdigit():
        run_realtime(int(src), detector)
    elif Path(src).is_file():
        ext = Path(src).suffix.lower()
        if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
            run_image(src, detector, args.save, args.output)
        elif ext in [".mp4",".avi",".mov",".mkv"]:
            run_video(src, detector, args.save, args.output, args.skip, display)
        else:
            print(f"Unsupported file type: {ext}")
    elif Path(src).is_dir():
        imgs = list(Path(src).glob("*.jpg")) + list(Path(src).glob("*.png"))
        logger.info(f"Processing {len(imgs)} images from {src}")
        for img_path in imgs:
            run_image(str(img_path), detector, args.save, args.output)
    else:
        print(f"Source not found: {src}")


if __name__ == "__main__":
    main()